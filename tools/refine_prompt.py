#!/usr/bin/env python3
"""
Extract rules from PR data using LLM and generate patches.

Usage:
    python tools/refine_prompt.py --sample-size 5
"""

import json
import argparse
import os
import difflib
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


def log(message: str) -> None:
    """Print log message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def load_training_data(input_path: str) -> list[dict]:
    """Load PR data from JSONL file."""
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def sample_prs(data: list[dict], sample_size: int) -> list[dict]:
    """Sample recent PRs, preferring those with substantive review feedback."""
    # Sort by merged_at descending (most recent first)
    sorted_data = sorted(data, key=lambda x: x.get("merged_at", ""), reverse=True)

    # Prioritize PRs with review comments or detailed reviews
    def has_feedback(pr: dict) -> bool:
        has_review_comments = len(pr.get("review_comments", [])) > 0
        has_substantive_reviews = any(
            len(r.get("body", "")) > 20
            for r in pr.get("reviews", [])
        )
        has_comments = any(
            len(c.get("body", "")) > 50
            for c in pr.get("comments", [])
        )
        return has_review_comments or has_substantive_reviews or has_comments

    with_feedback = [pr for pr in sorted_data if has_feedback(pr)]
    without_feedback = [pr for pr in sorted_data if not has_feedback(pr)]

    # Prefer PRs with feedback
    selected = with_feedback[:sample_size]
    if len(selected) < sample_size:
        remaining = sample_size - len(selected)
        selected.extend(without_feedback[:remaining])

    return selected[:sample_size]


def format_pr_for_llm(pr: dict) -> str:
    """Format PR data for LLM input."""
    parts = []

    parts.append(f"## PR #{pr['pr_number']}: {pr['title']}")
    parts.append(f"URL: {pr['url']}")
    parts.append(f"Changes: +{pr['additions']}/-{pr['deletions']} in {pr['files_changed']} files")

    if pr.get("body"):
        body_text = pr["body"][:1000]  # Truncate long descriptions
        parts.append(f"\n### Description:\n{body_text}")

    # Include linked issues
    if pr.get("linked_issues"):
        parts.append("\n### Linked Issues:")
        for issue in pr["linked_issues"]:
            parts.append(f"- #{issue['number']}: {issue['title']}")
            if issue.get("labels"):
                parts.append(f"  Labels: {', '.join(issue['labels'])}")

    # Include reviews with substantive feedback
    if pr.get("reviews"):
        reviews_with_content = [r for r in pr["reviews"] if len(r.get("body", "")) > 10]
        if reviews_with_content:
            parts.append("\n### Review Feedback:")
            for review in reviews_with_content:
                parts.append(f"- [{review['state']}] @{review['author']}: {review['body']}")

    # Include issue comments
    if pr.get("comments"):
        comments_with_content = [c for c in pr["comments"] if len(c.get("body", "")) > 20]
        if comments_with_content:
            parts.append("\n### Discussion Comments:")
            for comment in comments_with_content[:5]:  # Limit to 5 comments
                body_text = comment["body"][:500]
                parts.append(f"- @{comment['author']}: {body_text}")

    # Include inline code review comments
    if pr.get("review_comments"):
        parts.append("\n### Inline Code Review Comments:")
        for comment in pr["review_comments"][:10]:  # Limit to 10 inline comments
            body_text = comment["body"][:300]
            parts.append(f"- @{comment['author']} on `{comment['path']}`:")
            parts.append(f"  {body_text}")
            if comment.get("diff_hunk"):
                hunk_text = comment["diff_hunk"][:200]
                parts.append(f"  ```\n{hunk_text}\n  ```")

    # Include a summary of the diff (not full diff to save tokens)
    if pr.get("diff"):
        diff_preview = pr["diff"][:2000]  # First 2000 chars
        parts.append(f"\n### Diff Preview:\n```diff\n{diff_preview}\n```")

    return "\n".join(parts)


def create_extraction_prompt(prs_text: str) -> str:
    """Create the prompt for rule extraction."""
    return f"""あなたはシニアソフトウェアエンジニアとして、PRレビューのフィードバックを分析し、一般化されたコーディングルールとベストプラクティスを抽出します。

以下は複数のマージ済みPRのデータです：
- PRの説明とタイトル
- レビューフィードバックとコメント
- インラインコードレビューコメント（コードコンテキスト付き）
- チームメンバー間のディスカッション

あなたのタスク：
1. レビューフィードバックとディスカッションを分析する
2. 繰り返し出現するパターン、提案、修正を特定する
3. 将来のコードレビューに役立つ一般化されたルールを抽出する
4. 具体的で実行可能なガイドラインに焦点を当てる（曖昧なアドバイスは避ける）

出力形式：
- Markdown形式で明確なセクションに分けて記述する
- カテゴリ別にルールをグループ化する（例：コードスタイル、アーキテクチャ、テスト、ドキュメントなど）
- 各ルールは簡潔かつ具体的に記述する
- 必要に応じて簡単な理由を含める
- 複数回出現するルールや重要な問題に対応するルールを優先する

**重要：すべての出力は日本語で記述してください。**

---

## PRデータ:

{prs_text}

---

一般化されたコーディングルールとベストプラクティスを日本語で抽出してください："""


def call_openai_api(prompt: str, model: str = "gpt-4") -> str:
    """Call OpenAI API to extract rules."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=api_key)

    log(f"Calling OpenAI API ({model})...")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "あなたはコードレビューを分析し、実行可能なコーディングガイドラインを抽出する専門家です。すべての出力は日本語で記述してください。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,  # Lower temperature for more consistent output
        max_tokens=4000
    )

    return response.choices[0].message.content


def generate_unified_diff(original: str, updated: str, from_file: str, to_file: str) -> str:
    """Generate unified diff between original and updated content."""
    original_lines = original.splitlines(keepends=True)
    updated_lines = updated.splitlines(keepends=True)

    # Ensure lines end with newlines for proper diff output
    if original_lines and not original_lines[-1].endswith('\n'):
        original_lines[-1] += '\n'
    if updated_lines and not updated_lines[-1].endswith('\n'):
        updated_lines[-1] += '\n'

    diff = difflib.unified_diff(
        original_lines,
        updated_lines,
        fromfile=from_file,
        tofile=to_file
    )

    return "".join(diff)


def merge_rules_into_prompt(original_prompt: str, new_rules: str) -> str:
    """Merge extracted rules into the existing prompt file."""
    timestamp_marker = "<!-- Auto-generated rules section -->"

    if timestamp_marker in original_prompt:
        # Replace existing auto-generated section
        parts = original_prompt.split(timestamp_marker)
        return parts[0].rstrip() + "\n\n" + timestamp_marker + "\n\n" + new_rules
    else:
        # Append to end
        return original_prompt.rstrip() + "\n\n" + timestamp_marker + "\n\n" + new_rules


def main():
    parser = argparse.ArgumentParser(description="Extract rules from PR data and generate patches")
    parser.add_argument("--input", default="data/training.jsonl", help="Input JSONL file")
    parser.add_argument("--prompt-file", default="prompts/review_rules.md", help="Prompt file to update")
    parser.add_argument("--output-rules", default="reports/prompt_update.md", help="Output rules file")
    parser.add_argument("--output-patch", default="reports/prompt_patch.diff", help="Output patch file")
    parser.add_argument("--sample-size", type=int, default=5, help="Number of PRs to sample")
    parser.add_argument("--model", default="gpt-4", help="OpenAI model")
    args = parser.parse_args()

    # Ensure output directories exist
    Path(args.output_rules).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_patch).parent.mkdir(parents=True, exist_ok=True)

    # Task overview
    log("=" * 60)
    log("Task: Extract rules from PR data and generate patches")
    log(f"  Input: {args.input}")
    log(f"  Sample size: {args.sample_size}")
    log(f"  Model: {args.model}")
    log(f"  Output rules: {args.output_rules}")
    log(f"  Output patch: {args.output_patch}")
    log("=" * 60)

    # Step 1: Load training data
    log("[Step 1/5] Loading training data...")
    data = load_training_data(args.input)
    log(f"[Step 1/5] Loaded {len(data)} PRs")

    if not data:
        log("Error: No PR data found. Run collect.py first.")
        return

    # Step 2: Sample PRs
    log("[Step 2/5] Sampling PRs with feedback...")
    sampled = sample_prs(data, args.sample_size)
    log(f"[Step 2/5] Sampled {len(sampled)} PRs for analysis")

    # Step 3: Call OpenAI API
    log("[Step 3/5] Extracting rules via LLM...")
    prs_text = "\n\n---\n\n".join(format_pr_for_llm(pr) for pr in sampled)
    extraction_prompt = create_extraction_prompt(prs_text)
    extracted_rules = call_openai_api(extraction_prompt, args.model)
    log("[Step 3/5] Rules extracted successfully")

    # Step 4: Save extracted rules
    log("[Step 4/5] Saving extracted rules...")
    with open(args.output_rules, "w", encoding="utf-8") as f:
        f.write(extracted_rules)
        # Ensure file ends with newline
        if not extracted_rules.endswith("\n"):
            f.write("\n")
    log(f"[Step 4/5] Saved to {args.output_rules}")

    # Step 5: Generate patch
    log("[Step 5/5] Generating patch...")
    prompt_path = Path(args.prompt_file)
    if prompt_path.exists():
        original_prompt = prompt_path.read_text(encoding="utf-8")
    else:
        original_prompt = "# Code Review Rules\n\nThis file contains coding guidelines extracted from PR reviews.\n"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(original_prompt, encoding="utf-8")

    updated_prompt = merge_rules_into_prompt(original_prompt, extracted_rules)
    patch = generate_unified_diff(
        original_prompt,
        updated_prompt,
        f"a/{args.prompt_file}",
        f"b/{args.prompt_file}"
    )

    with open(args.output_patch, "w", encoding="utf-8") as f:
        f.write(patch)
    log(f"[Step 5/5] Saved to {args.output_patch}")

    # Summary
    log("=" * 60)
    log("Summary:")
    log(f"  PRs analyzed: {len(sampled)}")
    log(f"  Rules extracted: {args.output_rules}")
    log(f"  Patch generated: {args.output_patch}")
    log("")
    log("To apply the patch:")
    log(f"  patch -p1 < {args.output_patch}")
    log("=" * 60)


if __name__ == "__main__":
    main()
