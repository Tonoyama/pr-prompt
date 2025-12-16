#!/usr/bin/env python3
"""
Collect PR data from GitHub using the gh CLI.

Usage:
    python tools/collect.py --repo owner/repo --limit 10
"""

import subprocess
import json
import argparse
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


def log(message: str) -> None:
    """Print log message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def run_gh_command(args: list[str], retries: int = 3) -> str:
    """Execute gh CLI command and return output."""
    # Use existing gh auth (keyring). Remove GH_TOKEN if present to avoid conflicts.
    env = {k: v for k, v in os.environ.items() if k != "GH_TOKEN"}

    for attempt in range(retries):
        try:
            result = subprocess.run(
                ["gh"] + args,
                capture_output=True,
                text=True,
                timeout=60,
                env=env
            )
            if result.returncode != 0:
                if "rate limit" in result.stderr.lower():
                    wait_time = 2 ** attempt * 10
                    log(f"  Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                raise RuntimeError(f"gh command failed: {result.stderr}")
            return result.stdout
        except subprocess.TimeoutExpired:
            log(f"  Command timed out (attempt {attempt + 1}/{retries})")
            if attempt < retries - 1:
                continue
            raise
    raise RuntimeError("Max retries exceeded")


def get_merged_prs(repo: str, limit: int) -> list[dict]:
    """Get list of merged PRs."""
    fields = "number,url,title,body,additions,deletions,changedFiles,mergedAt"
    cmd = ["pr", "list", "--repo", repo, "--state", "merged",
           "--limit", str(limit), "--json", fields]
    output = run_gh_command(cmd)
    return json.loads(output)


def get_pr_diff(repo: str, pr_number: int) -> str:
    """Get PR diff."""
    cmd = ["pr", "diff", str(pr_number), "--repo", repo]
    try:
        return run_gh_command(cmd)
    except RuntimeError as e:
        log(f"  Warning: Failed to get diff for PR #{pr_number}: {e}")
        return ""


def get_pr_reviews(repo: str, pr_number: int) -> list[dict]:
    """Get PR reviews via API."""
    cmd = ["api", f"repos/{repo}/pulls/{pr_number}/reviews"]
    try:
        output = run_gh_command(cmd)
        reviews = json.loads(output)
        return [
            {
                "author": r.get("user", {}).get("login", "unknown"),
                "state": r.get("state", ""),
                "body": r.get("body", ""),
                "submitted_at": r.get("submitted_at", "")
            }
            for r in reviews
        ]
    except RuntimeError as e:
        log(f"  Warning: Failed to get reviews for PR #{pr_number}: {e}")
        return []


def get_pr_comments(repo: str, pr_number: int) -> list[dict]:
    """Get PR issue comments (general discussion)."""
    cmd = ["api", f"repos/{repo}/issues/{pr_number}/comments"]
    try:
        output = run_gh_command(cmd)
        comments = json.loads(output)
        return [
            {
                "author": c.get("user", {}).get("login", "unknown"),
                "body": c.get("body", ""),
                "created_at": c.get("created_at", "")
            }
            for c in comments
        ]
    except RuntimeError as e:
        log(f"  Warning: Failed to get comments for PR #{pr_number}: {e}")
        return []


def get_pr_review_comments(repo: str, pr_number: int) -> list[dict]:
    """Get PR inline review comments (code-level)."""
    cmd = ["api", f"repos/{repo}/pulls/{pr_number}/comments"]
    try:
        output = run_gh_command(cmd)
        comments = json.loads(output)
        return [
            {
                "author": c.get("user", {}).get("login", "unknown"),
                "body": c.get("body", ""),
                "path": c.get("path", ""),
                "diff_hunk": c.get("diff_hunk", ""),
                "created_at": c.get("created_at", "")
            }
            for c in comments
        ]
    except RuntimeError as e:
        log(f"  Warning: Failed to get review comments for PR #{pr_number}: {e}")
        return []


def extract_issue_references(body: str) -> list[int]:
    """Extract issue numbers from PR body (close #123, fixes #456, etc.)."""
    if not body:
        return []
    pattern = r'(?:close|closes|closed|fix|fixes|fixed|resolve|resolves|resolved)\s*#(\d+)'
    matches = re.findall(pattern, body, re.IGNORECASE)
    return [int(m) for m in matches]


def get_issue_data(repo: str, issue_number: int) -> Optional[dict]:
    """Get issue data."""
    cmd = ["api", f"repos/{repo}/issues/{issue_number}"]
    try:
        output = run_gh_command(cmd)
        issue = json.loads(output)
        return {
            "number": issue.get("number"),
            "title": issue.get("title", ""),
            "body": issue.get("body", ""),
            "labels": [lbl.get("name", "") for lbl in issue.get("labels", [])]
        }
    except RuntimeError:
        return None


def collect_pr_data(repo: str, pr: dict) -> dict:
    """Collect all data for a single PR."""
    pr_number = pr["number"]

    # Get diff
    diff = get_pr_diff(repo, pr_number)

    # Get reviews and comments
    reviews = get_pr_reviews(repo, pr_number)
    comments = get_pr_comments(repo, pr_number)
    review_comments = get_pr_review_comments(repo, pr_number)

    # Get linked issues
    issue_refs = extract_issue_references(pr.get("body", ""))
    linked_issues = []
    for issue_num in issue_refs:
        issue_data = get_issue_data(repo, issue_num)
        if issue_data:
            linked_issues.append(issue_data)

    return {
        "pr_number": pr_number,
        "url": pr.get("url", ""),
        "title": pr.get("title", ""),
        "body": pr.get("body", ""),
        "diff": diff,
        "reviews": reviews,
        "comments": comments,
        "review_comments": review_comments,
        "linked_issues": linked_issues,
        "files_changed": pr.get("changedFiles", 0),
        "additions": pr.get("additions", 0),
        "deletions": pr.get("deletions", 0),
        "merged_at": pr.get("mergedAt", "")
    }


def main():
    parser = argparse.ArgumentParser(description="Collect PR data from GitHub")
    parser.add_argument("--repo", required=True, help="Repository (owner/repo)")
    parser.add_argument("--limit", type=int, default=10, help="Max PRs to collect")
    parser.add_argument("--output", default="data/training.jsonl", help="Output file")
    args = parser.parse_args()

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Task overview
    log("=" * 60)
    log("Task: Collect PR data from GitHub")
    log(f"  Repository: {args.repo}")
    log(f"  Limit: {args.limit}")
    log(f"  Output: {args.output}")
    log("=" * 60)

    # Step 1: Get merged PRs
    log("[Step 1/2] Fetching merged PRs...")
    prs = get_merged_prs(args.repo, args.limit)
    log(f"[Step 1/2] Found {len(prs)} merged PRs")

    # Step 2: Collect detailed data for each PR
    log("[Step 2/2] Collecting detailed PR data...")
    with open(output_path, "w", encoding="utf-8") as f:
        for i, pr in enumerate(prs):
            log(f"  Processing PR #{pr['number']} ({i+1}/{len(prs)})...")
            try:
                pr_data = collect_pr_data(args.repo, pr)
                f.write(json.dumps(pr_data, ensure_ascii=False) + "\n")
            except Exception as e:
                log(f"  Warning: Failed to collect PR #{pr['number']}: {e}")

    log("=" * 60)
    log(f"Done! Data saved to {output_path}")
    log("=" * 60)


if __name__ == "__main__":
    main()
