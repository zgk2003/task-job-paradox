"""
GitHub Real Data Fetcher for Task-Job Paradox Analysis

Fetches real PR data from GitHub API to validate our predictions
against actual software development patterns.

Strategy:
1. Sample popular repositories in high AI-exposure languages (Python, JS, Java)
2. Sample repositories in low AI-exposure languages (Haskell, Erlang, etc.)
3. Fetch PR data spanning 2021-2025 to cover pre/post AI adoption
4. Extract timestamps for task-level and job-level metrics
"""

import urllib.request
import urllib.error
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import ssl

# Disable SSL verification for API calls (common in restricted environments)
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE


@dataclass
class RealPRData:
    """Real PR data from GitHub API."""
    repo_name: str
    repo_language: str
    pr_number: int
    created_at: datetime
    merged_at: Optional[datetime]
    closed_at: Optional[datetime]
    commits: int
    review_comments: int
    additions: int
    deletions: int
    changed_files: int

    @property
    def lead_time_hours(self) -> Optional[float]:
        if self.merged_at:
            return (self.merged_at - self.created_at).total_seconds() / 3600
        return None

    @property
    def is_merged(self) -> bool:
        return self.merged_at is not None


class GitHubDataFetcher:
    """Fetches real data from GitHub API."""

    BASE_URL = "https://api.github.com"

    # Repositories to sample (popular, active repos)
    # High AI exposure languages
    HIGH_EXPOSURE_REPOS = [
        # Python
        "python/cpython",
        "django/django",
        "pallets/flask",
        "psf/requests",
        "numpy/numpy",
        "pandas-dev/pandas",
        "scikit-learn/scikit-learn",
        "pytorch/pytorch",
        # JavaScript/TypeScript
        "facebook/react",
        "vuejs/vue",
        "angular/angular",
        "nodejs/node",
        "microsoft/vscode",
        "vercel/next.js",
        # Java
        "spring-projects/spring-boot",
        "elastic/elasticsearch",
        "apache/kafka",
    ]

    # Low AI exposure languages
    LOW_EXPOSURE_REPOS = [
        # Haskell
        "ghc/ghc",
        "haskell/haskell-language-server",
        "commercialhaskell/stack",
        # Erlang/Elixir
        "erlang/otp",
        "elixir-lang/elixir",
        "phoenixframework/phoenix",
        # Rust (moderate exposure, but different from mainstream)
        "rust-lang/rust",
        "denoland/deno",
        # Scala
        "scala/scala",
        "akka/akka",
        # Go (for comparison - moderate exposure)
        "golang/go",
        "kubernetes/kubernetes",
    ]

    def __init__(self, token: Optional[str] = None):
        self.token = token
        self.request_count = 0
        self.rate_limit_remaining = 60  # Default for unauthenticated

    def _make_request(self, url: str) -> Optional[Dict]:
        """Make a rate-limited request to GitHub API."""
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "TaskJobParadox-Research"
        }
        if self.token:
            headers["Authorization"] = f"token {self.token}"

        # Rate limiting
        if self.rate_limit_remaining <= 1:
            print("Rate limit nearly exhausted, waiting 60 seconds...")
            time.sleep(60)

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, context=ssl_context, timeout=30) as response:
                self.request_count += 1

                # Track rate limit
                remaining = response.headers.get('X-RateLimit-Remaining')
                if remaining:
                    self.rate_limit_remaining = int(remaining)

                return json.loads(response.read().decode())

        except urllib.error.HTTPError as e:
            if e.code == 403:
                print(f"Rate limited. Waiting...")
                time.sleep(60)
                return self._make_request(url)
            elif e.code == 404:
                return None
            else:
                print(f"HTTP Error {e.code}: {e.reason}")
                return None
        except Exception as e:
            print(f"Request error: {e}")
            return None

    def get_repo_info(self, repo: str) -> Optional[Dict]:
        """Get repository information."""
        url = f"{self.BASE_URL}/repos/{repo}"
        return self._make_request(url)

    def get_merged_prs(
        self,
        repo: str,
        since: datetime,
        until: datetime,
        max_prs: int = 100
    ) -> List[Dict]:
        """Fetch merged PRs for a repository in a date range."""
        prs = []
        page = 1
        per_page = 30

        while len(prs) < max_prs:
            url = (f"{self.BASE_URL}/repos/{repo}/pulls?"
                   f"state=closed&sort=updated&direction=desc&"
                   f"per_page={per_page}&page={page}")

            data = self._make_request(url)
            if not data or len(data) == 0:
                break

            for pr in data:
                if not pr.get('merged_at'):
                    continue

                merged_at = datetime.fromisoformat(pr['merged_at'].replace('Z', '+00:00'))
                merged_at = merged_at.replace(tzinfo=None)

                if merged_at < since:
                    # PRs are sorted by update time, so we might find older ones
                    continue
                if merged_at > until:
                    continue

                prs.append(pr)

                if len(prs) >= max_prs:
                    break

            page += 1

            # Don't hammer the API
            time.sleep(0.5)

            # Stop if we've gone through many pages
            if page > 10:
                break

        return prs

    def parse_pr_data(self, pr: Dict, repo_language: str, repo_name: str) -> RealPRData:
        """Parse PR API response into our data structure."""
        created_at = datetime.fromisoformat(pr['created_at'].replace('Z', '+00:00'))
        created_at = created_at.replace(tzinfo=None)

        merged_at = None
        if pr.get('merged_at'):
            merged_at = datetime.fromisoformat(pr['merged_at'].replace('Z', '+00:00'))
            merged_at = merged_at.replace(tzinfo=None)

        closed_at = None
        if pr.get('closed_at'):
            closed_at = datetime.fromisoformat(pr['closed_at'].replace('Z', '+00:00'))
            closed_at = closed_at.replace(tzinfo=None)

        return RealPRData(
            repo_name=repo_name,
            repo_language=repo_language,
            pr_number=pr['number'],
            created_at=created_at,
            merged_at=merged_at,
            closed_at=closed_at,
            commits=pr.get('commits', 0),
            review_comments=pr.get('review_comments', 0),
            additions=pr.get('additions', 0),
            deletions=pr.get('deletions', 0),
            changed_files=pr.get('changed_files', 0),
        )

    def fetch_dataset(
        self,
        since: datetime = datetime(2021, 1, 1),
        until: datetime = datetime(2025, 6, 30),
        prs_per_repo: int = 50,
        max_repos_per_category: int = 8
    ) -> List[RealPRData]:
        """Fetch a dataset of real PRs from GitHub."""
        all_prs = []

        print("="*60)
        print("FETCHING REAL GITHUB DATA")
        print("="*60)
        print(f"Date range: {since.date()} to {until.date()}")
        print(f"PRs per repo: {prs_per_repo}")
        print()

        # Fetch high exposure repos
        print("HIGH AI EXPOSURE REPOSITORIES:")
        print("-"*40)
        for i, repo in enumerate(self.HIGH_EXPOSURE_REPOS[:max_repos_per_category]):
            print(f"  Fetching {repo}...", end=" ", flush=True)

            repo_info = self.get_repo_info(repo)
            if not repo_info:
                print("SKIPPED (not found)")
                continue

            language = repo_info.get('language', 'Unknown')
            prs = self.get_merged_prs(repo, since, until, prs_per_repo)

            for pr in prs:
                pr_data = self.parse_pr_data(pr, language, repo)
                all_prs.append(pr_data)

            print(f"OK ({len(prs)} PRs, {language})")
            time.sleep(1)  # Be nice to API

        print()
        print("LOW AI EXPOSURE REPOSITORIES:")
        print("-"*40)
        for i, repo in enumerate(self.LOW_EXPOSURE_REPOS[:max_repos_per_category]):
            print(f"  Fetching {repo}...", end=" ", flush=True)

            repo_info = self.get_repo_info(repo)
            if not repo_info:
                print("SKIPPED (not found)")
                continue

            language = repo_info.get('language', 'Unknown')
            prs = self.get_merged_prs(repo, since, until, prs_per_repo)

            for pr in prs:
                pr_data = self.parse_pr_data(pr, language, repo)
                all_prs.append(pr_data)

            print(f"OK ({len(prs)} PRs, {language})")
            time.sleep(1)

        print()
        print(f"Total PRs fetched: {len(all_prs)}")
        print(f"API requests made: {self.request_count}")

        return all_prs


def fetch_and_save_real_data(output_path: str = "../data/real_github_data.json"):
    """Fetch real GitHub data and save to file."""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fetcher = GitHubDataFetcher()
    prs = fetcher.fetch_dataset(
        prs_per_repo=50,
        max_repos_per_category=6  # Limit to stay within rate limits
    )

    # Convert to serializable format
    data = []
    for pr in prs:
        data.append({
            'repo_name': pr.repo_name,
            'repo_language': pr.repo_language,
            'pr_number': pr.pr_number,
            'created_at': pr.created_at.isoformat(),
            'merged_at': pr.merged_at.isoformat() if pr.merged_at else None,
            'closed_at': pr.closed_at.isoformat() if pr.closed_at else None,
            'commits': pr.commits,
            'review_comments': pr.review_comments,
            'additions': pr.additions,
            'deletions': pr.deletions,
            'changed_files': pr.changed_files,
        })

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved {len(data)} PRs to {output_path}")
    return prs


if __name__ == "__main__":
    fetch_and_save_real_data()
