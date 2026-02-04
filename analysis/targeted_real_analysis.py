#!/usr/bin/env python3
"""
Targeted Real Data Analysis using GitHub Search API

Uses date-filtered searches to get balanced pre/post samples.
"""

import urllib.request
import urllib.parse
import urllib.error
import json
import ssl
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import time

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

TREATMENT_DATE = datetime(2022, 11, 30)


def github_search(query: str, per_page: int = 30) -> Dict:
    """Search GitHub using the search API."""
    encoded_query = urllib.parse.quote(query)
    url = f"https://api.github.com/search/issues?q={encoded_query}&per_page={per_page}&sort=created&order=desc"

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "TaskJobParadox-Research"
    }

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, context=ssl_context, timeout=30) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 403:
            print("Rate limited, waiting 60s...")
            time.sleep(60)
            return github_search(query, per_page)
        print(f"HTTP Error {e.code}")
        return {'items': [], 'total_count': 0}
    except Exception as e:
        print(f"Error: {e}")
        return {'items': [], 'total_count': 0}


def get_pr_details(repo: str, pr_number: int) -> Optional[Dict]:
    """Get detailed PR information including commits."""
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "TaskJobParadox-Research"
    }

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, context=ssl_context, timeout=30) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        return None


def search_merged_prs(language: str, start_date: str, end_date: str, limit: int = 20) -> List[Dict]:
    """Search for merged PRs in a language within a date range."""
    # Search for merged PRs in specific language, merged in date range
    query = f"is:pr is:merged language:{language} merged:{start_date}..{end_date}"

    result = github_search(query, per_page=limit)
    return result.get('items', [])


def analyze_prs_with_details(prs: List[Dict]) -> Dict:
    """Analyze PRs by fetching details for each."""
    lead_times = []
    commits_list = []
    additions_list = []
    deletions_list = []

    for pr in prs[:15]:  # Limit to conserve API calls
        # Extract repo from URL
        # URL format: https://api.github.com/repos/owner/repo/issues/123
        url_parts = pr.get('repository_url', '').split('/')
        if len(url_parts) >= 2:
            repo = f"{url_parts[-2]}/{url_parts[-1]}"
        else:
            continue

        pr_number = pr.get('number')
        if not pr_number:
            continue

        # Get detailed PR info
        details = get_pr_details(repo, pr_number)
        if not details:
            continue

        if details.get('merged_at') and details.get('created_at'):
            created = datetime.fromisoformat(details['created_at'].replace('Z', '+00:00'))
            merged = datetime.fromisoformat(details['merged_at'].replace('Z', '+00:00'))
            lead_time = (merged - created).total_seconds() / 3600

            lead_times.append(lead_time)
            commits_list.append(details.get('commits', 0))
            additions_list.append(details.get('additions', 0))
            deletions_list.append(details.get('deletions', 0))

        time.sleep(0.5)  # Rate limit

    def safe_mean(lst):
        return sum(lst) / len(lst) if lst else 0

    return {
        'n': len(lead_times),
        'lead_time_mean': safe_mean(lead_times),
        'commits_mean': safe_mean(commits_list),
        'additions_mean': safe_mean(additions_list),
        'deletions_mean': safe_mean(deletions_list),
    }


def run_targeted_analysis():
    """Run analysis with targeted date ranges."""

    print("="*70)
    print("TARGETED REAL DATA ANALYSIS")
    print("="*70)
    print(f"Treatment date: {TREATMENT_DATE.date()}")
    print()

    # Define periods
    pre_period = ("2021-06-01", "2022-10-31")  # ~16 months before AI
    post_period = ("2023-02-01", "2024-06-30")  # ~16 months after AI adoption

    # Languages to analyze
    high_exposure = ['Python', 'JavaScript', 'TypeScript', 'Java']
    low_exposure = ['Haskell', 'Erlang', 'Scala', 'Rust']

    results = {}

    # Analyze high exposure languages
    print("HIGH AI EXPOSURE LANGUAGES:")
    print("-"*50)

    for lang in high_exposure:
        print(f"\n  {lang}:")

        # Pre-treatment
        print(f"    Pre-treatment ({pre_period[0]} to {pre_period[1]})...", end=" ", flush=True)
        pre_prs = search_merged_prs(lang, pre_period[0], pre_period[1], limit=25)
        pre_stats = analyze_prs_with_details(pre_prs)
        print(f"n={pre_stats['n']}")
        time.sleep(2)

        # Post-treatment
        print(f"    Post-treatment ({post_period[0]} to {post_period[1]})...", end=" ", flush=True)
        post_prs = search_merged_prs(lang, post_period[0], post_period[1], limit=25)
        post_stats = analyze_prs_with_details(post_prs)
        print(f"n={post_stats['n']}")
        time.sleep(2)

        if pre_stats['n'] >= 3 and post_stats['n'] >= 3:
            lt_change = ((post_stats['lead_time_mean'] - pre_stats['lead_time_mean'])
                        / pre_stats['lead_time_mean'] * 100) if pre_stats['lead_time_mean'] > 0 else 0
            c_change = ((post_stats['commits_mean'] - pre_stats['commits_mean'])
                       / pre_stats['commits_mean'] * 100) if pre_stats['commits_mean'] > 0 else 0

            print(f"    Lead Time: {pre_stats['lead_time_mean']:.1f}h → {post_stats['lead_time_mean']:.1f}h ({lt_change:+.1f}%)")
            print(f"    Commits: {pre_stats['commits_mean']:.1f} → {post_stats['commits_mean']:.1f} ({c_change:+.1f}%)")

            results[f"high_{lang}"] = {
                'pre': pre_stats,
                'post': post_stats,
                'lead_time_change': lt_change,
                'commits_change': c_change,
            }

    print("\n" + "="*70)
    print("LOW AI EXPOSURE LANGUAGES:")
    print("-"*50)

    for lang in low_exposure:
        print(f"\n  {lang}:")

        # Pre-treatment
        print(f"    Pre-treatment ({pre_period[0]} to {pre_period[1]})...", end=" ", flush=True)
        pre_prs = search_merged_prs(lang, pre_period[0], pre_period[1], limit=25)
        pre_stats = analyze_prs_with_details(pre_prs)
        print(f"n={pre_stats['n']}")
        time.sleep(2)

        # Post-treatment
        print(f"    Post-treatment ({post_period[0]} to {post_period[1]})...", end=" ", flush=True)
        post_prs = search_merged_prs(lang, post_period[0], post_period[1], limit=25)
        post_stats = analyze_prs_with_details(post_prs)
        print(f"n={post_stats['n']}")
        time.sleep(2)

        if pre_stats['n'] >= 3 and post_stats['n'] >= 3:
            lt_change = ((post_stats['lead_time_mean'] - pre_stats['lead_time_mean'])
                        / pre_stats['lead_time_mean'] * 100) if pre_stats['lead_time_mean'] > 0 else 0
            c_change = ((post_stats['commits_mean'] - pre_stats['commits_mean'])
                       / pre_stats['commits_mean'] * 100) if pre_stats['commits_mean'] > 0 else 0

            print(f"    Lead Time: {pre_stats['lead_time_mean']:.1f}h → {post_stats['lead_time_mean']:.1f}h ({lt_change:+.1f}%)")
            print(f"    Commits: {pre_stats['commits_mean']:.1f} → {post_stats['commits_mean']:.1f} ({c_change:+.1f}%)")

            results[f"low_{lang}"] = {
                'pre': pre_stats,
                'post': post_stats,
                'lead_time_change': lt_change,
                'commits_change': c_change,
            }

    # Aggregate comparison
    print("\n" + "="*70)
    print("AGGREGATE COMPARISON")
    print("="*70)

    high_results = [v for k, v in results.items() if k.startswith('high_')]
    low_results = [v for k, v in results.items() if k.startswith('low_')]

    if high_results and low_results:
        # Aggregate high exposure
        high_pre_lt = sum(r['pre']['lead_time_mean'] * r['pre']['n'] for r in high_results) / sum(r['pre']['n'] for r in high_results) if sum(r['pre']['n'] for r in high_results) > 0 else 0
        high_post_lt = sum(r['post']['lead_time_mean'] * r['post']['n'] for r in high_results) / sum(r['post']['n'] for r in high_results) if sum(r['post']['n'] for r in high_results) > 0 else 0

        high_pre_c = sum(r['pre']['commits_mean'] * r['pre']['n'] for r in high_results) / sum(r['pre']['n'] for r in high_results) if sum(r['pre']['n'] for r in high_results) > 0 else 0
        high_post_c = sum(r['post']['commits_mean'] * r['post']['n'] for r in high_results) / sum(r['post']['n'] for r in high_results) if sum(r['post']['n'] for r in high_results) > 0 else 0

        # Aggregate low exposure
        low_pre_lt = sum(r['pre']['lead_time_mean'] * r['pre']['n'] for r in low_results) / sum(r['pre']['n'] for r in low_results) if sum(r['pre']['n'] for r in low_results) > 0 else 0
        low_post_lt = sum(r['post']['lead_time_mean'] * r['post']['n'] for r in low_results) / sum(r['post']['n'] for r in low_results) if sum(r['post']['n'] for r in low_results) > 0 else 0

        low_pre_c = sum(r['pre']['commits_mean'] * r['pre']['n'] for r in low_results) / sum(r['pre']['n'] for r in low_results) if sum(r['pre']['n'] for r in low_results) > 0 else 0
        low_post_c = sum(r['post']['commits_mean'] * r['post']['n'] for r in low_results) / sum(r['post']['n'] for r in low_results) if sum(r['post']['n'] for r in low_results) > 0 else 0

        print("\nPR Lead Time (Job-Level Metric):")
        print(f"  High AI exposure: {high_pre_lt:.1f}h → {high_post_lt:.1f}h")
        print(f"  Low AI exposure:  {low_pre_lt:.1f}h → {low_post_lt:.1f}h")

        high_lt_change = high_post_lt - high_pre_lt
        low_lt_change = low_post_lt - low_pre_lt
        did_lt = high_lt_change - low_lt_change
        print(f"  DiD estimate: {did_lt:+.1f} hours")

        print("\nCommits per PR (Iteration Metric):")
        print(f"  High AI exposure: {high_pre_c:.1f} → {high_post_c:.1f}")
        print(f"  Low AI exposure:  {low_pre_c:.1f} → {low_post_c:.1f}")

        high_c_change = high_post_c - high_pre_c
        low_c_change = low_post_c - low_pre_c
        did_c = high_c_change - low_c_change
        print(f"  DiD estimate: {did_c:+.1f} commits")

        # Summary
        print("\n" + "="*70)
        print("PRELIMINARY FINDINGS")
        print("="*70)

        high_lt_pct = (high_post_lt - high_pre_lt) / high_pre_lt * 100 if high_pre_lt > 0 else 0
        low_lt_pct = (low_post_lt - low_pre_lt) / low_pre_lt * 100 if low_pre_lt > 0 else 0

        print(f"\n1. JOB-LEVEL (Lead Time):")
        print(f"   High exposure change: {high_lt_pct:+.1f}%")
        print(f"   Low exposure change:  {low_lt_pct:+.1f}%")

        if did_lt < 0:
            print(f"   → High-exposure improved MORE (supports AI effect hypothesis)")
        else:
            print(f"   → Low-exposure improved more (needs investigation)")

        high_c_pct = (high_post_c - high_pre_c) / high_pre_c * 100 if high_pre_c > 0 else 0
        low_c_pct = (low_post_c - low_pre_c) / low_pre_c * 100 if low_pre_c > 0 else 0

        print(f"\n2. ITERATION (Commits per PR):")
        print(f"   High exposure change: {high_c_pct:+.1f}%")
        print(f"   Low exposure change:  {low_c_pct:+.1f}%")

        if did_c > 0:
            print(f"   → High-exposure has MORE iteration increase (supports H4)")

        print("\n3. PARADOX CHECK:")
        if abs(high_lt_pct) < 20 and high_c_pct > 10:
            print("   ✓ PATTERN CONSISTENT WITH PARADOX:")
            print("     Modest lead time improvement + increased iteration")
        elif high_lt_pct < -20:
            print("   ? Strong lead time improvement - may not support paradox")
        else:
            print("   ~ Mixed evidence - needs larger sample")

    print("\n" + "="*70)
    print("LIMITATIONS:")
    print("  - Small sample sizes due to API rate limits")
    print("  - Search API may have selection bias")
    print("  - Full study requires GitHub Archive via BigQuery")
    print("="*70)

    return results


if __name__ == "__main__":
    results = run_targeted_analysis()
