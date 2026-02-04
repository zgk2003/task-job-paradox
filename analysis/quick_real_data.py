#!/usr/bin/env python3
"""
Quick Real Data Analysis - Targeted approach with minimal API calls.

Uses GitHub's search API to get aggregate statistics about PRs,
avoiding the need to fetch individual PR details.
"""

import urllib.request
import urllib.error
import json
import ssl
from datetime import datetime
from typing import Dict, List, Tuple
import time

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

TREATMENT_DATE = datetime(2022, 11, 30)


def github_request(url: str) -> Dict:
    """Make a request to GitHub API."""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "TaskJobParadox-Research"
    }

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, context=ssl_context, timeout=30) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"Error: {e}")
        return {}


def count_merged_prs(repo: str, since: str, until: str) -> int:
    """Count merged PRs for a repo in a date range using search API."""
    # Search API is more efficient for counts
    query = f"repo:{repo} is:pr is:merged merged:{since}..{until}"
    url = f"https://api.github.com/search/issues?q={urllib.parse.quote(query)}&per_page=1"

    data = github_request(url)
    return data.get('total_count', 0)


def get_recent_prs_stats(repo: str, state: str = 'closed', count: int = 30) -> List[Dict]:
    """Get recent PR statistics."""
    url = f"https://api.github.com/repos/{repo}/pulls?state={state}&sort=updated&direction=desc&per_page={count}"
    return github_request(url) or []


def analyze_pr_list(prs: List[Dict], cutoff_date: datetime) -> Tuple[Dict, Dict]:
    """Analyze a list of PRs, splitting by pre/post cutoff date."""
    pre_stats = {'lead_times': [], 'commits': [], 'comments': []}
    post_stats = {'lead_times': [], 'commits': [], 'comments': []}

    for pr in prs:
        if not pr.get('merged_at'):
            continue

        merged_at = datetime.fromisoformat(pr['merged_at'].replace('Z', '+00:00'))
        merged_at = merged_at.replace(tzinfo=None)
        created_at = datetime.fromisoformat(pr['created_at'].replace('Z', '+00:00'))
        created_at = created_at.replace(tzinfo=None)

        lead_time = (merged_at - created_at).total_seconds() / 3600  # hours

        stats = post_stats if merged_at >= cutoff_date else pre_stats
        stats['lead_times'].append(lead_time)
        stats['commits'].append(pr.get('commits', 0))
        stats['comments'].append(pr.get('review_comments', 0))

    return pre_stats, post_stats


def safe_mean(lst):
    return sum(lst) / len(lst) if lst else 0


def run_quick_analysis():
    """Run quick analysis on a few key repositories."""

    # Target repositories (well-known, active)
    repos = {
        'high_exposure': [
            ('pallets/flask', 'Python'),
            ('psf/requests', 'Python'),
            ('expressjs/express', 'JavaScript'),
            ('lodash/lodash', 'JavaScript'),
        ],
        'low_exposure': [
            ('erlang/otp', 'Erlang'),
            ('elixir-lang/elixir', 'Elixir'),
            ('scala/scala', 'Scala'),
            ('rust-lang/rust', 'Rust'),
        ]
    }

    print("="*70)
    print("QUICK REAL DATA ANALYSIS")
    print("="*70)
    print(f"Treatment date: {TREATMENT_DATE.date()}")
    print()

    all_results = {}

    for exposure_level, repo_list in repos.items():
        print(f"\n{exposure_level.upper().replace('_', ' ')} LANGUAGES:")
        print("-"*50)

        all_pre = {'lead_times': [], 'commits': [], 'comments': []}
        all_post = {'lead_times': [], 'commits': [], 'comments': []}

        for repo, language in repo_list:
            print(f"  {repo} ({language})...", end=" ", flush=True)

            try:
                prs = get_recent_prs_stats(repo, 'closed', 100)
                merged_prs = [p for p in prs if p.get('merged_at')]

                if not merged_prs:
                    print("no merged PRs")
                    continue

                pre_stats, post_stats = analyze_pr_list(merged_prs, TREATMENT_DATE)

                # Aggregate
                all_pre['lead_times'].extend(pre_stats['lead_times'])
                all_post['lead_times'].extend(post_stats['lead_times'])
                all_pre['commits'].extend(pre_stats['commits'])
                all_post['commits'].extend(post_stats['commits'])

                print(f"OK (pre={len(pre_stats['lead_times'])}, post={len(post_stats['lead_times'])})")

                time.sleep(2)  # Be nice to API

            except Exception as e:
                print(f"Error: {e}")
                continue

        # Calculate aggregates for this exposure level
        if all_pre['lead_times'] and all_post['lead_times']:
            pre_lead = safe_mean(all_pre['lead_times'])
            post_lead = safe_mean(all_post['lead_times'])
            lead_change = (post_lead - pre_lead) / pre_lead * 100 if pre_lead else 0

            pre_commits = safe_mean(all_pre['commits'])
            post_commits = safe_mean(all_post['commits'])
            commits_change = (post_commits - pre_commits) / pre_commits * 100 if pre_commits else 0

            all_results[exposure_level] = {
                'pre_lead_time': pre_lead,
                'post_lead_time': post_lead,
                'lead_time_change': lead_change,
                'pre_commits': pre_commits,
                'post_commits': post_commits,
                'commits_change': commits_change,
                'n_pre': len(all_pre['lead_times']),
                'n_post': len(all_post['lead_times']),
            }

            print(f"\n  Aggregate Results ({exposure_level}):")
            print(f"    Lead Time: {pre_lead:.1f}h → {post_lead:.1f}h ({lead_change:+.1f}%)")
            print(f"    Commits/PR: {pre_commits:.1f} → {post_commits:.1f} ({commits_change:+.1f}%)")
            print(f"    Sample: {len(all_pre['lead_times'])} pre, {len(all_post['lead_times'])} post")

    # DiD comparison
    if 'high_exposure' in all_results and 'low_exposure' in all_results:
        print("\n" + "="*70)
        print("DIFFERENCE-IN-DIFFERENCES COMPARISON")
        print("="*70)

        high = all_results['high_exposure']
        low = all_results['low_exposure']

        # Lead time DiD
        high_change_lt = high['post_lead_time'] - high['pre_lead_time']
        low_change_lt = low['post_lead_time'] - low['pre_lead_time']
        did_lt = high_change_lt - low_change_lt

        print(f"\nPR Lead Time (Job-Level):")
        print(f"  High exposure: {high['pre_lead_time']:.1f}h → {high['post_lead_time']:.1f}h ({high_change_lt:+.1f}h)")
        print(f"  Low exposure:  {low['pre_lead_time']:.1f}h → {low['post_lead_time']:.1f}h ({low_change_lt:+.1f}h)")
        print(f"  DiD estimate: {did_lt:+.1f} hours")

        # Commits DiD
        high_change_c = high['post_commits'] - high['pre_commits']
        low_change_c = low['post_commits'] - low['pre_commits']
        did_c = high_change_c - low_change_c

        print(f"\nCommits per PR (Iteration):")
        print(f"  High exposure: {high['pre_commits']:.1f} → {high['post_commits']:.1f} ({high_change_c:+.1f})")
        print(f"  Low exposure:  {low['pre_commits']:.1f} → {low['post_commits']:.1f} ({low_change_c:+.1f})")
        print(f"  DiD estimate: {did_c:+.1f} commits")

        print("\n" + "="*70)
        print("INTERPRETATION")
        print("="*70)

        print("\n1. LEAD TIME (Job-Level Metric):")
        if did_lt < 0:
            print(f"   ✓ High-exposure repos improved {abs(did_lt):.1f}h MORE than low-exposure")
            print(f"   → Consistent with AI helping job completion")
        else:
            print(f"   ? Low-exposure repos improved more")
            print(f"   → May reflect other factors (repo maturity, team size, etc.)")

        print(f"\n2. COMMITS (Iteration Proxy):")
        if did_c > 0:
            print(f"   ✓ High-exposure repos added {did_c:.1f} MORE commits per PR")
            print(f"   → Supports H4: AI enables more iteration")
        else:
            print(f"   ? High-exposure repos have fewer commits")

        # Overall assessment
        overall_lead_change = (high['lead_time_change'] + low['lead_time_change']) / 2
        overall_commits_change = (high['commits_change'] + low['commits_change']) / 2

        print(f"\n3. OVERALL PATTERN:")
        print(f"   Lead time change: {overall_lead_change:+.1f}%")
        print(f"   Commits change: {overall_commits_change:+.1f}%")

        if abs(overall_lead_change) < 15 and overall_commits_change > 5:
            print(f"\n   → CONSISTENT WITH PARADOX:")
            print(f"     Job-level metric changed modestly while iteration increased")
            print(f"     This suggests AI time savings may be reallocated to more iteration")
        elif overall_lead_change < -20:
            print(f"\n   → STRONG JOB-LEVEL IMPROVEMENT:")
            print(f"     More improvement than expected - may not support paradox hypothesis")

    print("\n" + "="*70)
    print("NOTE: This is preliminary analysis with limited data")
    print("Full study requires GitHub Archive data via BigQuery")
    print("="*70)

    return all_results


if __name__ == "__main__":
    import urllib.parse
    results = run_quick_analysis()
