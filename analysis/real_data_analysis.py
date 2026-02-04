#!/usr/bin/env python3
"""
Real Data Analysis for Task-Job Paradox

Analyzes actual GitHub PR data to test whether the task-job paradox
is observable in real software development patterns.

Key questions:
1. Did PR lead time (job-level) decrease after AI adoption (Nov 2022)?
2. Did commits per PR (iteration proxy) increase after AI adoption?
3. Are effects different for high vs low AI exposure languages?
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import os
import sys

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_model import TREATMENT_START

# High AI exposure languages
HIGH_EXPOSURE_LANGUAGES = {'Python', 'JavaScript', 'TypeScript', 'Java', 'Kotlin'}
LOW_EXPOSURE_LANGUAGES = {'Haskell', 'Erlang', 'Elixir', 'Scala', 'Rust', 'Go', 'C', 'C++'}


def load_real_data(path: str) -> pd.DataFrame:
    """Load real GitHub data from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Parse dates
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['merged_at'] = pd.to_datetime(df['merged_at'])

    # Calculate derived metrics
    df['lead_time_hours'] = (df['merged_at'] - df['created_at']).dt.total_seconds() / 3600
    df['code_churn'] = df['additions'] + df['deletions']

    # Treatment indicator
    df['post_treatment'] = df['created_at'] >= TREATMENT_START

    # AI exposure classification
    df['high_exposure'] = df['repo_language'].isin(HIGH_EXPOSURE_LANGUAGES)

    # Time variables
    df['year_month'] = df['created_at'].dt.to_period('M')
    df['year'] = df['created_at'].dt.year

    return df


def analyze_real_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze real GitHub data for task-job paradox patterns.

    Since we don't have direct task-level metrics (review-response latency)
    from the basic PR API, we use proxies:
    - commits: More commits might indicate more iteration
    - review_comments: More comments might indicate more review cycles

    Job-level metric:
    - lead_time_hours: Direct measure of PR completion time
    """
    results = {}

    print("\n" + "="*70)
    print("REAL GITHUB DATA ANALYSIS")
    print("="*70)

    # Basic statistics
    print(f"\nDataset Overview:")
    print(f"  Total PRs: {len(df):,}")
    print(f"  Repositories: {df['repo_name'].nunique()}")
    print(f"  Languages: {df['repo_language'].nunique()}")
    print(f"  Date range: {df['created_at'].min().date()} to {df['created_at'].max().date()}")

    pre_df = df[~df['post_treatment']]
    post_df = df[df['post_treatment']]

    print(f"\n  Pre-treatment (before Nov 2022): {len(pre_df):,} PRs")
    print(f"  Post-treatment (after Nov 2022): {len(post_df):,} PRs")

    high_exp = df[df['high_exposure']]
    low_exp = df[~df['high_exposure']]
    print(f"\n  High AI exposure (Python/JS/Java): {len(high_exp):,} PRs")
    print(f"  Low AI exposure (other): {len(low_exp):,} PRs")

    # Overall pre/post comparison
    print("\n" + "-"*70)
    print("OVERALL PRE/POST COMPARISON")
    print("-"*70)

    metrics = {
        'lead_time_hours': 'PR Lead Time (hours)',
        'commits': 'Commits per PR',
        'review_comments': 'Review Comments',
        'code_churn': 'Code Churn (lines)',
        'changed_files': 'Changed Files',
    }

    overall_results = {}
    for metric, label in metrics.items():
        pre_mean = pre_df[metric].mean()
        post_mean = post_df[metric].mean()
        pct_change = (post_mean - pre_mean) / pre_mean * 100 if pre_mean != 0 else 0

        overall_results[metric] = {
            'pre_mean': pre_mean,
            'post_mean': post_mean,
            'pct_change': pct_change
        }

        direction = "↓" if pct_change < 0 else "↑"
        print(f"  {label}:")
        print(f"    Pre:  {pre_mean:.1f}")
        print(f"    Post: {post_mean:.1f}")
        print(f"    Change: {pct_change:+.1f}% {direction}")
        print()

    results['overall'] = overall_results

    # By AI exposure level
    print("-"*70)
    print("BY AI EXPOSURE LEVEL")
    print("-"*70)

    for exposure_level, exposure_df in [('High', high_exp), ('Low', low_exp)]:
        print(f"\n{exposure_level} AI Exposure Languages:")

        pre = exposure_df[~exposure_df['post_treatment']]
        post = exposure_df[exposure_df['post_treatment']]

        if len(pre) < 10 or len(post) < 10:
            print(f"  Insufficient data (pre={len(pre)}, post={len(post)})")
            continue

        exposure_results = {}
        for metric in ['lead_time_hours', 'commits']:
            pre_mean = pre[metric].mean()
            post_mean = post[metric].mean()
            pct_change = (post_mean - pre_mean) / pre_mean * 100 if pre_mean != 0 else 0

            exposure_results[metric] = {
                'pre_mean': pre_mean,
                'post_mean': post_mean,
                'pct_change': pct_change
            }

            label = 'Lead Time' if metric == 'lead_time_hours' else 'Commits/PR'
            print(f"  {label}: {pre_mean:.1f} → {post_mean:.1f} ({pct_change:+.1f}%)")

        results[f'{exposure_level.lower()}_exposure'] = exposure_results

    # Difference-in-differences style comparison
    print("\n" + "-"*70)
    print("DIFFERENCE-IN-DIFFERENCES ESTIMATE")
    print("-"*70)

    # Calculate DiD for lead time
    high_pre = high_exp[~high_exp['post_treatment']]['lead_time_hours'].mean()
    high_post = high_exp[high_exp['post_treatment']]['lead_time_hours'].mean()
    low_pre = low_exp[~low_exp['post_treatment']]['lead_time_hours'].mean()
    low_post = low_exp[low_exp['post_treatment']]['lead_time_hours'].mean()

    high_change = high_post - high_pre
    low_change = low_post - low_pre
    did_estimate = high_change - low_change

    print(f"\nPR Lead Time (Job-Level Metric):")
    print(f"  High exposure change: {high_pre:.1f}h → {high_post:.1f}h ({high_change:+.1f}h)")
    print(f"  Low exposure change:  {low_pre:.1f}h → {low_post:.1f}h ({low_change:+.1f}h)")
    print(f"  DiD estimate: {did_estimate:+.1f} hours")

    if did_estimate < 0:
        print(f"  → High exposure repos improved MORE than low exposure (expected if AI helps)")
    else:
        print(f"  → High exposure repos improved LESS than low exposure")

    results['did_lead_time'] = {
        'high_pre': high_pre, 'high_post': high_post,
        'low_pre': low_pre, 'low_post': low_post,
        'did_estimate': did_estimate
    }

    # Same for commits (mechanism)
    high_pre_c = high_exp[~high_exp['post_treatment']]['commits'].mean()
    high_post_c = high_exp[high_exp['post_treatment']]['commits'].mean()
    low_pre_c = low_exp[~low_exp['post_treatment']]['commits'].mean()
    low_post_c = low_exp[low_exp['post_treatment']]['commits'].mean()

    high_change_c = high_post_c - high_pre_c
    low_change_c = low_post_c - low_pre_c
    did_commits = high_change_c - low_change_c

    print(f"\nCommits per PR (Iteration Proxy):")
    print(f"  High exposure change: {high_pre_c:.1f} → {high_post_c:.1f} ({high_change_c:+.1f})")
    print(f"  Low exposure change:  {low_pre_c:.1f} → {low_post_c:.1f} ({low_change_c:+.1f})")
    print(f"  DiD estimate: {did_commits:+.1f} commits")

    if did_commits > 0:
        print(f"  → High exposure repos have MORE iteration increase (supports H4)")

    results['did_commits'] = {
        'high_pre': high_pre_c, 'high_post': high_post_c,
        'low_pre': low_pre_c, 'low_post': low_post_c,
        'did_estimate': did_commits
    }

    # Time series by month
    print("\n" + "-"*70)
    print("MONTHLY TIME SERIES")
    print("-"*70)

    monthly = df.groupby(['year_month', 'high_exposure']).agg({
        'lead_time_hours': 'mean',
        'commits': 'mean',
        'pr_number': 'count'
    }).rename(columns={'pr_number': 'n_prs'})

    results['monthly'] = monthly.reset_index().to_dict('records')

    # Summary assessment
    print("\n" + "="*70)
    print("PRELIMINARY ASSESSMENT")
    print("="*70)

    lead_time_change = overall_results['lead_time_hours']['pct_change']
    commits_change = overall_results['commits']['pct_change']

    print(f"\n1. JOB-LEVEL METRIC (PR Lead Time):")
    if lead_time_change < -5:
        print(f"   ✓ Decreased by {abs(lead_time_change):.1f}% - directionally consistent with AI helping")
    elif lead_time_change > 5:
        print(f"   ✗ Increased by {lead_time_change:.1f}% - opposite of prediction")
    else:
        print(f"   ~ Minimal change ({lead_time_change:+.1f}%) - consistent with paradox")

    print(f"\n2. ITERATION METRIC (Commits per PR):")
    if commits_change > 5:
        print(f"   ✓ Increased by {commits_change:.1f}% - supports reallocation hypothesis (H4)")
    elif commits_change < -5:
        print(f"   ? Decreased by {abs(commits_change):.1f}% - unexpected")
    else:
        print(f"   ~ Minimal change ({commits_change:+.1f}%)")

    print(f"\n3. DIFFERENTIAL EFFECTS (DiD):")
    if did_estimate < 0:
        print(f"   ✓ High AI exposure repos improved more ({did_estimate:.1f}h) - supports H3")
    else:
        print(f"   ? Low AI exposure repos improved more - needs investigation")

    print("\n" + "="*70)

    return results


def create_real_data_figures(df: pd.DataFrame, output_dir: str = '../figures'):
    """Generate figures from real data analysis."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("matplotlib not available, skipping figures")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Figure: Lead time over time by exposure
    fig, ax = plt.subplots(figsize=(12, 6))

    for exposure, label, color in [(True, 'High AI Exposure', 'blue'),
                                    (False, 'Low AI Exposure', 'orange')]:
        subset = df[df['high_exposure'] == exposure]
        monthly = subset.groupby(subset['created_at'].dt.to_period('M'))['lead_time_hours'].mean()
        monthly.index = monthly.index.to_timestamp()
        ax.plot(monthly.index, monthly.values, color=color, label=label, linewidth=2)

    ax.axvline(x=TREATMENT_START, color='red', linestyle='--', linewidth=2, label='AI Adoption (Nov 2022)')
    ax.set_xlabel('Date')
    ax.set_ylabel('PR Lead Time (hours)')
    ax.set_title('Real GitHub Data: PR Lead Time Over Time\nby AI Exposure Level')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/real_data_lead_time.png', dpi=150)
    plt.close()
    print(f"Saved {output_dir}/real_data_lead_time.png")

    # Figure: Commits over time
    fig, ax = plt.subplots(figsize=(12, 6))

    for exposure, label, color in [(True, 'High AI Exposure', 'blue'),
                                    (False, 'Low AI Exposure', 'orange')]:
        subset = df[df['high_exposure'] == exposure]
        monthly = subset.groupby(subset['created_at'].dt.to_period('M'))['commits'].mean()
        monthly.index = monthly.index.to_timestamp()
        ax.plot(monthly.index, monthly.values, color=color, label=label, linewidth=2)

    ax.axvline(x=TREATMENT_START, color='red', linestyle='--', linewidth=2, label='AI Adoption (Nov 2022)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Commits per PR')
    ax.set_title('Real GitHub Data: Commits per PR Over Time\n(Iteration Intensity)')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/real_data_commits.png', dpi=150)
    plt.close()
    print(f"Saved {output_dir}/real_data_commits.png")


def main():
    """Run real data analysis."""
    from github_data_fetcher import fetch_and_save_real_data

    data_path = "../data/real_github_data.json"

    # Check if we already have data
    if not os.path.exists(data_path):
        print("Fetching real data from GitHub API...")
        fetch_and_save_real_data(data_path)
    else:
        print(f"Using existing data from {data_path}")

    # Load and analyze
    df = load_real_data(data_path)
    results = analyze_real_data(df)

    # Generate figures
    create_real_data_figures(df)

    return df, results


if __name__ == "__main__":
    df, results = main()
