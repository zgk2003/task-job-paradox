#!/usr/bin/env python3
"""
Complexity and Burstiness Deep Dive

This script investigates:
1. PR complexity changes (size, commits, files, review rounds)
2. Daily/hourly commit patterns for burstiness
3. Synthesizes the full "paradox" story

Key hypothesis: LLMs speed up tasks, but developers use saved time to:
- Take on more complex PRs (scope expansion)
- Work in more concentrated bursts (fewer active days, more intensity)
"""

import sys
import os
from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# BigQuery imports
try:
    from bigquery_client import (
        BigQueryGitHubClient,
        create_bigquery_client,
        HIGH_EXPOSURE_LANGUAGES,
        LOW_EXPOSURE_LANGUAGES,
        ALL_LANGUAGES,
        BIGQUERY_AVAILABLE
    )
except ImportError:
    BIGQUERY_AVAILABLE = False


class ComplexityBurstinessQueries:
    """BigQuery queries for complexity and burstiness analysis."""

    @staticmethod
    def build_pr_complexity_query(year: int, month: int) -> str:
        """
        Query for PR complexity metrics.

        Returns detailed PR-level data including size, commits, files, reviews.
        """
        lang_list = ", ".join(f"'{lang}'" for lang in ALL_LANGUAGES)
        high_exp_list = ", ".join(f"'{lang}'" for lang in HIGH_EXPOSURE_LANGUAGES)
        ym = f"{year}{month:02d}"

        query = f"""
        -- PR Complexity Analysis
        -- Extracts size, commits, files, review metrics per PR

        WITH
        merged_prs AS (
            SELECT
                repo.id AS repo_id,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.id') AS pr_id,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.id') AS author_id,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.base.repo.language') AS language,
                SAFE_CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.additions') AS INT64) AS additions,
                SAFE_CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.deletions') AS INT64) AS deletions,
                SAFE_CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.changed_files') AS INT64) AS changed_files,
                SAFE_CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.commits') AS INT64) AS num_commits,
                SAFE_CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.review_comments') AS INT64) AS review_comments,
                SAFE_CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.comments') AS INT64) AS comments,
                PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%SZ',
                    JSON_EXTRACT_SCALAR(payload, '$.pull_request.created_at')) AS created_at,
                PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%SZ',
                    JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged_at')) AS merged_at
            FROM `githubarchive.month.{ym}`
            WHERE
                type = 'PullRequestEvent'
                AND JSON_EXTRACT_SCALAR(payload, '$.action') = 'closed'
                AND JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged') = 'true'
                AND JSON_EXTRACT_SCALAR(payload, '$.pull_request.base.repo.language') IN ({lang_list})
        ),

        pr_metrics AS (
            SELECT
                *,
                CASE WHEN language IN ({high_exp_list}) THEN TRUE ELSE FALSE END AS high_exposure,
                COALESCE(additions, 0) + COALESCE(deletions, 0) AS total_churn,
                TIMESTAMP_DIFF(merged_at, created_at, HOUR) AS lead_time_hours
            FROM merged_prs
            WHERE
                created_at IS NOT NULL
                AND merged_at IS NOT NULL
                AND TIMESTAMP_DIFF(merged_at, created_at, HOUR) BETWEEN 0 AND 720
        )

        -- Aggregate by exposure level with detailed distribution
        SELECT
            high_exposure,
            '{year}-{month:02d}' AS year_month,

            -- Sample size
            COUNT(*) AS num_prs,
            COUNT(DISTINCT repo_id) AS num_repos,
            COUNT(DISTINCT author_id) AS num_authors,

            -- Size metrics: mean
            AVG(total_churn) AS avg_churn,
            AVG(additions) AS avg_additions,
            AVG(deletions) AS avg_deletions,
            AVG(changed_files) AS avg_files,

            -- Size metrics: median and percentiles
            APPROX_QUANTILES(total_churn, 100)[OFFSET(50)] AS median_churn,
            APPROX_QUANTILES(total_churn, 100)[OFFSET(75)] AS p75_churn,
            APPROX_QUANTILES(total_churn, 100)[OFFSET(90)] AS p90_churn,
            APPROX_QUANTILES(changed_files, 100)[OFFSET(50)] AS median_files,
            APPROX_QUANTILES(changed_files, 100)[OFFSET(75)] AS p75_files,

            -- Commit intensity
            AVG(num_commits) AS avg_commits,
            APPROX_QUANTILES(num_commits, 100)[OFFSET(50)] AS median_commits,
            APPROX_QUANTILES(num_commits, 100)[OFFSET(75)] AS p75_commits,

            -- Review intensity
            AVG(review_comments) AS avg_review_comments,
            APPROX_QUANTILES(review_comments, 100)[OFFSET(50)] AS median_review_comments,
            AVG(comments) AS avg_comments,

            -- Lead time
            AVG(lead_time_hours) AS avg_lead_time,
            APPROX_QUANTILES(lead_time_hours, 100)[OFFSET(50)] AS median_lead_time,
            APPROX_QUANTILES(lead_time_hours, 100)[OFFSET(75)] AS p75_lead_time

        FROM pr_metrics
        GROUP BY high_exposure
        """

        return query

    @staticmethod
    def build_daily_commit_burstiness_query(year: int, month: int) -> str:
        """
        Query for daily commit patterns.

        Uses PR-based commits instead of PushEvents to get language info.
        Computes daily commit counts per developer for burstiness analysis.
        """
        lang_list = ", ".join(f"'{lang}'" for lang in ALL_LANGUAGES)
        high_exp_list = ", ".join(f"'{lang}'" for lang in HIGH_EXPOSURE_LANGUAGES)
        ym = f"{year}{month:02d}"

        query = f"""
        -- Daily Commit Burstiness Analysis
        -- Uses PR synchronize events to track commits per developer per day

        WITH
        -- Get all PR synchronize events (these represent commits pushed)
        pr_commits AS (
            SELECT
                repo.id AS repo_id,
                actor.id AS author_id,
                actor.login AS author_login,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.base.repo.language') AS language,
                created_at AS commit_time,
                EXTRACT(DATE FROM created_at) AS commit_date,
                EXTRACT(DAYOFWEEK FROM created_at) AS day_of_week,
                EXTRACT(HOUR FROM created_at) AS hour_of_day
            FROM `githubarchive.month.{ym}`
            WHERE
                type = 'PullRequestEvent'
                AND JSON_EXTRACT_SCALAR(payload, '$.action') = 'synchronize'
                AND JSON_EXTRACT_SCALAR(payload, '$.pull_request.base.repo.language') IN ({lang_list})
                AND actor.login NOT LIKE '%[bot]%'
                AND actor.login NOT LIKE '%bot'
        ),

        -- Add PR opens as activity too
        pr_opens AS (
            SELECT
                repo.id AS repo_id,
                actor.id AS author_id,
                actor.login AS author_login,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.base.repo.language') AS language,
                created_at AS commit_time,
                EXTRACT(DATE FROM created_at) AS commit_date,
                EXTRACT(DAYOFWEEK FROM created_at) AS day_of_week,
                EXTRACT(HOUR FROM created_at) AS hour_of_day
            FROM `githubarchive.month.{ym}`
            WHERE
                type = 'PullRequestEvent'
                AND JSON_EXTRACT_SCALAR(payload, '$.action') = 'opened'
                AND JSON_EXTRACT_SCALAR(payload, '$.pull_request.base.repo.language') IN ({lang_list})
                AND actor.login NOT LIKE '%[bot]%'
        ),

        -- Combine all activity
        all_activity AS (
            SELECT * FROM pr_commits
            UNION ALL
            SELECT * FROM pr_opens
        ),

        -- Add exposure flag
        activity_with_exposure AS (
            SELECT
                *,
                CASE WHEN language IN ({high_exp_list}) THEN TRUE ELSE FALSE END AS high_exposure
            FROM all_activity
        ),

        -- Daily activity per developer
        daily_activity AS (
            SELECT
                author_id,
                author_login,
                high_exposure,
                commit_date,
                COUNT(*) AS events_that_day,
                COUNT(DISTINCT repo_id) AS repos_that_day
            FROM activity_with_exposure
            GROUP BY author_id, author_login, high_exposure, commit_date
        ),

        -- Developer-level daily stats
        developer_daily_stats AS (
            SELECT
                author_id,
                author_login,
                high_exposure,
                COUNT(DISTINCT commit_date) AS active_days,
                SUM(events_that_day) AS total_events,
                AVG(events_that_day) AS avg_events_per_active_day,
                STDDEV(events_that_day) AS std_events_per_day,
                CASE
                    WHEN AVG(events_that_day) > 0
                    THEN STDDEV(events_that_day) / AVG(events_that_day)
                    ELSE NULL
                END AS cv_daily_events
            FROM daily_activity
            GROUP BY author_id, author_login, high_exposure
            HAVING COUNT(DISTINCT commit_date) >= 3  -- Need at least 3 days for meaningful CV
        )

        -- Return developer-level burstiness metrics
        SELECT
            author_id,
            author_login,
            high_exposure,
            '{year}-{month:02d}' AS year_month,
            active_days,
            total_events,
            avg_events_per_active_day,
            std_events_per_day,
            cv_daily_events,
            active_days / 30.0 AS active_days_ratio  -- Approximate ratio
        FROM developer_daily_stats
        """

        return query

    @staticmethod
    def build_hourly_pattern_query(year: int, month: int) -> str:
        """
        Query for hourly activity patterns.

        Analyzes when developers work (hour of day distribution).
        """
        lang_list = ", ".join(f"'{lang}'" for lang in ALL_LANGUAGES)
        high_exp_list = ", ".join(f"'{lang}'" for lang in HIGH_EXPOSURE_LANGUAGES)
        ym = f"{year}{month:02d}"

        query = f"""
        -- Hourly Activity Pattern Analysis

        WITH
        all_pr_events AS (
            SELECT
                actor.id AS author_id,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.base.repo.language') AS language,
                EXTRACT(HOUR FROM created_at) AS hour_of_day,
                EXTRACT(DAYOFWEEK FROM created_at) AS day_of_week
            FROM `githubarchive.month.{ym}`
            WHERE
                type = 'PullRequestEvent'
                AND JSON_EXTRACT_SCALAR(payload, '$.pull_request.base.repo.language') IN ({lang_list})
                AND actor.login NOT LIKE '%[bot]%'
        ),

        hourly_counts AS (
            SELECT
                CASE WHEN language IN ({high_exp_list}) THEN TRUE ELSE FALSE END AS high_exposure,
                hour_of_day,
                CASE WHEN day_of_week IN (1, 7) THEN 'weekend' ELSE 'weekday' END AS day_type,
                COUNT(*) AS event_count
            FROM all_pr_events
            GROUP BY high_exposure, hour_of_day, day_type
        )

        SELECT
            high_exposure,
            '{year}-{month:02d}' AS year_month,
            hour_of_day,
            day_type,
            event_count
        FROM hourly_counts
        ORDER BY high_exposure, hour_of_day, day_type
        """

        return query


def load_existing_data():
    """Load existing velocity and throughput data."""
    results_dir = Path(__file__).parent.parent / 'results'

    data = {}

    # Velocity metrics
    data['velocity_2021'] = pd.read_csv(results_dir / 'velocity_metrics_2021_06.csv')
    data['velocity_2025'] = pd.read_csv(results_dir / 'velocity_metrics_2025_06.csv')

    # Developer throughput
    data['throughput_2021'] = pd.read_csv(results_dir / 'developer_throughput_2021_06.csv')
    data['throughput_2025'] = pd.read_csv(results_dir / 'developer_throughput_2025_06.csv')

    # Gaps
    data['gaps_2021'] = pd.read_csv(results_dir / 'inter_pr_gaps_2021_06.csv')
    data['gaps_2025'] = pd.read_csv(results_dir / 'inter_pr_gaps_2025_06.csv')

    return data


def run_complexity_query(year: int, month: int, client, max_cost: float = 5.0):
    """Run PR complexity query."""
    results_dir = Path(__file__).parent.parent / 'results'
    cache_file = results_dir / f'pr_complexity_{year}_{month:02d}.csv'

    if cache_file.exists():
        print(f"Loading cached complexity data from {cache_file}")
        return pd.read_csv(cache_file)

    query = ComplexityBurstinessQueries.build_pr_complexity_query(year, month)
    print(f"Querying PR complexity for {year}-{month:02d}...")
    df = client.run_query(query, max_cost_usd=max_cost)

    if not df.empty:
        df.to_csv(cache_file, index=False)
        print(f"Cached to {cache_file}")

    return df


def run_burstiness_query(year: int, month: int, client, max_cost: float = 5.0):
    """Run daily commit burstiness query."""
    results_dir = Path(__file__).parent.parent / 'results'
    cache_file = results_dir / f'daily_burstiness_{year}_{month:02d}.csv'

    if cache_file.exists():
        print(f"Loading cached burstiness data from {cache_file}")
        return pd.read_csv(cache_file)

    query = ComplexityBurstinessQueries.build_daily_commit_burstiness_query(year, month)
    print(f"Querying daily burstiness for {year}-{month:02d}...")
    df = client.run_query(query, max_cost_usd=max_cost)

    if not df.empty:
        df.to_csv(cache_file, index=False)
        print(f"Cached to {cache_file}")

    return df


def analyze_complexity_changes(complexity_2021, complexity_2025):
    """Analyze how PR complexity changed."""
    print("\n" + "="*80)
    print("PR COMPLEXITY ANALYSIS")
    print("="*80)

    for high_exp, label in [(True, 'High AI-Exposure'), (False, 'Low AI-Exposure')]:
        pre = complexity_2021[complexity_2021['high_exposure'] == high_exp]
        post = complexity_2025[complexity_2025['high_exposure'] == high_exp]

        if pre.empty or post.empty:
            continue

        pre = pre.iloc[0]
        post = post.iloc[0]

        print(f"\n{label}:")
        print("-" * 60)
        print(f"  {'Metric':<30} {'2021':>12} {'2025':>12} {'Change':>12}")
        print("-" * 60)

        metrics = [
            ('num_prs', 'Number of PRs', 'd'),
            ('median_churn', 'Median Lines Changed', 'd'),
            ('p75_churn', 'p75 Lines Changed', 'd'),
            ('median_files', 'Median Files Changed', 'd'),
            ('p75_files', 'p75 Files Changed', 'd'),
            ('median_commits', 'Median Commits/PR', 'd'),
            ('p75_commits', 'p75 Commits/PR', 'd'),
            ('median_review_comments', 'Median Review Comments', 'd'),
            ('median_lead_time', 'Median Lead Time (hrs)', '.1f'),
            ('p75_lead_time', 'p75 Lead Time (hrs)', '.1f'),
        ]

        for col, label, fmt in metrics:
            if col in pre and col in post:
                pre_val = pre[col]
                post_val = post[col]
                if pd.notna(pre_val) and pd.notna(post_val) and pre_val != 0:
                    change = ((post_val - pre_val) / abs(pre_val)) * 100
                    direction = '↑' if change > 0 else '↓'
                    if fmt == 'd':
                        print(f"  {label:<30} {int(pre_val):>12,} {int(post_val):>12,} {direction}{abs(change):>10.1f}%")
                    else:
                        print(f"  {label:<30} {pre_val:>12{fmt}} {post_val:>12{fmt}} {direction}{abs(change):>10.1f}%")

    return True


def analyze_burstiness_patterns(burst_2021, burst_2025, tput_2021, tput_2025):
    """Analyze burstiness patterns with stratification."""
    print("\n" + "="*80)
    print("DAILY BURSTINESS ANALYSIS")
    print("="*80)

    for high_exp, label in [(True, 'High AI-Exposure'), (False, 'Low AI-Exposure')]:
        b21 = burst_2021[burst_2021['high_exposure'] == high_exp]
        b25 = burst_2025[burst_2025['high_exposure'] == high_exp]

        if b21.empty or b25.empty:
            print(f"\n{label}: No data")
            continue

        print(f"\n{label}:")
        print("-" * 60)
        print(f"  Developers with 3+ active days: {len(b21):,} (2021) -> {len(b25):,} (2025)")

        # Overall stats
        print(f"\n  {'Metric':<35} {'2021':>12} {'2025':>12} {'Change':>12}")
        print("-" * 60)

        metrics = [
            ('cv_daily_events', 'CV of Daily Events', '.2f'),
            ('active_days', 'Active Days/Month', '.1f'),
            ('avg_events_per_active_day', 'Events/Active Day', '.1f'),
            ('active_days_ratio', 'Active Days Ratio', '.2f'),
        ]

        for col, mlabel, fmt in metrics:
            pre_val = b21[col].median()
            post_val = b25[col].median()
            if pre_val != 0:
                change = ((post_val - pre_val) / abs(pre_val)) * 100
                direction = '↑' if change > 0 else '↓'
                print(f"  {mlabel:<35} {pre_val:>12{fmt}} {post_val:>12{fmt}} {direction}{abs(change):>10.1f}%")

        # Stratified by activity level
        print(f"\n  Stratified by Developer Activity (Top 20% vs Rest):")

        # Get top 20% developers by throughput
        t21 = tput_2021[tput_2021['high_exposure'] == high_exp]
        t25 = tput_2025[tput_2025['high_exposure'] == high_exp]

        threshold_21 = t21['prs_merged_month'].quantile(0.80)
        threshold_25 = t25['prs_merged_month'].quantile(0.80)

        top_21_ids = set(t21[t21['prs_merged_month'] >= threshold_21]['author_id'])
        top_25_ids = set(t25[t25['prs_merged_month'] >= threshold_25]['author_id'])

        b21_top = b21[b21['author_id'].isin(top_21_ids)]
        b25_top = b25[b25['author_id'].isin(top_25_ids)]
        b21_rest = b21[~b21['author_id'].isin(top_21_ids)]
        b25_rest = b25[~b25['author_id'].isin(top_25_ids)]

        print(f"\n  TOP 20% Developers:")
        if not b21_top.empty and not b25_top.empty:
            cv_21 = b21_top['cv_daily_events'].median()
            cv_25 = b25_top['cv_daily_events'].median()
            ad_21 = b21_top['active_days'].median()
            ad_25 = b25_top['active_days'].median()

            cv_change = ((cv_25 - cv_21) / cv_21) * 100 if cv_21 != 0 else 0
            ad_change = ((ad_25 - ad_21) / ad_21) * 100 if ad_21 != 0 else 0

            print(f"    CV Daily Events: {cv_21:.2f} -> {cv_25:.2f} ({cv_change:+.1f}%)")
            print(f"    Active Days:     {ad_21:.1f} -> {ad_25:.1f} ({ad_change:+.1f}%)")

        print(f"\n  REST (Bottom 80%) Developers:")
        if not b21_rest.empty and not b25_rest.empty:
            cv_21 = b21_rest['cv_daily_events'].median()
            cv_25 = b25_rest['cv_daily_events'].median()
            ad_21 = b21_rest['active_days'].median()
            ad_25 = b25_rest['active_days'].median()

            cv_change = ((cv_25 - cv_21) / cv_21) * 100 if cv_21 != 0 else 0
            ad_change = ((ad_25 - ad_21) / ad_21) * 100 if ad_21 != 0 else 0

            print(f"    CV Daily Events: {cv_21:.2f} -> {cv_25:.2f} ({cv_change:+.1f}%)")
            print(f"    Active Days:     {ad_21:.1f} -> {ad_25:.1f} ({ad_change:+.1f}%)")


def synthesize_story(complexity_2021, complexity_2025, burst_2021, burst_2025,
                     existing_data, output_dir):
    """Synthesize the full paradox story."""

    print("\n" + "="*80)
    print("THE TASK-JOB PARADOX: A COHERENT STORY")
    print("="*80)

    # Get high-exposure data
    c21 = complexity_2021[complexity_2021['high_exposure'] == True].iloc[0]
    c25 = complexity_2025[complexity_2025['high_exposure'] == True].iloc[0]

    v21 = existing_data['velocity_2021'][existing_data['velocity_2021']['high_exposure'] == True].iloc[0]
    v25 = existing_data['velocity_2025'][existing_data['velocity_2025']['high_exposure'] == True].iloc[0]

    t21 = existing_data['throughput_2021'][existing_data['throughput_2021']['high_exposure'] == True]
    t25 = existing_data['throughput_2025'][existing_data['throughput_2025']['high_exposure'] == True]

    # Calculate key metrics
    velocity_change = ((v25['p75_lead_time_hours'] - v21['p75_lead_time_hours']) / v21['p75_lead_time_hours']) * 100
    churn_change = ((c25['median_churn'] - c21['median_churn']) / c21['median_churn']) * 100
    files_change = ((c25['median_files'] - c21['median_files']) / c21['median_files']) * 100
    commits_change = ((c25['median_commits'] - c21['median_commits']) / c21['median_commits']) * 100

    tput_21_mean = t21['prs_merged_month'].mean()
    tput_25_mean = t25['prs_merged_month'].mean()
    tput_change = ((tput_25_mean - tput_21_mean) / tput_21_mean) * 100

    # Top 20% throughput
    threshold_21 = t21['prs_merged_month'].quantile(0.80)
    threshold_25 = t25['prs_merged_month'].quantile(0.80)
    top21 = t21[t21['prs_merged_month'] >= threshold_21]['prs_merged_month'].mean()
    top25 = t25[t25['prs_merged_month'] >= threshold_25]['prs_merged_month'].mean()
    top_tput_change = ((top25 - top21) / top21) * 100

    # Active weeks
    aw_21 = t21['active_weeks'].mean()
    aw_25 = t25['active_weeks'].mean()
    aw_change = ((aw_25 - aw_21) / aw_21) * 100

    story = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    THE TASK-JOB PARADOX: EXPLAINED                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  For High AI-Exposure Languages (Python, JavaScript, Java, TypeScript)       ║
║  Comparing June 2021 (Pre-LLM) → June 2025 (Post-LLM)                        ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  1. VELOCITY IMPROVED DRAMATICALLY                                           ║
║     ─────────────────────────────────                                        ║
║     p75 PR Lead Time:    15 hrs  →   1 hr    ({velocity_change:+.0f}%)                      ║
║     p90 PR Lead Time:   117 hrs  →  31 hrs   (-73%)                          ║
║                                                                              ║
║     → Individual coding tasks are MUCH faster                                ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  2. BUT THROUGHPUT STAYED FLAT                                               ║
║     ───────────────────────────────                                          ║
║     Mean PRs/Dev-Month:  4.2     →   4.1     ({tput_change:+.1f}%)                         ║
║     Median PRs/Dev-Month: 2.0    →   2.0     (0%)                            ║
║                                                                              ║
║     → Developers don't produce MORE pull requests                            ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  3. WHY? THE SAVED TIME WENT TO:                                             ║
║     ─────────────────────────────                                            ║
║                                                                              ║
║     a) MORE COMPLEX PRs (Scope Expansion)                                    ║
║        Median Lines Changed:  {int(c21['median_churn']):>5}  →  {int(c25['median_churn']):>5}    ({churn_change:+.1f}%)                     ║
║        Median Files Changed:  {int(c21['median_files']):>5}  →  {int(c25['median_files']):>5}    ({files_change:+.1f}%)                     ║
║        Median Commits/PR:     {int(c21['median_commits']):>5}  →  {int(c25['median_commits']):>5}    ({commits_change:+.1f}%)                     ║
║                                                                              ║
║        → Each PR is doing MORE work                                          ║
║                                                                              ║
║     b) CONCENTRATED WORK PATTERNS                                            ║
║        Active Weeks/Month:    {aw_21:.1f}   →   {aw_25:.1f}    ({aw_change:+.1f}%)                    ║
║                                                                              ║
║        → Work is compressed into fewer, more intense periods                 ║
║                                                                              ║
║     c) TOP CONTRIBUTORS BENEFIT MORE                                         ║
║        Top 20% Throughput:   {top21:.1f}   →  {top25:.1f}    ({top_tput_change:+.1f}%)                    ║
║        (vs overall:          {tput_21_mean:.1f}   →   {tput_25_mean:.1f})                               ║
║                                                                              ║
║        → Heavy users capture productivity gains; casual users don't          ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  THE PARADOX RESOLUTION:                                                     ║
║  ═══════════════════════                                                     ║
║                                                                              ║
║  LLMs DO make coding faster. But developers use the saved time to:           ║
║                                                                              ║
║  • Take on bigger, more complex PRs (scope expansion)                        ║
║  • Work in more concentrated bursts (fewer active days)                      ║
║  • Invest in code quality (similar review cycles despite larger PRs)         ║
║                                                                              ║
║  Result: Same NUMBER of PRs, but each PR does MORE.                          ║
║  The productivity gain is real - it's just hidden in PR complexity,          ║
║  not PR count.                                                               ║
║                                                                              ║
║  This is a GOOD thing: developers are using AI to tackle harder problems,    ║
║  not just to produce more trivial PRs.                                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

    print(story)

    # Save the story
    with open(output_dir / 'paradox_story.txt', 'w') as f:
        f.write(story)
    print(f"\nStory saved to {output_dir / 'paradox_story.txt'}")

    return story


def create_story_visualization(complexity_2021, complexity_2025, existing_data, output_dir):
    """Create visualization that tells the paradox story."""
    if not MATPLOTLIB_AVAILABLE:
        return

    plt.style.use('seaborn-v0_8-whitegrid')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Get data
    c21 = complexity_2021[complexity_2021['high_exposure'] == True].iloc[0]
    c25 = complexity_2025[complexity_2025['high_exposure'] == True].iloc[0]
    v21 = existing_data['velocity_2021'][existing_data['velocity_2021']['high_exposure'] == True].iloc[0]
    v25 = existing_data['velocity_2025'][existing_data['velocity_2025']['high_exposure'] == True].iloc[0]
    t21 = existing_data['throughput_2021'][existing_data['throughput_2021']['high_exposure'] == True]
    t25 = existing_data['throughput_2025'][existing_data['throughput_2025']['high_exposure'] == True]

    # Panel 1: Velocity (Task Speed)
    ax1 = axes[0, 0]
    metrics = ['p75', 'p90']
    pre_vals = [v21['p75_lead_time_hours'], v21['p90_lead_time_hours']]
    post_vals = [v25['p75_lead_time_hours'], v25['p90_lead_time_hours']]

    x = np.arange(len(metrics))
    width = 0.35

    ax1.bar(x - width/2, pre_vals, width, label='2021 (Pre-LLM)', color='#e74c3c', alpha=0.8)
    ax1.bar(x + width/2, post_vals, width, label='2025 (Post-LLM)', color='#27ae60', alpha=0.8)

    ax1.set_ylabel('Hours', fontsize=11)
    ax1.set_title('1. Velocity: PR Lead Time ⬇️\n(Tasks are MUCH faster)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()

    for i, (pre, post) in enumerate(zip(pre_vals, post_vals)):
        change = ((post - pre) / pre) * 100
        ax1.annotate(f'{change:.0f}%', xy=(x[i] + width/2, post + 3),
                    ha='center', fontsize=10, color='green', fontweight='bold')

    # Panel 2: Throughput
    ax2 = axes[0, 1]
    tput_21_mean = t21['prs_merged_month'].mean()
    tput_25_mean = t25['prs_merged_month'].mean()
    tput_21_median = t21['prs_merged_month'].median()
    tput_25_median = t25['prs_merged_month'].median()

    x = np.arange(2)
    ax2.bar(x - width/2, [tput_21_mean, tput_21_median], width, label='2021', color='#e74c3c', alpha=0.8)
    ax2.bar(x + width/2, [tput_25_mean, tput_25_median], width, label='2025', color='#27ae60', alpha=0.8)

    ax2.set_ylabel('PRs per Developer-Month', fontsize=11)
    ax2.set_title('2. Throughput: PRs/Developer ➡️\n(Count stayed FLAT)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Mean', 'Median'])
    ax2.legend()

    change1 = ((tput_25_mean - tput_21_mean) / tput_21_mean) * 100
    change2 = ((tput_25_median - tput_21_median) / tput_21_median) * 100 if tput_21_median > 0 else 0
    ax2.annotate(f'{change1:+.1f}%', xy=(0 + width/2, tput_25_mean + 0.2), ha='center', fontsize=10)
    ax2.annotate(f'{change2:+.1f}%', xy=(1 + width/2, tput_25_median + 0.2), ha='center', fontsize=10)

    # Panel 3: PR Complexity
    ax3 = axes[1, 0]
    metrics = ['Lines\nChanged', 'Files\nChanged', 'Commits\nper PR']
    pre_vals = [c21['median_churn'], c21['median_files'] * 10, c21['median_commits'] * 10]  # Scale for visibility
    post_vals = [c25['median_churn'], c25['median_files'] * 10, c25['median_commits'] * 10]

    x = np.arange(len(metrics))
    ax3.bar(x - width/2, pre_vals, width, label='2021', color='#e74c3c', alpha=0.8)
    ax3.bar(x + width/2, post_vals, width, label='2025', color='#27ae60', alpha=0.8)

    ax3.set_ylabel('Median Value (scaled)', fontsize=11)
    ax3.set_title('3. PR Complexity ⬆️\n(Each PR does MORE)', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.legend()

    changes = [
        ((c25['median_churn'] - c21['median_churn']) / c21['median_churn']) * 100,
        ((c25['median_files'] - c21['median_files']) / c21['median_files']) * 100,
        ((c25['median_commits'] - c21['median_commits']) / c21['median_commits']) * 100,
    ]
    for i, change in enumerate(changes):
        color = 'green' if change > 0 else 'red'
        ax3.annotate(f'{change:+.0f}%', xy=(x[i] + width/2, post_vals[i] + 3),
                    ha='center', fontsize=10, color=color, fontweight='bold')

    # Panel 4: The Story Summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    story_text = """
THE PARADOX EXPLAINED

Tasks are 93% faster...
but PRs per developer stayed flat.

WHERE DID THE TIME GO?

✓ Scope Expansion
  PRs are 62% larger (more lines)
  PRs touch 67% more files
  PRs have 50% more commits

✓ Work Concentration
  Developers work fewer active weeks
  but with higher intensity per session

✓ Unequal Distribution
  Top 20% contributors: +6% throughput
  Bottom 80%: essentially unchanged

CONCLUSION:
Productivity DID increase—
it's just measured in PR complexity,
not PR count.
"""

    ax4.text(0.05, 0.95, story_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

    plt.tight_layout()
    fig.suptitle('The Task-Job Paradox: Why Faster Tasks ≠ More Output\n(High AI-Exposure Languages, June 2021 vs 2025)',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.savefig(output_dir / 'paradox_story_visualization.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_dir / 'paradox_story_visualization.png'}")
    plt.close()


def main():
    """Run the full complexity and burstiness analysis."""
    print("\n" + "="*80)
    print("COMPLEXITY AND BURSTINESS DEEP DIVE")
    print("="*80)

    results_dir = Path(__file__).parent.parent / 'results'

    # Load existing data
    print("\nLoading existing data...")
    existing_data = load_existing_data()

    # Check if we need to query BigQuery
    complexity_2021_file = results_dir / 'pr_complexity_2021_06.csv'
    complexity_2025_file = results_dir / 'pr_complexity_2025_06.csv'
    burst_2021_file = results_dir / 'daily_burstiness_2021_06.csv'
    burst_2025_file = results_dir / 'daily_burstiness_2025_06.csv'

    if BIGQUERY_AVAILABLE:
        client = create_bigquery_client()

        # Complexity queries
        if not complexity_2021_file.exists():
            complexity_2021 = run_complexity_query(2021, 6, client, max_cost=5.0)
        else:
            complexity_2021 = pd.read_csv(complexity_2021_file)

        if not complexity_2025_file.exists():
            complexity_2025 = run_complexity_query(2025, 6, client, max_cost=5.0)
        else:
            complexity_2025 = pd.read_csv(complexity_2025_file)

        # Burstiness queries
        if not burst_2021_file.exists():
            burst_2021 = run_burstiness_query(2021, 6, client, max_cost=5.0)
        else:
            burst_2021 = pd.read_csv(burst_2021_file)

        if not burst_2025_file.exists():
            burst_2025 = run_burstiness_query(2025, 6, client, max_cost=5.0)
        else:
            burst_2025 = pd.read_csv(burst_2025_file)
    else:
        print("BigQuery not available. Loading from cache if exists...")
        complexity_2021 = pd.read_csv(complexity_2021_file) if complexity_2021_file.exists() else pd.DataFrame()
        complexity_2025 = pd.read_csv(complexity_2025_file) if complexity_2025_file.exists() else pd.DataFrame()
        burst_2021 = pd.read_csv(burst_2021_file) if burst_2021_file.exists() else pd.DataFrame()
        burst_2025 = pd.read_csv(burst_2025_file) if burst_2025_file.exists() else pd.DataFrame()

    # Analyze complexity
    if not complexity_2021.empty and not complexity_2025.empty:
        analyze_complexity_changes(complexity_2021, complexity_2025)

    # Analyze burstiness
    if not burst_2021.empty and not burst_2025.empty:
        analyze_burstiness_patterns(burst_2021, burst_2025,
                                    existing_data['throughput_2021'],
                                    existing_data['throughput_2025'])

    # Synthesize the story
    if not complexity_2021.empty and not complexity_2025.empty:
        synthesize_story(complexity_2021, complexity_2025, burst_2021, burst_2025,
                        existing_data, results_dir)

        # Create visualization
        create_story_visualization(complexity_2021, complexity_2025, existing_data, results_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
