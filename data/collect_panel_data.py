#!/usr/bin/env python3
"""
Panel Data Collection Script for Task-Job Paradox

Collects monthly data from GitHub Archive (2021-01 to 2025-06) for:
- Velocity (PR lead times, review times)
- Throughput (developer-level PR counts)
- Complexity (PR size metrics)
- Burstiness (daily activity patterns)
- Slack (inter-PR gaps with multi-month lookback)
- Controls (repo/author characteristics)

Usage:
    python collect_panel_data.py --project-id YOUR_PROJECT_ID [options]

Options:
    --start-year 2021       Start year
    --start-month 1         Start month
    --end-year 2025         End year
    --end-month 6           End month
    --metrics velocity,throughput,complexity,burstiness,slack,controls
    --skip-existing         Skip months already collected
    --dry-run               Print queries without executing
    --max-cost 100.0        Maximum total cost in USD
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, date
from typing import List, Dict, Optional, Tuple
from calendar import monthrange
import time

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.bigquery_client import create_bigquery_client

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent
PANEL_DIR = DATA_DIR / 'panel'
LOG_FILE = DATA_DIR / 'collection_log.json'

HIGH_EXPOSURE_LANGUAGES = ['Python', 'JavaScript', 'Java', 'TypeScript']
LOW_EXPOSURE_LANGUAGES = ['Fortran', 'COBOL', 'Assembly', 'Erlang', 'Haskell']
ALL_LANGUAGES = HIGH_EXPOSURE_LANGUAGES + LOW_EXPOSURE_LANGUAGES

# Cost estimates per query type (USD)
COST_ESTIMATES = {
    'velocity': 0.10,
    'throughput': 0.50,
    'complexity': 0.20,
    'burstiness': 0.30,
    'slack': 0.40,
    'controls': 0.15,
}


# =============================================================================
# IMPROVED QUERIES (addressing advisor feedback)
# =============================================================================

def get_velocity_query(year: int, month: int) -> str:
    """
    Velocity metrics with review time from adjacent months.

    Improvement: Joins review events from current AND next month to avoid
    right-censoring of reviews that happen after month boundary.
    """
    days_in_month = monthrange(year, month)[1]
    start_date = f'{year}-{month:02d}-01'
    end_date = f'{year}-{month:02d}-{days_in_month}'

    # For review lookback, include next month
    if month == 12:
        next_year, next_month = year + 1, 1
    else:
        next_year, next_month = year, month + 1

    next_days = monthrange(next_year, next_month)[1]
    review_end_date = f'{next_year}-{next_month:02d}-{next_days}'

    languages_sql = ", ".join([f"'{lang}'" for lang in ALL_LANGUAGES])
    high_exp_sql = ", ".join([f"'{lang}'" for lang in HIGH_EXPOSURE_LANGUAGES])

    return f"""
    WITH pr_events AS (
        SELECT
            JSON_EXTRACT_SCALAR(payload, '$.pull_request.id') as pr_id,
            JSON_EXTRACT_SCALAR(payload, '$.pull_request.number') as pr_number,
            repo.name as repo_name,
            JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.login') as author,
            JSON_EXTRACT_SCALAR(payload, '$.action') as action,
            created_at,
            TIMESTAMP(JSON_EXTRACT_SCALAR(payload, '$.pull_request.created_at')) as pr_created_at,
            TIMESTAMP(JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged_at')) as pr_merged_at,
            CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.additions') AS INT64) as additions,
            CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.deletions') AS INT64) as deletions,
            repo.id as repo_id
        FROM `githubarchive.month.{year}{month:02d}`
        WHERE type = 'PullRequestEvent'
            AND JSON_EXTRACT_SCALAR(payload, '$.action') IN ('opened', 'closed')
    ),

    -- Get repo languages
    repo_languages AS (
        SELECT DISTINCT
            repo.id as repo_id,
            repo.name as repo_name,
            (SELECT l.name FROM UNNEST([
                STRUCT('Python' as name, REGEXP_CONTAINS(LOWER(repo.name), r'python|django|flask|pytorch|tensorflow') as match),
                STRUCT('JavaScript' as name, REGEXP_CONTAINS(LOWER(repo.name), r'javascript|nodejs|react|vue|angular') as match),
                STRUCT('Java' as name, REGEXP_CONTAINS(LOWER(repo.name), r'java|spring|android') as match),
                STRUCT('TypeScript' as name, REGEXP_CONTAINS(LOWER(repo.name), r'typescript|angular|nest') as match)
            ]) l WHERE l.match LIMIT 1) as detected_lang
        FROM `githubarchive.month.{year}{month:02d}`
        WHERE type = 'PushEvent'
    ),

    -- Merged PRs created this month
    merged_prs AS (
        SELECT
            pr_id,
            pr_number,
            repo_name,
            author,
            pr_created_at,
            pr_merged_at,
            additions,
            deletions,
            repo_id,
            TIMESTAMP_DIFF(pr_merged_at, pr_created_at, HOUR) as lead_time_hours
        FROM pr_events
        WHERE action = 'closed'
            AND pr_merged_at IS NOT NULL
            AND pr_created_at >= '{start_date}'
            AND pr_created_at < '{end_date}'
            AND TIMESTAMP_DIFF(pr_merged_at, pr_created_at, HOUR) BETWEEN 0 AND 720
    ),

    -- Review events (current + next month for censoring fix)
    review_events AS (
        SELECT
            JSON_EXTRACT_SCALAR(payload, '$.pull_request.id') as pr_id,
            created_at as review_time
        FROM `githubarchive.month.{year}{month:02d}`
        WHERE type = 'PullRequestReviewEvent'

        UNION ALL

        SELECT
            JSON_EXTRACT_SCALAR(payload, '$.pull_request.id') as pr_id,
            created_at as review_time
        FROM `githubarchive.month.{next_year}{next_month:02d}`
        WHERE type = 'PullRequestReviewEvent'
    ),

    -- First review per PR
    first_reviews AS (
        SELECT
            pr_id,
            MIN(review_time) as first_review_time
        FROM review_events
        GROUP BY pr_id
    ),

    -- Join with language info
    prs_with_lang AS (
        SELECT
            m.*,
            COALESCE(rl.detected_lang, 'Unknown') as language,
            CASE WHEN COALESCE(rl.detected_lang, 'Unknown') IN ({high_exp_sql}) THEN TRUE ELSE FALSE END as high_exposure,
            fr.first_review_time,
            TIMESTAMP_DIFF(fr.first_review_time, m.pr_created_at, HOUR) as time_to_review_hours
        FROM merged_prs m
        LEFT JOIN repo_languages rl ON m.repo_id = rl.repo_id
        LEFT JOIN first_reviews fr ON m.pr_id = fr.pr_id
    )

    SELECT
        high_exposure,
        '{year}-{month:02d}' as year_month,
        COUNT(*) as num_prs,
        COUNT(DISTINCT repo_name) as num_repos,
        COUNT(DISTINCT author) as num_authors,
        AVG(lead_time_hours) as avg_lead_time_hours,
        APPROX_QUANTILES(lead_time_hours, 100)[OFFSET(50)] as median_lead_time_hours,
        APPROX_QUANTILES(lead_time_hours, 100)[OFFSET(75)] as p75_lead_time_hours,
        APPROX_QUANTILES(lead_time_hours, 100)[OFFSET(90)] as p90_lead_time_hours,
        STDDEV(lead_time_hours) as std_lead_time_hours,
        AVG(time_to_review_hours) as avg_time_to_review_hours,
        APPROX_QUANTILES(time_to_review_hours, 100)[OFFSET(50)] as median_time_to_review_hours,
        AVG(additions + deletions) as avg_pr_size,
        APPROX_QUANTILES(additions + deletions, 100)[OFFSET(50)] as median_pr_size
    FROM prs_with_lang
    WHERE high_exposure IS NOT NULL
    GROUP BY high_exposure
    ORDER BY high_exposure DESC
    """


def get_throughput_query(year: int, month: int) -> str:
    """
    Developer-level throughput with proper weekly CV calculation.

    Improvement: Generates full week calendar and fills zeros for inactive weeks.
    """
    days_in_month = monthrange(year, month)[1]
    start_date = f'{year}-{month:02d}-01'
    end_date = f'{year}-{month:02d}-{days_in_month}'

    high_exp_sql = ", ".join([f"'{lang}'" for lang in HIGH_EXPOSURE_LANGUAGES])

    return f"""
    WITH merged_prs AS (
        SELECT
            JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.login') as author,
            repo.name as repo_name,
            TIMESTAMP(JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged_at')) as merged_at,
            CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.additions') AS INT64) as additions,
            CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.deletions') AS INT64) as deletions,
            EXTRACT(WEEK FROM TIMESTAMP(JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged_at'))) as week_num
        FROM `githubarchive.month.{year}{month:02d}`
        WHERE type = 'PullRequestEvent'
            AND JSON_EXTRACT_SCALAR(payload, '$.action') = 'closed'
            AND JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged') = 'true'
            AND JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.login') NOT LIKE '%[bot]%'
    ),

    -- Detect primary language per author
    author_languages AS (
        SELECT
            author,
            -- Simplified: mark as high exposure if majority of repos suggest it
            TRUE as high_exposure  -- Will be refined with actual language detection
        FROM merged_prs
        GROUP BY author
    ),

    -- Generate all weeks in the month for each author
    weeks_in_month AS (
        SELECT DISTINCT week_num
        FROM UNNEST(GENERATE_ARRAY(
            EXTRACT(WEEK FROM DATE '{start_date}'),
            EXTRACT(WEEK FROM DATE '{end_date}')
        )) as week_num
    ),

    -- All author-week combinations
    author_weeks AS (
        SELECT DISTINCT
            m.author,
            w.week_num
        FROM merged_prs m
        CROSS JOIN weeks_in_month w
    ),

    -- PRs per author per week (with zeros)
    weekly_counts AS (
        SELECT
            aw.author,
            aw.week_num,
            COALESCE(COUNT(m.merged_at), 0) as prs_this_week
        FROM author_weeks aw
        LEFT JOIN merged_prs m ON aw.author = m.author AND aw.week_num = m.week_num
        GROUP BY aw.author, aw.week_num
    ),

    -- Compute CV including zeros
    weekly_stats AS (
        SELECT
            author,
            AVG(prs_this_week) as avg_weekly_prs,
            STDDEV(prs_this_week) as std_weekly_prs,
            COUNT(*) as num_weeks
        FROM weekly_counts
        GROUP BY author
    ),

    -- Author-level aggregates
    author_summary AS (
        SELECT
            m.author,
            COUNT(*) as prs_merged_month,
            COUNT(DISTINCT m.week_num) as active_weeks,
            SUM(m.additions) as total_additions,
            SUM(m.deletions) as total_deletions
        FROM merged_prs m
        GROUP BY m.author
    )

    SELECT
        a.author,
        TRUE as high_exposure,  -- Simplified; refine with language detection
        a.prs_merged_month,
        a.active_weeks,
        SAFE_DIVIDE(ws.std_weekly_prs, ws.avg_weekly_prs) as cv_weekly_prs,
        a.total_additions,
        a.total_deletions
    FROM author_summary a
    LEFT JOIN weekly_stats ws ON a.author = ws.author
    WHERE a.prs_merged_month >= 1
    ORDER BY a.prs_merged_month DESC
    """


def get_complexity_query(year: int, month: int) -> str:
    """PR complexity metrics by exposure group."""
    days_in_month = monthrange(year, month)[1]
    start_date = f'{year}-{month:02d}-01'
    end_date = f'{year}-{month:02d}-{days_in_month}'

    high_exp_sql = ", ".join([f"'{lang}'" for lang in HIGH_EXPOSURE_LANGUAGES])

    return f"""
    WITH merged_prs AS (
        SELECT
            JSON_EXTRACT_SCALAR(payload, '$.pull_request.id') as pr_id,
            repo.name as repo_name,
            CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.additions') AS INT64) as additions,
            CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.deletions') AS INT64) as deletions,
            CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.changed_files') AS INT64) as files_changed,
            CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.commits') AS INT64) as commits,
            CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.review_comments') AS INT64) as review_comments,
            TIMESTAMP(JSON_EXTRACT_SCALAR(payload, '$.pull_request.created_at')) as created_at,
            TIMESTAMP(JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged_at')) as merged_at,
            TIMESTAMP_DIFF(
                TIMESTAMP(JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged_at')),
                TIMESTAMP(JSON_EXTRACT_SCALAR(payload, '$.pull_request.created_at')),
                HOUR
            ) as lead_time_hours
        FROM `githubarchive.month.{year}{month:02d}`
        WHERE type = 'PullRequestEvent'
            AND JSON_EXTRACT_SCALAR(payload, '$.action') = 'closed'
            AND JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged') = 'true'
            AND JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.login') NOT LIKE '%[bot]%'
    ),

    prs_with_exposure AS (
        SELECT
            *,
            additions + deletions as churn,
            TRUE as high_exposure  -- Simplified
        FROM merged_prs
        WHERE additions + deletions BETWEEN 1 AND 100000
            AND files_changed BETWEEN 1 AND 1000
    )

    SELECT
        high_exposure,
        '{year}-{month:02d}' as year_month,
        COUNT(*) as num_prs,
        COUNT(DISTINCT repo_name) as num_repos,
        COUNT(DISTINCT pr_id) as num_authors,
        AVG(churn) as avg_churn,
        AVG(additions) as avg_additions,
        AVG(deletions) as avg_deletions,
        AVG(files_changed) as avg_files,
        APPROX_QUANTILES(churn, 100)[OFFSET(50)] as median_churn,
        APPROX_QUANTILES(churn, 100)[OFFSET(75)] as p75_churn,
        APPROX_QUANTILES(churn, 100)[OFFSET(90)] as p90_churn,
        APPROX_QUANTILES(files_changed, 100)[OFFSET(50)] as median_files,
        APPROX_QUANTILES(files_changed, 100)[OFFSET(75)] as p75_files,
        AVG(commits) as avg_commits,
        APPROX_QUANTILES(commits, 100)[OFFSET(50)] as median_commits,
        APPROX_QUANTILES(commits, 100)[OFFSET(75)] as p75_commits,
        AVG(review_comments) as avg_review_comments,
        APPROX_QUANTILES(review_comments, 100)[OFFSET(50)] as median_review_comments,
        AVG(CAST(comments AS FLOAT64)) as avg_comments,
        AVG(lead_time_hours) as avg_lead_time,
        APPROX_QUANTILES(lead_time_hours, 100)[OFFSET(50)] as median_lead_time,
        APPROX_QUANTILES(lead_time_hours, 100)[OFFSET(75)] as p75_lead_time
    FROM prs_with_exposure
    GROUP BY high_exposure
    """


def get_burstiness_query(year: int, month: int) -> str:
    """
    Daily activity patterns with correct days-in-month calculation.

    Improvement: Uses actual days in month instead of fixed 30.
    """
    days_in_month = monthrange(year, month)[1]
    start_date = f'{year}-{month:02d}-01'
    end_date = f'{year}-{month:02d}-{days_in_month}'

    return f"""
    WITH daily_events AS (
        SELECT
            JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.login') as author,
            DATE(created_at) as event_date,
            COUNT(*) as events_today
        FROM `githubarchive.month.{year}{month:02d}`
        WHERE type = 'PullRequestEvent'
            AND JSON_EXTRACT_SCALAR(payload, '$.action') IN ('opened', 'synchronize', 'closed')
            AND JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.login') NOT LIKE '%[bot]%'
        GROUP BY author, event_date
    ),

    -- Generate all days in month
    all_days AS (
        SELECT day_date
        FROM UNNEST(GENERATE_DATE_ARRAY(DATE '{start_date}', DATE '{end_date}')) as day_date
    ),

    -- All author-day combinations (for authors with any activity)
    authors AS (
        SELECT DISTINCT author FROM daily_events
    ),

    author_days AS (
        SELECT
            a.author,
            d.day_date
        FROM authors a
        CROSS JOIN all_days d
    ),

    -- Fill in zeros
    daily_with_zeros AS (
        SELECT
            ad.author,
            ad.day_date,
            COALESCE(de.events_today, 0) as events
        FROM author_days ad
        LEFT JOIN daily_events de ON ad.author = de.author AND ad.day_date = de.event_date
    ),

    -- Compute per-author stats
    author_stats AS (
        SELECT
            author,
            COUNT(CASE WHEN events > 0 THEN 1 END) as active_days,
            {days_in_month} as total_days_in_month,
            SUM(events) as total_events,
            AVG(events) as avg_daily_events,
            STDDEV(events) as std_daily_events
        FROM daily_with_zeros
        GROUP BY author
    )

    SELECT
        author,
        TRUE as high_exposure,  -- Simplified
        active_days,
        total_days_in_month,
        SAFE_DIVIDE(active_days, total_days_in_month) as active_days_ratio,
        SAFE_DIVIDE(std_daily_events, avg_daily_events) as cv_daily_events,
        total_events
    FROM author_stats
    WHERE total_events >= 3  -- Minimum activity for meaningful CV
    ORDER BY total_events DESC
    """


def get_slack_query(year: int, month: int) -> str:
    """
    Inter-PR gaps with 3-month lookback to fix censoring.

    Improvement: Looks at PRs from 3 months to find "next PR" events,
    avoiding underestimation of gaps at month boundaries.
    """
    days_in_month = monthrange(year, month)[1]

    # Build 3-month lookback
    months_to_query = []
    for offset in range(3):
        m = month + offset
        y = year
        if m > 12:
            m -= 12
            y += 1
        if y <= 2025 or (y == 2025 and m <= 6):  # Don't exceed data range
            months_to_query.append((y, m))

    union_parts = []
    for y, m in months_to_query:
        union_parts.append(f"""
        SELECT
            JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.login') as author,
            TIMESTAMP(JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged_at')) as merged_at,
            TIMESTAMP(JSON_EXTRACT_SCALAR(payload, '$.pull_request.created_at')) as created_at
        FROM `githubarchive.month.{y}{m:02d}`
        WHERE type = 'PullRequestEvent'
            AND JSON_EXTRACT_SCALAR(payload, '$.action') = 'closed'
            AND JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged') = 'true'
            AND JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.login') NOT LIKE '%[bot]%'
        """)

    all_prs_sql = " UNION ALL ".join(union_parts)

    return f"""
    WITH all_merged_prs AS (
        {all_prs_sql}
    ),

    -- PRs merged in the target month
    target_month_prs AS (
        SELECT *
        FROM all_merged_prs
        WHERE EXTRACT(YEAR FROM merged_at) = {year}
            AND EXTRACT(MONTH FROM merged_at) = {month}
    ),

    -- Find next PR for each author after each merge
    prs_with_next AS (
        SELECT
            t.author,
            t.merged_at,
            MIN(a.created_at) as next_pr_created
        FROM target_month_prs t
        LEFT JOIN all_merged_prs a ON t.author = a.author
            AND a.created_at > t.merged_at
        GROUP BY t.author, t.merged_at
    ),

    -- Compute gaps
    gaps AS (
        SELECT
            author,
            merged_at,
            next_pr_created,
            TIMESTAMP_DIFF(next_pr_created, merged_at, HOUR) as gap_hours,
            CASE WHEN next_pr_created IS NULL THEN TRUE ELSE FALSE END as right_censored
        FROM prs_with_next
    ),

    -- Filter valid gaps (exclude very long ones for outliers, but keep right-censored info)
    valid_gaps AS (
        SELECT *
        FROM gaps
        WHERE gap_hours IS NULL OR gap_hours BETWEEN 0 AND 2160  -- 90 days max
    )

    SELECT
        author,
        TRUE as high_exposure,  -- Simplified
        COUNT(*) as num_gaps,
        AVG(CASE WHEN NOT right_censored THEN gap_hours END) as avg_gap_hours,
        APPROX_QUANTILES(CASE WHEN NOT right_censored THEN gap_hours END, 100)[OFFSET(50)] as median_gap_hours,
        APPROX_QUANTILES(CASE WHEN NOT right_censored THEN gap_hours END, 100)[OFFSET(75)] as p75_gap_hours,
        COUNTIF(right_censored) > 0 as has_right_censored
    FROM valid_gaps
    GROUP BY author
    HAVING COUNT(*) >= 2
    ORDER BY num_gaps DESC
    """


def get_controls_query(year: int, month: int) -> str:
    """Control variables for regression analysis."""

    return f"""
    WITH repo_stats AS (
        SELECT
            repo.id as repo_id,
            repo.name as repo_name,
            COUNT(*) as monthly_events,
            COUNT(DISTINCT actor.login) as unique_contributors
        FROM `githubarchive.month.{year}{month:02d}`
        WHERE type IN ('PushEvent', 'PullRequestEvent', 'IssuesEvent')
        GROUP BY repo.id, repo.name
    ),

    pr_stats AS (
        SELECT
            repo.id as repo_id,
            COUNT(*) as total_prs,
            COUNTIF(JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.login') LIKE '%[bot]%') as bot_prs,
            COUNTIF(EXTRACT(DAYOFWEEK FROM created_at) IN (1, 7)) as weekend_prs
        FROM `githubarchive.month.{year}{month:02d}`
        WHERE type = 'PullRequestEvent'
        GROUP BY repo.id
    )

    SELECT
        '{year}-{month:02d}' as year_month,
        TRUE as high_exposure,  -- Simplified
        AVG(rs.unique_contributors) as avg_team_size,
        AVG(SAFE_DIVIDE(ps.bot_prs, ps.total_prs)) as pct_bot_prs,
        AVG(SAFE_DIVIDE(ps.weekend_prs, ps.total_prs)) as weekend_pct,
        COUNT(DISTINCT rs.repo_id) as num_repos
    FROM repo_stats rs
    LEFT JOIN pr_stats ps ON rs.repo_id = ps.repo_id
    GROUP BY year_month, high_exposure
    """


# =============================================================================
# COLLECTION LOGIC
# =============================================================================

QUERY_FUNCTIONS = {
    'velocity': get_velocity_query,
    'throughput': get_throughput_query,
    'complexity': get_complexity_query,
    'burstiness': get_burstiness_query,
    'slack': get_slack_query,
    'controls': get_controls_query,
}


def load_collection_log() -> Dict:
    """Load collection progress log."""
    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            return json.load(f)
    return {'collected': {}, 'errors': {}, 'last_updated': None}


def save_collection_log(log: Dict):
    """Save collection progress."""
    log['last_updated'] = datetime.now().isoformat()
    with open(LOG_FILE, 'w') as f:
        json.dump(log, f, indent=2)


def is_collected(log: Dict, metric: str, year: int, month: int) -> bool:
    """Check if a specific month/metric is already collected."""
    key = f"{metric}_{year}_{month:02d}"
    return key in log.get('collected', {})


def mark_collected(log: Dict, metric: str, year: int, month: int, path: str, rows: int):
    """Mark a month/metric as collected."""
    key = f"{metric}_{year}_{month:02d}"
    log['collected'][key] = {
        'path': str(path),
        'rows': rows,
        'collected_at': datetime.now().isoformat()
    }


def generate_month_range(
    start_year: int, start_month: int,
    end_year: int, end_month: int
) -> List[Tuple[int, int]]:
    """Generate list of (year, month) tuples."""
    months = []
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        months.append((year, month))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months


def collect_metric(
    client,
    metric: str,
    year: int,
    month: int,
    dry_run: bool = False
) -> Tuple[Optional[Path], int]:
    """Collect a single metric for a single month."""

    query_fn = QUERY_FUNCTIONS[metric]
    query = query_fn(year, month)

    output_dir = PANEL_DIR / metric
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{metric}_{year}_{month:02d}.csv"

    if dry_run:
        print(f"[DRY RUN] Would execute query for {metric} {year}-{month:02d}")
        print(f"Query preview:\n{query[:500]}...")
        return None, 0

    print(f"Collecting {metric} for {year}-{month:02d}...")

    try:
        df = client.run_query(query, max_cost_usd=COST_ESTIMATES[metric] * 2)

        if df is not None and not df.empty:
            df.to_csv(output_path, index=False)
            print(f"  Saved {len(df)} rows to {output_path}")
            return output_path, len(df)
        else:
            print(f"  No data returned")
            return None, 0

    except Exception as e:
        print(f"  ERROR: {e}")
        return None, -1


def main():
    parser = argparse.ArgumentParser(description='Collect panel data for Task-Job Paradox')
    parser.add_argument('--project-id', required=True, help='BigQuery project ID')
    parser.add_argument('--start-year', type=int, default=2021)
    parser.add_argument('--start-month', type=int, default=1)
    parser.add_argument('--end-year', type=int, default=2025)
    parser.add_argument('--end-month', type=int, default=6)
    parser.add_argument('--metrics', default='velocity,throughput,complexity,burstiness,slack,controls',
                        help='Comma-separated metrics to collect')
    parser.add_argument('--skip-existing', action='store_true', help='Skip already collected data')
    parser.add_argument('--dry-run', action='store_true', help='Print queries without executing')
    parser.add_argument('--max-cost', type=float, default=100.0, help='Maximum total cost in USD')
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation prompt')

    args = parser.parse_args()

    metrics = [m.strip() for m in args.metrics.split(',')]
    months = generate_month_range(args.start_year, args.start_month, args.end_year, args.end_month)

    print("=" * 60)
    print("TASK-JOB PARADOX: PANEL DATA COLLECTION")
    print("=" * 60)
    print(f"Period: {args.start_year}-{args.start_month:02d} to {args.end_year}-{args.end_month:02d}")
    print(f"Metrics: {metrics}")
    print(f"Total months: {len(months)}")
    print(f"Total queries: {len(months) * len(metrics)}")

    # Estimate cost
    estimated_cost = sum(COST_ESTIMATES.get(m, 0.20) for m in metrics) * len(months)
    print(f"Estimated cost: ${estimated_cost:.2f}")

    if estimated_cost > args.max_cost:
        print(f"ERROR: Estimated cost exceeds max-cost (${args.max_cost})")
        print("Use --max-cost to increase limit or reduce metrics/months")
        return

    if not args.dry_run and not args.yes:
        confirm = input(f"\nProceed with collection? (y/n): ")
        if confirm.lower() != 'y':
            print("Aborted.")
            return

    # Initialize
    log = load_collection_log()
    client = None if args.dry_run else create_bigquery_client(args.project_id)

    collected = 0
    skipped = 0
    errors = 0

    for metric in metrics:
        print(f"\n{'='*60}")
        print(f"METRIC: {metric.upper()}")
        print('='*60)

        for year, month in months:
            if args.skip_existing and is_collected(log, metric, year, month):
                print(f"  Skipping {metric} {year}-{month:02d} (already collected)")
                skipped += 1
                continue

            path, rows = collect_metric(client, metric, year, month, args.dry_run)

            if rows > 0:
                mark_collected(log, metric, year, month, path, rows)
                save_collection_log(log)
                collected += 1
            elif rows == -1:
                errors += 1
                log.setdefault('errors', {})[f"{metric}_{year}_{month:02d}"] = datetime.now().isoformat()
                save_collection_log(log)

            # Rate limiting
            if not args.dry_run:
                time.sleep(1)

    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)
    print(f"Collected: {collected}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")
    print(f"\nLog saved to: {LOG_FILE}")
    print(f"Data saved to: {PANEL_DIR}")


if __name__ == '__main__':
    main()
