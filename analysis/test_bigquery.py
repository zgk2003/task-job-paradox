#!/usr/bin/env python3
"""
Test Script for BigQuery GitHub Archive Integration

This script tests the BigQuery queries with a small date range to validate
the integration without incurring large costs.

Usage:
    # Dry run (estimate costs only, no data)
    python test_bigquery.py --dry-run

    # Test with one month of data (minimal cost)
    python test_bigquery.py --project-id YOUR_PROJECT

    # Test with custom date range
    python test_bigquery.py --project-id YOUR_PROJECT --start 2024-01 --end 2024-03

Prerequisites:
    1. Google Cloud project with BigQuery API enabled
    2. Authentication configured:
       - Option A: gcloud auth application-default login
       - Option B: GOOGLE_APPLICATION_CREDENTIALS env var pointing to service account JSON
    3. GOOGLE_CLOUD_PROJECT env var (or use --project-id flag)
"""

import os
import sys
from pathlib import Path
from datetime import date
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Check for BigQuery availability
try:
    from bigquery_client import (
        BigQueryGitHubClient,
        MonthlyMetricsQuery,
        GitHubArchiveDataLoader,
        create_bigquery_client,
        BIGQUERY_AVAILABLE,
        ALL_LANGUAGES,
        HIGH_EXPOSURE_LANGUAGES,
        LOW_EXPOSURE_LANGUAGES
    )
except ImportError as e:
    print(f"Import error: {e}")
    BIGQUERY_AVAILABLE = False


def print_sql_preview(query: str, max_lines: int = 50):
    """Print a preview of the SQL query."""
    lines = query.strip().split('\n')
    print("\n--- SQL Query Preview ---")
    for i, line in enumerate(lines[:max_lines]):
        print(line)
    if len(lines) > max_lines:
        print(f"... ({len(lines) - max_lines} more lines)")
    print("--- End SQL ---\n")


def test_query_generation():
    """Test that queries are generated correctly."""
    print("\n" + "="*60)
    print("TEST 1: Query Generation")
    print("="*60)

    # Test monthly PR metrics query
    query = MonthlyMetricsQuery.build_monthly_pr_metrics_query(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 3, 31),
        languages=['Python', 'JavaScript'],
        min_prs_per_repo=5
    )

    print("\nMonthly PR Metrics Query (2024 Q1, Python/JS):")
    print_sql_preview(query, max_lines=30)

    # Test review latency query
    review_query = MonthlyMetricsQuery.build_review_latency_query(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 3, 31),
        languages=['Python']
    )

    print("\nReview Latency Query (2024 Q1, Python):")
    print_sql_preview(review_query, max_lines=30)

    print("Query generation: PASSED")
    return True


def test_bigquery_connection(project_id: str, credentials_path: str = None):
    """Test BigQuery connection and authentication."""
    print("\n" + "="*60)
    print("TEST 2: BigQuery Connection")
    print("="*60)

    try:
        client = create_bigquery_client(
            project_id=project_id,
            credentials_path=credentials_path
        )
        print(f"Connected to project: {client.client.project}")
        print("BigQuery connection: PASSED")
        return client
    except Exception as e:
        print(f"BigQuery connection: FAILED")
        print(f"Error: {e}")
        return None


def test_cost_estimation(client: BigQueryGitHubClient):
    """Test query cost estimation with dry run."""
    print("\n" + "="*60)
    print("TEST 3: Cost Estimation (Dry Run)")
    print("="*60)

    # Small query - one month
    query_1m = MonthlyMetricsQuery.build_monthly_pr_metrics_query(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31)
    )

    print("\nEstimating cost for 1 month of data (Jan 2024):")
    estimate_1m = client.estimate_query_cost(query_1m)
    print(f"  Bytes to process: {estimate_1m['bytes_processed']:,}")
    print(f"  GB to process: {estimate_1m['gb_processed']}")
    print(f"  Estimated cost: ${estimate_1m['estimated_cost_usd']}")
    print(f"  Free tier usage: {estimate_1m['free_tier_usage_pct']}%")

    # Quarterly query
    query_3m = MonthlyMetricsQuery.build_monthly_pr_metrics_query(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 3, 31)
    )

    print("\nEstimating cost for 3 months of data (Q1 2024):")
    estimate_3m = client.estimate_query_cost(query_3m)
    print(f"  Bytes to process: {estimate_3m['bytes_processed']:,}")
    print(f"  GB to process: {estimate_3m['gb_processed']}")
    print(f"  Estimated cost: ${estimate_3m['estimated_cost_usd']}")
    print(f"  Free tier usage: {estimate_3m['free_tier_usage_pct']}%")

    # Full study period
    query_full = MonthlyMetricsQuery.build_monthly_pr_metrics_query(
        start_date=date(2021, 1, 1),
        end_date=date(2025, 6, 30)
    )

    print("\nEstimating cost for FULL study period (2021-2025):")
    estimate_full = client.estimate_query_cost(query_full)
    print(f"  Bytes to process: {estimate_full['bytes_processed']:,}")
    print(f"  GB to process: {estimate_full['gb_processed']}")
    print(f"  Estimated cost: ${estimate_full['estimated_cost_usd']}")
    print(f"  Free tier usage: {estimate_full['free_tier_usage_pct']}%")

    print("\nCost estimation: PASSED")
    return {
        '1_month': estimate_1m,
        '3_months': estimate_3m,
        'full_period': estimate_full
    }


def test_small_query(client: BigQueryGitHubClient, start: date, end: date):
    """Run a small query to validate results structure."""
    print("\n" + "="*60)
    print(f"TEST 4: Small Query Execution ({start} to {end})")
    print("="*60)

    loader = GitHubArchiveDataLoader(client)

    print(f"\nLoading data for {start} to {end}...")
    df = loader.load_monthly_metrics(
        start_date=start,
        end_date=end,
        include_review_latency=True,
        dry_run=False,
        max_cost_usd=2.0  # Safety limit
    )

    if df.empty:
        print("WARNING: No data returned")
        return None

    print(f"\nResults shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")

    print("\n--- Sample Data ---")
    print(df.head(10).to_string())

    print("\n--- Summary Statistics ---")
    print(f"Date range: {df['year_month'].min()} to {df['year_month'].max()}")
    print(f"Total PRs: {df['num_prs'].sum():,}")
    print(f"High exposure PRs: {df[df['high_exposure']]['num_prs'].sum():,}")
    print(f"Low exposure PRs: {df[~df['high_exposure']]['num_prs'].sum():,}")

    if 'avg_lead_time_hours' in df.columns:
        print(f"\nAvg Lead Time (hours): {df['avg_lead_time_hours'].mean():.1f}")
    if 'avg_commits_per_pr' in df.columns:
        print(f"Avg Commits per PR: {df['avg_commits_per_pr'].mean():.1f}")
    if 'avg_review_response_latency' in df.columns:
        avg_review = df['avg_review_response_latency'].mean()
        if not pd.isna(avg_review):
            print(f"Avg Review Response Latency (hours): {avg_review:.1f}")

    # Validate data quality
    quality = loader.validate_data_quality(df)
    if quality['warnings']:
        print("\n--- Quality Warnings ---")
        for w in quality['warnings']:
            print(f"  - {w}")

    print("\nSmall query execution: PASSED")
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Test BigQuery GitHub Archive integration'
    )
    parser.add_argument(
        '--project-id', type=str,
        default=os.environ.get('GOOGLE_CLOUD_PROJECT'),
        help='Google Cloud project ID'
    )
    parser.add_argument(
        '--credentials', type=str,
        help='Path to service account JSON'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Only test query generation and cost estimation'
    )
    parser.add_argument(
        '--start', type=str, default='2024-01',
        help='Start month (YYYY-MM format, default: 2024-01)'
    )
    parser.add_argument(
        '--end', type=str, default='2024-01',
        help='End month (YYYY-MM format, default: 2024-01)'
    )

    args = parser.parse_args()

    print("="*60)
    print("BIGQUERY GITHUB ARCHIVE INTEGRATION TEST")
    print("="*60)

    # Test 1: Query generation (no BigQuery needed)
    test_query_generation()

    if not BIGQUERY_AVAILABLE:
        print("\n" + "="*60)
        print("BigQuery not available - skipping connection tests")
        print("Install with: pip install google-cloud-bigquery pyarrow db-dtypes")
        print("="*60)
        return

    if not args.project_id:
        print("\n" + "="*60)
        print("No project ID provided - skipping BigQuery tests")
        print("Set GOOGLE_CLOUD_PROJECT env var or use --project-id flag")
        print("="*60)
        return

    # Test 2: Connection
    client = test_bigquery_connection(args.project_id, args.credentials)
    if client is None:
        return

    # Test 3: Cost estimation
    costs = test_cost_estimation(client)

    if args.dry_run:
        print("\n" + "="*60)
        print("DRY RUN COMPLETE")
        print("="*60)
        print("\nTo run actual queries, remove --dry-run flag")
        return

    # Test 4: Small query
    # Parse date range
    start_parts = args.start.split('-')
    end_parts = args.end.split('-')
    start_date = date(int(start_parts[0]), int(start_parts[1]), 1)

    # End date is last day of the month
    import calendar
    end_year, end_month = int(end_parts[0]), int(end_parts[1])
    last_day = calendar.monthrange(end_year, end_month)[1]
    end_date = date(end_year, end_month, last_day)

    df = test_small_query(client, start_date, end_date)

    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)

    if df is not None:
        # Save test results
        output_path = Path(__file__).parent.parent / 'results' / 'test_bigquery_results.csv'
        output_path.parent.mkdir(exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nTest results saved to: {output_path}")


# For imports
import pandas as pd

if __name__ == "__main__":
    main()
