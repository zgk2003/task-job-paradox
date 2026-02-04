"""
BigQuery Client for GitHub Archive Data

This module provides efficient access to GitHub Archive data via BigQuery,
performing all heavy aggregations server-side to handle the massive dataset.

Key Design Principles:
1. Server-side aggregation: All metric calculations happen in BigQuery
2. Minimize data transfer: Only download aggregated results
3. Cost efficiency: Use monthly tables and date filtering to reduce scanned data
4. Incremental queries: Support querying specific date ranges

Event Types Used:
- PullRequestEvent: PR opens, closes, merges (for lead time, code churn)
- PullRequestReviewEvent: Code reviews (for review metrics)
- PushEvent: Commits (for iteration intensity)

References:
- GH Archive: https://www.gharchive.org/
- BigQuery examples: https://davelester.github.io/gharchive-bigquery-examples/
- GitHub Event Types: https://docs.github.com/en/rest/using-the-rest-api/github-event-types
"""

import os
from datetime import datetime, date
from typing import Optional, Dict, List, Any, Tuple
import pandas as pd
import numpy as np

# BigQuery imports with fallback
try:
    from google.cloud import bigquery
    from google.oauth2 import service_account
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False
    print("Warning: google-cloud-bigquery not installed. Install with:")
    print("  pip install google-cloud-bigquery pyarrow")


# High AI exposure languages (61-80% Copilot accuracy)
HIGH_EXPOSURE_LANGUAGES = ['Python', 'JavaScript', 'Java', 'TypeScript']

# Low AI exposure languages (~30% Copilot accuracy)
LOW_EXPOSURE_LANGUAGES = ['Fortran', 'COBOL', 'Assembly', 'Erlang', 'Haskell']

# All languages for filtering
ALL_LANGUAGES = HIGH_EXPOSURE_LANGUAGES + LOW_EXPOSURE_LANGUAGES

# Treatment date: ChatGPT public launch
TREATMENT_DATE = date(2022, 11, 30)


class BigQueryGitHubClient:
    """
    Client for querying GitHub Archive data from BigQuery.

    Handles authentication, query execution, and result formatting
    for the Task-Job Paradox analysis.
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        credentials_path: Optional[str] = None
    ):
        """
        Initialize the BigQuery client.

        Args:
            project_id: Google Cloud project ID. If None, uses GOOGLE_CLOUD_PROJECT env var.
            credentials_path: Path to service account JSON. If None, uses default credentials.
        """
        if not BIGQUERY_AVAILABLE:
            raise ImportError(
                "google-cloud-bigquery is required. Install with: "
                "pip install google-cloud-bigquery pyarrow"
            )

        self.project_id = project_id or os.environ.get('GOOGLE_CLOUD_PROJECT')

        if credentials_path:
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path
            )
            self.client = bigquery.Client(
                project=self.project_id,
                credentials=credentials
            )
        else:
            # Use default credentials (gcloud auth, env var, etc.)
            self.client = bigquery.Client(project=self.project_id)

        print(f"BigQuery client initialized for project: {self.client.project}")

    def estimate_query_cost(self, query: str) -> Dict[str, Any]:
        """
        Estimate the cost of a query before running it.

        Returns:
            Dict with estimated bytes processed and cost.
        """
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        query_job = self.client.query(query, job_config=job_config)

        bytes_processed = query_job.total_bytes_processed
        gb_processed = bytes_processed / (1024 ** 3)
        # BigQuery pricing: $5 per TB (as of 2024)
        estimated_cost = (bytes_processed / (1024 ** 4)) * 5

        return {
            'bytes_processed': bytes_processed,
            'gb_processed': round(gb_processed, 2),
            'estimated_cost_usd': round(estimated_cost, 4),
            'free_tier_usage_pct': round((gb_processed / 1024) * 100, 2)  # 1TB free/month
        }

    def run_query(
        self,
        query: str,
        dry_run: bool = False,
        max_cost_usd: float = 1.0
    ) -> pd.DataFrame:
        """
        Execute a BigQuery query and return results as DataFrame.

        Args:
            query: SQL query string
            dry_run: If True, only estimate cost without running
            max_cost_usd: Maximum allowed cost before confirmation

        Returns:
            pandas DataFrame with query results
        """
        # First estimate cost
        estimate = self.estimate_query_cost(query)
        print(f"Query will process ~{estimate['gb_processed']} GB "
              f"(est. ${estimate['estimated_cost_usd']})")

        if dry_run:
            return pd.DataFrame()

        if estimate['estimated_cost_usd'] > max_cost_usd:
            raise ValueError(
                f"Query cost ${estimate['estimated_cost_usd']} exceeds "
                f"max allowed ${max_cost_usd}. Set dry_run=True to test, "
                f"or increase max_cost_usd."
            )

        # Execute query
        print("Executing query...")
        query_job = self.client.query(query)
        results = query_job.result()

        # Convert to DataFrame
        df = results.to_dataframe()
        print(f"Query returned {len(df):,} rows")

        return df


class MonthlyMetricsQuery:
    """
    Generates optimized BigQuery SQL for monthly PR metrics aggregation.

    This is the primary query class for Interrupted Time Series analysis,
    computing monthly averages of task-level and job-level metrics.
    """

    @staticmethod
    def _get_table_expression(start_date: date, end_date: date) -> str:
        """
        Generate table reference for date range using table wildcards.

        Uses monthly tables for efficiency when spanning multiple months.
        """
        # For date ranges spanning multiple months, use monthly tables with wildcard
        start_ym = start_date.strftime('%Y%m')
        end_ym = end_date.strftime('%Y%m')

        # Using monthly tables for efficiency
        return f"`githubarchive.month.*` WHERE _TABLE_SUFFIX BETWEEN '{start_ym}' AND '{end_ym}'"

    @staticmethod
    def build_monthly_pr_metrics_query(
        start_date: date = date(2021, 1, 1),
        end_date: date = date(2025, 6, 30),
        languages: Optional[List[str]] = None,
        min_prs_per_repo: int = 10
    ) -> str:
        """
        Build query for monthly PR metrics aggregated by language exposure.

        This query extracts:
        - Job-level: PR lead time (open to merge)
        - Mechanism: Code churn (additions + deletions), PR size

        Aggregates to monthly level for ITS analysis.

        Args:
            start_date: Start of observation window
            end_date: End of observation window
            languages: List of languages to include (None = all target languages)
            min_prs_per_repo: Minimum merged PRs per repo for inclusion

        Returns:
            SQL query string
        """
        if languages is None:
            languages = ALL_LANGUAGES

        # Format language list for SQL
        lang_list = ", ".join(f"'{lang}'" for lang in languages)

        # High exposure languages for SQL
        high_exp_list = ", ".join(f"'{lang}'" for lang in HIGH_EXPOSURE_LANGUAGES)

        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        start_ym = start_date.strftime('%Y%m')
        end_ym = end_date.strftime('%Y%m')
        treatment_str = TREATMENT_DATE.strftime('%Y-%m-%d')

        query = f"""
        -- Monthly PR Metrics for Task-Job Paradox Analysis
        -- Aggregates pull request data by month and language exposure level

        WITH
        -- Step 1: Extract all PR events with relevant fields
        pr_events AS (
            SELECT
                repo.id AS repo_id,
                repo.name AS repo_name,
                JSON_EXTRACT_SCALAR(payload, '$.action') AS action,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.id') AS pr_id,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.number') AS pr_number,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged') AS merged,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.additions') AS additions,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.deletions') AS deletions,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.changed_files') AS changed_files,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.commits') AS commits,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.review_comments') AS review_comments,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.created_at') AS pr_created_at,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged_at') AS pr_merged_at,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.base.repo.language') AS language,
                created_at AS event_time
            FROM `githubarchive.month.*`
            WHERE
                _TABLE_SUFFIX BETWEEN '{start_ym}' AND '{end_ym}'
                AND type = 'PullRequestEvent'
                AND JSON_EXTRACT_SCALAR(payload, '$.action') IN ('closed', 'opened')
        ),

        -- Step 2: Filter to merged PRs with valid language
        merged_prs AS (
            SELECT
                repo_id,
                repo_name,
                pr_id,
                pr_number,
                language,
                CASE WHEN language IN ({high_exp_list}) THEN TRUE ELSE FALSE END AS high_exposure,
                SAFE_CAST(additions AS INT64) AS additions,
                SAFE_CAST(deletions AS INT64) AS deletions,
                SAFE_CAST(changed_files AS INT64) AS changed_files,
                SAFE_CAST(commits AS INT64) AS commits,
                SAFE_CAST(review_comments AS INT64) AS review_comments,
                PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%SZ', pr_created_at) AS created_at,
                PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%SZ', pr_merged_at) AS merged_at
            FROM pr_events
            WHERE
                action = 'closed'
                AND merged = 'true'
                AND language IN ({lang_list})
                AND pr_created_at IS NOT NULL
                AND pr_merged_at IS NOT NULL
        ),

        -- Step 3: Calculate PR-level metrics
        pr_metrics AS (
            SELECT
                repo_id,
                repo_name,
                pr_id,
                language,
                high_exposure,
                additions,
                deletions,
                additions + deletions AS code_churn,
                changed_files,
                commits AS num_commits,
                review_comments AS num_review_comments,
                created_at,
                merged_at,
                TIMESTAMP_DIFF(merged_at, created_at, HOUR) AS lead_time_hours,
                FORMAT_TIMESTAMP('%Y-%m', created_at) AS year_month,
                EXTRACT(YEAR FROM created_at) AS year,
                EXTRACT(MONTH FROM created_at) AS month,
                CASE WHEN created_at >= '{treatment_str}' THEN TRUE ELSE FALSE END AS post_treatment
            FROM merged_prs
            WHERE
                -- Filter out PRs with unrealistic lead times (>30 days likely data issues)
                TIMESTAMP_DIFF(merged_at, created_at, HOUR) BETWEEN 0 AND 720
                -- Filter out bots and trivial PRs
                AND additions + deletions > 0
                AND additions + deletions < 100000  -- Exclude bulk changes
        ),

        -- Step 4: Filter to active repos (minimum PR count for reliability)
        active_repos AS (
            SELECT repo_id
            FROM pr_metrics
            GROUP BY repo_id
            HAVING COUNT(*) >= {min_prs_per_repo}
        )

        -- Step 5: Aggregate to monthly level by exposure
        SELECT
            year_month,
            year,
            month,
            high_exposure,
            post_treatment,

            -- Sample sizes
            COUNT(DISTINCT repo_id) AS num_repos,
            COUNT(*) AS num_prs,

            -- Job-level metrics
            AVG(lead_time_hours) AS avg_lead_time_hours,
            APPROX_QUANTILES(lead_time_hours, 100)[OFFSET(50)] AS median_lead_time_hours,
            STDDEV(lead_time_hours) AS std_lead_time_hours,

            -- Mechanism metrics
            AVG(num_commits) AS avg_commits_per_pr,
            AVG(num_review_comments) AS avg_review_comments,
            AVG(code_churn) AS avg_code_churn,
            AVG(changed_files) AS avg_changed_files,

            -- Size distribution
            AVG(additions) AS avg_additions,
            AVG(deletions) AS avg_deletions

        FROM pr_metrics
        WHERE repo_id IN (SELECT repo_id FROM active_repos)
        GROUP BY year_month, year, month, high_exposure, post_treatment
        ORDER BY year_month, high_exposure
        """

        return query

    @staticmethod
    def build_review_latency_query(
        start_date: date = date(2021, 1, 1),
        end_date: date = date(2025, 6, 30),
        languages: Optional[List[str]] = None
    ) -> str:
        """
        Build query for review-response latency metrics.

        This measures the task-level metric: time from review submission
        to author's next push/commit in response.

        Note: This is more complex as it requires joining review events
        with subsequent push events on the same PR.

        Args:
            start_date: Start of observation window
            end_date: End of observation window
            languages: List of languages to include

        Returns:
            SQL query string
        """
        if languages is None:
            languages = ALL_LANGUAGES

        lang_list = ", ".join(f"'{lang}'" for lang in languages)
        high_exp_list = ", ".join(f"'{lang}'" for lang in HIGH_EXPOSURE_LANGUAGES)

        start_ym = start_date.strftime('%Y%m')
        end_ym = end_date.strftime('%Y%m')
        treatment_str = TREATMENT_DATE.strftime('%Y-%m-%d')

        query = f"""
        -- Review-Response Latency Analysis
        -- Task-level metric: time from review to author's response

        WITH
        -- Get PR review events requesting changes
        reviews AS (
            SELECT
                repo.id AS repo_id,
                repo.name AS repo_name,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.id') AS pr_id,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.number') AS pr_number,
                JSON_EXTRACT_SCALAR(payload, '$.review.state') AS review_state,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.base.repo.language') AS language,
                created_at AS review_time,
                actor.login AS reviewer
            FROM `githubarchive.month.*`
            WHERE
                _TABLE_SUFFIX BETWEEN '{start_ym}' AND '{end_ym}'
                AND type = 'PullRequestReviewEvent'
                AND JSON_EXTRACT_SCALAR(payload, '$.review.state') = 'changes_requested'
        ),

        -- Get push events (commits) on PRs - these indicate author responses
        -- Note: PullRequestEvent with synchronize action indicates new commits pushed
        pr_updates AS (
            SELECT
                repo.id AS repo_id,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.id') AS pr_id,
                created_at AS update_time,
                JSON_EXTRACT_SCALAR(payload, '$.sender.login') AS updater
            FROM `githubarchive.month.*`
            WHERE
                _TABLE_SUFFIX BETWEEN '{start_ym}' AND '{end_ym}'
                AND type = 'PullRequestEvent'
                AND JSON_EXTRACT_SCALAR(payload, '$.action') = 'synchronize'
        ),

        -- Match reviews with subsequent updates
        review_response_pairs AS (
            SELECT
                r.repo_id,
                r.repo_name,
                r.pr_id,
                r.language,
                CASE WHEN r.language IN ({high_exp_list}) THEN TRUE ELSE FALSE END AS high_exposure,
                r.review_time,
                MIN(u.update_time) AS response_time
            FROM reviews r
            INNER JOIN pr_updates u
                ON r.repo_id = u.repo_id
                AND r.pr_id = u.pr_id
                AND u.update_time > r.review_time
                AND r.reviewer != u.updater  -- Exclude self-reviews
            WHERE r.language IN ({lang_list})
            GROUP BY r.repo_id, r.repo_name, r.pr_id, r.language, high_exposure, r.review_time
        ),

        -- Calculate latencies
        latencies AS (
            SELECT
                repo_id,
                pr_id,
                language,
                high_exposure,
                review_time,
                response_time,
                TIMESTAMP_DIFF(response_time, review_time, HOUR) AS response_latency_hours,
                FORMAT_TIMESTAMP('%Y-%m', review_time) AS year_month,
                EXTRACT(YEAR FROM review_time) AS year,
                EXTRACT(MONTH FROM review_time) AS month,
                CASE WHEN review_time >= '{treatment_str}' THEN TRUE ELSE FALSE END AS post_treatment
            FROM review_response_pairs
            WHERE
                -- Filter unrealistic latencies (>7 days likely abandoned)
                TIMESTAMP_DIFF(response_time, review_time, HOUR) BETWEEN 0 AND 168
        )

        -- Aggregate monthly by exposure
        SELECT
            year_month,
            year,
            month,
            high_exposure,
            post_treatment,

            COUNT(DISTINCT repo_id) AS num_repos,
            COUNT(*) AS num_review_response_pairs,

            -- Task-level metric: Review response latency
            AVG(response_latency_hours) AS avg_review_response_latency,
            APPROX_QUANTILES(response_latency_hours, 100)[OFFSET(50)] AS median_review_response_latency,
            STDDEV(response_latency_hours) AS std_review_response_latency

        FROM latencies
        GROUP BY year_month, year, month, high_exposure, post_treatment
        ORDER BY year_month, high_exposure
        """

        return query


class GitHubArchiveDataLoader:
    """
    High-level interface for loading GitHub Archive data for analysis.

    Orchestrates queries and combines results into analysis-ready DataFrames.
    """

    def __init__(self, client: BigQueryGitHubClient):
        """
        Initialize with a BigQuery client.

        Args:
            client: Configured BigQueryGitHubClient instance
        """
        self.client = client
        self.query_builder = MonthlyMetricsQuery()

    def load_monthly_metrics(
        self,
        start_date: date = date(2021, 1, 1),
        end_date: date = date(2025, 6, 30),
        include_review_latency: bool = True,
        dry_run: bool = False,
        max_cost_usd: float = 5.0
    ) -> pd.DataFrame:
        """
        Load monthly aggregated metrics for ITS analysis.

        Args:
            start_date: Start of observation window
            end_date: End of observation window
            include_review_latency: Whether to also query review latency (additional cost)
            dry_run: If True, only estimate costs
            max_cost_usd: Maximum cost per query

        Returns:
            DataFrame with monthly metrics by exposure level
        """
        print("\n" + "="*60)
        print("LOADING MONTHLY PR METRICS FROM GITHUB ARCHIVE")
        print("="*60)
        print(f"Date range: {start_date} to {end_date}")
        print(f"Languages: {', '.join(ALL_LANGUAGES)}")

        # Query 1: PR-level metrics (lead time, commits, churn)
        print("\n--- Query 1: PR Metrics (lead time, commits, churn) ---")
        pr_query = self.query_builder.build_monthly_pr_metrics_query(
            start_date=start_date,
            end_date=end_date
        )

        pr_df = self.client.run_query(pr_query, dry_run=dry_run, max_cost_usd=max_cost_usd)

        if dry_run:
            print("Dry run - no data returned")
            return pd.DataFrame()

        # Query 2: Review latency (optional, additional cost)
        if include_review_latency:
            print("\n--- Query 2: Review Response Latency ---")
            review_query = self.query_builder.build_review_latency_query(
                start_date=start_date,
                end_date=end_date
            )

            review_df = self.client.run_query(
                review_query, dry_run=dry_run, max_cost_usd=max_cost_usd
            )

            # Merge review metrics into main DataFrame
            if not review_df.empty:
                merge_cols = ['year_month', 'high_exposure']
                review_cols = ['year_month', 'high_exposure',
                              'avg_review_response_latency',
                              'median_review_response_latency',
                              'num_review_response_pairs']
                pr_df = pr_df.merge(
                    review_df[review_cols],
                    on=merge_cols,
                    how='left'
                )

        # Post-process
        pr_df['date'] = pd.to_datetime(pr_df['year_month'] + '-01')
        pr_df = pr_df.sort_values(['year_month', 'high_exposure'])

        print(f"\nLoaded {len(pr_df)} monthly observations")
        print(f"Date range in data: {pr_df['year_month'].min()} to {pr_df['year_month'].max()}")
        print(f"High exposure rows: {pr_df['high_exposure'].sum()}")
        print(f"Low exposure rows: {(~pr_df['high_exposure']).sum()}")

        return pr_df

    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run data quality checks on loaded data.

        Args:
            df: DataFrame from load_monthly_metrics

        Returns:
            Dict with quality metrics and warnings
        """
        quality = {
            'total_rows': len(df),
            'date_range': (df['year_month'].min(), df['year_month'].max()),
            'missing_months': [],
            'low_sample_months': [],
            'warnings': []
        }

        # Check for missing months
        all_months = pd.date_range(
            df['date'].min(), df['date'].max(), freq='MS'
        ).strftime('%Y-%m').tolist()

        present_months = df['year_month'].unique().tolist()
        missing = set(all_months) - set(present_months)
        quality['missing_months'] = sorted(list(missing))

        if missing:
            quality['warnings'].append(
                f"Missing data for {len(missing)} months: {sorted(list(missing))[:5]}..."
            )

        # Check for low sample sizes
        low_sample = df[df['num_prs'] < 100]['year_month'].unique().tolist()
        quality['low_sample_months'] = low_sample

        if low_sample:
            quality['warnings'].append(
                f"{len(low_sample)} months have <100 PRs"
            )

        # Check treatment balance
        pre = df[~df['post_treatment']]
        post = df[df['post_treatment']]
        quality['pre_treatment_months'] = len(pre['year_month'].unique())
        quality['post_treatment_months'] = len(post['year_month'].unique())

        if quality['pre_treatment_months'] < 12:
            quality['warnings'].append(
                f"Only {quality['pre_treatment_months']} pre-treatment months "
                "(recommend 12+ for robust ITS)"
            )

        return quality


def create_bigquery_client(
    project_id: Optional[str] = None,
    credentials_path: Optional[str] = None
) -> BigQueryGitHubClient:
    """
    Factory function to create a configured BigQuery client.

    Args:
        project_id: Google Cloud project ID
        credentials_path: Path to service account credentials JSON

    Returns:
        Configured BigQueryGitHubClient
    """
    return BigQueryGitHubClient(
        project_id=project_id,
        credentials_path=credentials_path
    )


# Example usage and testing
if __name__ == "__main__":
    print("BigQuery GitHub Archive Client")
    print("="*60)

    if not BIGQUERY_AVAILABLE:
        print("\nBigQuery not available. Install with:")
        print("  pip install google-cloud-bigquery pyarrow")
        print("\nTo use this module:")
        print("  1. Create a Google Cloud project")
        print("  2. Enable the BigQuery API")
        print("  3. Set up authentication (gcloud auth application-default login)")
        print("  4. Set GOOGLE_CLOUD_PROJECT environment variable")
    else:
        print("\nBigQuery is available!")
        print("\nTo test with dry run (no cost):")
        print("  client = create_bigquery_client()")
        print("  loader = GitHubArchiveDataLoader(client)")
        print("  loader.load_monthly_metrics(dry_run=True)")
