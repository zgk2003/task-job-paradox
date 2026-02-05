"""
Revised Empirical Strategy Analysis: Peaks and Slack Hypothesis

This module implements the revised empirical strategy that tests whether
LLM productivity gains are absorbed as slack rather than increased throughput.

Key Metrics (measured at multiple granularities):
1. Velocity: PR lead time, time to first review
2. Throughput: PRs per developer per time period
3. Burstiness: CV of commits (daily, weekly)
4. Slack: Inter-PR gaps

Identification Strategy:
- ITS: Sharp discontinuity at Nov 30, 2022 (ChatGPT launch)
- DiD: High vs low AI-exposure languages
"""

import os
import sys
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Dict, List, Any, Tuple
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# BigQuery imports with fallback
try:
    from google.cloud import bigquery
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False

# Import from existing module
from bigquery_client import (
    HIGH_EXPOSURE_LANGUAGES,
    LOW_EXPOSURE_LANGUAGES,
    ALL_LANGUAGES,
    TREATMENT_DATE,
    BigQueryGitHubClient,
    create_bigquery_client
)


class RevisedStrategyQueries:
    """
    BigQuery queries for the revised empirical strategy.

    These queries extract developer-level data for computing:
    - Throughput (PRs per developer)
    - Burstiness (commit patterns)
    - Slack (gaps between PRs)
    """

    @staticmethod
    def build_developer_throughput_query(
        year: int,
        month: int,
        languages: Optional[List[str]] = None
    ) -> str:
        """
        Query for developer-level throughput metrics.

        Returns PRs per developer aggregated at weekly and monthly levels.
        """
        if languages is None:
            languages = ALL_LANGUAGES

        lang_list = ", ".join(f"'{lang}'" for lang in languages)
        high_exp_list = ", ".join(f"'{lang}'" for lang in HIGH_EXPOSURE_LANGUAGES)

        ym = f"{year}{month:02d}"
        treatment_str = TREATMENT_DATE.strftime('%Y-%m-%d')

        query = f"""
        -- Developer Throughput: PRs merged per developer
        -- Granularities: weekly, monthly

        WITH
        -- Get merged PRs with author info
        merged_prs AS (
            SELECT
                repo.id AS repo_id,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.id') AS pr_id,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.id') AS author_id,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.login') AS author_login,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.base.repo.language') AS language,
                PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%SZ',
                    JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged_at')) AS merged_at,
                PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%SZ',
                    JSON_EXTRACT_SCALAR(payload, '$.pull_request.created_at')) AS created_at
            FROM `githubarchive.month.{ym}`
            WHERE
                type = 'PullRequestEvent'
                AND JSON_EXTRACT_SCALAR(payload, '$.action') = 'closed'
                AND JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged') = 'true'
                AND JSON_EXTRACT_SCALAR(payload, '$.pull_request.base.repo.language') IN ({lang_list})
                AND JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.id') IS NOT NULL
        ),

        -- Filter valid PRs and add time dimensions
        valid_prs AS (
            SELECT
                repo_id,
                pr_id,
                author_id,
                author_login,
                language,
                CASE WHEN language IN ({high_exp_list}) THEN TRUE ELSE FALSE END AS high_exposure,
                merged_at,
                created_at,
                TIMESTAMP_DIFF(merged_at, created_at, HOUR) AS lead_time_hours,
                EXTRACT(WEEK FROM merged_at) AS week_of_year,
                EXTRACT(DAYOFWEEK FROM merged_at) AS day_of_week,
                FORMAT_TIMESTAMP('%Y-%m', merged_at) AS year_month
            FROM merged_prs
            WHERE
                merged_at IS NOT NULL
                AND created_at IS NOT NULL
                AND TIMESTAMP_DIFF(merged_at, created_at, HOUR) BETWEEN 0 AND 720
                -- Exclude likely bots (>50 PRs in a month is suspicious)
                AND author_login NOT LIKE '%[bot]%'
                AND author_login NOT LIKE '%bot'
        ),

        -- Aggregate by developer per week
        developer_weekly AS (
            SELECT
                author_id,
                author_login,
                high_exposure,
                year_month,
                week_of_year,
                COUNT(DISTINCT pr_id) AS prs_merged,
                COUNT(DISTINCT repo_id) AS repos_contributed,
                AVG(lead_time_hours) AS avg_lead_time_hours
            FROM valid_prs
            GROUP BY author_id, author_login, high_exposure, year_month, week_of_year
        ),

        -- Aggregate by developer per month
        developer_monthly AS (
            SELECT
                author_id,
                author_login,
                high_exposure,
                year_month,
                COUNT(DISTINCT pr_id) AS prs_merged_month,
                COUNT(DISTINCT repo_id) AS repos_contributed_month,
                AVG(lead_time_hours) AS avg_lead_time_hours_month,
                COUNT(DISTINCT week_of_year) AS active_weeks
            FROM valid_prs
            GROUP BY author_id, author_login, high_exposure, year_month
            -- Filter to reasonably active developers (at least 1 PR)
            HAVING COUNT(DISTINCT pr_id) >= 1
        )

        -- Return developer-level monthly metrics with weekly stats
        SELECT
            m.author_id,
            m.author_login,
            m.high_exposure,
            m.year_month,
            m.prs_merged_month,
            m.repos_contributed_month,
            m.avg_lead_time_hours_month,
            m.active_weeks,

            -- Weekly throughput stats for this developer
            AVG(w.prs_merged) AS avg_prs_per_active_week,
            STDDEV(w.prs_merged) AS std_prs_per_week,
            CASE
                WHEN AVG(w.prs_merged) > 0
                THEN STDDEV(w.prs_merged) / AVG(w.prs_merged)
                ELSE NULL
            END AS cv_weekly_prs

        FROM developer_monthly m
        LEFT JOIN developer_weekly w
            ON m.author_id = w.author_id AND m.year_month = w.year_month
        GROUP BY
            m.author_id, m.author_login, m.high_exposure, m.year_month,
            m.prs_merged_month, m.repos_contributed_month,
            m.avg_lead_time_hours_month, m.active_weeks
        """

        return query

    @staticmethod
    def build_commit_burstiness_query(
        year: int,
        month: int,
        languages: Optional[List[str]] = None
    ) -> str:
        """
        Query for commit burstiness metrics.

        Computes CV of commits at daily and weekly granularities per developer.
        """
        if languages is None:
            languages = ALL_LANGUAGES

        lang_list = ", ".join(f"'{lang}'" for lang in languages)
        high_exp_list = ", ".join(f"'{lang}'" for lang in HIGH_EXPOSURE_LANGUAGES)

        ym = f"{year}{month:02d}"

        query = f"""
        -- Commit Burstiness: CV of commits per developer at different granularities

        WITH
        -- Get push events with commit info
        push_events AS (
            SELECT
                repo.id AS repo_id,
                actor.id AS author_id,
                actor.login AS author_login,
                JSON_EXTRACT_SCALAR(payload, '$.repository.language') AS language,
                created_at AS push_time,
                CAST(JSON_EXTRACT_SCALAR(payload, '$.size') AS INT64) AS num_commits,
                EXTRACT(DATE FROM created_at) AS push_date,
                EXTRACT(WEEK FROM created_at) AS week_of_year,
                EXTRACT(HOUR FROM created_at) AS hour_of_day,
                EXTRACT(DAYOFWEEK FROM created_at) AS day_of_week
            FROM `githubarchive.month.{ym}`
            WHERE
                type = 'PushEvent'
                AND JSON_EXTRACT_SCALAR(payload, '$.repository.language') IN ({lang_list})
                AND actor.login NOT LIKE '%[bot]%'
                AND actor.login NOT LIKE '%bot'
        ),

        -- Add exposure flag
        pushes_with_exposure AS (
            SELECT
                *,
                CASE WHEN language IN ({high_exp_list}) THEN TRUE ELSE FALSE END AS high_exposure
            FROM push_events
            WHERE num_commits > 0 AND num_commits < 100  -- Filter outliers
        ),

        -- Daily commits per developer
        daily_commits AS (
            SELECT
                author_id,
                author_login,
                high_exposure,
                push_date,
                SUM(num_commits) AS commits_that_day,
                COUNT(*) AS pushes_that_day
            FROM pushes_with_exposure
            GROUP BY author_id, author_login, high_exposure, push_date
        ),

        -- Weekly commits per developer
        weekly_commits AS (
            SELECT
                author_id,
                author_login,
                high_exposure,
                week_of_year,
                SUM(num_commits) AS commits_that_week,
                COUNT(DISTINCT push_date) AS active_days_that_week
            FROM pushes_with_exposure
            GROUP BY author_id, author_login, high_exposure, week_of_year
        ),

        -- Hourly activity (for within-day burstiness)
        hourly_commits AS (
            SELECT
                author_id,
                high_exposure,
                push_date,
                hour_of_day,
                SUM(num_commits) AS commits_that_hour
            FROM pushes_with_exposure
            GROUP BY author_id, high_exposure, push_date, hour_of_day
        ),

        -- Developer-level daily stats
        developer_daily_stats AS (
            SELECT
                author_id,
                author_login,
                high_exposure,
                COUNT(DISTINCT push_date) AS active_days,
                SUM(commits_that_day) AS total_commits,
                AVG(commits_that_day) AS avg_commits_per_active_day,
                STDDEV(commits_that_day) AS std_commits_per_day,
                CASE
                    WHEN AVG(commits_that_day) > 0
                    THEN STDDEV(commits_that_day) / AVG(commits_that_day)
                    ELSE NULL
                END AS cv_daily_commits
            FROM daily_commits
            GROUP BY author_id, author_login, high_exposure
            HAVING COUNT(DISTINCT push_date) >= 3  -- Need at least 3 days for meaningful CV
        ),

        -- Developer-level weekly stats
        developer_weekly_stats AS (
            SELECT
                author_id,
                high_exposure,
                COUNT(DISTINCT week_of_year) AS active_weeks,
                AVG(commits_that_week) AS avg_commits_per_active_week,
                STDDEV(commits_that_week) AS std_commits_per_week,
                CASE
                    WHEN AVG(commits_that_week) > 0
                    THEN STDDEV(commits_that_week) / AVG(commits_that_week)
                    ELSE NULL
                END AS cv_weekly_commits,
                AVG(active_days_that_week) AS avg_active_days_per_week
            FROM weekly_commits
            GROUP BY author_id, high_exposure
            HAVING COUNT(DISTINCT week_of_year) >= 2  -- Need at least 2 weeks
        )

        -- Combine daily and weekly burstiness metrics
        SELECT
            d.author_id,
            d.author_login,
            d.high_exposure,
            '{year}-{month:02d}' AS year_month,

            -- Volume metrics
            d.total_commits,
            d.active_days,
            w.active_weeks,

            -- Daily burstiness
            d.avg_commits_per_active_day,
            d.std_commits_per_day,
            d.cv_daily_commits,

            -- Weekly burstiness
            w.avg_commits_per_active_week,
            w.std_commits_per_week,
            w.cv_weekly_commits,

            -- Active time ratio
            d.active_days / 30.0 AS active_days_ratio_month,  -- Approximate
            w.avg_active_days_per_week / 7.0 AS active_days_ratio_week

        FROM developer_daily_stats d
        LEFT JOIN developer_weekly_stats w
            ON d.author_id = w.author_id AND d.high_exposure = w.high_exposure
        """

        return query

    @staticmethod
    def build_inter_pr_gap_query(
        year: int,
        month: int,
        languages: Optional[List[str]] = None
    ) -> str:
        """
        Query for inter-PR gap (slack) metrics.

        Measures time from PR merge to next PR creation by same author.
        """
        if languages is None:
            languages = ALL_LANGUAGES

        lang_list = ", ".join(f"'{lang}'" for lang in languages)
        high_exp_list = ", ".join(f"'{lang}'" for lang in HIGH_EXPOSURE_LANGUAGES)

        ym = f"{year}{month:02d}"

        query = f"""
        -- Inter-PR Gap: Time between PR merge and next PR creation (slack measure)

        WITH
        -- Get all PR events (both opens and merges)
        pr_events AS (
            SELECT
                repo.id AS repo_id,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.id') AS pr_id,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.id') AS author_id,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.login') AS author_login,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.base.repo.language') AS language,
                JSON_EXTRACT_SCALAR(payload, '$.action') AS action,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged') AS merged,
                PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%SZ',
                    JSON_EXTRACT_SCALAR(payload, '$.pull_request.created_at')) AS pr_created_at,
                PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%SZ',
                    JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged_at')) AS pr_merged_at,
                created_at AS event_time
            FROM `githubarchive.month.{ym}`
            WHERE
                type = 'PullRequestEvent'
                AND JSON_EXTRACT_SCALAR(payload, '$.pull_request.base.repo.language') IN ({lang_list})
                AND JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.id') IS NOT NULL
                AND JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.login') NOT LIKE '%[bot]%'
        ),

        -- Get merged PRs
        merged_prs AS (
            SELECT DISTINCT
                author_id,
                author_login,
                language,
                CASE WHEN language IN ({high_exp_list}) THEN TRUE ELSE FALSE END AS high_exposure,
                pr_id,
                pr_merged_at AS merged_at
            FROM pr_events
            WHERE action = 'closed' AND merged = 'true' AND pr_merged_at IS NOT NULL
        ),

        -- Get PR creations
        pr_creations AS (
            SELECT DISTINCT
                author_id,
                pr_id,
                pr_created_at AS created_at
            FROM pr_events
            WHERE action = 'opened' AND pr_created_at IS NOT NULL
        ),

        -- For each merged PR, find the next PR creation by same author
        pr_with_next AS (
            SELECT
                m.author_id,
                m.author_login,
                m.high_exposure,
                m.pr_id AS merged_pr_id,
                m.merged_at,
                MIN(c.created_at) AS next_pr_created_at
            FROM merged_prs m
            LEFT JOIN pr_creations c
                ON m.author_id = c.author_id
                AND c.created_at > m.merged_at
                AND c.pr_id != m.pr_id
            GROUP BY m.author_id, m.author_login, m.high_exposure, m.pr_id, m.merged_at
        ),

        -- Calculate gaps
        gaps AS (
            SELECT
                author_id,
                author_login,
                high_exposure,
                merged_pr_id,
                merged_at,
                next_pr_created_at,
                TIMESTAMP_DIFF(next_pr_created_at, merged_at, HOUR) AS gap_hours,
                TIMESTAMP_DIFF(next_pr_created_at, merged_at, MINUTE) / 60.0 AS gap_hours_precise,
                -- Business hours calculation (rough approximation)
                CASE
                    WHEN EXTRACT(DAYOFWEEK FROM merged_at) IN (1, 7) THEN 0  -- Weekend
                    ELSE 1
                END AS merge_on_weekday
            FROM pr_with_next
            WHERE
                next_pr_created_at IS NOT NULL
                -- Filter reasonable gaps (1 hour to 30 days)
                AND TIMESTAMP_DIFF(next_pr_created_at, merged_at, HOUR) BETWEEN 1 AND 720
        )

        -- Aggregate to developer level
        SELECT
            author_id,
            author_login,
            high_exposure,
            '{year}-{month:02d}' AS year_month,

            -- Gap statistics
            COUNT(*) AS num_gaps,
            AVG(gap_hours) AS avg_gap_hours,
            APPROX_QUANTILES(gap_hours, 100)[OFFSET(50)] AS median_gap_hours,
            APPROX_QUANTILES(gap_hours, 100)[OFFSET(75)] AS p75_gap_hours,
            STDDEV(gap_hours) AS std_gap_hours,
            MIN(gap_hours) AS min_gap_hours,
            MAX(gap_hours) AS max_gap_hours,

            -- Gap in business days (approximate: hours / 8)
            AVG(gap_hours) / 8.0 AS avg_gap_business_days,
            APPROX_QUANTILES(gap_hours, 100)[OFFSET(50)] / 8.0 AS median_gap_business_days,

            -- Weekday vs weekend merges
            COUNTIF(merge_on_weekday = 1) AS weekday_merges,
            COUNTIF(merge_on_weekday = 0) AS weekend_merges

        FROM gaps
        GROUP BY author_id, author_login, high_exposure
        HAVING COUNT(*) >= 2  -- Need at least 2 gaps for meaningful stats
        """

        return query

    @staticmethod
    def build_velocity_metrics_query(
        year: int,
        month: int,
        languages: Optional[List[str]] = None
    ) -> str:
        """
        Query for velocity metrics (PR lead time, time to first review).

        Aggregated at multiple levels for robustness checks.
        """
        if languages is None:
            languages = ALL_LANGUAGES

        lang_list = ", ".join(f"'{lang}'" for lang in languages)
        high_exp_list = ", ".join(f"'{lang}'" for lang in HIGH_EXPOSURE_LANGUAGES)

        ym = f"{year}{month:02d}"

        query = f"""
        -- Velocity Metrics: PR lead time and time to first review

        WITH
        -- Merged PRs with timing info
        merged_prs AS (
            SELECT
                repo.id AS repo_id,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.id') AS pr_id,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.id') AS author_id,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.base.repo.language') AS language,
                CASE WHEN JSON_EXTRACT_SCALAR(payload, '$.pull_request.base.repo.language')
                     IN ({high_exp_list}) THEN TRUE ELSE FALSE END AS high_exposure,
                PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%SZ',
                    JSON_EXTRACT_SCALAR(payload, '$.pull_request.created_at')) AS created_at,
                PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%SZ',
                    JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged_at')) AS merged_at,
                SAFE_CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.additions') AS INT64) AS additions,
                SAFE_CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.deletions') AS INT64) AS deletions
            FROM `githubarchive.month.{ym}`
            WHERE
                type = 'PullRequestEvent'
                AND JSON_EXTRACT_SCALAR(payload, '$.action') = 'closed'
                AND JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged') = 'true'
                AND JSON_EXTRACT_SCALAR(payload, '$.pull_request.base.repo.language') IN ({lang_list})
        ),

        -- First review times
        first_reviews AS (
            SELECT
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.id') AS pr_id,
                MIN(created_at) AS first_review_time
            FROM `githubarchive.month.{ym}`
            WHERE
                type = 'PullRequestReviewEvent'
            GROUP BY pr_id
        ),

        -- Combined metrics
        pr_metrics AS (
            SELECT
                m.repo_id,
                m.pr_id,
                m.author_id,
                m.high_exposure,
                m.created_at,
                m.merged_at,
                r.first_review_time,

                -- Lead time in hours
                TIMESTAMP_DIFF(m.merged_at, m.created_at, HOUR) AS lead_time_hours,
                TIMESTAMP_DIFF(m.merged_at, m.created_at, MINUTE) / 60.0 AS lead_time_hours_precise,

                -- Time to first review
                TIMESTAMP_DIFF(r.first_review_time, m.created_at, HOUR) AS time_to_first_review_hours,

                -- PR size
                COALESCE(m.additions, 0) + COALESCE(m.deletions, 0) AS total_churn

            FROM merged_prs m
            LEFT JOIN first_reviews r ON m.pr_id = r.pr_id
            WHERE
                m.created_at IS NOT NULL
                AND m.merged_at IS NOT NULL
                AND TIMESTAMP_DIFF(m.merged_at, m.created_at, HOUR) BETWEEN 0 AND 720
        )

        -- Aggregate by exposure level
        SELECT
            high_exposure,
            '{year}-{month:02d}' AS year_month,

            -- Sample size
            COUNT(*) AS num_prs,
            COUNT(DISTINCT repo_id) AS num_repos,
            COUNT(DISTINCT author_id) AS num_authors,

            -- Lead time (hours)
            AVG(lead_time_hours) AS avg_lead_time_hours,
            APPROX_QUANTILES(lead_time_hours, 100)[OFFSET(50)] AS median_lead_time_hours,
            APPROX_QUANTILES(lead_time_hours, 100)[OFFSET(75)] AS p75_lead_time_hours,
            APPROX_QUANTILES(lead_time_hours, 100)[OFFSET(90)] AS p90_lead_time_hours,
            STDDEV(lead_time_hours) AS std_lead_time_hours,

            -- Time to first review (hours)
            AVG(time_to_first_review_hours) AS avg_time_to_review_hours,
            APPROX_QUANTILES(time_to_first_review_hours, 100)[OFFSET(50)] AS median_time_to_review_hours,

            -- PR size stats (for controlling)
            AVG(total_churn) AS avg_pr_size,
            APPROX_QUANTILES(total_churn, 100)[OFFSET(50)] AS median_pr_size

        FROM pr_metrics
        GROUP BY high_exposure
        """

        return query


class RevisedStrategyAnalyzer:
    """
    Analyzer for the revised empirical strategy.

    Handles data loading, metric computation, and multi-granularity analysis.
    """

    def __init__(self, client: Optional[BigQueryGitHubClient] = None):
        """
        Initialize the analyzer.

        Args:
            client: BigQuery client. If None, will try to create one.
        """
        self.client = client
        self.queries = RevisedStrategyQueries()
        self.results_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'results'
        )

    def _get_client(self) -> BigQueryGitHubClient:
        """Get or create BigQuery client."""
        if self.client is None:
            self.client = create_bigquery_client()
        return self.client

    def load_developer_throughput(
        self,
        year: int,
        month: int,
        max_cost_usd: float = 5.0,
        cache: bool = True
    ) -> pd.DataFrame:
        """
        Load developer-level throughput data.

        Args:
            year: Year to query
            month: Month to query
            max_cost_usd: Maximum query cost
            cache: Whether to cache/load from cache

        Returns:
            DataFrame with developer throughput metrics
        """
        cache_file = os.path.join(
            self.results_dir,
            f'developer_throughput_{year}_{month:02d}.csv'
        )

        if cache and os.path.exists(cache_file):
            print(f"Loading cached data from {cache_file}")
            return pd.read_csv(cache_file)

        client = self._get_client()
        query = self.queries.build_developer_throughput_query(year, month)

        print(f"\nQuerying developer throughput for {year}-{month:02d}...")
        df = client.run_query(query, max_cost_usd=max_cost_usd)

        if cache and not df.empty:
            df.to_csv(cache_file, index=False)
            print(f"Cached to {cache_file}")

        return df

    def load_commit_burstiness(
        self,
        year: int,
        month: int,
        max_cost_usd: float = 5.0,
        cache: bool = True
    ) -> pd.DataFrame:
        """
        Load commit burstiness data.

        Args:
            year: Year to query
            month: Month to query
            max_cost_usd: Maximum query cost
            cache: Whether to cache/load from cache

        Returns:
            DataFrame with burstiness metrics per developer
        """
        cache_file = os.path.join(
            self.results_dir,
            f'commit_burstiness_{year}_{month:02d}.csv'
        )

        if cache and os.path.exists(cache_file):
            print(f"Loading cached data from {cache_file}")
            return pd.read_csv(cache_file)

        client = self._get_client()
        query = self.queries.build_commit_burstiness_query(year, month)

        print(f"\nQuerying commit burstiness for {year}-{month:02d}...")
        df = client.run_query(query, max_cost_usd=max_cost_usd)

        if cache and not df.empty:
            df.to_csv(cache_file, index=False)
            print(f"Cached to {cache_file}")

        return df

    def load_inter_pr_gaps(
        self,
        year: int,
        month: int,
        max_cost_usd: float = 5.0,
        cache: bool = True
    ) -> pd.DataFrame:
        """
        Load inter-PR gap (slack) data.

        Args:
            year: Year to query
            month: Month to query
            max_cost_usd: Maximum query cost
            cache: Whether to cache/load from cache

        Returns:
            DataFrame with inter-PR gap metrics per developer
        """
        cache_file = os.path.join(
            self.results_dir,
            f'inter_pr_gaps_{year}_{month:02d}.csv'
        )

        if cache and os.path.exists(cache_file):
            print(f"Loading cached data from {cache_file}")
            return pd.read_csv(cache_file)

        client = self._get_client()
        query = self.queries.build_inter_pr_gap_query(year, month)

        print(f"\nQuerying inter-PR gaps for {year}-{month:02d}...")
        df = client.run_query(query, max_cost_usd=max_cost_usd)

        if cache and not df.empty:
            df.to_csv(cache_file, index=False)
            print(f"Cached to {cache_file}")

        return df

    def load_velocity_metrics(
        self,
        year: int,
        month: int,
        max_cost_usd: float = 5.0,
        cache: bool = True
    ) -> pd.DataFrame:
        """
        Load velocity metrics (lead time, review time).

        Args:
            year: Year to query
            month: Month to query
            max_cost_usd: Maximum query cost
            cache: Whether to cache/load from cache

        Returns:
            DataFrame with velocity metrics
        """
        cache_file = os.path.join(
            self.results_dir,
            f'velocity_metrics_{year}_{month:02d}.csv'
        )

        if cache and os.path.exists(cache_file):
            print(f"Loading cached data from {cache_file}")
            return pd.read_csv(cache_file)

        client = self._get_client()
        query = self.queries.build_velocity_metrics_query(year, month)

        print(f"\nQuerying velocity metrics for {year}-{month:02d}...")
        df = client.run_query(query, max_cost_usd=max_cost_usd)

        if cache and not df.empty:
            df.to_csv(cache_file, index=False)
            print(f"Cached to {cache_file}")

        return df

    def load_all_metrics(
        self,
        year: int,
        month: int,
        max_cost_usd: float = 5.0,
        cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all metrics for the revised strategy.

        Args:
            year: Year to query
            month: Month to query
            max_cost_usd: Maximum cost per query
            cache: Whether to use caching

        Returns:
            Dict with DataFrames for each metric type
        """
        print(f"\n{'='*60}")
        print(f"LOADING ALL REVISED STRATEGY METRICS FOR {year}-{month:02d}")
        print(f"{'='*60}")

        results = {}

        # 1. Velocity metrics
        print("\n[1/4] Loading velocity metrics...")
        results['velocity'] = self.load_velocity_metrics(year, month, max_cost_usd, cache)

        # 2. Developer throughput
        print("\n[2/4] Loading developer throughput...")
        results['throughput'] = self.load_developer_throughput(year, month, max_cost_usd, cache)

        # 3. Commit burstiness
        print("\n[3/4] Loading commit burstiness...")
        results['burstiness'] = self.load_commit_burstiness(year, month, max_cost_usd, cache)

        # 4. Inter-PR gaps
        print("\n[4/4] Loading inter-PR gaps...")
        results['gaps'] = self.load_inter_pr_gaps(year, month, max_cost_usd, cache)

        print(f"\n{'='*60}")
        print("DATA LOADING COMPLETE")
        for name, df in results.items():
            if df is not None and not df.empty:
                print(f"  {name}: {len(df)} rows")

        return results

    def compute_aggregate_metrics(
        self,
        data: Dict[str, pd.DataFrame],
        year_month: str
    ) -> pd.DataFrame:
        """
        Compute aggregate metrics from developer-level data.

        Args:
            data: Dict of DataFrames from load_all_metrics
            year_month: Period label (e.g., '2021-06')

        Returns:
            DataFrame with aggregate metrics by exposure level
        """
        results = []

        for high_exp in [True, False]:
            row = {
                'year_month': year_month,
                'high_exposure': high_exp
            }

            # Velocity metrics (already aggregated)
            if 'velocity' in data and not data['velocity'].empty:
                vel = data['velocity'][data['velocity']['high_exposure'] == high_exp]
                if not vel.empty:
                    row['num_prs'] = vel['num_prs'].values[0]
                    row['num_repos'] = vel['num_repos'].values[0]
                    row['num_authors'] = vel['num_authors'].values[0]
                    row['avg_lead_time_hours'] = vel['avg_lead_time_hours'].values[0]
                    row['median_lead_time_hours'] = vel['median_lead_time_hours'].values[0]
                    row['p75_lead_time_hours'] = vel['p75_lead_time_hours'].values[0]
                    row['avg_time_to_review_hours'] = vel['avg_time_to_review_hours'].values[0]
                    row['median_time_to_review_hours'] = vel['median_time_to_review_hours'].values[0]

            # Throughput metrics (aggregate from developer level)
            if 'throughput' in data and not data['throughput'].empty:
                tput = data['throughput'][data['throughput']['high_exposure'] == high_exp]
                if not tput.empty:
                    row['num_active_developers'] = len(tput)
                    row['avg_prs_per_developer_month'] = tput['prs_merged_month'].mean()
                    row['median_prs_per_developer_month'] = tput['prs_merged_month'].median()
                    row['std_prs_per_developer_month'] = tput['prs_merged_month'].std()
                    row['total_prs_from_throughput'] = tput['prs_merged_month'].sum()
                    row['avg_active_weeks'] = tput['active_weeks'].mean()

            # Burstiness metrics (aggregate from developer level)
            if 'burstiness' in data and not data['burstiness'].empty:
                burst = data['burstiness'][data['burstiness']['high_exposure'] == high_exp]
                if not burst.empty:
                    # CV of daily commits
                    row['avg_cv_daily_commits'] = burst['cv_daily_commits'].mean()
                    row['median_cv_daily_commits'] = burst['cv_daily_commits'].median()
                    # CV of weekly commits
                    row['avg_cv_weekly_commits'] = burst['cv_weekly_commits'].mean()
                    row['median_cv_weekly_commits'] = burst['cv_weekly_commits'].median()
                    # Active time ratio
                    row['avg_active_days_ratio'] = burst['active_days_ratio_month'].mean()
                    row['median_active_days_ratio'] = burst['active_days_ratio_month'].median()
                    # Volume
                    row['avg_commits_per_developer'] = burst['total_commits'].mean()

            # Slack metrics (aggregate from developer level)
            if 'gaps' in data and not data['gaps'].empty:
                gaps = data['gaps'][data['gaps']['high_exposure'] == high_exp]
                if not gaps.empty:
                    # Inter-PR gap in hours
                    row['avg_inter_pr_gap_hours'] = gaps['avg_gap_hours'].mean()
                    row['median_inter_pr_gap_hours'] = gaps['median_gap_hours'].median()
                    row['avg_p75_gap_hours'] = gaps['p75_gap_hours'].mean()
                    # In business days
                    row['avg_inter_pr_gap_business_days'] = gaps['avg_gap_business_days'].mean()
                    row['median_inter_pr_gap_business_days'] = gaps['median_gap_business_days'].median()
                    # Num developers with gap data
                    row['num_developers_with_gaps'] = len(gaps)

            results.append(row)

        return pd.DataFrame(results)

    def compare_periods(
        self,
        pre_data: Dict[str, pd.DataFrame],
        post_data: Dict[str, pd.DataFrame],
        pre_label: str = '2021-06',
        post_label: str = '2025-06'
    ) -> pd.DataFrame:
        """
        Compare metrics between pre and post periods.

        Args:
            pre_data: Data from pre-treatment period
            post_data: Data from post-treatment period
            pre_label: Label for pre period
            post_label: Label for post period

        Returns:
            DataFrame with comparison metrics
        """
        pre_agg = self.compute_aggregate_metrics(pre_data, pre_label)
        post_agg = self.compute_aggregate_metrics(post_data, post_label)

        # Combine
        combined = pd.concat([pre_agg, post_agg], ignore_index=True)
        combined['post_treatment'] = combined['year_month'] == post_label

        return combined

    def compute_changes(
        self,
        comparison_df: pd.DataFrame,
        pre_label: str = '2021-06',
        post_label: str = '2025-06'
    ) -> pd.DataFrame:
        """
        Compute percentage changes between periods.

        Args:
            comparison_df: Output from compare_periods
            pre_label: Pre-period label
            post_label: Post-period label

        Returns:
            DataFrame with changes by exposure level
        """
        results = []

        for high_exp in [True, False]:
            pre = comparison_df[
                (comparison_df['year_month'] == pre_label) &
                (comparison_df['high_exposure'] == high_exp)
            ]
            post = comparison_df[
                (comparison_df['year_month'] == post_label) &
                (comparison_df['high_exposure'] == high_exp)
            ]

            if pre.empty or post.empty:
                continue

            row = {
                'high_exposure': high_exp,
                'exposure_label': 'High (Python/JS/Java/TS)' if high_exp else 'Low (Fortran/COBOL/etc)'
            }

            # Compute changes for key metrics
            metrics = [
                ('median_lead_time_hours', 'Velocity: Median PR Lead Time'),
                ('median_time_to_review_hours', 'Velocity: Median Time to Review'),
                ('avg_prs_per_developer_month', 'Throughput: PRs per Dev-Month'),
                ('median_cv_daily_commits', 'Burstiness: CV Daily Commits'),
                ('median_cv_weekly_commits', 'Burstiness: CV Weekly Commits'),
                ('avg_active_days_ratio', 'Burstiness: Active Days Ratio'),
                ('median_inter_pr_gap_hours', 'Slack: Median Inter-PR Gap (hrs)'),
                ('median_inter_pr_gap_business_days', 'Slack: Median Inter-PR Gap (days)')
            ]

            for col, label in metrics:
                if col in pre.columns and col in post.columns:
                    pre_val = pre[col].values[0]
                    post_val = post[col].values[0]

                    if pd.notna(pre_val) and pd.notna(post_val) and pre_val != 0:
                        pct_change = ((post_val - pre_val) / abs(pre_val)) * 100
                        row[f'{col}_pre'] = pre_val
                        row[f'{col}_post'] = post_val
                        row[f'{col}_change_pct'] = pct_change

            results.append(row)

        return pd.DataFrame(results)


def run_revised_analysis(
    max_cost_usd: float = 5.0,
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Run the full revised empirical strategy analysis.

    Compares June 2021 (pre-LLM) vs June 2025 (post-LLM).

    Args:
        max_cost_usd: Maximum cost per query
        use_cache: Whether to use cached data

    Returns:
        Dict with all analysis results
    """
    print("\n" + "="*70)
    print("REVISED EMPIRICAL STRATEGY ANALYSIS")
    print("Peaks and Slack Hypothesis")
    print("="*70)
    print("\nComparing June 2021 (pre-LLM) vs June 2025 (post-LLM)")
    print(f"Max cost per query: ${max_cost_usd}")
    print(f"Using cache: {use_cache}")

    analyzer = RevisedStrategyAnalyzer()
    results = {}

    # Load pre-treatment data (June 2021)
    print("\n" + "-"*50)
    print("LOADING PRE-TREATMENT DATA (June 2021)")
    print("-"*50)
    results['pre_data'] = analyzer.load_all_metrics(2021, 6, max_cost_usd, use_cache)

    # Load post-treatment data (June 2025)
    print("\n" + "-"*50)
    print("LOADING POST-TREATMENT DATA (June 2025)")
    print("-"*50)
    results['post_data'] = analyzer.load_all_metrics(2025, 6, max_cost_usd, use_cache)

    # Compute comparison
    print("\n" + "-"*50)
    print("COMPUTING COMPARISONS")
    print("-"*50)

    results['comparison'] = analyzer.compare_periods(
        results['pre_data'],
        results['post_data']
    )

    results['changes'] = analyzer.compute_changes(results['comparison'])

    # Save results
    results_dir = analyzer.results_dir

    results['comparison'].to_csv(
        os.path.join(results_dir, 'revised_strategy_comparison.csv'),
        index=False
    )
    print(f"\nSaved comparison to revised_strategy_comparison.csv")

    results['changes'].to_csv(
        os.path.join(results_dir, 'revised_strategy_changes.csv'),
        index=False
    )
    print(f"Saved changes to revised_strategy_changes.csv")

    # Print summary
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)

    print("\n### Key Findings (High AI-Exposure Languages) ###\n")

    if not results['changes'].empty:
        high_exp = results['changes'][results['changes']['high_exposure'] == True]
        if not high_exp.empty:
            row = high_exp.iloc[0]

            metrics_to_show = [
                ('median_lead_time_hours', 'PR Lead Time (median hours)', '↓ expected'),
                ('median_time_to_review_hours', 'Time to Review (median hours)', '↓ expected'),
                ('avg_prs_per_developer_month', 'PRs per Developer-Month', '≈ flat expected'),
                ('median_cv_daily_commits', 'CV Daily Commits', '↑ expected'),
                ('median_cv_weekly_commits', 'CV Weekly Commits', '↑ expected'),
                ('avg_active_days_ratio', 'Active Days Ratio', '↓ expected'),
                ('median_inter_pr_gap_hours', 'Inter-PR Gap (hours)', '↑ expected'),
            ]

            for col, label, expected in metrics_to_show:
                pre_col = f'{col}_pre'
                post_col = f'{col}_post'
                change_col = f'{col}_change_pct'

                if change_col in row and pd.notna(row[change_col]):
                    change = row[change_col]
                    direction = '↑' if change > 0 else '↓'
                    print(f"{label}:")
                    print(f"  Pre:  {row[pre_col]:.2f}")
                    print(f"  Post: {row[post_col]:.2f}")
                    print(f"  Change: {direction} {abs(change):.1f}% ({expected})")
                    print()

    return results


# Entry point
if __name__ == "__main__":
    results = run_revised_analysis(max_cost_usd=5.0, use_cache=True)
