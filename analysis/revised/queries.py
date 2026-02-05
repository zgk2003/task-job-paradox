"""
BigQuery Queries for Revised Empirical Strategy

This module contains all BigQuery SQL queries for extracting:
1. Velocity metrics (PR lead time, review time)
2. Throughput metrics (PRs per developer)
3. Burstiness metrics (CV of daily/weekly activity)
4. Slack metrics (inter-PR gaps)
5. Complexity metrics (PR size, commits, files)

All queries support the multi-granularity measurement strategy.
"""

from typing import List, Optional
from .config import (
    HIGH_EXPOSURE_LANGUAGES,
    LOW_EXPOSURE_LANGUAGES,
    ALL_LANGUAGES,
    TREATMENT_DATE_STR,
)


def _format_lang_list(languages: List[str]) -> str:
    """Format language list for SQL IN clause."""
    return ", ".join(f"'{lang}'" for lang in languages)


class VelocityQueries:
    """Queries for velocity metrics (task speed)."""

    @staticmethod
    def pr_lead_time(year: int, month: int, languages: Optional[List[str]] = None) -> str:
        """
        Query for PR lead time metrics.

        Returns aggregated PR lead time statistics by exposure level.
        Granularities: hours, business hours approximation, percentiles.
        """
        langs = languages or ALL_LANGUAGES
        lang_list = _format_lang_list(langs)
        high_exp_list = _format_lang_list(HIGH_EXPOSURE_LANGUAGES)
        ym = f"{year}{month:02d}"

        return f"""
        -- Velocity: PR Lead Time
        -- Granularities: hours (median, p75, p90), business hours approximation

        WITH merged_prs AS (
            SELECT
                repo.id AS repo_id,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.id') AS pr_id,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.id') AS author_id,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.base.repo.language') AS language,
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
                TIMESTAMP_DIFF(merged_at, created_at, HOUR) AS lead_time_hours,
                TIMESTAMP_DIFF(merged_at, created_at, MINUTE) / 60.0 AS lead_time_hours_precise
            FROM merged_prs
            WHERE
                created_at IS NOT NULL
                AND merged_at IS NOT NULL
                AND TIMESTAMP_DIFF(merged_at, created_at, HOUR) BETWEEN 0 AND 720
        )

        SELECT
            high_exposure,
            '{year}-{month:02d}' AS year_month,

            -- Sample size
            COUNT(*) AS num_prs,
            COUNT(DISTINCT repo_id) AS num_repos,
            COUNT(DISTINCT author_id) AS num_authors,

            -- Lead time in hours (multiple aggregations)
            AVG(lead_time_hours) AS avg_lead_time_hours,
            APPROX_QUANTILES(lead_time_hours, 100)[OFFSET(50)] AS median_lead_time_hours,
            APPROX_QUANTILES(lead_time_hours, 100)[OFFSET(75)] AS p75_lead_time_hours,
            APPROX_QUANTILES(lead_time_hours, 100)[OFFSET(90)] AS p90_lead_time_hours,
            STDDEV(lead_time_hours) AS std_lead_time_hours,

            -- Business hours approximation (divide by ~2 for weekday work hours)
            AVG(lead_time_hours) / 2.0 AS avg_lead_time_business_hours_approx,
            APPROX_QUANTILES(lead_time_hours, 100)[OFFSET(50)] / 2.0 AS median_lead_time_business_hours_approx

        FROM pr_metrics
        GROUP BY high_exposure
        """

    @staticmethod
    def time_to_first_review(year: int, month: int, languages: Optional[List[str]] = None) -> str:
        """Query for time to first review metrics."""
        langs = languages or ALL_LANGUAGES
        lang_list = _format_lang_list(langs)
        high_exp_list = _format_lang_list(HIGH_EXPOSURE_LANGUAGES)
        ym = f"{year}{month:02d}"

        return f"""
        -- Velocity: Time to First Review

        WITH pr_opens AS (
            SELECT
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.id') AS pr_id,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.base.repo.language') AS language,
                PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%SZ',
                    JSON_EXTRACT_SCALAR(payload, '$.pull_request.created_at')) AS pr_created_at
            FROM `githubarchive.month.{ym}`
            WHERE
                type = 'PullRequestEvent'
                AND JSON_EXTRACT_SCALAR(payload, '$.action') = 'opened'
                AND JSON_EXTRACT_SCALAR(payload, '$.pull_request.base.repo.language') IN ({lang_list})
        ),

        first_reviews AS (
            SELECT
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.id') AS pr_id,
                MIN(created_at) AS first_review_time
            FROM `githubarchive.month.{ym}`
            WHERE type = 'PullRequestReviewEvent'
            GROUP BY pr_id
        ),

        pr_with_review AS (
            SELECT
                p.pr_id,
                p.language,
                CASE WHEN p.language IN ({high_exp_list}) THEN TRUE ELSE FALSE END AS high_exposure,
                p.pr_created_at,
                r.first_review_time,
                TIMESTAMP_DIFF(r.first_review_time, p.pr_created_at, HOUR) AS time_to_review_hours
            FROM pr_opens p
            INNER JOIN first_reviews r ON p.pr_id = r.pr_id
            WHERE TIMESTAMP_DIFF(r.first_review_time, p.pr_created_at, HOUR) BETWEEN 0 AND 720
        )

        SELECT
            high_exposure,
            '{year}-{month:02d}' AS year_month,
            COUNT(*) AS num_prs_with_review,
            AVG(time_to_review_hours) AS avg_time_to_review_hours,
            APPROX_QUANTILES(time_to_review_hours, 100)[OFFSET(50)] AS median_time_to_review_hours,
            APPROX_QUANTILES(time_to_review_hours, 100)[OFFSET(75)] AS p75_time_to_review_hours
        FROM pr_with_review
        GROUP BY high_exposure
        """


class ThroughputQueries:
    """Queries for throughput metrics (output volume)."""

    @staticmethod
    def developer_throughput(year: int, month: int, languages: Optional[List[str]] = None) -> str:
        """
        Query for developer-level throughput metrics.

        Returns PRs per developer at weekly and monthly granularities.
        """
        langs = languages or ALL_LANGUAGES
        lang_list = _format_lang_list(langs)
        high_exp_list = _format_lang_list(HIGH_EXPOSURE_LANGUAGES)
        ym = f"{year}{month:02d}"

        return f"""
        -- Throughput: PRs per Developer (weekly and monthly)

        WITH merged_prs AS (
            SELECT
                repo.id AS repo_id,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.id') AS pr_id,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.id') AS author_id,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.login') AS author_login,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.base.repo.language') AS language,
                PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%SZ',
                    JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged_at')) AS merged_at,
                TIMESTAMP_DIFF(
                    PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%SZ',
                        JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged_at')),
                    PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%SZ',
                        JSON_EXTRACT_SCALAR(payload, '$.pull_request.created_at')),
                    HOUR
                ) AS lead_time_hours
            FROM `githubarchive.month.{ym}`
            WHERE
                type = 'PullRequestEvent'
                AND JSON_EXTRACT_SCALAR(payload, '$.action') = 'closed'
                AND JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged') = 'true'
                AND JSON_EXTRACT_SCALAR(payload, '$.pull_request.base.repo.language') IN ({lang_list})
                AND JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.id') IS NOT NULL
                AND JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.login') NOT LIKE '%[bot]%'
        ),

        valid_prs AS (
            SELECT
                *,
                CASE WHEN language IN ({high_exp_list}) THEN TRUE ELSE FALSE END AS high_exposure,
                EXTRACT(WEEK FROM merged_at) AS week_of_year
            FROM merged_prs
            WHERE
                merged_at IS NOT NULL
                AND lead_time_hours BETWEEN 0 AND 720
        ),

        -- Weekly aggregation per developer
        developer_weekly AS (
            SELECT
                author_id,
                author_login,
                high_exposure,
                week_of_year,
                COUNT(DISTINCT pr_id) AS prs_merged_week,
                AVG(lead_time_hours) AS avg_lead_time_week
            FROM valid_prs
            GROUP BY author_id, author_login, high_exposure, week_of_year
        ),

        -- Monthly aggregation per developer
        developer_monthly AS (
            SELECT
                author_id,
                author_login,
                high_exposure,
                COUNT(DISTINCT pr_id) AS prs_merged_month,
                COUNT(DISTINCT repo_id) AS repos_contributed_month,
                AVG(lead_time_hours) AS avg_lead_time_month,
                COUNT(DISTINCT week_of_year) AS active_weeks
            FROM valid_prs
            GROUP BY author_id, author_login, high_exposure
            HAVING COUNT(DISTINCT pr_id) >= 1
        )

        -- Combine monthly with weekly stats
        SELECT
            m.author_id,
            m.author_login,
            m.high_exposure,
            '{year}-{month:02d}' AS year_month,

            -- Monthly throughput
            m.prs_merged_month,
            m.repos_contributed_month,
            m.avg_lead_time_month AS avg_lead_time_hours,
            m.active_weeks,

            -- Weekly throughput stats (for burstiness)
            AVG(w.prs_merged_week) AS avg_prs_per_active_week,
            STDDEV(w.prs_merged_week) AS std_prs_per_week,
            CASE
                WHEN AVG(w.prs_merged_week) > 0
                THEN STDDEV(w.prs_merged_week) / AVG(w.prs_merged_week)
                ELSE NULL
            END AS cv_weekly_prs

        FROM developer_monthly m
        LEFT JOIN developer_weekly w
            ON m.author_id = w.author_id AND m.high_exposure = w.high_exposure
        GROUP BY
            m.author_id, m.author_login, m.high_exposure,
            m.prs_merged_month, m.repos_contributed_month,
            m.avg_lead_time_month, m.active_weeks
        """


class BurstinessQueries:
    """Queries for burstiness metrics (work pattern concentration)."""

    @staticmethod
    def daily_activity_burstiness(year: int, month: int, languages: Optional[List[str]] = None) -> str:
        """
        Query for daily activity burstiness.

        Computes CV of daily activity per developer.
        Uses PR events (opens, syncs) as activity proxy.
        """
        langs = languages or ALL_LANGUAGES
        lang_list = _format_lang_list(langs)
        high_exp_list = _format_lang_list(HIGH_EXPOSURE_LANGUAGES)
        ym = f"{year}{month:02d}"

        return f"""
        -- Burstiness: CV of Daily Activity

        WITH pr_activity AS (
            SELECT
                actor.id AS author_id,
                actor.login AS author_login,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.base.repo.language') AS language,
                EXTRACT(DATE FROM created_at) AS activity_date,
                EXTRACT(HOUR FROM created_at) AS hour_of_day
            FROM `githubarchive.month.{ym}`
            WHERE
                type = 'PullRequestEvent'
                AND JSON_EXTRACT_SCALAR(payload, '$.action') IN ('opened', 'synchronize')
                AND JSON_EXTRACT_SCALAR(payload, '$.pull_request.base.repo.language') IN ({lang_list})
                AND actor.login NOT LIKE '%[bot]%'
        ),

        activity_with_exposure AS (
            SELECT
                *,
                CASE WHEN language IN ({high_exp_list}) THEN TRUE ELSE FALSE END AS high_exposure
            FROM pr_activity
        ),

        -- Daily activity per developer
        daily_activity AS (
            SELECT
                author_id,
                author_login,
                high_exposure,
                activity_date,
                COUNT(*) AS events_that_day
            FROM activity_with_exposure
            GROUP BY author_id, author_login, high_exposure, activity_date
        ),

        -- Developer-level daily burstiness
        developer_burstiness AS (
            SELECT
                author_id,
                author_login,
                high_exposure,
                COUNT(DISTINCT activity_date) AS active_days,
                SUM(events_that_day) AS total_events,
                AVG(events_that_day) AS avg_events_per_active_day,
                STDDEV(events_that_day) AS std_events_per_day,
                CASE
                    WHEN AVG(events_that_day) > 0
                    THEN STDDEV(events_that_day) / AVG(events_that_day)
                    ELSE NULL
                END AS cv_daily_events,
                COUNT(DISTINCT activity_date) / 30.0 AS active_days_ratio
            FROM daily_activity
            GROUP BY author_id, author_login, high_exposure
            HAVING COUNT(DISTINCT activity_date) >= 3
        )

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
            active_days_ratio
        FROM developer_burstiness
        """


class SlackQueries:
    """Queries for slack metrics (gaps between work)."""

    @staticmethod
    def inter_pr_gaps(year: int, month: int, languages: Optional[List[str]] = None) -> str:
        """
        Query for inter-PR gap metrics.

        Measures time from PR merge to next PR creation (same author).
        """
        langs = languages or ALL_LANGUAGES
        lang_list = _format_lang_list(langs)
        high_exp_list = _format_lang_list(HIGH_EXPOSURE_LANGUAGES)
        ym = f"{year}{month:02d}"

        return f"""
        -- Slack: Inter-PR Gaps

        WITH pr_events AS (
            SELECT
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.id') AS pr_id,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.id') AS author_id,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.login') AS author_login,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.base.repo.language') AS language,
                JSON_EXTRACT_SCALAR(payload, '$.action') AS action,
                JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged') AS merged,
                PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%SZ',
                    JSON_EXTRACT_SCALAR(payload, '$.pull_request.created_at')) AS pr_created_at,
                PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%SZ',
                    JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged_at')) AS pr_merged_at
            FROM `githubarchive.month.{ym}`
            WHERE
                type = 'PullRequestEvent'
                AND JSON_EXTRACT_SCALAR(payload, '$.pull_request.base.repo.language') IN ({lang_list})
                AND JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.id') IS NOT NULL
                AND JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.login') NOT LIKE '%[bot]%'
        ),

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

        pr_creations AS (
            SELECT DISTINCT
                author_id,
                pr_id,
                pr_created_at AS created_at
            FROM pr_events
            WHERE action = 'opened' AND pr_created_at IS NOT NULL
        ),

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

        gaps AS (
            SELECT
                author_id,
                author_login,
                high_exposure,
                TIMESTAMP_DIFF(next_pr_created_at, merged_at, HOUR) AS gap_hours
            FROM pr_with_next
            WHERE
                next_pr_created_at IS NOT NULL
                AND TIMESTAMP_DIFF(next_pr_created_at, merged_at, HOUR) BETWEEN 1 AND 720
        )

        SELECT
            author_id,
            author_login,
            high_exposure,
            '{year}-{month:02d}' AS year_month,
            COUNT(*) AS num_gaps,
            AVG(gap_hours) AS avg_gap_hours,
            APPROX_QUANTILES(gap_hours, 100)[OFFSET(50)] AS median_gap_hours,
            APPROX_QUANTILES(gap_hours, 100)[OFFSET(75)] AS p75_gap_hours,
            STDDEV(gap_hours) AS std_gap_hours,
            AVG(gap_hours) / 8.0 AS avg_gap_business_days,
            APPROX_QUANTILES(gap_hours, 100)[OFFSET(50)] / 8.0 AS median_gap_business_days
        FROM gaps
        GROUP BY author_id, author_login, high_exposure
        HAVING COUNT(*) >= 2
        """


class ComplexityQueries:
    """Queries for complexity metrics (scope expansion)."""

    @staticmethod
    def pr_complexity(year: int, month: int, languages: Optional[List[str]] = None) -> str:
        """
        Query for PR complexity metrics.

        Measures PR size (lines, files, commits) to detect scope expansion.
        """
        langs = languages or ALL_LANGUAGES
        lang_list = _format_lang_list(langs)
        high_exp_list = _format_lang_list(HIGH_EXPOSURE_LANGUAGES)
        ym = f"{year}{month:02d}"

        return f"""
        -- Complexity: PR Size and Scope

        WITH merged_prs AS (
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

        SELECT
            high_exposure,
            '{year}-{month:02d}' AS year_month,

            -- Sample size
            COUNT(*) AS num_prs,
            COUNT(DISTINCT repo_id) AS num_repos,
            COUNT(DISTINCT author_id) AS num_authors,

            -- Lines changed (churn)
            AVG(total_churn) AS avg_churn,
            APPROX_QUANTILES(total_churn, 100)[OFFSET(50)] AS median_churn,
            APPROX_QUANTILES(total_churn, 100)[OFFSET(75)] AS p75_churn,
            APPROX_QUANTILES(total_churn, 100)[OFFSET(90)] AS p90_churn,

            -- Files changed
            AVG(changed_files) AS avg_files,
            APPROX_QUANTILES(changed_files, 100)[OFFSET(50)] AS median_files,
            APPROX_QUANTILES(changed_files, 100)[OFFSET(75)] AS p75_files,

            -- Commits per PR
            AVG(num_commits) AS avg_commits,
            APPROX_QUANTILES(num_commits, 100)[OFFSET(50)] AS median_commits,
            APPROX_QUANTILES(num_commits, 100)[OFFSET(75)] AS p75_commits,

            -- Review intensity
            AVG(review_comments) AS avg_review_comments,
            APPROX_QUANTILES(review_comments, 100)[OFFSET(50)] AS median_review_comments,

            -- Lead time for reference
            AVG(lead_time_hours) AS avg_lead_time,
            APPROX_QUANTILES(lead_time_hours, 100)[OFFSET(50)] AS median_lead_time,
            APPROX_QUANTILES(lead_time_hours, 100)[OFFSET(75)] AS p75_lead_time

        FROM pr_metrics
        GROUP BY high_exposure
        """


# =============================================================================
# QUERY REGISTRY
# =============================================================================

QUERY_REGISTRY = {
    'velocity_lead_time': VelocityQueries.pr_lead_time,
    'velocity_review_time': VelocityQueries.time_to_first_review,
    'throughput': ThroughputQueries.developer_throughput,
    'burstiness': BurstinessQueries.daily_activity_burstiness,
    'slack': SlackQueries.inter_pr_gaps,
    'complexity': ComplexityQueries.pr_complexity,
}


def get_all_queries(year: int, month: int, languages: Optional[List[str]] = None) -> dict:
    """Get all queries for a given month."""
    return {
        name: query_fn(year, month, languages)
        for name, query_fn in QUERY_REGISTRY.items()
    }
