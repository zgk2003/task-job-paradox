"""
Metrics Extraction Module for Task-Job Paradox Analysis

This module calculates the core metrics at both task and job levels:

TASK-LEVEL METRICS (Execution-proximal):
- Review-response latency: Time from review feedback to author's revision
- CI-fix latency: Time from test failure to successful fix

JOB-LEVEL METRICS (Broader work objectives):
- PR lead time: Time from PR creation to merge
- Release inclusion time: Time from commit to release

MECHANISM METRICS (Explaining the paradox):
- Commits per PR: Iteration intensity
- Review rounds: Feedback cycles
- Code churn: Lines changed/rework

The key hypothesis is that task-level metrics improve substantially with AI,
but job-level metrics improve much less due to structural bottlenecks.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
from collections import defaultdict

from .data_model import (
    Repository, PullRequest, Language,
    TREATMENT_START, PRE_PERIOD_START, POST_PERIOD_END
)


class MetricsExtractor:
    """Extract and aggregate metrics from repository data."""

    def __init__(self, repositories: List[Repository]):
        self.repositories = repositories

    def extract_pr_level_metrics(self) -> pd.DataFrame:
        """Extract metrics at the pull request level.

        Returns a DataFrame with one row per PR containing:
        - Identifiers: repo_id, pr_id
        - Timing: created_at, merged_at, year_month
        - Treatment: post_treatment, high_exposure
        - Task metrics: review_response_latency, ci_fix_latency
        - Job metrics: lead_time_hours
        - Mechanism metrics: num_commits, num_review_rounds, code_churn
        """
        records = []

        for repo in self.repositories:
            for pr in repo.pull_requests:
                if pr.merged_at is None:
                    continue  # Only analyze merged PRs

                # Calculate task-level metrics
                review_latencies = pr.review_response_latencies()
                ci_latencies = pr.ci_fix_latencies()

                record = {
                    # Identifiers
                    'repo_id': repo.repo_id,
                    'pr_id': pr.pr_id,
                    'repo_name': repo.name,

                    # Timing
                    'created_at': pr.created_at,
                    'merged_at': pr.merged_at,
                    'year_month': pr.created_at.strftime('%Y-%m'),
                    'year': pr.created_at.year,
                    'month': pr.created_at.month,

                    # Treatment indicators
                    'post_treatment': pr.created_at >= TREATMENT_START,
                    'high_exposure': repo.high_ai_exposure,
                    'language': repo.primary_language.value,

                    # Moderators (for H5)
                    'num_contributors': repo.num_contributors,
                    'high_coordination': repo.high_coordination,

                    # Task-level metrics (H1)
                    'review_response_latency': (
                        np.mean(review_latencies) if review_latencies else np.nan
                    ),
                    'ci_fix_latency': (
                        np.mean(ci_latencies) if ci_latencies else np.nan
                    ),

                    # Job-level metrics (H2)
                    'lead_time_hours': pr.lead_time_hours,

                    # Mechanism metrics (H4)
                    'num_commits': pr.num_commits,
                    'num_review_rounds': pr.num_review_rounds,
                    'code_churn': pr.total_code_churn,
                }
                records.append(record)

        return pd.DataFrame(records)

    def aggregate_monthly(self, pr_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate PR-level metrics to monthly time series.

        This is the primary unit of analysis for interrupted time series.
        """
        # Group by year-month and exposure level
        grouped = pr_df.groupby(['year_month', 'high_exposure'])

        monthly = grouped.agg({
            # Task-level metrics
            'review_response_latency': 'mean',
            'ci_fix_latency': 'mean',

            # Job-level metrics
            'lead_time_hours': 'mean',

            # Mechanism metrics
            'num_commits': 'mean',
            'num_review_rounds': 'mean',
            'code_churn': 'mean',

            # Sample size
            'pr_id': 'count'
        }).rename(columns={'pr_id': 'num_prs'})

        monthly = monthly.reset_index()
        monthly['date'] = pd.to_datetime(monthly['year_month'] + '-01')
        monthly['post_treatment'] = monthly['date'] >= TREATMENT_START

        return monthly

    def compute_summary_statistics(self, pr_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute summary statistics for the dataset."""
        pre = pr_df[~pr_df['post_treatment']]
        post = pr_df[pr_df['post_treatment']]

        high_exp = pr_df[pr_df['high_exposure']]
        low_exp = pr_df[~pr_df['high_exposure']]

        return {
            'total_prs': len(pr_df),
            'total_repos': pr_df['repo_id'].nunique(),
            'pre_treatment_prs': len(pre),
            'post_treatment_prs': len(post),
            'high_exposure_prs': len(high_exp),
            'low_exposure_prs': len(low_exp),

            # Pre-treatment means
            'pre_review_latency_mean': pre['review_response_latency'].mean(),
            'pre_ci_fix_latency_mean': pre['ci_fix_latency'].mean(),
            'pre_lead_time_mean': pre['lead_time_hours'].mean(),
            'pre_commits_mean': pre['num_commits'].mean(),

            # Post-treatment means
            'post_review_latency_mean': post['review_response_latency'].mean(),
            'post_ci_fix_latency_mean': post['ci_fix_latency'].mean(),
            'post_lead_time_mean': post['lead_time_hours'].mean(),
            'post_commits_mean': post['num_commits'].mean(),
        }


def calculate_percent_change(pre_mean: float, post_mean: float) -> float:
    """Calculate percentage change from pre to post period."""
    if pre_mean == 0:
        return np.nan
    return 100 * (post_mean - pre_mean) / pre_mean


def create_analysis_dataset(repositories: List[Repository]) -> Dict[str, pd.DataFrame]:
    """Create all analysis datasets from raw repository data.

    Returns:
        Dictionary containing:
        - 'pr_level': PR-level observations
        - 'monthly': Monthly aggregated time series
        - 'monthly_by_exposure': Separate time series by AI exposure
    """
    extractor = MetricsExtractor(repositories)

    # PR-level data
    pr_df = extractor.extract_pr_level_metrics()
    pr_df = pr_df.sort_values('created_at')

    # Monthly aggregations
    monthly_df = extractor.aggregate_monthly(pr_df)

    # Summary stats
    summary = extractor.compute_summary_statistics(pr_df)

    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Total PRs: {summary['total_prs']:,}")
    print(f"Total Repos: {summary['total_repos']:,}")
    print(f"Pre-treatment PRs: {summary['pre_treatment_prs']:,}")
    print(f"Post-treatment PRs: {summary['post_treatment_prs']:,}")
    print(f"High AI exposure PRs: {summary['high_exposure_prs']:,}")
    print(f"Low AI exposure PRs: {summary['low_exposure_prs']:,}")

    print("\n" + "-"*60)
    print("METRIC CHANGES (Pre vs Post Treatment)")
    print("-"*60)

    # Task-level changes
    review_change = calculate_percent_change(
        summary['pre_review_latency_mean'],
        summary['post_review_latency_mean']
    )
    ci_change = calculate_percent_change(
        summary['pre_ci_fix_latency_mean'],
        summary['post_ci_fix_latency_mean']
    )

    # Job-level changes
    lead_time_change = calculate_percent_change(
        summary['pre_lead_time_mean'],
        summary['post_lead_time_mean']
    )

    # Mechanism changes
    commits_change = calculate_percent_change(
        summary['pre_commits_mean'],
        summary['post_commits_mean']
    )

    print("\nTASK-LEVEL METRICS:")
    print(f"  Review-response latency: {summary['pre_review_latency_mean']:.1f}h → "
          f"{summary['post_review_latency_mean']:.1f}h ({review_change:+.1f}%)")
    print(f"  CI-fix latency: {summary['pre_ci_fix_latency_mean']:.1f}h → "
          f"{summary['post_ci_fix_latency_mean']:.1f}h ({ci_change:+.1f}%)")

    print("\nJOB-LEVEL METRICS:")
    print(f"  PR lead time: {summary['pre_lead_time_mean']:.1f}h → "
          f"{summary['post_lead_time_mean']:.1f}h ({lead_time_change:+.1f}%)")

    print("\nMECHANISM METRICS:")
    print(f"  Commits per PR: {summary['pre_commits_mean']:.1f} → "
          f"{summary['post_commits_mean']:.1f} ({commits_change:+.1f}%)")

    print("\n" + "="*60)
    print("THE PARADOX:")
    print(f"  Task-level improvement: ~{abs(review_change):.0f}%")
    print(f"  Job-level improvement:  ~{abs(lead_time_change):.0f}%")
    print(f"  Gap: {abs(review_change) - abs(lead_time_change):.0f} percentage points")
    print("="*60)

    return {
        'pr_level': pr_df,
        'monthly': monthly_df,
        'summary': summary
    }
