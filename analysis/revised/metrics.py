"""
Metrics Extraction and Computation for Revised Empirical Strategy

This module handles:
1. Loading data from BigQuery (with caching)
2. Computing aggregate metrics from developer-level data
3. Stratification by developer activity level
4. Multi-granularity metric computation
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

from .config import (
    HIGH_EXPOSURE_LANGUAGES,
    MetricCategory,
    AnalysisConfig,
    DEFAULT_CONFIG,
)
from .queries import QUERY_REGISTRY


# BigQuery imports with fallback
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from bigquery_client import BigQueryGitHubClient, create_bigquery_client
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False


@dataclass
class MetricResult:
    """Container for a metric result with metadata."""
    name: str
    category: MetricCategory
    value: float
    unit: str
    period: str
    high_exposure: bool
    granularity: str
    sample_size: int = 0


class DataLoader:
    """Loads data from BigQuery with caching support."""

    def __init__(
        self,
        client: Optional[BigQueryGitHubClient] = None,
        cache_dir: Optional[Path] = None,
        config: AnalysisConfig = DEFAULT_CONFIG
    ):
        self.client = client
        self.cache_dir = cache_dir or Path(__file__).parent.parent.parent / 'results'
        self.config = config
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_client(self) -> BigQueryGitHubClient:
        """Get or create BigQuery client."""
        if self.client is None:
            if not BIGQUERY_AVAILABLE:
                raise ImportError("BigQuery not available. Install google-cloud-bigquery.")
            self.client = create_bigquery_client()
        return self.client

    def _cache_path(self, query_name: str, year: int, month: int) -> Path:
        """Get cache file path for a query."""
        return self.cache_dir / f'{query_name}_{year}_{month:02d}.csv'

    def load_query(
        self,
        query_name: str,
        year: int,
        month: int,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Load data for a query, using cache if available.

        Args:
            query_name: Name of query from QUERY_REGISTRY
            year: Year to query
            month: Month to query
            force_refresh: If True, ignore cache and re-query

        Returns:
            DataFrame with query results
        """
        cache_path = self._cache_path(query_name, year, month)

        # Try cache first
        if not force_refresh and cache_path.exists():
            print(f"Loading cached {query_name} for {year}-{month:02d}")
            return pd.read_csv(cache_path)

        # Run query
        if query_name not in QUERY_REGISTRY:
            raise ValueError(f"Unknown query: {query_name}")

        query_fn = QUERY_REGISTRY[query_name]
        query = query_fn(year, month)

        client = self._get_client()
        print(f"Querying {query_name} for {year}-{month:02d}...")

        df = client.run_query(query, max_cost_usd=self.config.max_query_cost_usd)

        # Cache results
        if not df.empty:
            df.to_csv(cache_path, index=False)
            print(f"Cached to {cache_path}")

        return df

    def load_all_metrics(
        self,
        year: int,
        month: int,
        force_refresh: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all metrics for a given month.

        Returns:
            Dict mapping query name to DataFrame
        """
        results = {}
        for query_name in QUERY_REGISTRY:
            try:
                results[query_name] = self.load_query(query_name, year, month, force_refresh)
            except Exception as e:
                print(f"Error loading {query_name}: {e}")
                results[query_name] = pd.DataFrame()
        return results


class MetricsComputer:
    """Computes aggregate metrics from raw data."""

    def __init__(self, config: AnalysisConfig = DEFAULT_CONFIG):
        self.config = config

    def compute_velocity_metrics(
        self,
        velocity_df: pd.DataFrame,
        review_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Compute velocity metrics from query results."""
        results = {}

        for high_exp in [True, False]:
            key = 'high' if high_exp else 'low'
            vel = velocity_df[velocity_df['high_exposure'] == high_exp]

            if vel.empty:
                continue

            row = vel.iloc[0]
            results[key] = {
                'num_prs': row.get('num_prs', 0),
                'num_repos': row.get('num_repos', 0),
                'num_authors': row.get('num_authors', 0),
                'avg_lead_time_hours': row.get('avg_lead_time_hours'),
                'median_lead_time_hours': row.get('median_lead_time_hours'),
                'p75_lead_time_hours': row.get('p75_lead_time_hours'),
                'p90_lead_time_hours': row.get('p90_lead_time_hours'),
            }

            # Add review metrics if available
            if review_df is not None and not review_df.empty:
                rev = review_df[review_df['high_exposure'] == high_exp]
                if not rev.empty:
                    rev_row = rev.iloc[0]
                    results[key]['avg_time_to_review_hours'] = rev_row.get('avg_time_to_review_hours')
                    results[key]['median_time_to_review_hours'] = rev_row.get('median_time_to_review_hours')

        return results

    def compute_throughput_metrics(
        self,
        throughput_df: pd.DataFrame,
        stratify_by_activity: bool = True
    ) -> Dict[str, Any]:
        """
        Compute throughput metrics from developer-level data.

        Args:
            throughput_df: Developer-level throughput data
            stratify_by_activity: If True, compute metrics for activity quintiles

        Returns:
            Dict with aggregate and stratified metrics
        """
        results = {}

        for high_exp in [True, False]:
            key = 'high' if high_exp else 'low'
            tput = throughput_df[throughput_df['high_exposure'] == high_exp]

            if tput.empty:
                continue

            # Aggregate metrics
            results[key] = {
                'num_developers': len(tput),
                'total_prs': tput['prs_merged_month'].sum(),
                'avg_prs_per_dev': tput['prs_merged_month'].mean(),
                'median_prs_per_dev': tput['prs_merged_month'].median(),
                'std_prs_per_dev': tput['prs_merged_month'].std(),
                'p75_prs_per_dev': tput['prs_merged_month'].quantile(0.75),
                'p90_prs_per_dev': tput['prs_merged_month'].quantile(0.90),
                'avg_active_weeks': tput['active_weeks'].mean(),
            }

            # Burstiness from weekly CV
            cv_weekly = tput['cv_weekly_prs'].dropna()
            if len(cv_weekly) > 0:
                results[key]['avg_cv_weekly_prs'] = cv_weekly.mean()
                results[key]['median_cv_weekly_prs'] = cv_weekly.median()

            # Stratified by activity quintile
            if stratify_by_activity:
                results[key]['by_quintile'] = self._compute_quintile_metrics(tput)
                results[key]['top_20_pct'] = self._compute_top_contributor_metrics(tput)

        return results

    def _compute_quintile_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute metrics for each activity quintile."""
        df = df.copy()
        df['quintile'] = pd.qcut(
            df['prs_merged_month'].rank(method='first'),
            q=5,
            labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
        )

        quintile_metrics = {}
        for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
            q_data = df[df['quintile'] == q]
            quintile_metrics[q] = {
                'n': len(q_data),
                'avg_prs': q_data['prs_merged_month'].mean(),
                'median_prs': q_data['prs_merged_month'].median(),
                'avg_cv_weekly': q_data['cv_weekly_prs'].mean() if 'cv_weekly_prs' in q_data else None,
            }

        return quintile_metrics

    def _compute_top_contributor_metrics(
        self,
        df: pd.DataFrame,
        top_pct: float = 0.20
    ) -> Dict[str, Any]:
        """Compute metrics for top contributors."""
        threshold = df['prs_merged_month'].quantile(1 - top_pct)
        top = df[df['prs_merged_month'] >= threshold]

        return {
            'n_developers': len(top),
            'threshold_prs': threshold,
            'avg_prs': top['prs_merged_month'].mean(),
            'median_prs': top['prs_merged_month'].median(),
            'avg_cv_weekly': top['cv_weekly_prs'].mean() if 'cv_weekly_prs' in top else None,
            'avg_active_weeks': top['active_weeks'].mean(),
        }

    def compute_burstiness_metrics(self, burstiness_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute burstiness metrics from daily activity data."""
        results = {}

        for high_exp in [True, False]:
            key = 'high' if high_exp else 'low'
            burst = burstiness_df[burstiness_df['high_exposure'] == high_exp]

            if burst.empty:
                continue

            results[key] = {
                'num_developers': len(burst),
                'avg_cv_daily': burst['cv_daily_events'].mean(),
                'median_cv_daily': burst['cv_daily_events'].median(),
                'avg_active_days': burst['active_days'].mean(),
                'median_active_days': burst['active_days'].median(),
                'avg_active_days_ratio': burst['active_days_ratio'].mean(),
                'median_active_days_ratio': burst['active_days_ratio'].median(),
            }

        return results

    def compute_slack_metrics(self, slack_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute slack metrics from inter-PR gap data."""
        results = {}

        for high_exp in [True, False]:
            key = 'high' if high_exp else 'low'
            gaps = slack_df[slack_df['high_exposure'] == high_exp]

            if gaps.empty:
                continue

            results[key] = {
                'num_developers': len(gaps),
                'avg_gap_hours': gaps['avg_gap_hours'].mean(),
                'median_gap_hours': gaps['median_gap_hours'].median(),
                'avg_p75_gap_hours': gaps['p75_gap_hours'].mean(),
                'avg_gap_business_days': gaps['avg_gap_business_days'].mean(),
                'median_gap_business_days': gaps['median_gap_business_days'].median(),
            }

        return results

    def compute_complexity_metrics(self, complexity_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute complexity metrics from PR size data."""
        results = {}

        for high_exp in [True, False]:
            key = 'high' if high_exp else 'low'
            comp = complexity_df[complexity_df['high_exposure'] == high_exp]

            if comp.empty:
                continue

            row = comp.iloc[0]
            results[key] = {
                'num_prs': row.get('num_prs', 0),
                'median_churn': row.get('median_churn'),
                'p75_churn': row.get('p75_churn'),
                'p90_churn': row.get('p90_churn'),
                'median_files': row.get('median_files'),
                'p75_files': row.get('p75_files'),
                'median_commits': row.get('median_commits'),
                'p75_commits': row.get('p75_commits'),
                'median_review_comments': row.get('median_review_comments'),
            }

        return results


class PeriodComparison:
    """Compare metrics between pre and post treatment periods."""

    def __init__(self, pre_metrics: Dict, post_metrics: Dict):
        self.pre = pre_metrics
        self.post = post_metrics

    def compute_changes(self) -> Dict[str, Any]:
        """Compute percentage changes between periods."""
        changes = {}

        for exposure in ['high', 'low']:
            if exposure not in self.pre or exposure not in self.post:
                continue

            pre = self.pre[exposure]
            post = self.post[exposure]
            changes[exposure] = {}

            for key in pre:
                if key in post and isinstance(pre[key], (int, float)) and isinstance(post[key], (int, float)):
                    pre_val = pre[key]
                    post_val = post[key]

                    if pd.notna(pre_val) and pd.notna(post_val) and pre_val != 0:
                        pct_change = ((post_val - pre_val) / abs(pre_val)) * 100
                        changes[exposure][key] = {
                            'pre': pre_val,
                            'post': post_val,
                            'change_pct': pct_change
                        }

        return changes

    def to_dataframe(self) -> pd.DataFrame:
        """Convert comparison to DataFrame."""
        changes = self.compute_changes()
        rows = []

        for exposure, metrics in changes.items():
            for metric, values in metrics.items():
                rows.append({
                    'exposure': exposure,
                    'metric': metric,
                    'pre': values['pre'],
                    'post': values['post'],
                    'change_pct': values['change_pct']
                })

        return pd.DataFrame(rows)


def load_and_compute_all_metrics(
    year: int,
    month: int,
    loader: Optional[DataLoader] = None,
    computer: Optional[MetricsComputer] = None
) -> Dict[str, Any]:
    """
    Convenience function to load and compute all metrics for a month.

    Args:
        year: Year
        month: Month
        loader: DataLoader instance (created if None)
        computer: MetricsComputer instance (created if None)

    Returns:
        Dict with all computed metrics
    """
    loader = loader or DataLoader()
    computer = computer or MetricsComputer()

    # Load raw data
    raw_data = loader.load_all_metrics(year, month)

    # Compute metrics
    results = {
        'year_month': f'{year}-{month:02d}',
        'raw_data': raw_data,
    }

    if not raw_data.get('velocity_lead_time', pd.DataFrame()).empty:
        results['velocity'] = computer.compute_velocity_metrics(
            raw_data['velocity_lead_time'],
            raw_data.get('velocity_review_time')
        )

    if not raw_data.get('throughput', pd.DataFrame()).empty:
        results['throughput'] = computer.compute_throughput_metrics(raw_data['throughput'])

    if not raw_data.get('burstiness', pd.DataFrame()).empty:
        results['burstiness'] = computer.compute_burstiness_metrics(raw_data['burstiness'])

    if not raw_data.get('slack', pd.DataFrame()).empty:
        results['slack'] = computer.compute_slack_metrics(raw_data['slack'])

    if not raw_data.get('complexity', pd.DataFrame()).empty:
        results['complexity'] = computer.compute_complexity_metrics(raw_data['complexity'])

    return results
