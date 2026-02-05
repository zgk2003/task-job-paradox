"""
Statistical Analysis for Revised Empirical Strategy

Implements:
1. Interrupted Time Series (ITS) - Sharp discontinuity at treatment date
2. Difference-in-Differences (DiD) - High vs low AI-exposure languages
3. Heterogeneity analysis - By developer activity level
4. Statistical tests - Mann-Whitney U, t-tests
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy import stats

from .config import TREATMENT_DATE, HYPOTHESES, DEFAULT_CONFIG


@dataclass
class ITSResult:
    """Result from Interrupted Time Series analysis."""
    metric: str
    pre_mean: float
    post_mean: float
    change_pct: float
    t_statistic: float
    p_value: float
    significant: bool
    n_pre: int
    n_post: int


@dataclass
class DiDResult:
    """Result from Difference-in-Differences analysis."""
    metric: str
    treated_pre: float
    treated_post: float
    control_pre: float
    control_post: float
    treated_change: float
    control_change: float
    did_estimate: float
    se: Optional[float]
    p_value: Optional[float]
    significant: bool


@dataclass
class HeterogeneityResult:
    """Result from heterogeneity analysis."""
    metric: str
    subgroup: str
    pre: float
    post: float
    change_pct: float
    n: int


class InterruptedTimeSeries:
    """
    Interrupted Time Series Analysis.

    Tests for sharp changes at the treatment date (ChatGPT launch).
    """

    def __init__(self, treatment_date=TREATMENT_DATE):
        self.treatment_date = treatment_date

    def analyze(
        self,
        pre_values: np.ndarray,
        post_values: np.ndarray,
        metric_name: str = 'metric'
    ) -> ITSResult:
        """
        Perform ITS analysis comparing pre and post treatment periods.

        Args:
            pre_values: Array of metric values before treatment
            post_values: Array of metric values after treatment
            metric_name: Name of the metric

        Returns:
            ITSResult with analysis results
        """
        pre_values = np.array(pre_values)
        post_values = np.array(post_values)

        # Remove NaN
        pre_values = pre_values[~np.isnan(pre_values)]
        post_values = post_values[~np.isnan(post_values)]

        if len(pre_values) == 0 or len(post_values) == 0:
            return ITSResult(
                metric=metric_name,
                pre_mean=np.nan,
                post_mean=np.nan,
                change_pct=np.nan,
                t_statistic=np.nan,
                p_value=np.nan,
                significant=False,
                n_pre=0,
                n_post=0
            )

        pre_mean = np.mean(pre_values)
        post_mean = np.mean(post_values)
        change_pct = ((post_mean - pre_mean) / abs(pre_mean)) * 100 if pre_mean != 0 else np.nan

        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(pre_values, post_values)

        return ITSResult(
            metric=metric_name,
            pre_mean=pre_mean,
            post_mean=post_mean,
            change_pct=change_pct,
            t_statistic=t_stat,
            p_value=p_value,
            significant=p_value < 0.05,
            n_pre=len(pre_values),
            n_post=len(post_values)
        )

    def analyze_from_dataframes(
        self,
        pre_df: pd.DataFrame,
        post_df: pd.DataFrame,
        metric_col: str,
        group_col: Optional[str] = None
    ) -> Dict[str, ITSResult]:
        """
        Analyze from DataFrames, optionally grouping by a column.

        Args:
            pre_df: Pre-treatment DataFrame
            post_df: Post-treatment DataFrame
            metric_col: Column name for metric
            group_col: Optional column to group by (e.g., 'high_exposure')

        Returns:
            Dict mapping group name to ITSResult
        """
        results = {}

        if group_col is None:
            results['all'] = self.analyze(
                pre_df[metric_col].values,
                post_df[metric_col].values,
                metric_col
            )
        else:
            for group in pre_df[group_col].unique():
                pre_group = pre_df[pre_df[group_col] == group]
                post_group = post_df[post_df[group_col] == group]

                group_name = str(group)
                results[group_name] = self.analyze(
                    pre_group[metric_col].values,
                    post_group[metric_col].values,
                    f"{metric_col}_{group_name}"
                )

        return results


class DifferenceInDifferences:
    """
    Difference-in-Differences Analysis.

    Compares changes between treatment (high AI-exposure) and control
    (low AI-exposure) groups.
    """

    def analyze(
        self,
        treated_pre: float,
        treated_post: float,
        control_pre: float,
        control_post: float,
        metric_name: str = 'metric',
        treated_pre_n: int = 0,
        treated_post_n: int = 0,
        control_pre_n: int = 0,
        control_post_n: int = 0
    ) -> DiDResult:
        """
        Compute DiD estimate.

        DiD = (Treated_post - Treated_pre) - (Control_post - Control_pre)

        Args:
            treated_pre: Treated group pre-treatment mean
            treated_post: Treated group post-treatment mean
            control_pre: Control group pre-treatment mean
            control_post: Control group post-treatment mean
            metric_name: Name of metric

        Returns:
            DiDResult with estimates
        """
        treated_change = treated_post - treated_pre
        control_change = control_post - control_pre
        did_estimate = treated_change - control_change

        # Simple significance test (assumes independence)
        # In practice, should use clustered standard errors
        return DiDResult(
            metric=metric_name,
            treated_pre=treated_pre,
            treated_post=treated_post,
            control_pre=control_pre,
            control_post=control_post,
            treated_change=treated_change,
            control_change=control_change,
            did_estimate=did_estimate,
            se=None,  # Would need more data for proper SE
            p_value=None,
            significant=False  # Conservative without proper SE
        )

    def analyze_from_comparison(
        self,
        pre_metrics: Dict[str, Any],
        post_metrics: Dict[str, Any],
        metric_key: str
    ) -> Optional[DiDResult]:
        """
        Analyze from pre/post metric dictionaries.

        Args:
            pre_metrics: Dict with 'high' and 'low' exposure metrics for pre-period
            post_metrics: Dict with 'high' and 'low' exposure metrics for post-period
            metric_key: Key for the metric to analyze

        Returns:
            DiDResult or None if data missing
        """
        try:
            treated_pre = pre_metrics['high'][metric_key]
            treated_post = post_metrics['high'][metric_key]
            control_pre = pre_metrics['low'][metric_key]
            control_post = post_metrics['low'][metric_key]

            if any(pd.isna(v) for v in [treated_pre, treated_post, control_pre, control_post]):
                return None

            return self.analyze(
                treated_pre, treated_post,
                control_pre, control_post,
                metric_key
            )
        except (KeyError, TypeError):
            return None


class HeterogeneityAnalysis:
    """
    Heterogeneity Analysis by Developer Activity Level.

    Tests whether effects differ between top contributors and others.
    """

    def analyze_by_quintile(
        self,
        pre_df: pd.DataFrame,
        post_df: pd.DataFrame,
        metric_col: str,
        activity_col: str = 'prs_merged_month'
    ) -> List[HeterogeneityResult]:
        """
        Analyze effects by activity quintile.

        Args:
            pre_df: Pre-treatment developer-level data
            post_df: Post-treatment developer-level data
            metric_col: Column with metric to analyze
            activity_col: Column with activity level

        Returns:
            List of HeterogeneityResult for each quintile
        """
        results = []

        # Assign quintiles
        pre_df = pre_df.copy()
        post_df = post_df.copy()

        pre_df['quintile'] = pd.qcut(
            pre_df[activity_col].rank(method='first'),
            q=5,
            labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
        )
        post_df['quintile'] = pd.qcut(
            post_df[activity_col].rank(method='first'),
            q=5,
            labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
        )

        for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
            pre_q = pre_df[pre_df['quintile'] == q][metric_col]
            post_q = post_df[post_df['quintile'] == q][metric_col]

            if len(pre_q) == 0 or len(post_q) == 0:
                continue

            pre_mean = pre_q.mean()
            post_mean = post_q.mean()
            change = ((post_mean - pre_mean) / abs(pre_mean)) * 100 if pre_mean != 0 else np.nan

            results.append(HeterogeneityResult(
                metric=metric_col,
                subgroup=q,
                pre=pre_mean,
                post=post_mean,
                change_pct=change,
                n=len(pre_q) + len(post_q)
            ))

        return results

    def analyze_top_vs_rest(
        self,
        pre_df: pd.DataFrame,
        post_df: pd.DataFrame,
        metric_col: str,
        activity_col: str = 'prs_merged_month',
        top_pct: float = 0.20
    ) -> Tuple[HeterogeneityResult, HeterogeneityResult]:
        """
        Analyze effects for top X% vs rest.

        Args:
            pre_df: Pre-treatment data
            post_df: Post-treatment data
            metric_col: Metric column
            activity_col: Activity column
            top_pct: Top percentile (e.g., 0.20 for top 20%)

        Returns:
            Tuple of (top_result, rest_result)
        """
        pre_threshold = pre_df[activity_col].quantile(1 - top_pct)
        post_threshold = post_df[activity_col].quantile(1 - top_pct)

        pre_top = pre_df[pre_df[activity_col] >= pre_threshold][metric_col]
        pre_rest = pre_df[pre_df[activity_col] < pre_threshold][metric_col]
        post_top = post_df[post_df[activity_col] >= post_threshold][metric_col]
        post_rest = post_df[post_df[activity_col] < post_threshold][metric_col]

        def compute_result(pre, post, label):
            pre_mean = pre.mean()
            post_mean = post.mean()
            change = ((post_mean - pre_mean) / abs(pre_mean)) * 100 if pre_mean != 0 else np.nan
            return HeterogeneityResult(
                metric=metric_col,
                subgroup=label,
                pre=pre_mean,
                post=post_mean,
                change_pct=change,
                n=len(pre) + len(post)
            )

        return (
            compute_result(pre_top, post_top, f'Top {int(top_pct*100)}%'),
            compute_result(pre_rest, post_rest, f'Rest {int((1-top_pct)*100)}%')
        )


class StatisticalTests:
    """Statistical tests for hypothesis testing."""

    @staticmethod
    def mann_whitney_u(
        pre_values: np.ndarray,
        post_values: np.ndarray
    ) -> Tuple[float, float]:
        """
        Non-parametric Mann-Whitney U test.

        Returns:
            Tuple of (U-statistic, p-value)
        """
        pre = np.array(pre_values)
        post = np.array(post_values)

        pre = pre[~np.isnan(pre)]
        post = post[~np.isnan(post)]

        if len(pre) == 0 or len(post) == 0:
            return np.nan, np.nan

        stat, p_value = stats.mannwhitneyu(pre, post, alternative='two-sided')
        return stat, p_value

    @staticmethod
    def effect_size_cohens_d(
        pre_values: np.ndarray,
        post_values: np.ndarray
    ) -> float:
        """
        Compute Cohen's d effect size.

        Returns:
            Cohen's d value
        """
        pre = np.array(pre_values)
        post = np.array(post_values)

        pre = pre[~np.isnan(pre)]
        post = post[~np.isnan(post)]

        if len(pre) == 0 or len(post) == 0:
            return np.nan

        pooled_std = np.sqrt(
            ((len(pre) - 1) * np.var(pre) + (len(post) - 1) * np.var(post)) /
            (len(pre) + len(post) - 2)
        )

        if pooled_std == 0:
            return np.nan

        return (np.mean(post) - np.mean(pre)) / pooled_std


def run_full_statistical_analysis(
    pre_metrics: Dict[str, Any],
    post_metrics: Dict[str, Any],
    pre_throughput_df: Optional[pd.DataFrame] = None,
    post_throughput_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Run full statistical analysis suite.

    Args:
        pre_metrics: Computed metrics for pre-treatment period
        post_metrics: Computed metrics for post-treatment period
        pre_throughput_df: Optional developer-level throughput data (pre)
        post_throughput_df: Optional developer-level throughput data (post)

    Returns:
        Dict with all analysis results
    """
    results = {
        'its': {},
        'did': {},
        'heterogeneity': {},
        'tests': {}
    }

    did = DifferenceInDifferences()

    # Analyze key metrics with DiD
    metrics_to_analyze = [
        ('velocity', 'median_lead_time_hours'),
        ('velocity', 'p75_lead_time_hours'),
        ('throughput', 'avg_prs_per_dev'),
        ('throughput', 'median_prs_per_dev'),
        ('complexity', 'median_churn'),
        ('complexity', 'median_files'),
        ('burstiness', 'avg_active_days_ratio'),
        ('slack', 'median_gap_hours'),
    ]

    for category, metric in metrics_to_analyze:
        if category in pre_metrics and category in post_metrics:
            result = did.analyze_from_comparison(
                pre_metrics[category],
                post_metrics[category],
                metric
            )
            if result:
                results['did'][f'{category}_{metric}'] = result

    # Heterogeneity analysis if developer-level data available
    if pre_throughput_df is not None and post_throughput_df is not None:
        het = HeterogeneityAnalysis()

        # Filter to high exposure only
        pre_high = pre_throughput_df[pre_throughput_df['high_exposure'] == True]
        post_high = post_throughput_df[post_throughput_df['high_exposure'] == True]

        if not pre_high.empty and not post_high.empty:
            # By quintile
            results['heterogeneity']['by_quintile'] = het.analyze_by_quintile(
                pre_high, post_high, 'prs_merged_month'
            )

            # Top 20% vs rest
            top, rest = het.analyze_top_vs_rest(pre_high, post_high, 'prs_merged_month')
            results['heterogeneity']['top_20'] = top
            results['heterogeneity']['rest_80'] = rest

            # Mann-Whitney U test for throughput
            u_stat, p_val = StatisticalTests.mann_whitney_u(
                pre_high['prs_merged_month'].values,
                post_high['prs_merged_month'].values
            )
            results['tests']['throughput_mannwhitney'] = {
                'u_statistic': u_stat,
                'p_value': p_val,
                'significant': p_val < 0.05 if not np.isnan(p_val) else False
            }

    return results
