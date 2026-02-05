"""
Statistical Analysis Module for Task-Job Paradox

Implements two complementary identification strategies:

1. INTERRUPTED TIME SERIES (ITS):
   - Compares outcomes before/after November 2022 AI adoption
   - Model: Y_rt = Σ_k β_k × 1[t=k] + α_r + ε_rt
   - Tests H1 (task improvement) and H2 (job improvement < task improvement)

2. DIFFERENCE-IN-DIFFERENCES (DiD):
   - Compares high vs low AI exposure languages before/after treatment
   - Model: Y_rt = β(Post × HighExposure) + γPost + δHighExposure + α_r + ε_rt
   - Tests H3 (effects concentrated in high exposure)

3. HETEROGENEITY ANALYSIS:
   - Tests H5: Task-job gap larger in coordination-heavy projects
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Statistical imports with fallbacks
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.regression.linear_model import OLS
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

from .data_model import TREATMENT_START


@dataclass
class RegressionResult:
    """Container for regression results."""
    coefficient: float
    std_error: float
    t_stat: float
    p_value: float
    ci_lower: float
    ci_upper: float
    n_obs: int
    r_squared: float

    def is_significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha

    def __str__(self) -> str:
        sig = "***" if self.p_value < 0.01 else "**" if self.p_value < 0.05 else "*" if self.p_value < 0.1 else ""
        return (f"β = {self.coefficient:.3f}{sig} (SE = {self.std_error:.3f}), "
                f"p = {self.p_value:.3f}, n = {self.n_obs:,}")


class InterruptedTimeSeriesAnalysis:
    """
    Interrupted Time Series analysis for AI adoption effects.

    Tests whether metrics changed after AI tool adoption (Nov 2022).
    Separate analyses for task-level and job-level metrics.
    """

    def __init__(self, pr_df: pd.DataFrame):
        self.pr_df = pr_df
        self._prepare_data()

    def _prepare_data(self):
        """Prepare data for ITS analysis."""
        self.pr_df = self.pr_df.copy()

        # Time variables
        self.pr_df['months_since_start'] = (
            (self.pr_df['created_at'] - self.pr_df['created_at'].min()).dt.days / 30
        ).astype(int)

        # Treatment indicator
        self.pr_df['post'] = self.pr_df['post_treatment'].astype(int)

        # Months since treatment (for trend break)
        self.pr_df['months_since_treatment'] = np.where(
            self.pr_df['post_treatment'],
            (self.pr_df['created_at'] - TREATMENT_START).dt.days / 30,
            0
        )

    def run_its_regression(
        self,
        outcome: str,
        include_trend: bool = True
    ) -> RegressionResult:
        """
        Run interrupted time series regression.

        Model: Y = β0 + β1*time + β2*post + β3*time_since_treatment + ε

        Args:
            outcome: Name of outcome variable
            include_trend: Whether to include linear time trends

        Returns:
            RegressionResult for the post-treatment effect (β2)
        """
        df = self.pr_df.dropna(subset=[outcome])

        if len(df) < 100:
            raise ValueError(f"Insufficient observations for {outcome}")

        if not STATSMODELS_AVAILABLE:
            return self._simple_comparison(df, outcome)

        # Build formula
        if include_trend:
            formula = f"{outcome} ~ months_since_start + post + months_since_treatment"
        else:
            formula = f"{outcome} ~ post"

        model = smf.ols(formula, data=df).fit(cov_type='HC1')

        return RegressionResult(
            coefficient=model.params['post'],
            std_error=model.bse['post'],
            t_stat=model.tvalues['post'],
            p_value=model.pvalues['post'],
            ci_lower=model.conf_int().loc['post', 0],
            ci_upper=model.conf_int().loc['post', 1],
            n_obs=int(model.nobs),
            r_squared=model.rsquared
        )

    def _simple_comparison(
        self,
        df: pd.DataFrame,
        outcome: str
    ) -> RegressionResult:
        """Fallback: simple pre/post comparison when statsmodels unavailable."""
        pre = df[~df['post_treatment']][outcome]
        post = df[df['post_treatment']][outcome]

        diff = post.mean() - pre.mean()
        pooled_se = np.sqrt(pre.var()/len(pre) + post.var()/len(post))
        t_stat = diff / pooled_se
        p_value = 2 * (1 - self._t_cdf(abs(t_stat), len(df)-2))

        return RegressionResult(
            coefficient=diff,
            std_error=pooled_se,
            t_stat=t_stat,
            p_value=p_value,
            ci_lower=diff - 1.96*pooled_se,
            ci_upper=diff + 1.96*pooled_se,
            n_obs=len(df),
            r_squared=np.nan
        )

    def _t_cdf(self, t: float, df: int) -> float:
        """Approximate t-distribution CDF."""
        # Using normal approximation for large df
        from math import erf, sqrt
        return 0.5 * (1 + erf(t / sqrt(2)))

    def analyze_all_metrics(self) -> Dict[str, RegressionResult]:
        """Run ITS analysis for all key metrics."""
        metrics = {
            # Task-level
            'review_response_latency': 'Review-Response Latency (task)',
            'ci_fix_latency': 'CI-Fix Latency (task)',

            # Job-level
            'lead_time_hours': 'PR Lead Time (job)',

            # Mechanism
            'num_commits': 'Commits per PR',
            'num_review_rounds': 'Review Rounds',
            'code_churn': 'Code Churn',
        }

        results = {}
        for metric, label in metrics.items():
            try:
                results[metric] = self.run_its_regression(metric)
            except Exception as e:
                print(f"Warning: Could not analyze {metric}: {e}")

        return results


class DifferenceInDifferencesAnalysis:
    """
    Difference-in-Differences analysis exploiting language exposure variation.

    Compares high AI exposure languages (Python, JS, Java) vs low exposure
    languages before and after AI tool adoption.

    Key identifying assumption: Parallel trends in absence of treatment.
    """

    def __init__(self, pr_df: pd.DataFrame):
        self.pr_df = pr_df.copy()
        self._prepare_data()

    def _prepare_data(self):
        """Prepare data for DiD analysis."""
        self.pr_df['post'] = self.pr_df['post_treatment'].astype(int)
        self.pr_df['treated'] = self.pr_df['high_exposure'].astype(int)
        self.pr_df['post_treated'] = self.pr_df['post'] * self.pr_df['treated']

    def run_did_regression(
        self,
        outcome: str,
        include_repo_fe: bool = False
    ) -> RegressionResult:
        """
        Run difference-in-differences regression.

        Model: Y = β0 + β1*Post + β2*Treated + β3*Post×Treated + ε

        The coefficient of interest is β3 (post_treated): the differential
        effect of AI adoption on high-exposure vs low-exposure languages.

        Args:
            outcome: Name of outcome variable
            include_repo_fe: Whether to include repository fixed effects

        Returns:
            RegressionResult for the DiD coefficient (β3)
        """
        df = self.pr_df.dropna(subset=[outcome])

        if len(df) < 100:
            raise ValueError(f"Insufficient observations for {outcome}")

        if not STATSMODELS_AVAILABLE:
            return self._simple_did(df, outcome)

        if include_repo_fe:
            formula = f"{outcome} ~ post + treated + post_treated + C(repo_id)"
        else:
            formula = f"{outcome} ~ post + treated + post_treated"

        model = smf.ols(formula, data=df).fit(cov_type='HC1')

        return RegressionResult(
            coefficient=model.params['post_treated'],
            std_error=model.bse['post_treated'],
            t_stat=model.tvalues['post_treated'],
            p_value=model.pvalues['post_treated'],
            ci_lower=model.conf_int().loc['post_treated', 0],
            ci_upper=model.conf_int().loc['post_treated', 1],
            n_obs=int(model.nobs),
            r_squared=model.rsquared
        )

    def _simple_did(self, df: pd.DataFrame, outcome: str) -> RegressionResult:
        """Fallback: manual DiD calculation."""
        # Calculate 2x2 means
        pre_treat = df[(~df['post_treatment']) & (df['high_exposure'])][outcome].mean()
        pre_ctrl = df[(~df['post_treatment']) & (~df['high_exposure'])][outcome].mean()
        post_treat = df[(df['post_treatment']) & (df['high_exposure'])][outcome].mean()
        post_ctrl = df[(df['post_treatment']) & (~df['high_exposure'])][outcome].mean()

        # DiD estimate
        did = (post_treat - pre_treat) - (post_ctrl - pre_ctrl)

        # Approximate SE (simplified)
        n = len(df)
        se = df[outcome].std() / np.sqrt(n/4)

        t_stat = did / se
        p_value = 2 * (1 - self._t_cdf(abs(t_stat), n-4))

        return RegressionResult(
            coefficient=did,
            std_error=se,
            t_stat=t_stat,
            p_value=p_value,
            ci_lower=did - 1.96*se,
            ci_upper=did + 1.96*se,
            n_obs=n,
            r_squared=np.nan
        )

    def _t_cdf(self, t: float, df: int) -> float:
        from math import erf, sqrt
        return 0.5 * (1 + erf(t / sqrt(2)))

    def analyze_all_metrics(self) -> Dict[str, RegressionResult]:
        """Run DiD analysis for all key metrics."""
        metrics = {
            'review_response_latency': 'Review-Response Latency (task)',
            'ci_fix_latency': 'CI-Fix Latency (task)',
            'lead_time_hours': 'PR Lead Time (job)',
            'num_commits': 'Commits per PR',
        }

        results = {}
        for metric, label in metrics.items():
            try:
                results[metric] = self.run_did_regression(metric)
            except Exception as e:
                print(f"Warning: Could not analyze {metric}: {e}")

        return results


class HeterogeneityAnalysis:
    """
    Test H5: Task-job gap is larger in coordination-heavy projects.

    Coordination intensity proxied by:
    - Number of contributors
    - Average review rounds
    """

    def __init__(self, pr_df: pd.DataFrame):
        self.pr_df = pr_df.copy()

    def analyze_by_coordination(self) -> Dict[str, Any]:
        """Compare task-job gap by coordination intensity."""
        high_coord = self.pr_df[self.pr_df['high_coordination']]
        low_coord = self.pr_df[~self.pr_df['high_coordination']]

        def calc_gap(df):
            """Calculate task-job improvement gap."""
            pre = df[~df['post_treatment']]
            post = df[df['post_treatment']]

            # Task improvement (review latency)
            task_pre = pre['review_response_latency'].mean()
            task_post = post['review_response_latency'].mean()
            task_change = (task_post - task_pre) / task_pre * 100

            # Job improvement (lead time)
            job_pre = pre['lead_time_hours'].mean()
            job_post = post['lead_time_hours'].mean()
            job_change = (job_post - job_pre) / job_pre * 100

            return {
                'task_change_pct': task_change,
                'job_change_pct': job_change,
                'gap': abs(task_change) - abs(job_change),
                'n_pre': len(pre),
                'n_post': len(post)
            }

        return {
            'high_coordination': calc_gap(high_coord),
            'low_coordination': calc_gap(low_coord)
        }


def run_full_analysis(pr_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run complete statistical analysis and return results.

    Tests all hypotheses:
    - H1: Task-level metrics improved after AI adoption
    - H2: Job-level metrics improved less than task-level
    - H3: Effects concentrated in high AI exposure languages
    - H4: Iteration intensity increased
    - H5: Gap larger in coordination-heavy projects
    """
    results = {}

    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS RESULTS")
    print("="*70)

    # 1. Interrupted Time Series Analysis
    print("\n" + "-"*70)
    print("1. INTERRUPTED TIME SERIES ANALYSIS")
    print("   Testing: Did metrics change after AI adoption (Nov 2022)?")
    print("-"*70)

    its = InterruptedTimeSeriesAnalysis(pr_df)
    its_results = its.analyze_all_metrics()
    results['its'] = its_results

    print("\nTask-level metrics (H1: expecting significant improvement):")
    if 'review_response_latency' in its_results:
        r = its_results['review_response_latency']
        print(f"  Review-response latency: {r}")
    if 'ci_fix_latency' in its_results:
        r = its_results['ci_fix_latency']
        print(f"  CI-fix latency: {r}")

    print("\nJob-level metrics (H2: expecting smaller improvement):")
    if 'lead_time_hours' in its_results:
        r = its_results['lead_time_hours']
        print(f"  PR lead time: {r}")

    print("\nMechanism metrics (H4: expecting increase in iteration):")
    if 'num_commits' in its_results:
        r = its_results['num_commits']
        print(f"  Commits per PR: {r}")

    # 2. Difference-in-Differences Analysis
    print("\n" + "-"*70)
    print("2. DIFFERENCE-IN-DIFFERENCES ANALYSIS")
    print("   Testing: Are effects larger for high AI exposure languages?")
    print("-"*70)

    did = DifferenceInDifferencesAnalysis(pr_df)
    did_results = did.analyze_all_metrics()
    results['did'] = did_results

    print("\nDiD estimates (High vs Low exposure × Post treatment):")
    for metric, r in did_results.items():
        print(f"  {metric}: {r}")

    # 3. Heterogeneity Analysis
    print("\n" + "-"*70)
    print("3. HETEROGENEITY ANALYSIS")
    print("   Testing H5: Is the task-job gap larger in coordination-heavy projects?")
    print("-"*70)

    hetero = HeterogeneityAnalysis(pr_df)
    hetero_results = hetero.analyze_by_coordination()
    results['heterogeneity'] = hetero_results

    high = hetero_results['high_coordination']
    low = hetero_results['low_coordination']

    print(f"\nHigh coordination projects (>10 contributors):")
    print(f"  Task improvement: {high['task_change_pct']:.1f}%")
    print(f"  Job improvement:  {high['job_change_pct']:.1f}%")
    print(f"  Gap: {high['gap']:.1f} percentage points")

    print(f"\nLow coordination projects (≤10 contributors):")
    print(f"  Task improvement: {low['task_change_pct']:.1f}%")
    print(f"  Job improvement:  {low['job_change_pct']:.1f}%")
    print(f"  Gap: {low['gap']:.1f} percentage points")

    gap_diff = high['gap'] - low['gap']
    print(f"\nH5 Result: Gap difference = {gap_diff:.1f} pp")
    print(f"  {'SUPPORTED' if gap_diff > 0 else 'NOT SUPPORTED'}: "
          "Task-job gap is {'larger' if gap_diff > 0 else 'smaller'} in "
          "coordination-heavy projects")

    # Summary
    print("\n" + "="*70)
    print("HYPOTHESIS TESTING SUMMARY")
    print("="*70)

    # H1: Task metrics improved
    task_improved = (its_results.get('review_response_latency') and
                    its_results['review_response_latency'].coefficient < 0 and
                    its_results['review_response_latency'].is_significant())
    print(f"\nH1 (Task metrics improved): {'✓ SUPPORTED' if task_improved else '✗ NOT SUPPORTED'}")

    # H2: Job improved less than task
    if 'review_response_latency' in its_results and 'lead_time_hours' in its_results:
        task_effect = abs(its_results['review_response_latency'].coefficient)
        job_effect = abs(its_results['lead_time_hours'].coefficient)
        h2_supported = task_effect > job_effect
        print(f"H2 (Job improved less than task): {'✓ SUPPORTED' if h2_supported else '✗ NOT SUPPORTED'}")
        print(f"    Task effect magnitude: {task_effect:.2f}")
        print(f"    Job effect magnitude: {job_effect:.2f}")

    # H3: Effects in high exposure
    h3_supported = (did_results.get('review_response_latency') and
                   did_results['review_response_latency'].coefficient < 0 and
                   did_results['review_response_latency'].is_significant())
    print(f"H3 (Effects in high exposure): {'✓ SUPPORTED' if h3_supported else '✗ NOT SUPPORTED'}")

    # H4: Iteration increased
    h4_supported = (its_results.get('num_commits') and
                   its_results['num_commits'].coefficient > 0)
    print(f"H4 (Iteration increased): {'✓ SUPPORTED' if h4_supported else '✗ NOT SUPPORTED'}")

    # H5: Gap larger in coordination-heavy
    h5_supported = gap_diff > 5  # 5pp threshold
    print(f"H5 (Gap larger with coordination): {'✓ SUPPORTED' if h5_supported else '✗ NOT SUPPORTED'}")

    print("\n" + "="*70)

    return results
