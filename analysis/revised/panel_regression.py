"""
Panel Regression Analysis for Task-Job Paradox

Implements proper econometric methods:
1. Interrupted Time Series (ITS) with segmented regression
2. Difference-in-Differences (DiD) with panel data
3. Event study design for parallel trends
4. Heterogeneity analysis with interaction terms

Addresses reviewer concerns about simple pre/post comparison.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import date

# Optional statistical imports
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.regression.linear_model import OLS
    from statsmodels.stats.sandwich_covariance import cov_hac
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

from .config import TREATMENT_DATE, HIGH_EXPOSURE_LANGUAGES


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ITSResult:
    """Result from Interrupted Time Series regression."""
    outcome: str
    # Pre-treatment
    intercept: float
    pre_trend: float  # Monthly slope before treatment
    # Treatment effects
    level_change: float  # Immediate jump at treatment
    slope_change: float  # Change in trend after treatment
    # Post-treatment implied
    post_trend: float  # = pre_trend + slope_change
    # Inference
    level_se: float
    slope_se: float
    level_pvalue: float
    slope_pvalue: float
    # Model fit
    r_squared: float
    n_observations: int
    # Robust SEs
    robust: bool


@dataclass
class DiDResult:
    """Result from Difference-in-Differences regression."""
    outcome: str
    # Coefficients
    treated_effect: float  # High exposure baseline difference
    post_effect: float  # Time effect (common trend)
    did_effect: float  # Treatment effect (the key estimate!)
    # Inference
    did_se: float
    did_pvalue: float
    did_ci_lower: float
    did_ci_upper: float
    # Model fit
    r_squared: float
    n_observations: int
    # Controls included
    controls: List[str]


@dataclass
class EventStudyResult:
    """Result from event study (leads and lags)."""
    outcome: str
    # Coefficients by period relative to treatment
    period_effects: Dict[int, float]  # period -> coefficient
    period_ses: Dict[int, float]  # period -> SE
    period_pvalues: Dict[int, float]
    # Test for parallel trends
    pre_trend_fstat: float
    pre_trend_pvalue: float
    parallel_trends_pass: bool


# =============================================================================
# DATA PREPARATION
# =============================================================================

def load_panel_data(data_dir: Path, metric: str) -> pd.DataFrame:
    """
    Load all monthly data for a metric into a panel DataFrame.

    Returns DataFrame with columns:
    - year_month: YYYY-MM string
    - month_num: Integer months since start (for trend)
    - post: 1 if after treatment, 0 otherwise
    - high_exposure: 1 for treated group, 0 for control
    - [metric columns]
    """
    panel_dir = data_dir / 'panel' / metric
    if not panel_dir.exists():
        raise FileNotFoundError(f"No panel data found at {panel_dir}")

    dfs = []
    for csv_file in sorted(panel_dir.glob(f"{metric}_*.csv")):
        df = pd.read_csv(csv_file)
        dfs.append(df)

    if not dfs:
        raise ValueError(f"No data files found for {metric}")

    panel = pd.concat(dfs, ignore_index=True)

    # Add time variables
    panel['year'] = panel['year_month'].str[:4].astype(int)
    panel['month'] = panel['year_month'].str[5:7].astype(int)
    panel['date'] = pd.to_datetime(panel['year_month'] + '-01')

    # Month number (for trend)
    min_date = panel['date'].min()
    panel['month_num'] = ((panel['date'].dt.year - min_date.year) * 12 +
                           panel['date'].dt.month - min_date.month)

    # Post-treatment indicator
    treatment_date = pd.Timestamp(TREATMENT_DATE)
    panel['post'] = (panel['date'] >= treatment_date).astype(int)

    # Months since treatment (for ITS)
    panel['months_since_treatment'] = np.where(
        panel['post'] == 1,
        panel['month_num'] - panel[panel['post'] == 1]['month_num'].min() + 1,
        0
    )

    # High exposure indicator
    panel['high_exposure'] = panel['high_exposure'].astype(int)

    return panel


def prepare_aggregated_panel(
    data_dir: Path,
    metrics: List[str] = ['velocity', 'throughput', 'complexity']
) -> pd.DataFrame:
    """
    Load and merge multiple metrics into single panel.

    For regression, we typically want monthly aggregates.
    """
    panels = {}
    for metric in metrics:
        try:
            df = load_panel_data(data_dir, metric)
            # Keep only aggregate rows (not developer-level)
            if 'author' not in df.columns:
                panels[metric] = df
        except Exception as e:
            print(f"Warning: Could not load {metric}: {e}")

    if not panels:
        raise ValueError("No metrics could be loaded")

    # Merge on year_month and high_exposure
    base = list(panels.values())[0][['year_month', 'high_exposure', 'month_num', 'post', 'date']]

    for metric, df in panels.items():
        # Prefix columns with metric name
        metric_cols = [c for c in df.columns
                       if c not in ['year_month', 'high_exposure', 'month_num', 'post', 'date', 'year', 'month', 'months_since_treatment']]
        df_renamed = df[['year_month', 'high_exposure'] + metric_cols].copy()
        df_renamed.columns = ['year_month', 'high_exposure'] + [f"{metric}_{c}" for c in metric_cols]
        base = base.merge(df_renamed, on=['year_month', 'high_exposure'], how='left')

    return base


# =============================================================================
# INTERRUPTED TIME SERIES
# =============================================================================

class InterruptedTimeSeriesRegression:
    """
    Proper ITS with segmented regression.

    Model:
    Y_t = β₀ + β₁·time + β₂·post + β₃·time_since_treatment + ε_t

    Where:
    - β₀: Intercept (baseline level)
    - β₁: Pre-treatment trend
    - β₂: Level change at treatment (immediate effect)
    - β₃: Slope change (change in trend)
    """

    def __init__(self, robust_se: bool = True):
        self.robust_se = robust_se

    def fit(
        self,
        data: pd.DataFrame,
        outcome: str,
        time_var: str = 'month_num',
        post_var: str = 'post',
        time_since_var: str = 'months_since_treatment',
        subset: Optional[str] = None
    ) -> ITSResult:
        """
        Fit ITS model.

        Args:
            data: Panel DataFrame
            outcome: Column name for outcome variable
            time_var: Column for time trend
            post_var: Column for post-treatment indicator
            time_since_var: Column for time since treatment
            subset: Optional filter (e.g., "high_exposure == 1")
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for regression")

        df = data.copy()
        if subset:
            df = df.query(subset)

        df = df.dropna(subset=[outcome, time_var, post_var])

        if len(df) < 10:
            raise ValueError(f"Insufficient observations: {len(df)}")

        # Build regression
        X = df[[time_var, post_var, time_since_var]].copy()
        X = sm.add_constant(X)
        y = df[outcome]

        model = OLS(y, X).fit()

        # Robust standard errors (HAC for autocorrelation)
        if self.robust_se:
            robust_cov = cov_hac(model)
            model = model.get_robustcov_results(cov_type='HAC', maxlags=3)

        return ITSResult(
            outcome=outcome,
            intercept=model.params['const'],
            pre_trend=model.params[time_var],
            level_change=model.params[post_var],
            slope_change=model.params[time_since_var],
            post_trend=model.params[time_var] + model.params[time_since_var],
            level_se=model.bse[post_var],
            slope_se=model.bse[time_since_var],
            level_pvalue=model.pvalues[post_var],
            slope_pvalue=model.pvalues[time_since_var],
            r_squared=model.rsquared,
            n_observations=len(df),
            robust=self.robust_se
        )

    def fit_by_group(
        self,
        data: pd.DataFrame,
        outcome: str,
        group_var: str = 'high_exposure'
    ) -> Dict[str, ITSResult]:
        """Fit separate ITS for each group."""
        results = {}
        for group_val in data[group_var].unique():
            subset = f"{group_var} == {group_val}"
            try:
                results[str(group_val)] = self.fit(data, outcome, subset=subset)
            except Exception as e:
                print(f"Warning: Could not fit ITS for {group_var}={group_val}: {e}")
        return results


# =============================================================================
# DIFFERENCE-IN-DIFFERENCES
# =============================================================================

class DiDRegression:
    """
    Difference-in-Differences with panel data.

    Model:
    Y_it = β₀ + β₁·Treated_i + β₂·Post_t + β₃·(Treated_i × Post_t) + γ·X_it + ε_it

    Where:
    - β₃ is the DiD estimate (causal effect of treatment)
    - X_it are optional controls
    """

    def __init__(self, cluster_se: bool = True):
        self.cluster_se = cluster_se

    def fit(
        self,
        data: pd.DataFrame,
        outcome: str,
        treated_var: str = 'high_exposure',
        post_var: str = 'post',
        controls: Optional[List[str]] = None,
        cluster_var: Optional[str] = None
    ) -> DiDResult:
        """
        Fit DiD model.

        Args:
            data: Panel DataFrame
            outcome: Outcome variable
            treated_var: Treatment group indicator
            post_var: Post-treatment indicator
            controls: List of control variables
            cluster_var: Variable to cluster SEs on
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for regression")

        df = data.copy()
        df = df.dropna(subset=[outcome, treated_var, post_var])

        # Create interaction
        df['treated_x_post'] = df[treated_var] * df[post_var]

        # Build formula
        formula = f"{outcome} ~ {treated_var} + {post_var} + treated_x_post"
        if controls:
            formula += " + " + " + ".join(controls)

        # Fit model
        model = smf.ols(formula, data=df).fit()

        # Clustered standard errors if requested
        if self.cluster_se and cluster_var and cluster_var in df.columns:
            model = model.get_robustcov_results(
                cov_type='cluster',
                groups=df[cluster_var]
            )
        elif self.cluster_se:
            model = model.get_robustcov_results(cov_type='HC3')

        # Extract DiD coefficient
        did_coef = model.params['treated_x_post']
        did_se = model.bse['treated_x_post']
        did_pval = model.pvalues['treated_x_post']

        # 95% CI
        ci = model.conf_int().loc['treated_x_post']

        return DiDResult(
            outcome=outcome,
            treated_effect=model.params[treated_var],
            post_effect=model.params[post_var],
            did_effect=did_coef,
            did_se=did_se,
            did_pvalue=did_pval,
            did_ci_lower=ci[0],
            did_ci_upper=ci[1],
            r_squared=model.rsquared,
            n_observations=len(df),
            controls=controls or []
        )

    def fit_with_interactions(
        self,
        data: pd.DataFrame,
        outcome: str,
        interaction_vars: List[str],
        **kwargs
    ) -> Dict[str, DiDResult]:
        """
        Fit DiD with additional interaction terms.

        Tests heterogeneous treatment effects by interacting treatment
        with other variables.
        """
        results = {}

        # Base model
        results['base'] = self.fit(data, outcome, **kwargs)

        # Models with interactions
        for var in interaction_vars:
            if var not in data.columns:
                continue

            df = data.copy()
            df[f'{var}_x_treated_x_post'] = (
                df['high_exposure'] * df['post'] * df[var]
            )

            formula = (f"{outcome} ~ high_exposure + post + treated_x_post + "
                      f"{var} + {var}_x_treated_x_post")

            try:
                model = smf.ols(formula, data=df).fit()
                results[f'interaction_{var}'] = model.params.to_dict()
            except Exception as e:
                print(f"Warning: Could not fit interaction with {var}: {e}")

        return results


# =============================================================================
# EVENT STUDY
# =============================================================================

class EventStudy:
    """
    Event study design to test parallel trends assumption.

    Creates leads and lags around treatment to:
    1. Test pre-treatment parallel trends (leads should be ~0)
    2. Trace out dynamic treatment effects (lags)
    """

    def __init__(self, pre_periods: int = 6, post_periods: int = 12):
        self.pre_periods = pre_periods
        self.post_periods = post_periods

    def fit(
        self,
        data: pd.DataFrame,
        outcome: str,
        treated_var: str = 'high_exposure',
        time_var: str = 'month_num',
        treatment_time: Optional[int] = None
    ) -> EventStudyResult:
        """
        Fit event study model.

        Args:
            data: Panel DataFrame
            outcome: Outcome variable
            treated_var: Treatment indicator
            time_var: Time variable
            treatment_time: Month number of treatment (auto-detected if None)
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for regression")

        df = data.copy()

        # Auto-detect treatment time
        if treatment_time is None:
            treatment_time = df[df['post'] == 1][time_var].min()

        # Create relative time
        df['rel_time'] = df[time_var] - treatment_time

        # Create period dummies (excluding -1 as reference)
        period_range = range(-self.pre_periods, self.post_periods + 1)
        for p in period_range:
            if p != -1:  # Reference period
                df[f'period_{p}'] = ((df['rel_time'] == p) & (df[treated_var] == 1)).astype(int)

        # Build formula
        period_vars = [f'period_{p}' for p in period_range if p != -1]
        formula = f"{outcome} ~ {treated_var} + " + " + ".join(period_vars)

        model = smf.ols(formula, data=df).fit(cov_type='HC3')

        # Extract period effects
        period_effects = {}
        period_ses = {}
        period_pvalues = {}

        for p in period_range:
            if p == -1:
                period_effects[p] = 0  # Reference
                period_ses[p] = 0
                period_pvalues[p] = 1
            else:
                var = f'period_{p}'
                if var in model.params:
                    period_effects[p] = model.params[var]
                    period_ses[p] = model.bse[var]
                    period_pvalues[p] = model.pvalues[var]

        # Test pre-trends (joint F-test on pre-treatment periods)
        pre_period_vars = [f'period_{p}' for p in period_range if p < -1 and f'period_{p}' in model.params]
        if pre_period_vars:
            f_test = model.f_test(' = '.join([f'{v} = 0' for v in pre_period_vars]))
            pre_trend_fstat = f_test.fvalue
            pre_trend_pvalue = f_test.pvalue
        else:
            pre_trend_fstat = 0
            pre_trend_pvalue = 1

        return EventStudyResult(
            outcome=outcome,
            period_effects=period_effects,
            period_ses=period_ses,
            period_pvalues=period_pvalues,
            pre_trend_fstat=pre_trend_fstat,
            pre_trend_pvalue=pre_trend_pvalue,
            parallel_trends_pass=pre_trend_pvalue > 0.05
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_full_panel_analysis(
    data_dir: Path,
    outcomes: List[str] = ['velocity_median_lead_time_hours', 'complexity_median_churn']
) -> Dict[str, Any]:
    """
    Run complete panel analysis suite.

    Returns dict with:
    - its: ITS results by outcome and group
    - did: DiD results by outcome
    - event_study: Event study results
    - heterogeneity: Interaction effects
    """
    results = {
        'its': {},
        'did': {},
        'event_study': {},
        'heterogeneity': {}
    }

    # Load data
    try:
        panel = prepare_aggregated_panel(data_dir)
    except Exception as e:
        print(f"Error loading panel data: {e}")
        return results

    # ITS
    its = InterruptedTimeSeriesRegression()
    for outcome in outcomes:
        if outcome in panel.columns:
            results['its'][outcome] = its.fit_by_group(panel, outcome)

    # DiD
    did = DiDRegression()
    for outcome in outcomes:
        if outcome in panel.columns:
            results['did'][outcome] = did.fit(panel, outcome)

    # Event Study
    es = EventStudy()
    for outcome in outcomes:
        if outcome in panel.columns:
            try:
                results['event_study'][outcome] = es.fit(panel, outcome)
            except Exception as e:
                print(f"Could not run event study for {outcome}: {e}")

    return results


def print_regression_summary(results: Dict[str, Any]):
    """Print formatted summary of regression results."""

    print("\n" + "=" * 70)
    print("PANEL REGRESSION RESULTS")
    print("=" * 70)

    # ITS Results
    if results.get('its'):
        print("\n" + "-" * 70)
        print("INTERRUPTED TIME SERIES")
        print("-" * 70)
        for outcome, group_results in results['its'].items():
            print(f"\nOutcome: {outcome}")
            for group, its_result in group_results.items():
                print(f"\n  Group: {'High Exposure' if group == '1' else 'Low Exposure'}")
                print(f"    Pre-trend: {its_result.pre_trend:.4f}/month")
                print(f"    Level change: {its_result.level_change:.4f} (p={its_result.level_pvalue:.4f})")
                print(f"    Slope change: {its_result.slope_change:.4f} (p={its_result.slope_pvalue:.4f})")
                print(f"    Post-trend: {its_result.post_trend:.4f}/month")

    # DiD Results
    if results.get('did'):
        print("\n" + "-" * 70)
        print("DIFFERENCE-IN-DIFFERENCES")
        print("-" * 70)
        for outcome, did_result in results['did'].items():
            print(f"\nOutcome: {outcome}")
            print(f"  DiD Effect: {did_result.did_effect:.4f}")
            print(f"  SE: {did_result.did_se:.4f}")
            print(f"  p-value: {did_result.did_pvalue:.4f}")
            print(f"  95% CI: [{did_result.did_ci_lower:.4f}, {did_result.did_ci_upper:.4f}]")
            sig = "***" if did_result.did_pvalue < 0.01 else "**" if did_result.did_pvalue < 0.05 else "*" if did_result.did_pvalue < 0.1 else ""
            print(f"  Significance: {sig}")

    # Event Study
    if results.get('event_study'):
        print("\n" + "-" * 70)
        print("EVENT STUDY (Parallel Trends Test)")
        print("-" * 70)
        for outcome, es_result in results['event_study'].items():
            print(f"\nOutcome: {outcome}")
            print(f"  Pre-trend F-stat: {es_result.pre_trend_fstat:.4f}")
            print(f"  Pre-trend p-value: {es_result.pre_trend_pvalue:.4f}")
            status = "PASS" if es_result.parallel_trends_pass else "FAIL"
            print(f"  Parallel trends: {status}")
