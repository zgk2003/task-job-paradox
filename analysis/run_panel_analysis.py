#!/usr/bin/env python3
"""
Panel Analysis: ITS and DiD with Time Series Visualizations

Runs proper econometric analysis on the collected panel data and
generates publication-quality time series figures.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import date

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

# Configuration
DATA_DIR = Path(__file__).parent.parent / 'data' / 'panel'
RESULTS_DIR = Path(__file__).parent.parent / 'results' / 'panel_analysis'
TREATMENT_DATE = pd.Timestamp('2022-11-30')

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_panel_data():
    """Load all panel data into DataFrames."""
    print("Loading panel data...")

    # Load velocity
    velocity_files = sorted(DATA_DIR.glob('velocity/velocity_*.csv'))
    velocity_dfs = [pd.read_csv(f) for f in velocity_files]
    velocity = pd.concat(velocity_dfs, ignore_index=True)

    # Load complexity
    complexity_files = sorted(DATA_DIR.glob('complexity/complexity_*.csv'))
    complexity_dfs = [pd.read_csv(f) for f in complexity_files]
    complexity = pd.concat(complexity_dfs, ignore_index=True)

    # Load throughput (aggregated)
    throughput_files = sorted(DATA_DIR.glob('throughput/throughput_*.csv'))
    throughput_dfs = []
    for f in throughput_files:
        df = pd.read_csv(f)
        # Extract year_month from filename if not in data
        if 'year_month' not in df.columns:
            parts = f.stem.split('_')
            df['year_month'] = f"{parts[1]}-{parts[2]}"
        throughput_dfs.append(df)
    throughput = pd.concat(throughput_dfs, ignore_index=True)

    print(f"  Velocity: {len(velocity)} rows")
    print(f"  Complexity: {len(complexity)} rows")
    print(f"  Throughput: {len(throughput)} rows")

    return velocity, complexity, throughput


def prepare_time_series(df, value_col, exposure_filter=True):
    """Prepare data for time series analysis."""
    df = df.copy()

    # Filter to high exposure if requested
    if exposure_filter and 'high_exposure' in df.columns:
        df = df[df['high_exposure'] == True]

    # Parse date
    df['date'] = pd.to_datetime(df['year_month'] + '-01')
    df = df.sort_values('date')

    # Add time variables
    df['month_num'] = range(len(df))
    df['post'] = (df['date'] >= TREATMENT_DATE).astype(int)

    # Months since treatment (for ITS slope change)
    treatment_month = df[df['post'] == 1]['month_num'].min() if df['post'].sum() > 0 else len(df)
    df['months_since_treatment'] = np.maximum(0, df['month_num'] - treatment_month + 1) * df['post']

    return df


def run_its_analysis(df, outcome_col, name):
    """Run Interrupted Time Series regression."""
    print(f"\n{'='*60}")
    print(f"ITS Analysis: {name}")
    print('='*60)

    try:
        import statsmodels.api as sm

        # Prepare data
        ts = prepare_time_series(df, outcome_col)
        ts = ts.dropna(subset=[outcome_col])

        if len(ts) < 10:
            print("Insufficient data for ITS")
            return None

        # Build regression: Y = β0 + β1*time + β2*post + β3*time_since_treatment
        X = ts[['month_num', 'post', 'months_since_treatment']].copy()
        X = sm.add_constant(X)
        y = ts[outcome_col]

        model = sm.OLS(y, X).fit(cov_type='HC3')  # Robust SEs

        print(f"\nModel: {outcome_col} ~ time + post + time_since_treatment")
        print(f"N observations: {len(ts)}")
        print(f"R-squared: {model.rsquared:.4f}")
        print(f"\nCoefficients:")
        print(f"  Intercept (baseline):     {model.params['const']:.4f}")
        print(f"  Pre-trend (monthly):      {model.params['month_num']:.4f}")
        print(f"  Level change (at treat):  {model.params['post']:.4f} (p={model.pvalues['post']:.4f})")
        print(f"  Slope change (post):      {model.params['months_since_treatment']:.4f} (p={model.pvalues['months_since_treatment']:.4f})")

        post_trend = model.params['month_num'] + model.params['months_since_treatment']
        print(f"  Post-treatment trend:     {post_trend:.4f}/month")

        return {
            'outcome': outcome_col,
            'name': name,
            'n': len(ts),
            'r_squared': model.rsquared,
            'intercept': model.params['const'],
            'pre_trend': model.params['month_num'],
            'level_change': model.params['post'],
            'level_pvalue': model.pvalues['post'],
            'slope_change': model.params['months_since_treatment'],
            'slope_pvalue': model.pvalues['months_since_treatment'],
            'post_trend': post_trend,
            'model': model,
            'data': ts
        }

    except ImportError:
        print("statsmodels not available, using simple before/after comparison")
        ts = prepare_time_series(df, outcome_col)
        pre = ts[ts['post'] == 0][outcome_col]
        post = ts[ts['post'] == 1][outcome_col]

        print(f"Pre-treatment mean: {pre.mean():.4f}")
        print(f"Post-treatment mean: {post.mean():.4f}")
        print(f"Change: {post.mean() - pre.mean():.4f} ({((post.mean() - pre.mean()) / pre.mean()) * 100:.1f}%)")

        return {
            'outcome': outcome_col,
            'name': name,
            'pre_mean': pre.mean(),
            'post_mean': post.mean(),
            'change': post.mean() - pre.mean(),
            'change_pct': ((post.mean() - pre.mean()) / pre.mean()) * 100,
            'data': ts
        }


def run_did_analysis(df, outcome_col, name):
    """Run Difference-in-Differences regression."""
    print(f"\n{'='*60}")
    print(f"DiD Analysis: {name}")
    print('='*60)

    try:
        import statsmodels.formula.api as smf

        df = df.copy()
        df['date'] = pd.to_datetime(df['year_month'] + '-01')
        df['post'] = (df['date'] >= TREATMENT_DATE).astype(int)
        df['high_exposure'] = df['high_exposure'].astype(int)
        df['treated_x_post'] = df['high_exposure'] * df['post']

        df = df.dropna(subset=[outcome_col])

        # DiD regression
        formula = f"{outcome_col} ~ high_exposure + post + treated_x_post"
        model = smf.ols(formula, data=df).fit(cov_type='HC3')

        print(f"\nModel: {outcome_col} ~ high_exposure + post + high_exposure*post")
        print(f"N observations: {len(df)}")
        print(f"R-squared: {model.rsquared:.4f}")
        print(f"\nCoefficients:")
        print(f"  Intercept (Low, Pre):     {model.params['Intercept']:.4f}")
        print(f"  High Exposure effect:     {model.params['high_exposure']:.4f}")
        print(f"  Post-treatment effect:    {model.params['post']:.4f}")
        print(f"  DiD (Treatment Effect):   {model.params['treated_x_post']:.4f} (p={model.pvalues['treated_x_post']:.4f})")

        # Calculate means for interpretation
        high_pre = df[(df['high_exposure']==1) & (df['post']==0)][outcome_col].mean()
        high_post = df[(df['high_exposure']==1) & (df['post']==1)][outcome_col].mean()
        low_pre = df[(df['high_exposure']==0) & (df['post']==0)][outcome_col].mean()
        low_post = df[(df['high_exposure']==0) & (df['post']==1)][outcome_col].mean()

        print(f"\nGroup Means:")
        print(f"  High Exposure Pre:  {high_pre:.2f}")
        print(f"  High Exposure Post: {high_post:.2f} (change: {high_post - high_pre:.2f})")
        print(f"  Low Exposure Pre:   {low_pre:.2f}")
        print(f"  Low Exposure Post:  {low_post:.2f} (change: {low_post - low_pre:.2f})")
        print(f"  DiD Estimate:       {(high_post - high_pre) - (low_post - low_pre):.2f}")

        return {
            'outcome': outcome_col,
            'name': name,
            'n': len(df),
            'r_squared': model.rsquared,
            'did_effect': model.params['treated_x_post'],
            'did_se': model.bse['treated_x_post'],
            'did_pvalue': model.pvalues['treated_x_post'],
            'high_pre': high_pre,
            'high_post': high_post,
            'low_pre': low_pre,
            'low_post': low_post,
            'model': model,
            'data': df
        }

    except ImportError:
        print("statsmodels not available")
        return None


def create_time_series_plot(df, value_col, title, ylabel, filename,
                            show_treatment=True, log_scale=False):
    """Create a time series plot with treatment line."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data for both exposure groups
    for exposure, color, label in [(True, '#2E86AB', 'High AI Exposure'),
                                    (False, '#A23B72', 'Low AI Exposure')]:
        if 'high_exposure' in df.columns:
            data = df[df['high_exposure'] == exposure].copy()
        else:
            data = df.copy()

        if data.empty:
            continue

        data['date'] = pd.to_datetime(data['year_month'] + '-01')
        data = data.sort_values('date')

        ax.plot(data['date'], data[value_col], 'o-', color=color,
                label=label, linewidth=2, markersize=4)

    # Add treatment line
    if show_treatment:
        ax.axvline(x=TREATMENT_DATE, color='red', linestyle='--',
                   linewidth=2, label='ChatGPT Launch')

        # Shade pre/post regions
        ax.axvspan(ax.get_xlim()[0], mdates.date2num(TREATMENT_DATE),
                   alpha=0.1, color='blue', label='Pre-treatment')
        ax.axvspan(mdates.date2num(TREATMENT_DATE), ax.get_xlim()[1],
                   alpha=0.1, color='green', label='Post-treatment')

    if log_scale:
        ax.set_yscale('log')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')

    # Format x-axis
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def create_its_visualization(its_result, filename):
    """Create ITS plot with fitted regression lines."""
    if its_result is None or 'model' not in its_result:
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))

    data = its_result['data']
    outcome = its_result['outcome']
    model = its_result['model']

    # Plot actual data
    ax.scatter(data['date'], data[outcome], color='#2E86AB', s=50,
               alpha=0.7, label='Observed', zorder=5)

    # Generate fitted values
    X_pred = data[['month_num', 'post', 'months_since_treatment']].copy()
    import statsmodels.api as sm
    X_pred = sm.add_constant(X_pred)
    data['fitted'] = model.predict(X_pred)

    # Plot pre-treatment trend
    pre_data = data[data['post'] == 0]
    if not pre_data.empty:
        ax.plot(pre_data['date'], pre_data['fitted'], '-', color='#F18F01',
                linewidth=3, label='Pre-treatment trend')

    # Plot post-treatment trend
    post_data = data[data['post'] == 1]
    if not post_data.empty:
        ax.plot(post_data['date'], post_data['fitted'], '-', color='#C73E1D',
                linewidth=3, label='Post-treatment trend')

    # Treatment line
    ax.axvline(x=TREATMENT_DATE, color='black', linestyle='--',
               linewidth=2, label='ChatGPT Launch')

    # Annotations
    level_change = its_result['level_change']
    slope_change = its_result['slope_change']
    level_p = its_result['level_pvalue']
    slope_p = its_result['slope_pvalue']

    annotation = f"Level change: {level_change:.2f} (p={level_p:.3f})\n"
    annotation += f"Slope change: {slope_change:.3f}/mo (p={slope_p:.3f})"

    ax.annotate(annotation, xy=(0.02, 0.98), xycoords='axes fraction',
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(its_result['name'], fontsize=12)
    ax.set_title(f"Interrupted Time Series: {its_result['name']}",
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def create_did_visualization(did_result, filename):
    """Create DiD visualization showing parallel trends."""
    if did_result is None:
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bar positions
    x = np.array([0, 1, 3, 4])
    width = 0.8

    values = [did_result['high_pre'], did_result['high_post'],
              did_result['low_pre'], did_result['low_post']]
    colors = ['#2E86AB', '#1a5276', '#A23B72', '#6c1f4a']
    labels = ['High (Pre)', 'High (Post)', 'Low (Pre)', 'Low (Post)']

    bars = ax.bar(x, values, width, color=colors)

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11)

    # Add change arrows
    high_change = did_result['high_post'] - did_result['high_pre']
    low_change = did_result['low_post'] - did_result['low_pre']

    ax.annotate('', xy=(1, did_result['high_post']),
                xytext=(0, did_result['high_pre']),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.annotate(f'{high_change:+.1f}', xy=(0.5, (did_result['high_pre'] + did_result['high_post'])/2),
                ha='center', fontsize=10, color='green')

    ax.annotate('', xy=(4, did_result['low_post']),
                xytext=(3, did_result['low_pre']),
                arrowprops=dict(arrowstyle='->', color='purple', lw=2))
    ax.annotate(f'{low_change:+.1f}', xy=(3.5, (did_result['low_pre'] + did_result['low_post'])/2),
                ha='center', fontsize=10, color='purple')

    # DiD annotation
    did_effect = did_result['did_effect']
    did_p = did_result['did_pvalue']
    sig = '***' if did_p < 0.01 else '**' if did_p < 0.05 else '*' if did_p < 0.1 else ''

    ax.annotate(f"DiD = {did_effect:.2f}{sig}\n(p = {did_p:.4f})",
                xy=(2, max(values) * 0.9),
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    ax.set_xticks([0.5, 3.5])
    ax.set_xticklabels(['High AI Exposure\n(Python, JS, etc.)',
                        'Low AI Exposure\n(Fortran, COBOL, etc.)'])
    ax.set_ylabel(did_result['name'], fontsize=12)
    ax.set_title(f"Difference-in-Differences: {did_result['name']}",
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def create_summary_figure(its_results, did_results):
    """Create comprehensive summary figure."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(16, 12))

    # Panel A: Velocity time series
    ax1 = fig.add_subplot(2, 2, 1)
    if 'velocity' in its_results and its_results['velocity'] is not None:
        data = its_results['velocity']['data']
        ax1.plot(data['date'], data['median_lead_time_hours'], 'o-',
                 color='#2E86AB', linewidth=2, markersize=4)
        ax1.axvline(x=TREATMENT_DATE, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Median Lead Time (hours)')
        ax1.set_title('A. Velocity: PR Lead Time', fontweight='bold')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Panel B: Complexity time series
    ax2 = fig.add_subplot(2, 2, 2)
    if 'complexity' in its_results and its_results['complexity'] is not None:
        data = its_results['complexity']['data']
        ax2.plot(data['date'], data['median_churn'], 'o-',
                 color='#F18F01', linewidth=2, markersize=4)
        ax2.axvline(x=TREATMENT_DATE, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Median Lines Changed')
        ax2.set_title('B. Complexity: PR Size (Scope Expansion)', fontweight='bold')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Panel C: ITS Summary
    ax3 = fig.add_subplot(2, 2, 3)
    metrics = []
    level_changes = []
    pvalues = []

    for name, result in its_results.items():
        if result is not None and 'level_change' in result:
            metrics.append(result['name'])
            level_changes.append(result['level_change'])
            pvalues.append(result['level_pvalue'])

    if metrics:
        colors = ['green' if lc < 0 else 'red' for lc in level_changes]
        bars = ax3.barh(range(len(metrics)), level_changes, color=colors, alpha=0.7)
        ax3.set_yticks(range(len(metrics)))
        ax3.set_yticklabels(metrics)
        ax3.axvline(x=0, color='black', linewidth=1)
        ax3.set_xlabel('Level Change at Treatment')
        ax3.set_title('C. ITS: Immediate Effects', fontweight='bold')

        # Add significance stars
        for i, (bar, p) in enumerate(zip(bars, pvalues)):
            sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
            ax3.annotate(sig, xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                        xytext=(5, 0), textcoords='offset points',
                        ha='left', va='center', fontsize=12)

    # Panel D: DiD Summary
    ax4 = fig.add_subplot(2, 2, 4)
    metrics = []
    did_effects = []
    pvalues = []

    for name, result in did_results.items():
        if result is not None and 'did_effect' in result:
            metrics.append(result['name'])
            did_effects.append(result['did_effect'])
            pvalues.append(result['did_pvalue'])

    if metrics:
        colors = ['green' if de < 0 else 'orange' for de in did_effects]
        bars = ax4.barh(range(len(metrics)), did_effects, color=colors, alpha=0.7)
        ax4.set_yticks(range(len(metrics)))
        ax4.set_yticklabels(metrics)
        ax4.axvline(x=0, color='black', linewidth=1)
        ax4.set_xlabel('DiD Effect (High - Low Exposure)')
        ax4.set_title('D. DiD: Treatment Effects', fontweight='bold')

        for i, (bar, p) in enumerate(zip(bars, pvalues)):
            sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
            ax4.annotate(sig, xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                        xytext=(5, 0), textcoords='offset points',
                        ha='left', va='center', fontsize=12)

    fig.suptitle('Task-Job Paradox: Panel Analysis Results',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'summary_panel_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: summary_panel_analysis.png")


def main():
    print("=" * 70)
    print("TASK-JOB PARADOX: PANEL REGRESSION ANALYSIS")
    print("=" * 70)
    print(f"\nTreatment Date: {TREATMENT_DATE.strftime('%Y-%m-%d')} (ChatGPT Launch)")
    print(f"Results directory: {RESULTS_DIR}")

    # Load data
    velocity, complexity, throughput = load_panel_data()

    # Store results
    its_results = {}
    did_results = {}

    # ==========================================================================
    # ITS ANALYSES
    # ==========================================================================
    print("\n" + "=" * 70)
    print("INTERRUPTED TIME SERIES ANALYSES")
    print("=" * 70)

    # Velocity ITS
    its_results['velocity'] = run_its_analysis(
        velocity[velocity['high_exposure'] == True],
        'median_lead_time_hours',
        'Median Lead Time (hours)'
    )

    # Velocity P75 ITS
    its_results['velocity_p75'] = run_its_analysis(
        velocity[velocity['high_exposure'] == True],
        'p75_lead_time_hours',
        'P75 Lead Time (hours)'
    )

    # Complexity ITS
    its_results['complexity'] = run_its_analysis(
        complexity[complexity['high_exposure'] == True],
        'median_churn',
        'Median Lines Changed'
    )

    # ==========================================================================
    # DiD ANALYSES
    # ==========================================================================
    print("\n" + "=" * 70)
    print("DIFFERENCE-IN-DIFFERENCES ANALYSES")
    print("=" * 70)

    # Velocity DiD
    did_results['velocity'] = run_did_analysis(
        velocity,
        'median_lead_time_hours',
        'Median Lead Time (hours)'
    )

    # Velocity P75 DiD
    did_results['velocity_p75'] = run_did_analysis(
        velocity,
        'p75_lead_time_hours',
        'P75 Lead Time (hours)'
    )

    # Complexity DiD
    did_results['complexity'] = run_did_analysis(
        complexity,
        'median_churn',
        'Median Lines Changed'
    )

    # ==========================================================================
    # VISUALIZATIONS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    # Time series plots
    print("\nTime Series Plots:")
    create_time_series_plot(velocity, 'median_lead_time_hours',
                            'PR Velocity Over Time (Median Lead Time)',
                            'Median Lead Time (hours)',
                            'ts_velocity_median.png')

    create_time_series_plot(velocity, 'p75_lead_time_hours',
                            'PR Velocity Over Time (P75 Lead Time)',
                            'P75 Lead Time (hours)',
                            'ts_velocity_p75.png')

    create_time_series_plot(complexity, 'median_churn',
                            'PR Complexity Over Time (Scope Expansion)',
                            'Median Lines Changed',
                            'ts_complexity.png')

    create_time_series_plot(complexity, 'median_files',
                            'PR Scope Over Time (Files Changed)',
                            'Median Files Changed',
                            'ts_files.png')

    # ITS plots
    print("\nITS Plots:")
    create_its_visualization(its_results['velocity'], 'its_velocity.png')
    create_its_visualization(its_results['velocity_p75'], 'its_velocity_p75.png')
    create_its_visualization(its_results['complexity'], 'its_complexity.png')

    # DiD plots
    print("\nDiD Plots:")
    create_did_visualization(did_results['velocity'], 'did_velocity.png')
    create_did_visualization(did_results['complexity'], 'did_complexity.png')

    # Summary figure
    print("\nSummary Figure:")
    create_summary_figure(its_results, did_results)

    # ==========================================================================
    # SUMMARY REPORT
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY OF FINDINGS")
    print("=" * 70)

    print("\n--- ITS Results (High Exposure Languages) ---")
    for name, result in its_results.items():
        if result is not None and 'level_change' in result:
            sig = '***' if result['level_pvalue'] < 0.01 else '**' if result['level_pvalue'] < 0.05 else '*' if result['level_pvalue'] < 0.1 else ''
            print(f"\n{result['name']}:")
            print(f"  Level change: {result['level_change']:.2f} {sig}")
            print(f"  Pre-trend: {result['pre_trend']:.4f}/month")
            print(f"  Post-trend: {result['post_trend']:.4f}/month")

    print("\n--- DiD Results (High vs Low Exposure) ---")
    for name, result in did_results.items():
        if result is not None and 'did_effect' in result:
            sig = '***' if result['did_pvalue'] < 0.01 else '**' if result['did_pvalue'] < 0.05 else '*' if result['did_pvalue'] < 0.1 else ''
            print(f"\n{result['name']}:")
            print(f"  DiD Effect: {result['did_effect']:.2f} {sig}")
            print(f"  High Exposure: {result['high_pre']:.1f} → {result['high_post']:.1f}")
            print(f"  Low Exposure: {result['low_pre']:.1f} → {result['low_post']:.1f}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nAll visualizations saved to: {RESULTS_DIR}")

    return its_results, did_results


if __name__ == '__main__':
    main()
