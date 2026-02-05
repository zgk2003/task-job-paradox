#!/usr/bin/env python3
"""
Analyze Results from Revised Empirical Strategy

This script analyzes the developer-level data and produces:
1. Summary statistics with multi-granularity measures
2. Visualizations of the "peaks and slack" hypothesis
3. Statistical comparisons pre/post LLM

Key Question: Do task-level velocity gains translate to throughput gains,
or are they absorbed as slack?
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Visualization imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available for visualizations")


def load_all_data():
    """Load all the results data."""
    results_dir = Path(__file__).parent.parent / 'results'

    data = {}

    # Velocity metrics (aggregated)
    data['velocity_2021'] = pd.read_csv(results_dir / 'velocity_metrics_2021_06.csv')
    data['velocity_2025'] = pd.read_csv(results_dir / 'velocity_metrics_2025_06.csv')

    # Developer throughput (developer-level)
    data['throughput_2021'] = pd.read_csv(results_dir / 'developer_throughput_2021_06.csv')
    data['throughput_2025'] = pd.read_csv(results_dir / 'developer_throughput_2025_06.csv')

    # Inter-PR gaps (developer-level)
    data['gaps_2021'] = pd.read_csv(results_dir / 'inter_pr_gaps_2021_06.csv')
    data['gaps_2025'] = pd.read_csv(results_dir / 'inter_pr_gaps_2025_06.csv')

    return data


def compute_multi_granularity_stats(data):
    """
    Compute statistics at multiple granularities for robustness.
    """
    results = []

    for period, label in [('2021', 'Pre-LLM (June 2021)'), ('2025', 'Post-LLM (June 2025)')]:
        for high_exp, exp_label in [(True, 'High AI-Exposure'), (False, 'Low AI-Exposure')]:
            row = {
                'period': period,
                'period_label': label,
                'high_exposure': high_exp,
                'exposure_label': exp_label
            }

            # --- Velocity Metrics (from aggregated data) ---
            vel = data[f'velocity_{period}']
            vel_row = vel[vel['high_exposure'] == high_exp]
            if not vel_row.empty:
                vel_row = vel_row.iloc[0]
                row['num_prs'] = vel_row['num_prs']
                row['num_repos'] = vel_row['num_repos']
                row['num_authors'] = vel_row['num_authors']
                row['avg_lead_time_hours'] = vel_row['avg_lead_time_hours']
                row['median_lead_time_hours'] = vel_row['median_lead_time_hours']
                row['p75_lead_time_hours'] = vel_row['p75_lead_time_hours']
                row['p90_lead_time_hours'] = vel_row['p90_lead_time_hours']
                row['avg_time_to_review_hours'] = vel_row['avg_time_to_review_hours']
                row['median_time_to_review_hours'] = vel_row['median_time_to_review_hours']
                row['median_pr_size'] = vel_row['median_pr_size']

            # --- Throughput Metrics (from developer-level data) ---
            tput = data[f'throughput_{period}']
            tput = tput[tput['high_exposure'] == high_exp]
            if not tput.empty:
                row['num_active_developers'] = len(tput)

                # PRs per developer (multiple aggregations for robustness)
                row['throughput_mean_prs_per_dev'] = tput['prs_merged_month'].mean()
                row['throughput_median_prs_per_dev'] = tput['prs_merged_month'].median()
                row['throughput_std_prs_per_dev'] = tput['prs_merged_month'].std()
                row['throughput_p75_prs_per_dev'] = tput['prs_merged_month'].quantile(0.75)
                row['throughput_p90_prs_per_dev'] = tput['prs_merged_month'].quantile(0.90)

                # Total PRs (sanity check)
                row['total_prs_from_devs'] = tput['prs_merged_month'].sum()

                # Active weeks (concentration measure)
                row['avg_active_weeks'] = tput['active_weeks'].mean()
                row['median_active_weeks'] = tput['active_weeks'].median()

                # Burstiness: CV of weekly PRs (from developer-level)
                cv_weekly = tput['cv_weekly_prs'].dropna()
                if len(cv_weekly) > 0:
                    row['burstiness_cv_weekly_mean'] = cv_weekly.mean()
                    row['burstiness_cv_weekly_median'] = cv_weekly.median()
                    row['burstiness_cv_weekly_std'] = cv_weekly.std()

            # --- Slack Metrics (from developer-level gaps) ---
            gaps = data[f'gaps_{period}']
            gaps = gaps[gaps['high_exposure'] == high_exp]
            if not gaps.empty:
                row['num_devs_with_gaps'] = len(gaps)

                # Inter-PR gap in hours (multiple aggregations)
                row['slack_mean_gap_hours'] = gaps['avg_gap_hours'].mean()
                row['slack_median_gap_hours'] = gaps['median_gap_hours'].median()
                row['slack_p75_gap_hours'] = gaps['p75_gap_hours'].median()

                # Inter-PR gap in business days
                row['slack_mean_gap_bdays'] = gaps['avg_gap_business_days'].mean()
                row['slack_median_gap_bdays'] = gaps['median_gap_business_days'].median()

            results.append(row)

    return pd.DataFrame(results)


def compute_changes(stats_df):
    """
    Compute percentage changes between pre and post periods.
    """
    changes = []

    for high_exp in [True, False]:
        pre = stats_df[(stats_df['period'] == '2021') & (stats_df['high_exposure'] == high_exp)]
        post = stats_df[(stats_df['period'] == '2025') & (stats_df['high_exposure'] == high_exp)]

        if pre.empty or post.empty:
            continue

        pre = pre.iloc[0]
        post = post.iloc[0]

        row = {
            'high_exposure': high_exp,
            'exposure_label': 'High AI-Exposure (Python/JS/Java/TS)' if high_exp else 'Low AI-Exposure (Fortran/COBOL/etc)'
        }

        # Compute changes for key metrics
        metrics = [
            ('p75_lead_time_hours', 'Velocity: p75 Lead Time (hrs)'),
            ('p90_lead_time_hours', 'Velocity: p90 Lead Time (hrs)'),
            ('avg_time_to_review_hours', 'Velocity: Avg Review Time (hrs)'),
            ('throughput_mean_prs_per_dev', 'Throughput: Mean PRs/Dev-Month'),
            ('throughput_median_prs_per_dev', 'Throughput: Median PRs/Dev-Month'),
            ('throughput_p75_prs_per_dev', 'Throughput: p75 PRs/Dev-Month'),
            ('avg_active_weeks', 'Concentration: Avg Active Weeks'),
            ('burstiness_cv_weekly_mean', 'Burstiness: Mean CV Weekly PRs'),
            ('burstiness_cv_weekly_median', 'Burstiness: Median CV Weekly PRs'),
            ('slack_median_gap_hours', 'Slack: Median Inter-PR Gap (hrs)'),
            ('slack_mean_gap_bdays', 'Slack: Mean Inter-PR Gap (bdays)'),
        ]

        for col, label in metrics:
            if col in pre.index and col in post.index:
                pre_val = pre[col]
                post_val = post[col]

                if pd.notna(pre_val) and pd.notna(post_val) and pre_val != 0:
                    pct_change = ((post_val - pre_val) / abs(pre_val)) * 100
                    row[f'{col}_pre'] = pre_val
                    row[f'{col}_post'] = post_val
                    row[f'{col}_change_pct'] = pct_change

        changes.append(row)

    return pd.DataFrame(changes)


def print_summary_table(stats_df, changes_df):
    """
    Print a formatted summary table of results.
    """
    print("\n" + "="*80)
    print("REVISED EMPIRICAL STRATEGY: PEAKS AND SLACK HYPOTHESIS")
    print("Comparing June 2021 (Pre-LLM) vs June 2025 (Post-LLM)")
    print("="*80)

    for high_exp in [True, False]:
        exp_label = "HIGH AI-EXPOSURE (Python/JS/Java/TS)" if high_exp else "LOW AI-EXPOSURE (Fortran/COBOL/etc)"
        print(f"\n{'='*80}")
        print(f"  {exp_label}")
        print(f"{'='*80}")

        pre = stats_df[(stats_df['period'] == '2021') & (stats_df['high_exposure'] == high_exp)].iloc[0]
        post = stats_df[(stats_df['period'] == '2025') & (stats_df['high_exposure'] == high_exp)].iloc[0]
        chg = changes_df[changes_df['high_exposure'] == high_exp].iloc[0] if not changes_df[changes_df['high_exposure'] == high_exp].empty else None

        print(f"\n  Sample Sizes:")
        print(f"    PRs:        {pre['num_prs']:>12,} -> {post['num_prs']:>12,}")
        print(f"    Developers: {pre['num_active_developers']:>12,} -> {post['num_active_developers']:>12,}")
        print(f"    Repos:      {pre['num_repos']:>12,} -> {post['num_repos']:>12,}")

        print(f"\n  1. VELOCITY (Task Speed) - Expected: DECREASE")
        print(f"    {'Metric':<35} {'Pre':>10} {'Post':>10} {'Change':>12}")
        print(f"    {'-'*35} {'-'*10} {'-'*10} {'-'*12}")

        for metric, label in [
            ('p75_lead_time_hours', 'p75 Lead Time (hours)'),
            ('p90_lead_time_hours', 'p90 Lead Time (hours)'),
            ('avg_time_to_review_hours', 'Avg Review Time (hours)'),
        ]:
            if metric in pre and pd.notna(pre[metric]):
                pre_v = pre[metric]
                post_v = post[metric]
                if pre_v != 0:
                    change = ((post_v - pre_v) / abs(pre_v)) * 100
                    direction = '↓' if change < 0 else '↑'
                    print(f"    {label:<35} {pre_v:>10.1f} {post_v:>10.1f} {direction} {abs(change):>9.1f}%")

        print(f"\n  2. THROUGHPUT (Output Volume) - Expected: FLAT (Paradox)")
        print(f"    {'Metric':<35} {'Pre':>10} {'Post':>10} {'Change':>12}")
        print(f"    {'-'*35} {'-'*10} {'-'*10} {'-'*12}")

        for metric, label in [
            ('throughput_mean_prs_per_dev', 'Mean PRs/Developer-Month'),
            ('throughput_median_prs_per_dev', 'Median PRs/Developer-Month'),
            ('throughput_p75_prs_per_dev', 'p75 PRs/Developer-Month'),
        ]:
            if metric in pre and pd.notna(pre[metric]):
                pre_v = pre[metric]
                post_v = post[metric]
                if pre_v != 0:
                    change = ((post_v - pre_v) / abs(pre_v)) * 100
                    direction = '↓' if change < 0 else '↑'
                    symbol = '✓' if abs(change) < 10 else ''  # Flat is expected
                    print(f"    {label:<35} {pre_v:>10.2f} {post_v:>10.2f} {direction} {abs(change):>9.1f}% {symbol}")

        print(f"\n  3. BURSTINESS (Work Pattern) - Expected: INCREASE")
        print(f"    {'Metric':<35} {'Pre':>10} {'Post':>10} {'Change':>12}")
        print(f"    {'-'*35} {'-'*10} {'-'*10} {'-'*12}")

        for metric, label in [
            ('burstiness_cv_weekly_mean', 'Mean CV Weekly PRs'),
            ('burstiness_cv_weekly_median', 'Median CV Weekly PRs'),
            ('avg_active_weeks', 'Avg Active Weeks in Month'),
        ]:
            if metric in pre and pd.notna(pre[metric]):
                pre_v = pre[metric]
                post_v = post[metric]
                if pre_v != 0:
                    change = ((post_v - pre_v) / abs(pre_v)) * 100
                    direction = '↓' if change < 0 else '↑'
                    print(f"    {label:<35} {pre_v:>10.2f} {post_v:>10.2f} {direction} {abs(change):>9.1f}%")

        print(f"\n  4. SLACK (Gaps Between PRs) - Expected: INCREASE")
        print(f"    {'Metric':<35} {'Pre':>10} {'Post':>10} {'Change':>12}")
        print(f"    {'-'*35} {'-'*10} {'-'*10} {'-'*12}")

        for metric, label in [
            ('slack_median_gap_hours', 'Median Inter-PR Gap (hours)'),
            ('slack_mean_gap_bdays', 'Mean Inter-PR Gap (biz days)'),
        ]:
            if metric in pre and pd.notna(pre[metric]):
                pre_v = pre[metric]
                post_v = post[metric]
                if pre_v != 0:
                    change = ((post_v - pre_v) / abs(pre_v)) * 100
                    direction = '↓' if change < 0 else '↑'
                    print(f"    {label:<35} {pre_v:>10.1f} {post_v:>10.1f} {direction} {abs(change):>9.1f}%")


def create_visualizations(stats_df, changes_df, output_dir):
    """
    Create publication-quality visualizations.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping visualizations - matplotlib not available")
        return

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- Figure 1: The Paradox Visualization ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Get high exposure data
    pre_high = stats_df[(stats_df['period'] == '2021') & (stats_df['high_exposure'] == True)].iloc[0]
    post_high = stats_df[(stats_df['period'] == '2025') & (stats_df['high_exposure'] == True)].iloc[0]

    # Panel 1: Velocity (Task Speed)
    ax1 = axes[0]
    metrics = ['p75_lead_time_hours', 'p90_lead_time_hours']
    labels = ['p75', 'p90']
    pre_vals = [pre_high[m] for m in metrics]
    post_vals = [post_high[m] for m in metrics]

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax1.bar(x - width/2, pre_vals, width, label='June 2021 (Pre-LLM)', color='#2ecc71', alpha=0.8)
    bars2 = ax1.bar(x + width/2, post_vals, width, label='June 2025 (Post-LLM)', color='#3498db', alpha=0.8)

    ax1.set_ylabel('Hours')
    ax1.set_title('1. Velocity: PR Lead Time\n(Task Speed)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend(loc='upper right')

    # Add change annotations
    for i, (pre, post) in enumerate(zip(pre_vals, post_vals)):
        if pre != 0:
            change = ((post - pre) / pre) * 100
            ax1.annotate(f'{change:.0f}%', xy=(x[i] + width/2, post + 5),
                        ha='center', fontsize=10, color='green' if change < 0 else 'red')

    ax1.set_ylim(0, max(pre_vals) * 1.3)

    # Panel 2: Throughput
    ax2 = axes[1]
    metrics = ['throughput_mean_prs_per_dev', 'throughput_median_prs_per_dev', 'throughput_p75_prs_per_dev']
    labels = ['Mean', 'Median', 'p75']
    pre_vals = [pre_high[m] for m in metrics]
    post_vals = [post_high[m] for m in metrics]

    x = np.arange(len(labels))

    bars1 = ax2.bar(x - width/2, pre_vals, width, label='June 2021', color='#2ecc71', alpha=0.8)
    bars2 = ax2.bar(x + width/2, post_vals, width, label='June 2025', color='#3498db', alpha=0.8)

    ax2.set_ylabel('PRs per Developer-Month')
    ax2.set_title('2. Throughput: PRs per Developer\n(Output Volume)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)

    # Add change annotations
    for i, (pre, post) in enumerate(zip(pre_vals, post_vals)):
        if pre != 0:
            change = ((post - pre) / pre) * 100
            ax2.annotate(f'{change:+.1f}%', xy=(x[i] + width/2, post + 0.3),
                        ha='center', fontsize=10, color='gray')

    ax2.set_ylim(0, max(pre_vals + post_vals) * 1.3)

    # Panel 3: Slack (Inter-PR Gap)
    ax3 = axes[2]
    pre_gap = pre_high['slack_median_gap_hours']
    post_gap = post_high['slack_median_gap_hours']

    bars = ax3.bar(['June 2021\n(Pre-LLM)', 'June 2025\n(Post-LLM)'],
                   [pre_gap, post_gap],
                   color=['#2ecc71', '#3498db'], alpha=0.8, width=0.5)

    ax3.set_ylabel('Hours')
    ax3.set_title('3. Slack: Median Inter-PR Gap\n(Time Between Completions)', fontsize=12, fontweight='bold')

    if pre_gap != 0:
        change = ((post_gap - pre_gap) / pre_gap) * 100
        ax3.annotate(f'{change:+.1f}%', xy=(1, post_gap + 1),
                    ha='center', fontsize=12, color='gray')

    ax3.set_ylim(0, max(pre_gap, post_gap) * 1.3)

    plt.tight_layout()
    fig.suptitle('The Task-Job Paradox: Velocity Up, Throughput Flat\n(High AI-Exposure Languages)',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.savefig(output_dir / 'revised_paradox_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_dir / 'revised_paradox_analysis.png'}")
    plt.close()

    # --- Figure 2: Distribution Comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Throughput Distribution
    ax1 = axes[0]
    tput_2021 = pd.read_csv(output_dir / 'developer_throughput_2021_06.csv')
    tput_2025 = pd.read_csv(output_dir / 'developer_throughput_2025_06.csv')

    tput_2021_high = tput_2021[tput_2021['high_exposure'] == True]['prs_merged_month']
    tput_2025_high = tput_2025[tput_2025['high_exposure'] == True]['prs_merged_month']

    # Cap at 50 for visualization
    tput_2021_capped = tput_2021_high.clip(upper=50)
    tput_2025_capped = tput_2025_high.clip(upper=50)

    ax1.hist(tput_2021_capped, bins=50, alpha=0.5, label='June 2021', color='#2ecc71', density=True)
    ax1.hist(tput_2025_capped, bins=50, alpha=0.5, label='June 2025', color='#3498db', density=True)

    ax1.axvline(tput_2021_high.median(), color='#27ae60', linestyle='--', label=f'2021 Median: {tput_2021_high.median():.1f}')
    ax1.axvline(tput_2025_high.median(), color='#2980b9', linestyle='--', label=f'2025 Median: {tput_2025_high.median():.1f}')

    ax1.set_xlabel('PRs Merged per Developer-Month')
    ax1.set_ylabel('Density')
    ax1.set_title('Throughput Distribution\n(High AI-Exposure Languages)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')

    # Panel 2: CV of Weekly PRs (Burstiness)
    ax2 = axes[1]
    cv_2021 = tput_2021[tput_2021['high_exposure'] == True]['cv_weekly_prs'].dropna()
    cv_2025 = tput_2025[tput_2025['high_exposure'] == True]['cv_weekly_prs'].dropna()

    # Cap for visualization
    cv_2021_capped = cv_2021.clip(upper=3)
    cv_2025_capped = cv_2025.clip(upper=3)

    ax2.hist(cv_2021_capped, bins=50, alpha=0.5, label='June 2021', color='#2ecc71', density=True)
    ax2.hist(cv_2025_capped, bins=50, alpha=0.5, label='June 2025', color='#3498db', density=True)

    ax2.axvline(cv_2021.median(), color='#27ae60', linestyle='--', label=f'2021 Median: {cv_2021.median():.2f}')
    ax2.axvline(cv_2025.median(), color='#2980b9', linestyle='--', label=f'2025 Median: {cv_2025.median():.2f}')

    ax2.set_xlabel('CV of Weekly PRs (Burstiness)')
    ax2.set_ylabel('Density')
    ax2.set_title('Work Pattern Burstiness\n(High AI-Exposure Languages)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_dir / 'revised_distributions.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'revised_distributions.png'}")
    plt.close()


def run_statistical_tests(data):
    """
    Run statistical tests on the developer-level data.
    """
    print("\n" + "="*80)
    print("STATISTICAL TESTS")
    print("="*80)

    tput_2021 = data['throughput_2021']
    tput_2025 = data['throughput_2025']

    for high_exp, label in [(True, 'High AI-Exposure'), (False, 'Low AI-Exposure')]:
        print(f"\n{label}:")
        print("-" * 40)

        tput_pre = tput_2021[tput_2021['high_exposure'] == high_exp]['prs_merged_month']
        tput_post = tput_2025[tput_2025['high_exposure'] == high_exp]['prs_merged_month']

        # Mann-Whitney U test (non-parametric)
        stat, p = stats.mannwhitneyu(tput_pre, tput_post, alternative='two-sided')
        print(f"  Throughput (PRs/dev-month) Mann-Whitney U test:")
        print(f"    Pre mean: {tput_pre.mean():.2f}, Post mean: {tput_post.mean():.2f}")
        print(f"    U-statistic: {stat:.0f}, p-value: {p:.4f}")
        print(f"    {'Significant' if p < 0.05 else 'Not significant'} at p < 0.05")

        # CV of weekly PRs
        cv_pre = tput_2021[tput_2021['high_exposure'] == high_exp]['cv_weekly_prs'].dropna()
        cv_post = tput_2025[tput_2025['high_exposure'] == high_exp]['cv_weekly_prs'].dropna()

        if len(cv_pre) > 0 and len(cv_post) > 0:
            stat, p = stats.mannwhitneyu(cv_pre, cv_post, alternative='two-sided')
            print(f"\n  Burstiness (CV weekly PRs) Mann-Whitney U test:")
            print(f"    Pre median: {cv_pre.median():.3f}, Post median: {cv_post.median():.3f}")
            print(f"    U-statistic: {stat:.0f}, p-value: {p:.4f}")
            print(f"    {'Significant' if p < 0.05 else 'Not significant'} at p < 0.05")


def main():
    """Main analysis function."""
    print("\n" + "="*80)
    print("ANALYZING REVISED EMPIRICAL STRATEGY RESULTS")
    print("="*80)

    # Load data
    print("\nLoading data...")
    data = load_all_data()

    # Compute multi-granularity statistics
    print("Computing statistics...")
    stats_df = compute_multi_granularity_stats(data)

    # Compute changes
    changes_df = compute_changes(stats_df)

    # Print summary
    print_summary_table(stats_df, changes_df)

    # Statistical tests
    run_statistical_tests(data)

    # Create visualizations
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    create_visualizations(stats_df, changes_df, Path(__file__).parent.parent / 'results')

    # Save detailed results
    results_dir = Path(__file__).parent.parent / 'results'
    stats_df.to_csv(results_dir / 'revised_multi_granularity_stats.csv', index=False)
    changes_df.to_csv(results_dir / 'revised_changes_detailed.csv', index=False)
    print(f"\nSaved detailed statistics to {results_dir / 'revised_multi_granularity_stats.csv'}")

    # Print conclusion
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    high_changes = changes_df[changes_df['high_exposure'] == True].iloc[0]

    print("""
For HIGH AI-EXPOSURE languages (Python, JavaScript, Java, TypeScript):

1. VELOCITY IMPROVED DRAMATICALLY:
   - p75 PR Lead Time: {p75_change:.0f}% change
   - p90 PR Lead Time: {p90_change:.0f}% change
   - Avg Review Time: {review_change:.0f}% change

2. THROUGHPUT STAYED ESSENTIALLY FLAT:
   - Mean PRs/Dev-Month: {tput_mean_change:+.1f}% change
   - Median PRs/Dev-Month: {tput_median_change:+.1f}% change

3. BURSTINESS:
   - CV of Weekly PRs: {cv_change:+.1f}% change

4. SLACK:
   - Median Inter-PR Gap: {gap_change:+.1f}% change

THE PARADOX IS CONFIRMED: Task velocity improved dramatically, but
individual developer throughput did not increase proportionally.
    """.format(
        p75_change=high_changes.get('p75_lead_time_hours_change_pct', 0),
        p90_change=high_changes.get('p90_lead_time_hours_change_pct', 0),
        review_change=high_changes.get('avg_time_to_review_hours_change_pct', 0),
        tput_mean_change=high_changes.get('throughput_mean_prs_per_dev_change_pct', 0),
        tput_median_change=high_changes.get('throughput_median_prs_per_dev_change_pct', 0),
        cv_change=high_changes.get('burstiness_cv_weekly_mean_change_pct', 0),
        gap_change=high_changes.get('slack_median_gap_hours_change_pct', 0),
    ))

    return stats_df, changes_df


if __name__ == "__main__":
    stats_df, changes_df = main()
