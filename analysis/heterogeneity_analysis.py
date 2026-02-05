#!/usr/bin/env python3
"""
Heterogeneity Analysis: Stratified by Developer Activity Level

The aggregate analysis shows a puzzle: velocity up 93%, throughput flat,
but burstiness and slack unchanged. This doesn't add up mathematically.

Hypothesis: The signal is diluted by casual/dormant contributors.
The "real" story is in the heavy contributors (top 20%).

This script stratifies the analysis by developer activity quintiles.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def load_developer_data():
    """Load developer-level data."""
    results_dir = Path(__file__).parent.parent / 'results'

    data = {
        'throughput_2021': pd.read_csv(results_dir / 'developer_throughput_2021_06.csv'),
        'throughput_2025': pd.read_csv(results_dir / 'developer_throughput_2025_06.csv'),
        'gaps_2021': pd.read_csv(results_dir / 'inter_pr_gaps_2021_06.csv'),
        'gaps_2025': pd.read_csv(results_dir / 'inter_pr_gaps_2025_06.csv'),
    }

    return data


def assign_activity_quintiles(df, col='prs_merged_month'):
    """
    Assign developers to activity quintiles based on their output.

    Q5 = Top 20% (most active)
    Q1 = Bottom 20% (least active)
    """
    df = df.copy()
    df['activity_quintile'] = pd.qcut(
        df[col].rank(method='first'),
        q=5,
        labels=['Q1 (Bottom 20%)', 'Q2', 'Q3', 'Q4', 'Q5 (Top 20%)']
    )
    return df


def analyze_by_quintile(data, high_exposure_only=True):
    """
    Analyze metrics stratified by developer activity quintile.
    """
    results = []

    for period in ['2021', '2025']:
        tput = data[f'throughput_{period}'].copy()
        gaps = data[f'gaps_{period}'].copy()

        if high_exposure_only:
            tput = tput[tput['high_exposure'] == True]
            gaps = gaps[gaps['high_exposure'] == True]

        # Assign quintiles
        tput = assign_activity_quintiles(tput, 'prs_merged_month')

        # Merge gaps data
        tput_with_gaps = tput.merge(
            gaps[['author_id', 'avg_gap_hours', 'median_gap_hours', 'num_gaps']],
            on='author_id',
            how='left'
        )

        # Analyze each quintile
        for quintile in tput_with_gaps['activity_quintile'].unique():
            q_data = tput_with_gaps[tput_with_gaps['activity_quintile'] == quintile]

            row = {
                'period': period,
                'quintile': quintile,
                'n_developers': len(q_data),
            }

            # Throughput metrics
            row['throughput_mean'] = q_data['prs_merged_month'].mean()
            row['throughput_median'] = q_data['prs_merged_month'].median()
            row['throughput_min'] = q_data['prs_merged_month'].min()
            row['throughput_max'] = q_data['prs_merged_month'].max()

            # Burstiness (CV of weekly PRs)
            cv_data = q_data['cv_weekly_prs'].dropna()
            if len(cv_data) > 0:
                row['burstiness_cv_mean'] = cv_data.mean()
                row['burstiness_cv_median'] = cv_data.median()

            # Active weeks
            row['active_weeks_mean'] = q_data['active_weeks'].mean()

            # Slack (inter-PR gap)
            gap_data = q_data['median_gap_hours'].dropna()
            if len(gap_data) > 0:
                row['slack_gap_mean'] = gap_data.mean()
                row['slack_gap_median'] = gap_data.median()

            # Lead time
            row['lead_time_mean'] = q_data['avg_lead_time_hours_month'].mean()

            results.append(row)

    return pd.DataFrame(results)


def compute_quintile_changes(quintile_df):
    """
    Compute changes between periods for each quintile.
    """
    changes = []

    for quintile in quintile_df['quintile'].unique():
        pre = quintile_df[(quintile_df['period'] == '2021') & (quintile_df['quintile'] == quintile)]
        post = quintile_df[(quintile_df['period'] == '2025') & (quintile_df['quintile'] == quintile)]

        if pre.empty or post.empty:
            continue

        pre = pre.iloc[0]
        post = post.iloc[0]

        row = {'quintile': quintile}

        for metric in ['throughput_mean', 'throughput_median', 'burstiness_cv_mean',
                       'active_weeks_mean', 'slack_gap_mean', 'slack_gap_median', 'lead_time_mean']:
            if metric in pre and metric in post:
                pre_val = pre[metric]
                post_val = post[metric]
                if pd.notna(pre_val) and pd.notna(post_val) and pre_val != 0:
                    row[f'{metric}_pre'] = pre_val
                    row[f'{metric}_post'] = post_val
                    row[f'{metric}_change_pct'] = ((post_val - pre_val) / abs(pre_val)) * 100

        changes.append(row)

    return pd.DataFrame(changes)


def analyze_top_contributors(data, top_pct=0.20, high_exposure_only=True):
    """
    Deep dive into top contributors only.
    """
    results = []

    for period in ['2021', '2025']:
        tput = data[f'throughput_{period}'].copy()
        gaps = data[f'gaps_{period}'].copy()

        if high_exposure_only:
            tput = tput[tput['high_exposure'] == True]
            gaps = gaps[gaps['high_exposure'] == True]

        # Get top X% by PRs merged
        threshold = tput['prs_merged_month'].quantile(1 - top_pct)
        top_devs = tput[tput['prs_merged_month'] >= threshold]

        # Merge with gaps
        top_with_gaps = top_devs.merge(
            gaps[['author_id', 'avg_gap_hours', 'median_gap_hours', 'p75_gap_hours', 'num_gaps']],
            on='author_id',
            how='left'
        )

        row = {
            'period': period,
            'n_developers': len(top_with_gaps),
            'threshold_prs': threshold,
        }

        # Throughput
        row['throughput_mean'] = top_with_gaps['prs_merged_month'].mean()
        row['throughput_median'] = top_with_gaps['prs_merged_month'].median()
        row['throughput_p25'] = top_with_gaps['prs_merged_month'].quantile(0.25)
        row['throughput_p75'] = top_with_gaps['prs_merged_month'].quantile(0.75)

        # Burstiness
        cv = top_with_gaps['cv_weekly_prs'].dropna()
        row['burstiness_cv_mean'] = cv.mean() if len(cv) > 0 else np.nan
        row['burstiness_cv_median'] = cv.median() if len(cv) > 0 else np.nan
        row['burstiness_cv_p75'] = cv.quantile(0.75) if len(cv) > 0 else np.nan

        # Active weeks
        row['active_weeks_mean'] = top_with_gaps['active_weeks'].mean()
        row['active_weeks_median'] = top_with_gaps['active_weeks'].median()

        # Slack - MULTIPLE GRANULARITIES
        gap_hours = top_with_gaps['median_gap_hours'].dropna()
        if len(gap_hours) > 0:
            row['slack_median_gap_hours'] = gap_hours.median()
            row['slack_mean_gap_hours'] = gap_hours.mean()
            row['slack_p25_gap_hours'] = gap_hours.quantile(0.25)
            row['slack_p75_gap_hours'] = gap_hours.quantile(0.75)

            # Convert to days
            row['slack_median_gap_days'] = gap_hours.median() / 24
            row['slack_mean_gap_days'] = gap_hours.mean() / 24

        # Lead time
        row['lead_time_mean'] = top_with_gaps['avg_lead_time_hours_month'].mean()
        row['lead_time_median'] = top_with_gaps['avg_lead_time_hours_month'].median()

        results.append(row)

    return pd.DataFrame(results)


def analyze_gap_distribution(data, high_exposure_only=True):
    """
    Analyze the full distribution of inter-PR gaps, not just aggregates.
    """
    results = []

    for period in ['2021', '2025']:
        gaps = data[f'gaps_{period}'].copy()
        tput = data[f'throughput_{period}'].copy()

        if high_exposure_only:
            gaps = gaps[gaps['high_exposure'] == True]
            tput = tput[tput['high_exposure'] == True]

        # Get top 20% developers
        threshold = tput['prs_merged_month'].quantile(0.80)
        top_dev_ids = tput[tput['prs_merged_month'] >= threshold]['author_id']
        top_gaps = gaps[gaps['author_id'].isin(top_dev_ids)]

        row = {'period': period}

        # All developers
        row['all_n'] = len(gaps)
        row['all_median_gap'] = gaps['median_gap_hours'].median()
        row['all_mean_gap'] = gaps['avg_gap_hours'].mean()

        # Top 20% only
        row['top20_n'] = len(top_gaps)
        row['top20_median_gap'] = top_gaps['median_gap_hours'].median()
        row['top20_mean_gap'] = top_gaps['avg_gap_hours'].mean()
        row['top20_p25_gap'] = top_gaps['median_gap_hours'].quantile(0.25)
        row['top20_p75_gap'] = top_gaps['median_gap_hours'].quantile(0.75)

        results.append(row)

    return pd.DataFrame(results)


def print_heterogeneity_results(quintile_df, changes_df, top_df, gap_dist_df):
    """Print formatted results."""

    print("\n" + "="*90)
    print("HETEROGENEITY ANALYSIS: STRATIFIED BY DEVELOPER ACTIVITY LEVEL")
    print("High AI-Exposure Languages Only (Python/JS/Java/TS)")
    print("="*90)

    # Quintile summary
    print("\n" + "-"*90)
    print("1. ANALYSIS BY ACTIVITY QUINTILE")
    print("-"*90)
    print("\nQ5 = Top 20% most active developers")
    print("Q1 = Bottom 20% least active developers\n")

    print(f"{'Quintile':<20} {'Period':<8} {'N Devs':>10} {'Throughput':>12} {'CV Weekly':>10} {'Gap (hrs)':>10}")
    print(f"{'':20} {'':8} {'':>10} {'(mean PRs)':>12} {'(mean)':>10} {'(median)':>10}")
    print("-"*90)

    for quintile in ['Q5 (Top 20%)', 'Q4', 'Q3', 'Q2', 'Q1 (Bottom 20%)']:
        for period in ['2021', '2025']:
            row = quintile_df[(quintile_df['quintile'] == quintile) & (quintile_df['period'] == period)]
            if not row.empty:
                r = row.iloc[0]
                tput = r.get('throughput_mean', np.nan)
                cv = r.get('burstiness_cv_mean', np.nan)
                gap = r.get('slack_gap_median', np.nan)
                print(f"{quintile:<20} {period:<8} {r['n_developers']:>10,} {tput:>12.1f} {cv:>10.2f} {gap:>10.1f}")
        print()

    # Changes by quintile
    print("\n" + "-"*90)
    print("2. CHANGES BY QUINTILE (2021 -> 2025)")
    print("-"*90)
    print(f"\n{'Quintile':<20} {'Throughput Δ':>15} {'Burstiness Δ':>15} {'Slack Gap Δ':>15}")
    print("-"*90)

    for _, row in changes_df.iterrows():
        tput_chg = row.get('throughput_mean_change_pct', np.nan)
        cv_chg = row.get('burstiness_cv_mean_change_pct', np.nan)
        gap_chg = row.get('slack_gap_median_change_pct', np.nan)

        tput_str = f"{tput_chg:+.1f}%" if pd.notna(tput_chg) else "N/A"
        cv_str = f"{cv_chg:+.1f}%" if pd.notna(cv_chg) else "N/A"
        gap_str = f"{gap_chg:+.1f}%" if pd.notna(gap_chg) else "N/A"

        print(f"{row['quintile']:<20} {tput_str:>15} {cv_str:>15} {gap_str:>15}")

    # Top 20% deep dive
    print("\n" + "-"*90)
    print("3. DEEP DIVE: TOP 20% CONTRIBUTORS ONLY")
    print("-"*90)

    pre = top_df[top_df['period'] == '2021'].iloc[0]
    post = top_df[top_df['period'] == '2025'].iloc[0]

    print(f"\n{'Metric':<35} {'2021':>12} {'2025':>12} {'Change':>12}")
    print("-"*90)

    metrics = [
        ('n_developers', 'Number of developers', ''),
        ('threshold_prs', 'Min PRs to qualify', ''),
        ('throughput_mean', 'Mean PRs/month', ' PRs'),
        ('throughput_median', 'Median PRs/month', ' PRs'),
        ('burstiness_cv_mean', 'Mean CV weekly PRs', ''),
        ('burstiness_cv_median', 'Median CV weekly PRs', ''),
        ('active_weeks_mean', 'Mean active weeks', ' wks'),
        ('slack_median_gap_hours', 'Median gap (hours)', ' hrs'),
        ('slack_mean_gap_hours', 'Mean gap (hours)', ' hrs'),
        ('slack_median_gap_days', 'Median gap (days)', ' days'),
        ('lead_time_mean', 'Mean lead time (hours)', ' hrs'),
    ]

    for col, label, unit in metrics:
        pre_val = pre.get(col, np.nan)
        post_val = post.get(col, np.nan)

        if pd.notna(pre_val) and pd.notna(post_val):
            if pre_val != 0:
                change = ((post_val - pre_val) / abs(pre_val)) * 100
                change_str = f"{change:+.1f}%"
            else:
                change_str = "N/A"

            if col == 'n_developers':
                print(f"{label:<35} {int(pre_val):>12,} {int(post_val):>12,} {change_str:>12}")
            else:
                print(f"{label:<35} {pre_val:>11.1f}{unit} {post_val:>11.1f}{unit} {change_str:>12}")

    # Gap distribution
    print("\n" + "-"*90)
    print("4. INTER-PR GAP DISTRIBUTION (ALL vs TOP 20%)")
    print("-"*90)

    print(f"\n{'Group':<25} {'2021 Median':>15} {'2025 Median':>15} {'Change':>12}")
    print("-"*90)

    pre_gap = gap_dist_df[gap_dist_df['period'] == '2021'].iloc[0]
    post_gap = gap_dist_df[gap_dist_df['period'] == '2025'].iloc[0]

    for prefix, label in [('all', 'All developers'), ('top20', 'Top 20% only')]:
        pre_val = pre_gap[f'{prefix}_median_gap']
        post_val = post_gap[f'{prefix}_median_gap']
        change = ((post_val - pre_val) / pre_val) * 100 if pre_val != 0 else 0
        print(f"{label:<25} {pre_val:>14.1f}h {post_val:>14.1f}h {change:>+11.1f}%")


def create_heterogeneity_visualizations(quintile_df, changes_df, top_df, output_dir):
    """Create visualizations for heterogeneity analysis."""
    if not MATPLOTLIB_AVAILABLE:
        return

    plt.style.use('seaborn-v0_8-whitegrid')

    # Figure 1: Changes by quintile
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    quintiles = ['Q1 (Bottom 20%)', 'Q2', 'Q3', 'Q4', 'Q5 (Top 20%)']
    x = np.arange(len(quintiles))

    # Panel 1: Throughput change by quintile
    ax1 = axes[0]
    tput_changes = [changes_df[changes_df['quintile'] == q]['throughput_mean_change_pct'].values[0]
                    if len(changes_df[changes_df['quintile'] == q]) > 0 else 0
                    for q in quintiles]
    colors = ['green' if c < 0 else 'red' for c in tput_changes]
    ax1.bar(x, tput_changes, color=colors, alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Q1\n(Bottom)', 'Q2', 'Q3', 'Q4', 'Q5\n(Top)'])
    ax1.set_ylabel('% Change')
    ax1.set_title('Throughput Change\n(PRs/Dev-Month)', fontweight='bold')

    # Panel 2: Burstiness change by quintile
    ax2 = axes[1]
    cv_changes = [changes_df[changes_df['quintile'] == q]['burstiness_cv_mean_change_pct'].values[0]
                  if len(changes_df[changes_df['quintile'] == q]) > 0 and
                     pd.notna(changes_df[changes_df['quintile'] == q]['burstiness_cv_mean_change_pct'].values[0])
                  else 0
                  for q in quintiles]
    colors = ['red' if c > 0 else 'green' for c in cv_changes]
    ax2.bar(x, cv_changes, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Q1\n(Bottom)', 'Q2', 'Q3', 'Q4', 'Q5\n(Top)'])
    ax2.set_ylabel('% Change')
    ax2.set_title('Burstiness Change\n(CV of Weekly PRs)', fontweight='bold')

    # Panel 3: Slack change by quintile
    ax3 = axes[2]
    gap_changes = [changes_df[changes_df['quintile'] == q]['slack_gap_median_change_pct'].values[0]
                   if len(changes_df[changes_df['quintile'] == q]) > 0 and
                      pd.notna(changes_df[changes_df['quintile'] == q]['slack_gap_median_change_pct'].values[0])
                   else 0
                   for q in quintiles]
    colors = ['red' if c > 0 else 'green' for c in gap_changes]
    ax3.bar(x, gap_changes, color=colors, alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Q1\n(Bottom)', 'Q2', 'Q3', 'Q4', 'Q5\n(Top)'])
    ax3.set_ylabel('% Change')
    ax3.set_title('Slack Change\n(Median Inter-PR Gap)', fontweight='bold')

    plt.tight_layout()
    fig.suptitle('Heterogeneity Analysis: Changes by Developer Activity Level\n(High AI-Exposure Languages, June 2021 → June 2025)',
                 fontsize=12, fontweight='bold', y=1.02)

    plt.savefig(output_dir / 'heterogeneity_by_quintile.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_dir / 'heterogeneity_by_quintile.png'}")
    plt.close()

    # Figure 2: Top 20% comparison
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    pre = top_df[top_df['period'] == '2021'].iloc[0]
    post = top_df[top_df['period'] == '2025'].iloc[0]

    # Throughput
    ax1 = axes[0]
    ax1.bar(['2021', '2025'], [pre['throughput_mean'], post['throughput_mean']],
            color=['#2ecc71', '#3498db'], alpha=0.8)
    ax1.set_ylabel('PRs/Developer-Month')
    ax1.set_title('Throughput\n(Top 20%)', fontweight='bold')
    change = ((post['throughput_mean'] - pre['throughput_mean']) / pre['throughput_mean']) * 100
    ax1.annotate(f'{change:+.1f}%', xy=(1, post['throughput_mean']),
                 ha='center', va='bottom', fontsize=11)

    # Burstiness
    ax2 = axes[1]
    ax2.bar(['2021', '2025'], [pre['burstiness_cv_mean'], post['burstiness_cv_mean']],
            color=['#2ecc71', '#3498db'], alpha=0.8)
    ax2.set_ylabel('CV of Weekly PRs')
    ax2.set_title('Burstiness\n(Top 20%)', fontweight='bold')
    change = ((post['burstiness_cv_mean'] - pre['burstiness_cv_mean']) / pre['burstiness_cv_mean']) * 100
    ax2.annotate(f'{change:+.1f}%', xy=(1, post['burstiness_cv_mean']),
                 ha='center', va='bottom', fontsize=11)

    # Slack
    ax3 = axes[2]
    ax3.bar(['2021', '2025'], [pre['slack_median_gap_hours'], post['slack_median_gap_hours']],
            color=['#2ecc71', '#3498db'], alpha=0.8)
    ax3.set_ylabel('Hours')
    ax3.set_title('Inter-PR Gap\n(Top 20%)', fontweight='bold')
    change = ((post['slack_median_gap_hours'] - pre['slack_median_gap_hours']) / pre['slack_median_gap_hours']) * 100
    ax3.annotate(f'{change:+.1f}%', xy=(1, post['slack_median_gap_hours']),
                 ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / 'heterogeneity_top20.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'heterogeneity_top20.png'}")
    plt.close()


def main():
    """Run heterogeneity analysis."""
    print("\n" + "="*90)
    print("RUNNING HETEROGENEITY ANALYSIS")
    print("="*90)

    # Load data
    print("\nLoading developer-level data...")
    data = load_developer_data()

    # Analyze by quintile
    print("Analyzing by activity quintile...")
    quintile_df = analyze_by_quintile(data, high_exposure_only=True)

    # Compute changes
    changes_df = compute_quintile_changes(quintile_df)

    # Top 20% deep dive
    print("Deep diving into top 20% contributors...")
    top_df = analyze_top_contributors(data, top_pct=0.20, high_exposure_only=True)

    # Gap distribution analysis
    print("Analyzing gap distribution...")
    gap_dist_df = analyze_gap_distribution(data, high_exposure_only=True)

    # Print results
    print_heterogeneity_results(quintile_df, changes_df, top_df, gap_dist_df)

    # Create visualizations
    output_dir = Path(__file__).parent.parent / 'results'
    create_heterogeneity_visualizations(quintile_df, changes_df, top_df, output_dir)

    # Save results
    quintile_df.to_csv(output_dir / 'heterogeneity_quintiles.csv', index=False)
    changes_df.to_csv(output_dir / 'heterogeneity_changes.csv', index=False)
    top_df.to_csv(output_dir / 'heterogeneity_top20.csv', index=False)
    gap_dist_df.to_csv(output_dir / 'heterogeneity_gap_distribution.csv', index=False)
    print(f"\nSaved detailed results to {output_dir}")

    # Conclusion
    print("\n" + "="*90)
    print("KEY INSIGHT")
    print("="*90)

    pre_top = top_df[top_df['period'] == '2021'].iloc[0]
    post_top = top_df[top_df['period'] == '2025'].iloc[0]

    tput_change = ((post_top['throughput_mean'] - pre_top['throughput_mean']) / pre_top['throughput_mean']) * 100
    cv_change = ((post_top['burstiness_cv_mean'] - pre_top['burstiness_cv_mean']) / pre_top['burstiness_cv_mean']) * 100
    gap_change = ((post_top['slack_median_gap_hours'] - pre_top['slack_median_gap_hours']) / pre_top['slack_median_gap_hours']) * 100

    print(f"""
For TOP 20% most active developers (High AI-Exposure languages):

  Throughput (PRs/dev-month):  {pre_top['throughput_mean']:.1f} → {post_top['throughput_mean']:.1f}  ({tput_change:+.1f}%)
  Burstiness (CV weekly PRs):  {pre_top['burstiness_cv_mean']:.2f} → {post_top['burstiness_cv_mean']:.2f}  ({cv_change:+.1f}%)
  Slack (median gap hours):    {pre_top['slack_median_gap_hours']:.1f} → {post_top['slack_median_gap_hours']:.1f}  ({gap_change:+.1f}%)

The pattern may differ from the aggregate analysis due to composition effects.
    """)

    return quintile_df, changes_df, top_df


if __name__ == "__main__":
    quintile_df, changes_df, top_df = main()
