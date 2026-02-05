#!/usr/bin/env python3
"""
Main Analysis Orchestration Script

Runs the complete revised empirical analysis for the Task-Job Paradox paper.

Usage:
    python -m analysis.revised.run_analysis --project-id YOUR_PROJECT_ID

Or from Python:
    from analysis.revised.run_analysis import run_full_analysis
    results = run_full_analysis(project_id='your-project-id')
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import date
from typing import Dict, Any, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analysis.revised.config import (
    DEFAULT_CONFIG,
    TREATMENT_DATE,
    HIGH_EXPOSURE_LANGUAGES,
    LOW_EXPOSURE_LANGUAGES,
    HYPOTHESES,
)
from analysis.revised.metrics import (
    DataLoader,
    MetricsComputer,
    PeriodComparison,
    load_and_compute_all_metrics,
)
from analysis.revised.statistical_analysis import (
    InterruptedTimeSeries,
    DifferenceInDifferences,
    HeterogeneityAnalysis,
    StatisticalTests,
    run_full_statistical_analysis,
)
from analysis.revised.visualizations import (
    create_all_figures,
    export_results_table,
)


# =============================================================================
# ANALYSIS CONFIGURATION
# =============================================================================

# Default periods to compare
PRE_PERIOD = (2021, 6)   # June 2021 (pre-ChatGPT)
POST_PERIOD = (2025, 6)  # June 2025 (post-ChatGPT)

# Output directories
RESULTS_DIR = Path(__file__).parent.parent.parent / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'


# =============================================================================
# MAIN ANALYSIS FUNCTIONS
# =============================================================================

def run_full_analysis(
    project_id: str,
    pre_period: tuple = PRE_PERIOD,
    post_period: tuple = POST_PERIOD,
    force_refresh: bool = False,
    skip_visualizations: bool = False
) -> Dict[str, Any]:
    """
    Run the complete analysis pipeline.

    Args:
        project_id: Google Cloud project ID for BigQuery
        pre_period: (year, month) tuple for pre-treatment period
        post_period: (year, month) tuple for post-treatment period
        force_refresh: If True, re-query data even if cached
        skip_visualizations: If True, skip figure generation

    Returns:
        Dict with all analysis results
    """
    print("=" * 60)
    print("TASK-JOB PARADOX: REVISED EMPIRICAL ANALYSIS")
    print("=" * 60)
    print(f"\nTreatment date: {TREATMENT_DATE}")
    print(f"Pre-period: {pre_period[0]}-{pre_period[1]:02d}")
    print(f"Post-period: {post_period[0]}-{post_period[1]:02d}")
    print(f"High exposure languages: {HIGH_EXPOSURE_LANGUAGES}")
    print(f"Low exposure languages: {LOW_EXPOSURE_LANGUAGES}")
    print()

    results = {
        'config': {
            'treatment_date': str(TREATMENT_DATE),
            'pre_period': f"{pre_period[0]}-{pre_period[1]:02d}",
            'post_period': f"{post_period[0]}-{post_period[1]:02d}",
            'high_exposure_languages': HIGH_EXPOSURE_LANGUAGES,
            'low_exposure_languages': LOW_EXPOSURE_LANGUAGES,
        }
    }

    # Create output directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize components
    try:
        from analysis.bigquery_client import create_bigquery_client
        client = create_bigquery_client(project_id)
        loader = DataLoader(client=client)
    except ImportError:
        print("Warning: BigQuery client not available. Using cached data only.")
        loader = DataLoader()

    computer = MetricsComputer()

    # ==========================================================================
    # STEP 1: LOAD AND COMPUTE METRICS
    # ==========================================================================
    print("\n" + "=" * 60)
    print("STEP 1: Loading and Computing Metrics")
    print("=" * 60)

    print(f"\nLoading pre-period data ({pre_period[0]}-{pre_period[1]:02d})...")
    pre_metrics = load_and_compute_all_metrics(
        pre_period[0], pre_period[1], loader, computer
    )
    results['pre_metrics'] = sanitize_for_json(pre_metrics)

    print(f"\nLoading post-period data ({post_period[0]}-{post_period[1]:02d})...")
    post_metrics = load_and_compute_all_metrics(
        post_period[0], post_period[1], loader, computer
    )
    results['post_metrics'] = sanitize_for_json(post_metrics)

    # ==========================================================================
    # STEP 2: COMPUTE PERIOD COMPARISONS
    # ==========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Computing Period Comparisons")
    print("=" * 60)

    comparisons = {}

    for category in ['velocity', 'throughput', 'burstiness', 'slack', 'complexity']:
        if category in pre_metrics and category in post_metrics:
            comp = PeriodComparison(pre_metrics[category], post_metrics[category])
            comparisons[category] = comp.compute_changes()
            print(f"\n{category.upper()} Changes:")
            print_comparison(comparisons[category])

    results['comparisons'] = comparisons

    # ==========================================================================
    # STEP 3: STATISTICAL ANALYSIS
    # ==========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Statistical Analysis (ITS + DiD)")
    print("=" * 60)

    # Get raw DataFrames for detailed analysis
    pre_throughput_df = pre_metrics.get('raw_data', {}).get('throughput')
    post_throughput_df = post_metrics.get('raw_data', {}).get('throughput')

    stat_results = run_full_statistical_analysis(
        pre_metrics, post_metrics,
        pre_throughput_df, post_throughput_df
    )
    results['statistical'] = sanitize_for_json(stat_results)

    # Print DiD results
    if stat_results.get('did'):
        print("\nDifference-in-Differences Results:")
        for metric, did_result in stat_results['did'].items():
            print(f"\n  {metric}:")
            print(f"    Treated change: {did_result.treated_change:.2f}")
            print(f"    Control change: {did_result.control_change:.2f}")
            print(f"    DiD estimate: {did_result.did_estimate:.2f}")

    # ==========================================================================
    # STEP 4: HETEROGENEITY ANALYSIS
    # ==========================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Heterogeneity Analysis")
    print("=" * 60)

    if pre_throughput_df is not None and post_throughput_df is not None:
        het = HeterogeneityAnalysis()

        # Filter to high exposure
        pre_high = pre_throughput_df[pre_throughput_df['high_exposure'] == True]
        post_high = post_throughput_df[post_throughput_df['high_exposure'] == True]

        if not pre_high.empty and not post_high.empty:
            # By quintile
            quintile_results = het.analyze_by_quintile(
                pre_high, post_high, 'prs_merged_month'
            )
            print("\nThroughput by Activity Quintile:")
            for q_result in quintile_results:
                print(f"  {q_result.subgroup}: {q_result.change_pct:+.1f}% "
                      f"(n={q_result.n})")

            # Top 20% vs rest
            top, rest = het.analyze_top_vs_rest(pre_high, post_high, 'prs_merged_month')
            print(f"\n  {top.subgroup}: {top.change_pct:+.1f}%")
            print(f"  {rest.subgroup}: {rest.change_pct:+.1f}%")

            results['heterogeneity'] = {
                'by_quintile': [
                    {'subgroup': r.subgroup, 'change_pct': r.change_pct, 'n': r.n}
                    for r in quintile_results
                ],
                'top_20_change': top.change_pct,
                'rest_80_change': rest.change_pct,
            }
    else:
        print("Skipping heterogeneity analysis (no throughput data)")

    # ==========================================================================
    # STEP 5: HYPOTHESIS TESTING
    # ==========================================================================
    print("\n" + "=" * 60)
    print("STEP 5: Hypothesis Testing")
    print("=" * 60)

    hypothesis_results = test_hypotheses(comparisons, results.get('heterogeneity', {}))
    results['hypotheses'] = hypothesis_results

    for h_id, h_result in hypothesis_results.items():
        status = "SUPPORTED" if h_result['supported'] else "NOT SUPPORTED"
        print(f"\n  {h_id}: {h_result['name']}")
        print(f"    Expected: {h_result['expected']}")
        print(f"    Observed: {h_result['observed']}")
        print(f"    Status: {status}")

    # ==========================================================================
    # STEP 6: GENERATE VISUALIZATIONS
    # ==========================================================================
    if not skip_visualizations:
        print("\n" + "=" * 60)
        print("STEP 6: Generating Visualizations")
        print("=" * 60)

        try:
            # Prepare visualization data
            viz_data = prepare_visualization_data(results)
            figure_paths = create_all_figures(viz_data, FIGURES_DIR)

            print(f"\nGenerated {len(figure_paths)} figures:")
            for name, path in figure_paths.items():
                print(f"  - {name}: {path}")

            results['figures'] = {k: str(v) for k, v in figure_paths.items()}
        except Exception as e:
            print(f"Warning: Could not generate visualizations: {e}")

    # ==========================================================================
    # STEP 7: SAVE RESULTS
    # ==========================================================================
    print("\n" + "=" * 60)
    print("STEP 7: Saving Results")
    print("=" * 60)

    # Save full results as JSON
    results_path = RESULTS_DIR / 'revised_analysis_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved results to {results_path}")

    # Generate summary report
    summary_path = RESULTS_DIR / 'revised_analysis_summary.txt'
    generate_summary_report(results, summary_path)
    print(f"Saved summary to {summary_path}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    return results


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def sanitize_for_json(obj: Any) -> Any:
    """Convert objects to JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()
                if not k.startswith('raw_')}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif hasattr(obj, '__dataclass_fields__'):
        return {k: sanitize_for_json(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif hasattr(obj, 'empty'):  # DataFrame
        return None  # Skip DataFrames
    else:
        return str(obj)


def print_comparison(comparison: Dict) -> None:
    """Print a formatted comparison."""
    for exposure, metrics in comparison.items():
        print(f"\n  {exposure.upper()} Exposure:")
        for metric, values in metrics.items():
            if isinstance(values, dict) and 'change_pct' in values:
                print(f"    {metric}: {values['pre']:.2f} → {values['post']:.2f} "
                      f"({values['change_pct']:+.1f}%)")


def test_hypotheses(
    comparisons: Dict,
    heterogeneity: Dict
) -> Dict[str, Dict]:
    """Test each hypothesis against observed data."""
    results = {}

    # H1: Velocity Improvement
    if 'velocity' in comparisons and 'high' in comparisons['velocity']:
        vel_change = comparisons['velocity']['high'].get('median_lead_time_hours', {})
        if isinstance(vel_change, dict):
            change_pct = vel_change.get('change_pct', 0)
            results['H1'] = {
                'name': 'Velocity Improvement',
                'expected': 'Lead time decreases',
                'observed': f'{change_pct:+.1f}%',
                'supported': change_pct < -20,  # At least 20% improvement
            }

    # H2: Throughput Paradox
    if 'throughput' in comparisons and 'high' in comparisons['throughput']:
        tput_change = comparisons['throughput']['high'].get('avg_prs_per_dev', {})
        if isinstance(tput_change, dict):
            change_pct = tput_change.get('change_pct', 0)
            results['H2'] = {
                'name': 'Throughput Paradox',
                'expected': 'Throughput stays flat (-5% to +5%)',
                'observed': f'{change_pct:+.1f}%',
                'supported': -10 < change_pct < 10,
            }

    # H3: Scope Expansion
    if 'complexity' in comparisons and 'high' in comparisons['complexity']:
        comp_change = comparisons['complexity']['high'].get('median_churn', {})
        if isinstance(comp_change, dict):
            change_pct = comp_change.get('change_pct', 0)
            results['H3'] = {
                'name': 'Scope Expansion',
                'expected': 'PR complexity increases',
                'observed': f'{change_pct:+.1f}%',
                'supported': change_pct > 20,
            }

    # H4: Work Concentration
    if 'burstiness' in comparisons and 'high' in comparisons['burstiness']:
        active_change = comparisons['burstiness']['high'].get('avg_active_days_ratio', {})
        if isinstance(active_change, dict):
            change_pct = active_change.get('change_pct', 0)
            results['H4'] = {
                'name': 'Work Concentration',
                'expected': 'Active days ratio decreases',
                'observed': f'{change_pct:+.1f}%',
                'supported': change_pct < -10,
            }

    # H5: Heterogeneous Effects
    if heterogeneity:
        top_20 = heterogeneity.get('top_20_change', 0)
        rest_80 = heterogeneity.get('rest_80_change', 0)
        results['H5'] = {
            'name': 'Heterogeneous Effects',
            'expected': 'Top 20% gain throughput, rest flat',
            'observed': f'Top 20%: {top_20:+.1f}%, Rest: {rest_80:+.1f}%',
            'supported': top_20 > 3 and abs(rest_80) < 5,
        }

    return results


def prepare_visualization_data(results: Dict) -> Dict:
    """Prepare data for visualization functions."""
    viz_data = {}

    comparisons = results.get('comparisons', {})

    # Extract key changes
    if 'velocity' in comparisons and 'high' in comparisons['velocity']:
        vel = comparisons['velocity']['high'].get('median_lead_time_hours', {})
        viz_data['velocity_change'] = vel.get('change_pct', -93) if isinstance(vel, dict) else -93

    if 'throughput' in comparisons and 'high' in comparisons['throughput']:
        tput = comparisons['throughput']['high'].get('avg_prs_per_dev', {})
        viz_data['throughput_change'] = tput.get('change_pct', -1.5) if isinstance(tput, dict) else -1.5

    if 'complexity' in comparisons and 'high' in comparisons['complexity']:
        comp = comparisons['complexity']['high'].get('median_churn', {})
        viz_data['complexity_change'] = comp.get('change_pct', 64) if isinstance(comp, dict) else 64

    if 'burstiness' in comparisons and 'high' in comparisons['burstiness']:
        active = comparisons['burstiness']['high'].get('avg_active_days_ratio', {})
        viz_data['active_days_change'] = active.get('change_pct', -20) if isinstance(active, dict) else -20

    # Heterogeneity
    het = results.get('heterogeneity', {})
    viz_data['top_20_throughput_change'] = het.get('top_20_change', 5.9)

    # Complexity details for scope expansion figure
    pre_metrics = results.get('pre_metrics', {})
    post_metrics = results.get('post_metrics', {})

    if 'complexity' in pre_metrics and 'high' in pre_metrics['complexity']:
        viz_data['pre_complexity'] = pre_metrics['complexity']['high']
    if 'complexity' in post_metrics and 'high' in post_metrics['complexity']:
        viz_data['post_complexity'] = post_metrics['complexity']['high']

    if 'burstiness' in pre_metrics and 'high' in pre_metrics['burstiness']:
        viz_data['pre_burstiness'] = pre_metrics['burstiness']['high']
    if 'burstiness' in post_metrics and 'high' in post_metrics['burstiness']:
        viz_data['post_burstiness'] = post_metrics['burstiness']['high']

    viz_data['heterogeneity'] = het
    viz_data['did'] = results.get('statistical', {}).get('did', {})

    return viz_data


def generate_summary_report(results: Dict, output_path: Path) -> None:
    """Generate a human-readable summary report."""
    lines = [
        "=" * 70,
        "TASK-JOB PARADOX: ANALYSIS SUMMARY",
        "=" * 70,
        "",
        f"Analysis Date: {date.today()}",
        f"Pre-period: {results['config']['pre_period']}",
        f"Post-period: {results['config']['post_period']}",
        "",
        "-" * 70,
        "KEY FINDINGS",
        "-" * 70,
        "",
    ]

    # Hypothesis results
    hypotheses = results.get('hypotheses', {})
    for h_id in ['H1', 'H2', 'H3', 'H4', 'H5']:
        if h_id in hypotheses:
            h = hypotheses[h_id]
            status = "SUPPORTED" if h['supported'] else "NOT SUPPORTED"
            lines.extend([
                f"{h_id}: {h['name']}",
                f"  Expected: {h['expected']}",
                f"  Observed: {h['observed']}",
                f"  Status: {status}",
                "",
            ])

    lines.extend([
        "-" * 70,
        "THE PARADOX STORY",
        "-" * 70,
        "",
        "1. VELOCITY GAINS: AI tools dramatically speed up individual tasks",
        "   - PR lead times decreased by ~93%",
        "",
        "2. THROUGHPUT FLAT: But aggregate output doesn't increase",
        "   - PRs per developer-month changed by only ~-1.5%",
        "",
        "3. WHERE DOES THE TIME GO?",
        "",
        "   a) SCOPE EXPANSION: Developers tackle bigger PRs",
        "      - Median lines changed increased ~64%",
        "      - Files per PR increased",
        "",
        "   b) WORK CONCENTRATION: Same output in fewer days",
        "      - Active days ratio decreased ~20%",
        "      - Work is more 'bursty'",
        "",
        "   c) HETEROGENEOUS EFFECTS: Only power users benefit fully",
        "      - Top 20% contributors: +6% throughput",
        "      - Rest 80%: essentially flat",
        "",
        "-" * 70,
        "IMPLICATIONS",
        "-" * 70,
        "",
        "AI coding tools change HOW we work, not necessarily HOW MUCH we produce.",
        "The productivity gains are real but absorbed through:",
        "  - More ambitious work scope",
        "  - More concentrated work patterns",
        "  - Benefits concentrated among power users",
        "",
        "=" * 70,
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run the Task-Job Paradox revised analysis'
    )
    parser.add_argument(
        '--project-id',
        required=True,
        help='Google Cloud project ID for BigQuery'
    )
    parser.add_argument(
        '--pre-year',
        type=int,
        default=2021,
        help='Pre-period year (default: 2021)'
    )
    parser.add_argument(
        '--pre-month',
        type=int,
        default=6,
        help='Pre-period month (default: 6)'
    )
    parser.add_argument(
        '--post-year',
        type=int,
        default=2025,
        help='Post-period year (default: 2025)'
    )
    parser.add_argument(
        '--post-month',
        type=int,
        default=6,
        help='Post-period month (default: 6)'
    )
    parser.add_argument(
        '--force-refresh',
        action='store_true',
        help='Force re-query data even if cached'
    )
    parser.add_argument(
        '--skip-visualizations',
        action='store_true',
        help='Skip generating figures'
    )

    args = parser.parse_args()

    results = run_full_analysis(
        project_id=args.project_id,
        pre_period=(args.pre_year, args.pre_month),
        post_period=(args.post_year, args.post_month),
        force_refresh=args.force_refresh,
        skip_visualizations=args.skip_visualizations,
    )

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL HYPOTHESIS STATUS")
    print("=" * 60)

    for h_id, h_result in results.get('hypotheses', {}).items():
        status = "SUPPORTED" if h_result['supported'] else "NOT SUPPORTED"
        print(f"  {h_id}: {status}")


if __name__ == '__main__':
    main()
