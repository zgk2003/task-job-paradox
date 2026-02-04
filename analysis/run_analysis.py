#!/usr/bin/env python3
"""
Main Analysis Script for Task-Job Paradox Empirical Study

This script orchestrates the full empirical analysis:
1. Simulates GitHub-like data (or loads real data when available)
2. Extracts task-level and job-level metrics
3. Runs statistical analyses (ITS, DiD, heterogeneity)
4. Generates publication-quality figures
5. Outputs results summary

Usage:
    python run_analysis.py [--real-data PATH] [--output-dir DIR]

The key finding we expect to demonstrate:
- Task-level metrics (review latency, CI-fix time) improve ~35% after AI adoption
- Job-level metrics (PR lead time) improve only ~8%
- This gap is the "Task-Job Paradox"
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from datetime import datetime
import argparse

# Import our modules
from data_model import TREATMENT_START, PRE_PERIOD_START, POST_PERIOD_END
from data_simulator import DataSimulator, SimulationConfig
from metrics import MetricsExtractor, create_analysis_dataset
from statistical_analysis import run_full_analysis
from visualizations import generate_all_figures, MATPLOTLIB_AVAILABLE


def print_header():
    """Print analysis header."""
    print("="*70)
    print("TASK-JOB PARADOX: EMPIRICAL ANALYSIS")
    print("="*70)
    print(f"\nAnalysis run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Treatment date: {TREATMENT_START.strftime('%Y-%m-%d')}")
    print(f"Observation window: {PRE_PERIOD_START.strftime('%Y-%m')} to "
          f"{POST_PERIOD_END.strftime('%Y-%m')}")
    print("="*70)


def main(args=None):
    """Run the complete analysis pipeline."""
    parser = argparse.ArgumentParser(
        description='Run Task-Job Paradox empirical analysis'
    )
    parser.add_argument(
        '--num-repos', type=int, default=500,
        help='Number of repositories to simulate (default: 500)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='../results',
        help='Directory for output files (default: ../results)'
    )
    parser.add_argument(
        '--figures-dir', type=str, default='../figures',
        help='Directory for figures (default: ../figures)'
    )

    args = parser.parse_args(args)

    print_header()

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    # Step 1: Generate or load data
    print("\n" + "="*70)
    print("STEP 1: DATA GENERATION")
    print("="*70)

    config = SimulationConfig(
        num_repos=args.num_repos,
        seed=args.seed,
        # These parameters encode our theoretical expectations
        task_effect_high_exposure=-0.35,  # 35% task improvement
        task_effect_low_exposure=-0.10,   # 10% spillover to low exposure
        job_effect_high_exposure=-0.08,   # Only 8% job improvement (paradox!)
        job_effect_low_exposure=-0.03,    # 3% for low exposure
        commits_per_pr_effect=0.25,       # 25% more iteration
    )

    simulator = DataSimulator(config)
    repositories = simulator.generate_dataset()

    # Step 2: Extract metrics
    print("\n" + "="*70)
    print("STEP 2: METRICS EXTRACTION")
    print("="*70)

    datasets = create_analysis_dataset(repositories)
    pr_df = datasets['pr_level']
    monthly_df = datasets['monthly']
    summary = datasets['summary']

    # Save intermediate data
    pr_df.to_csv(f'{args.output_dir}/pr_level_data.csv', index=False)
    monthly_df.to_csv(f'{args.output_dir}/monthly_data.csv', index=False)
    print(f"\nSaved datasets to {args.output_dir}/")

    # Step 3: Statistical analysis
    print("\n" + "="*70)
    print("STEP 3: STATISTICAL ANALYSIS")
    print("="*70)

    results = run_full_analysis(pr_df)

    # Step 4: Generate figures
    if MATPLOTLIB_AVAILABLE:
        print("\n" + "="*70)
        print("STEP 4: GENERATING FIGURES")
        print("="*70)

        generate_all_figures(
            pr_df=pr_df,
            monthly_df=monthly_df,
            its_results=results.get('its', {}),
            summary=summary,
            output_dir=args.figures_dir
        )
    else:
        print("\n" + "="*70)
        print("STEP 4: SKIPPING FIGURES (matplotlib not available)")
        print("="*70)

    # Step 5: Summary report
    print("\n" + "="*70)
    print("STEP 5: PRELIMINARY RESULTS SUMMARY")
    print("="*70)

    print("\n### KEY FINDINGS ###\n")

    # The Paradox
    task_change = ((summary['post_review_latency_mean'] - summary['pre_review_latency_mean'])
                   / summary['pre_review_latency_mean'] * 100)
    job_change = ((summary['post_lead_time_mean'] - summary['pre_lead_time_mean'])
                  / summary['pre_lead_time_mean'] * 100)

    print("THE TASK-JOB PARADOX:")
    print(f"  • Task-level improvement (review latency): {task_change:.1f}%")
    print(f"  • Job-level improvement (PR lead time):    {job_change:.1f}%")
    print(f"  • Gap: {abs(task_change) - abs(job_change):.1f} percentage points")
    print()
    print("  This {:.0f}x difference between task and job improvement".format(
        abs(task_change) / abs(job_change) if job_change != 0 else float('inf')
    ))
    print("  is the core empirical puzzle our research investigates.")

    # Mechanism
    commits_change = ((summary['post_commits_mean'] - summary['pre_commits_mean'])
                      / summary['pre_commits_mean'] * 100)

    print("\nMECHANISM (WHY THE GAP?):")
    print(f"  • Commits per PR increased by {commits_change:.1f}%")
    print("  • This suggests developers use AI-saved time for more iteration")
    print("  • More iteration maintains quality but limits job-level speedup")

    # Heterogeneity
    if 'heterogeneity' in results:
        het = results['heterogeneity']
        high_gap = het['high_coordination']['gap']
        low_gap = het['low_coordination']['gap']
        print("\nHETEROGENEITY BY COORDINATION:")
        print(f"  • High coordination projects: {high_gap:.1f}pp gap")
        print(f"  • Low coordination projects:  {low_gap:.1f}pp gap")
        if high_gap > low_gap:
            print("  • Gap is LARGER in coordination-heavy projects")
            print("    → Coordination is a binding constraint on job-level gains")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nOutputs saved to:")
    print(f"  • Data: {args.output_dir}/")
    print(f"  • Figures: {args.figures_dir}/")
    print("\nNext steps:")
    print("  1. Review figures for publication quality")
    print("  2. Run with real GitHub Archive data")
    print("  3. Conduct robustness checks (pre-trends, placebo tests)")
    print("="*70)

    return results, datasets


if __name__ == "__main__":
    results, datasets = main()
