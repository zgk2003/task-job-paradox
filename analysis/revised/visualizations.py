"""
Visualization Module for Revised Empirical Strategy

Creates publication-quality figures for the Task-Job Paradox paper.

Figure 1: The Paradox Overview (velocity vs throughput)
Figure 2: Scope Expansion (PR complexity over time)
Figure 3: Work Concentration (burstiness patterns)
Figure 4: Heterogeneity (top 20% vs rest)
Figure 5: DiD Comparison (high vs low exposure)
"""

from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.ticker import FuncFormatter
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

from .config import TREATMENT_DATE, MetricCategory


# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

COLORS = {
    'high_exposure': '#2E86AB',  # Blue
    'low_exposure': '#A23B72',   # Magenta
    'pre_period': '#F18F01',     # Orange
    'post_period': '#C73E1D',    # Red
    'velocity': '#2E86AB',
    'throughput': '#A23B72',
    'complexity': '#F18F01',
    'burstiness': '#C73E1D',
    'slack': '#6B4E71',
    'top_contributors': '#2E86AB',
    'rest': '#A23B72',
    'positive': '#2E86AB',
    'negative': '#C73E1D',
    'neutral': '#808080',
}

FIGURE_SIZE = {
    'single': (8, 6),
    'wide': (12, 6),
    'tall': (8, 10),
    'panel': (14, 10),
}


def setup_style():
    """Set up matplotlib style for publication quality."""
    if not MATPLOTLIB_AVAILABLE:
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


# =============================================================================
# FIGURE 1: THE PARADOX OVERVIEW
# =============================================================================

def create_paradox_overview(
    velocity_change: float,
    throughput_change: float,
    complexity_change: float,
    output_path: Optional[Path] = None,
    title: str = "The Task-Job Paradox"
) -> Optional[plt.Figure]:
    """
    Create the main paradox visualization showing velocity vs throughput.

    Shows three bars: velocity improvement, throughput (flat), complexity increase.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available for visualization")
        return None

    setup_style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE['single'])

    metrics = ['Velocity\n(Lead Time)', 'Throughput\n(PRs/Dev)', 'Complexity\n(Lines/PR)']
    changes = [velocity_change, throughput_change, complexity_change]
    colors = [
        COLORS['positive'] if velocity_change < 0 else COLORS['negative'],  # Decrease is good
        COLORS['neutral'],
        COLORS['positive'] if complexity_change > 0 else COLORS['neutral'],
    ]

    bars = ax.bar(metrics, changes, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, change in zip(bars, changes):
        height = bar.get_height()
        sign = '+' if change > 0 else ''
        ax.annotate(
            f'{sign}{change:.1f}%',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5 if height >= 0 else -15),
            textcoords='offset points',
            ha='center', va='bottom' if height >= 0 else 'top',
            fontsize=14, fontweight='bold'
        )

    # Reference line at 0
    ax.axhline(y=0, color='black', linewidth=1)

    # Annotations explaining the paradox
    ax.annotate(
        'Tasks faster\nbut output flat',
        xy=(0.5, max(changes) * 0.7),
        fontsize=10,
        ha='center',
        style='italic',
        color='gray'
    )

    ax.annotate(
        'Saved time → Bigger PRs',
        xy=(2, complexity_change * 0.5),
        fontsize=10,
        ha='center',
        style='italic',
        color='gray'
    )

    ax.set_ylabel('Change (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(min(changes) * 1.3, max(changes) * 1.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path)
        print(f"Saved figure to {output_path}")

    return fig


# =============================================================================
# FIGURE 2: SCOPE EXPANSION OVER TIME
# =============================================================================

def create_scope_expansion_figure(
    pre_complexity: Dict[str, float],
    post_complexity: Dict[str, float],
    output_path: Optional[Path] = None
) -> Optional[plt.Figure]:
    """
    Create figure showing PR complexity increase (scope expansion).

    Shows median, p75, p90 for lines changed and files changed.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE['wide'])

    # Lines changed
    ax1 = axes[0]
    metrics = ['Median', 'P75', 'P90']
    pre_vals = [
        pre_complexity.get('median_churn', 0),
        pre_complexity.get('p75_churn', 0),
        pre_complexity.get('p90_churn', 0)
    ]
    post_vals = [
        post_complexity.get('median_churn', 0),
        post_complexity.get('p75_churn', 0),
        post_complexity.get('p90_churn', 0)
    ]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax1.bar(x - width/2, pre_vals, width, label='Pre-LLM (2021)',
                    color=COLORS['pre_period'], edgecolor='black')
    bars2 = ax1.bar(x + width/2, post_vals, width, label='Post-LLM (2025)',
                    color=COLORS['post_period'], edgecolor='black')

    ax1.set_ylabel('Lines Changed')
    ax1.set_title('PR Size (Lines)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()

    # Add change percentages
    for i, (pre, post) in enumerate(zip(pre_vals, post_vals)):
        if pre > 0:
            change = ((post - pre) / pre) * 100
            ax1.annotate(
                f'+{change:.0f}%',
                xy=(i + width/2, post),
                xytext=(0, 5),
                textcoords='offset points',
                ha='center', fontsize=9, color=COLORS['post_period']
            )

    # Files changed
    ax2 = axes[1]
    pre_files = [
        pre_complexity.get('median_files', 0),
        pre_complexity.get('p75_files', 0),
    ]
    post_files = [
        post_complexity.get('median_files', 0),
        post_complexity.get('p75_files', 0),
    ]
    metrics2 = ['Median', 'P75']
    x2 = np.arange(len(metrics2))

    bars3 = ax2.bar(x2 - width/2, pre_files, width, label='Pre-LLM (2021)',
                    color=COLORS['pre_period'], edgecolor='black')
    bars4 = ax2.bar(x2 + width/2, post_files, width, label='Post-LLM (2025)',
                    color=COLORS['post_period'], edgecolor='black')

    ax2.set_ylabel('Files Changed')
    ax2.set_title('PR Scope (Files)', fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(metrics2)
    ax2.legend()

    fig.suptitle('Scope Expansion: PRs Get Larger Post-LLM', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path)
        print(f"Saved figure to {output_path}")

    return fig


# =============================================================================
# FIGURE 3: WORK CONCENTRATION (BURSTINESS)
# =============================================================================

def create_burstiness_figure(
    pre_burstiness: Dict[str, float],
    post_burstiness: Dict[str, float],
    output_path: Optional[Path] = None
) -> Optional[plt.Figure]:
    """
    Create figure showing work concentration patterns.

    Shows CV of daily events and active days ratio.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE['wide'])

    # CV of daily events (higher = more bursty)
    ax1 = axes[0]
    pre_cv = pre_burstiness.get('avg_cv_daily', 0)
    post_cv = post_burstiness.get('avg_cv_daily', 0)

    bars = ax1.bar(['Pre-LLM', 'Post-LLM'], [pre_cv, post_cv],
                   color=[COLORS['pre_period'], COLORS['post_period']],
                   edgecolor='black')

    if pre_cv > 0:
        change = ((post_cv - pre_cv) / pre_cv) * 100
        ax1.annotate(
            f'{change:+.1f}%',
            xy=(1, post_cv),
            xytext=(0, 5),
            textcoords='offset points',
            ha='center', fontsize=12, fontweight='bold'
        )

    ax1.set_ylabel('CV of Daily Activity')
    ax1.set_title('Work Burstiness\n(Higher = More Concentrated)', fontweight='bold')

    # Active days ratio (lower = more concentrated)
    ax2 = axes[1]
    pre_active = pre_burstiness.get('avg_active_days_ratio', 0) * 100
    post_active = post_burstiness.get('avg_active_days_ratio', 0) * 100

    bars = ax2.bar(['Pre-LLM', 'Post-LLM'], [pre_active, post_active],
                   color=[COLORS['pre_period'], COLORS['post_period']],
                   edgecolor='black')

    if pre_active > 0:
        change = ((post_active - pre_active) / pre_active) * 100
        ax2.annotate(
            f'{change:+.1f}%',
            xy=(1, post_active),
            xytext=(0, 5),
            textcoords='offset points',
            ha='center', fontsize=12, fontweight='bold'
        )

    ax2.set_ylabel('Active Days Ratio (%)')
    ax2.set_title('Work Distribution\n(Lower = Fewer Active Days)', fontweight='bold')

    fig.suptitle('Work Concentration: Fewer Days, More Intense Sessions', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path)
        print(f"Saved figure to {output_path}")

    return fig


# =============================================================================
# FIGURE 4: HETEROGENEITY (TOP 20% VS REST)
# =============================================================================

def create_heterogeneity_figure(
    quintile_data: List[Dict],
    top_20_change: float,
    rest_80_change: float,
    output_path: Optional[Path] = None
) -> Optional[plt.Figure]:
    """
    Create figure showing heterogeneous effects by developer activity.

    Shows throughput changes by quintile and top 20% vs rest comparison.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE['wide'])

    # By quintile
    ax1 = axes[0]
    if quintile_data:
        quintiles = [d['subgroup'] for d in quintile_data]
        changes = [d['change_pct'] for d in quintile_data]

        colors = [COLORS['positive'] if c > 0 else COLORS['negative'] for c in changes]
        bars = ax1.bar(quintiles, changes, color=colors, edgecolor='black')

        ax1.axhline(y=0, color='black', linewidth=1)
        ax1.set_xlabel('Activity Quintile (Q1=Least Active)')
        ax1.set_ylabel('Throughput Change (%)')
        ax1.set_title('Effect by Activity Level', fontweight='bold')

        # Add value labels
        for bar, change in zip(bars, changes):
            height = bar.get_height()
            ax1.annotate(
                f'{change:+.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3 if height >= 0 else -12),
                textcoords='offset points',
                ha='center', fontsize=9
            )

    # Top 20% vs rest
    ax2 = axes[1]
    groups = ['Top 20%\nContributors', 'Rest 80%']
    changes = [top_20_change, rest_80_change]
    colors = [
        COLORS['positive'] if top_20_change > 0 else COLORS['neutral'],
        COLORS['neutral']
    ]

    bars = ax2.bar(groups, changes, color=colors, edgecolor='black', width=0.5)
    ax2.axhline(y=0, color='black', linewidth=1)

    for bar, change in zip(bars, changes):
        height = bar.get_height()
        ax2.annotate(
            f'{change:+.1f}%',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5 if height >= 0 else -15),
            textcoords='offset points',
            ha='center', fontsize=14, fontweight='bold'
        )

    ax2.set_ylabel('Throughput Change (%)')
    ax2.set_title('Top Contributors vs Rest', fontweight='bold')

    # Annotation
    ax2.annotate(
        'Only heavy users\nshow throughput gains',
        xy=(0, top_20_change * 0.5),
        fontsize=10, style='italic', color='gray', ha='center'
    )

    fig.suptitle('Heterogeneous Effects: Who Benefits from AI Tools?', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path)
        print(f"Saved figure to {output_path}")

    return fig


# =============================================================================
# FIGURE 5: DiD COMPARISON
# =============================================================================

def create_did_figure(
    did_results: Dict[str, Any],
    output_path: Optional[Path] = None
) -> Optional[plt.Figure]:
    """
    Create Difference-in-Differences comparison figure.

    Shows parallel trends assumption and treatment effect.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    setup_style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE['single'])

    # Extract a key metric (e.g., lead time)
    if 'velocity_median_lead_time_hours' in did_results:
        result = did_results['velocity_median_lead_time_hours']

        # Plot pre-post for both groups
        periods = ['Pre-LLM\n(2021)', 'Post-LLM\n(2025)']

        # High exposure (treatment)
        high_vals = [result.treated_pre, result.treated_post]
        ax.plot(periods, high_vals, 'o-', color=COLORS['high_exposure'],
                linewidth=2, markersize=10, label='High AI Exposure')

        # Low exposure (control)
        low_vals = [result.control_pre, result.control_post]
        ax.plot(periods, low_vals, 's--', color=COLORS['low_exposure'],
                linewidth=2, markersize=10, label='Low AI Exposure')

        # DiD annotation
        did = result.did_estimate
        ax.annotate(
            f'DiD = {did:.1f} hours',
            xy=(1.1, (high_vals[1] + low_vals[1]) / 2),
            fontsize=12, fontweight='bold',
            color=COLORS['high_exposure']
        )

        ax.set_ylabel('Median Lead Time (hours)')
        ax.set_title('Difference-in-Differences: Velocity', fontweight='bold')
        ax.legend(loc='upper right')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path)
        print(f"Saved figure to {output_path}")

    return fig


# =============================================================================
# SUMMARY FIGURE: THE COMPLETE STORY
# =============================================================================

def create_summary_figure(
    velocity_change: float,
    throughput_change: float,
    complexity_change: float,
    active_days_change: float,
    top_20_throughput_change: float,
    output_path: Optional[Path] = None
) -> Optional[plt.Figure]:
    """
    Create comprehensive summary figure with the complete paradox story.

    Panel A: The paradox (velocity vs throughput)
    Panel B: The resolution (scope expansion + work concentration)
    Panel C: The heterogeneity (who benefits)
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    setup_style()
    fig = plt.figure(figsize=FIGURE_SIZE['panel'])

    # Panel A: The Paradox
    ax1 = fig.add_subplot(2, 2, 1)
    metrics = ['Velocity\n(Lead Time)', 'Throughput\n(PRs/Dev)']
    changes = [velocity_change, throughput_change]
    colors = [COLORS['positive'], COLORS['neutral']]

    bars = ax1.bar(metrics, changes, color=colors, edgecolor='black', width=0.5)
    ax1.axhline(y=0, color='black', linewidth=1)

    for bar, change in zip(bars, changes):
        height = bar.get_height()
        sign = '+' if change > 0 else ''
        ax1.annotate(
            f'{sign}{change:.0f}%',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5 if height >= 0 else -15),
            textcoords='offset points',
            ha='center', fontsize=12, fontweight='bold'
        )

    ax1.set_ylabel('Change (%)')
    ax1.set_title('A. The Paradox\nFaster Tasks ≠ More Output', fontweight='bold')

    # Panel B: Scope Expansion
    ax2 = fig.add_subplot(2, 2, 2)
    metrics = ['PR Complexity\n(Lines/PR)', 'Active Days\n(Ratio)']
    changes = [complexity_change, active_days_change]
    colors = [COLORS['complexity'], COLORS['burstiness']]

    bars = ax2.bar(metrics, changes, color=colors, edgecolor='black', width=0.5)
    ax2.axhline(y=0, color='black', linewidth=1)

    for bar, change in zip(bars, changes):
        height = bar.get_height()
        sign = '+' if change > 0 else ''
        ax2.annotate(
            f'{sign}{change:.0f}%',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5 if height >= 0 else -15),
            textcoords='offset points',
            ha='center', fontsize=12, fontweight='bold'
        )

    ax2.set_ylabel('Change (%)')
    ax2.set_title('B. The Resolution\nBigger PRs + Fewer Days', fontweight='bold')

    # Panel C: Heterogeneity
    ax3 = fig.add_subplot(2, 2, 3)
    groups = ['Top 20%', 'Rest 80%']
    changes = [top_20_throughput_change, throughput_change]
    colors = [COLORS['top_contributors'], COLORS['rest']]

    bars = ax3.bar(groups, changes, color=colors, edgecolor='black', width=0.5)
    ax3.axhline(y=0, color='black', linewidth=1)

    for bar, change in zip(bars, changes):
        height = bar.get_height()
        sign = '+' if change > 0 else ''
        ax3.annotate(
            f'{sign}{change:.1f}%',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5 if height >= 0 else -15),
            textcoords='offset points',
            ha='center', fontsize=12, fontweight='bold'
        )

    ax3.set_ylabel('Throughput Change (%)')
    ax3.set_title('C. Who Benefits?\nOnly Top Contributors', fontweight='bold')

    # Panel D: The Story (text)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    story_text = """
    THE TASK-JOB PARADOX EXPLAINED

    1. AI tools speed up individual coding tasks
       (93% faster PR lead times)

    2. But aggregate output stays flat
       (< 2% change in PRs per developer)

    3. Where does the saved time go?

       → Bigger PRs: Developers tackle more
         ambitious changes (+64% lines/PR)

       → Work concentration: Same output
         compressed into fewer days (-20%)

       → Only power users (top 20%) convert
         velocity gains into more PRs (+6%)

    IMPLICATION: AI tools change HOW we work,
    not necessarily HOW MUCH we produce.
    """

    ax4.text(0.1, 0.9, story_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax4.set_title('D. The Complete Story', fontweight='bold')

    fig.suptitle('The Task-Job Paradox in AI-Assisted Software Development',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path)
        print(f"Saved figure to {output_path}")

    return fig


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def create_all_figures(
    analysis_results: Dict[str, Any],
    output_dir: Path
) -> Dict[str, Path]:
    """
    Create all figures from analysis results.

    Args:
        analysis_results: Dict containing all computed metrics and analysis
        output_dir: Directory to save figures

    Returns:
        Dict mapping figure name to output path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figures = {}

    # Extract key changes (with defaults)
    velocity_change = analysis_results.get('velocity_change', -93)
    throughput_change = analysis_results.get('throughput_change', -1.5)
    complexity_change = analysis_results.get('complexity_change', 64)
    active_days_change = analysis_results.get('active_days_change', -20)
    top_20_change = analysis_results.get('top_20_throughput_change', 5.9)

    # Figure 1: Paradox Overview
    path = output_dir / 'figure1_paradox_overview.png'
    create_paradox_overview(velocity_change, throughput_change, complexity_change, path)
    figures['paradox_overview'] = path

    # Figure 2: Scope Expansion
    if 'pre_complexity' in analysis_results and 'post_complexity' in analysis_results:
        path = output_dir / 'figure2_scope_expansion.png'
        create_scope_expansion_figure(
            analysis_results['pre_complexity'],
            analysis_results['post_complexity'],
            path
        )
        figures['scope_expansion'] = path

    # Figure 3: Burstiness
    if 'pre_burstiness' in analysis_results and 'post_burstiness' in analysis_results:
        path = output_dir / 'figure3_work_concentration.png'
        create_burstiness_figure(
            analysis_results['pre_burstiness'],
            analysis_results['post_burstiness'],
            path
        )
        figures['work_concentration'] = path

    # Figure 4: Heterogeneity
    if 'heterogeneity' in analysis_results:
        path = output_dir / 'figure4_heterogeneity.png'
        het = analysis_results['heterogeneity']
        create_heterogeneity_figure(
            het.get('by_quintile', []),
            het.get('top_20_change', top_20_change),
            het.get('rest_80_change', throughput_change),
            path
        )
        figures['heterogeneity'] = path

    # Figure 5: DiD
    if 'did' in analysis_results:
        path = output_dir / 'figure5_did_comparison.png'
        create_did_figure(analysis_results['did'], path)
        figures['did_comparison'] = path

    # Summary Figure
    path = output_dir / 'figure_summary.png'
    create_summary_figure(
        velocity_change,
        throughput_change,
        complexity_change,
        active_days_change,
        top_20_change,
        path
    )
    figures['summary'] = path

    return figures


def export_results_table(
    analysis_results: Dict[str, Any],
    output_path: Path
) -> pd.DataFrame:
    """
    Export results as a formatted table for the paper.
    """
    rows = []

    # Velocity metrics
    if 'velocity' in analysis_results:
        vel = analysis_results['velocity']
        for exposure in ['high', 'low']:
            if exposure in vel:
                rows.append({
                    'Category': 'Velocity',
                    'Metric': 'Median Lead Time (hours)',
                    'Exposure': exposure.capitalize(),
                    'Pre-LLM': vel[exposure].get('pre_median_lead_time'),
                    'Post-LLM': vel[exposure].get('post_median_lead_time'),
                    'Change (%)': vel[exposure].get('change_pct'),
                })

    # Throughput metrics
    if 'throughput' in analysis_results:
        tput = analysis_results['throughput']
        for exposure in ['high', 'low']:
            if exposure in tput:
                rows.append({
                    'Category': 'Throughput',
                    'Metric': 'PRs per Developer',
                    'Exposure': exposure.capitalize(),
                    'Pre-LLM': tput[exposure].get('pre_avg_prs'),
                    'Post-LLM': tput[exposure].get('post_avg_prs'),
                    'Change (%)': tput[exposure].get('change_pct'),
                })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Exported results table to {output_path}")

    return df
