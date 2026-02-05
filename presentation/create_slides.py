#!/usr/bin/env python3
"""
Generate presentation slides for the Task-Job Paradox research.
Creates PNG images for each slide.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from pathlib import Path
import matplotlib.image as mpimg

# Setup
SLIDES_DIR = Path(__file__).parent / 'slides'
SLIDES_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path(__file__).parent.parent / 'results' / 'panel_analysis'

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
TITLE_FONT = 36
SUBTITLE_FONT = 24
BODY_FONT = 20
SMALL_FONT = 16

# Colors
PRIMARY = '#2E86AB'      # Blue
SECONDARY = '#A23B72'    # Magenta
ACCENT = '#F18F01'       # Orange
DARK = '#1A1A2E'         # Dark blue
LIGHT = '#F5F5F5'        # Light gray


def create_slide(figsize=(16, 9)):
    """Create a slide with standard dimensions."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.set_facecolor('white')
    ax.axis('off')
    fig.patch.set_facecolor('white')
    return fig, ax


def add_footer(ax, slide_num, total=8):
    """Add footer with slide number."""
    ax.text(15.5, 0.3, f'{slide_num}/{total}', fontsize=12,
            ha='right', color='gray', alpha=0.7)


def slide_1_title():
    """Title slide."""
    fig, ax = create_slide()

    # Main title
    ax.text(8, 5.5, 'The Task-Job Paradox', fontsize=48, fontweight='bold',
            ha='center', va='center', color=DARK)

    # Subtitle
    ax.text(8, 4.2, 'Why AI Coding Tools Speed Up Tasks\nBut Not Total Output',
            fontsize=28, ha='center', va='center', color=PRIMARY, linespacing=1.5)

    # Author/date
    ax.text(8, 2.2, 'Empirical Evidence from 5 Years of GitHub Data\n(2021-2025)',
            fontsize=20, ha='center', va='center', color='gray')

    # Treatment indicator
    ax.text(8, 1.2, 'Natural Experiment: ChatGPT Launch (Nov 30, 2022)',
            fontsize=16, ha='center', va='center', color=ACCENT, style='italic')

    add_footer(ax, 1)
    fig.savefig(SLIDES_DIR / 'slide_01_title.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created slide 1: Title")


def slide_2_puzzle():
    """The puzzle/problem."""
    fig, ax = create_slide()

    # Title
    ax.text(8, 8, 'The Puzzle', fontsize=TITLE_FONT, fontweight='bold',
            ha='center', color=DARK)

    # Two boxes showing the paradox
    # Left box - Task level
    rect1 = mpatches.FancyBboxPatch((1, 3.5), 6, 3.5, boxstyle="round,pad=0.1",
                                     facecolor='#E8F4EA', edgecolor=PRIMARY, linewidth=2)
    ax.add_patch(rect1)
    ax.text(4, 6.2, 'Task Level', fontsize=22, fontweight='bold', ha='center', color=PRIMARY)
    ax.text(4, 5.2, 'AI tools make developers\n30-50% faster on individual tasks',
            fontsize=18, ha='center', va='center', linespacing=1.3)
    ax.text(4, 4.2, '✓', fontsize=36, ha='center', color='green')

    # Right box - Job level
    rect2 = mpatches.FancyBboxPatch((9, 3.5), 6, 3.5, boxstyle="round,pad=0.1",
                                     facecolor='#FDEAEA', edgecolor=SECONDARY, linewidth=2)
    ax.add_patch(rect2)
    ax.text(12, 6.2, 'Job Level', fontsize=22, fontweight='bold', ha='center', color=SECONDARY)
    ax.text(12, 5.2, 'Total output (PRs, commits)\nremains roughly constant',
            fontsize=18, ha='center', va='center', linespacing=1.3)
    ax.text(12, 4.2, '?', fontsize=36, ha='center', color=SECONDARY)

    # Arrow
    ax.annotate('', xy=(8.5, 5.25), xytext=(7.5, 5.25),
                arrowprops=dict(arrowstyle='->', color=ACCENT, lw=3))

    # Key question
    ax.text(8, 2, 'Where does the saved time go?',
            fontsize=24, ha='center', va='center', color=ACCENT, fontweight='bold')

    add_footer(ax, 2)
    fig.savefig(SLIDES_DIR / 'slide_02_puzzle.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created slide 2: Puzzle")


def slide_3_method():
    """Data and methodology."""
    fig, ax = create_slide()

    # Title
    ax.text(8, 8.2, 'Our Approach', fontsize=TITLE_FONT, fontweight='bold',
            ha='center', color=DARK)

    # Data section
    ax.text(4, 6.8, 'Data', fontsize=24, fontweight='bold', ha='center', color=PRIMARY)
    data_text = """• 53 months of GitHub Archive data
• 11.6M+ developer-month observations
• Velocity, Complexity, Throughput metrics
• High vs Low AI exposure languages"""
    ax.text(4, 5.2, data_text, fontsize=16, ha='center', va='center', linespacing=1.6)

    # Method section
    ax.text(12, 6.8, 'Causal Identification', fontsize=24, fontweight='bold',
            ha='center', color=SECONDARY)
    method_text = """• Treatment: ChatGPT launch (Nov 2022)
• Difference-in-Differences (DiD)
• Interrupted Time Series (ITS)
• Control: AI-resistant languages"""
    ax.text(12, 5.2, method_text, fontsize=16, ha='center', va='center', linespacing=1.6)

    # Language groups
    ax.text(4, 2.5, 'High Exposure', fontsize=18, fontweight='bold',
            ha='center', color=PRIMARY)
    ax.text(4, 1.8, 'Python, JavaScript, TypeScript, Java',
            fontsize=14, ha='center', color='gray')

    ax.text(12, 2.5, 'Low Exposure (Control)', fontsize=18, fontweight='bold',
            ha='center', color=SECONDARY)
    ax.text(12, 1.8, 'Fortran, COBOL, Assembly',
            fontsize=14, ha='center', color='gray')

    # Dividing line
    ax.axvline(x=8, ymin=0.2, ymax=0.85, color='lightgray', linestyle='--', alpha=0.5)

    add_footer(ax, 3)
    fig.savefig(SLIDES_DIR / 'slide_03_method.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created slide 3: Method")


def slide_4_velocity():
    """Finding 1: Velocity improvement."""
    fig = plt.figure(figsize=(16, 9))

    # Title area
    ax_title = fig.add_axes([0, 0.85, 1, 0.15])
    ax_title.set_xlim(0, 1)
    ax_title.set_ylim(0, 1)
    ax_title.axis('off')
    ax_title.text(0.5, 0.5, 'Finding 1: Tasks Got Faster',
                  fontsize=TITLE_FONT, fontweight='bold', ha='center', color=DARK)

    # Load and display the time series
    ax_img = fig.add_axes([0.05, 0.15, 0.55, 0.65])
    try:
        img = mpimg.imread(RESULTS_DIR / 'ts_velocity_p75.png')
        ax_img.imshow(img)
    except:
        ax_img.text(0.5, 0.5, '[Velocity Time Series]', ha='center', va='center')
    ax_img.axis('off')

    # Key stats on right
    ax_stats = fig.add_axes([0.62, 0.15, 0.35, 0.65])
    ax_stats.set_xlim(0, 1)
    ax_stats.set_ylim(0, 1)
    ax_stats.axis('off')

    ax_stats.text(0.5, 0.9, 'DiD Result', fontsize=22, fontweight='bold',
                  ha='center', color=PRIMARY)

    ax_stats.text(0.5, 0.7, '-4.62 hours', fontsize=36, fontweight='bold',
                  ha='center', color='green')
    ax_stats.text(0.5, 0.55, 'reduction in P75 lead time', fontsize=16, ha='center')
    ax_stats.text(0.5, 0.42, 'p = 0.0145', fontsize=18, ha='center',
                  color=SECONDARY, fontweight='bold')

    ax_stats.text(0.5, 0.2, 'High-exposure languages show\nsignificantly faster PR completion\nafter ChatGPT launch',
                  fontsize=14, ha='center', va='center', linespacing=1.4, style='italic')

    # Footer
    ax_footer = fig.add_axes([0, 0, 1, 0.08])
    ax_footer.set_xlim(0, 16)
    ax_footer.set_ylim(0, 1)
    ax_footer.axis('off')
    ax_footer.text(15.5, 0.5, '4/8', fontsize=12, ha='right', color='gray')

    fig.savefig(SLIDES_DIR / 'slide_04_velocity.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created slide 4: Velocity")


def slide_5_scope():
    """Finding 2: Scope expansion."""
    fig = plt.figure(figsize=(16, 9))

    # Title area
    ax_title = fig.add_axes([0, 0.85, 1, 0.15])
    ax_title.set_xlim(0, 1)
    ax_title.set_ylim(0, 1)
    ax_title.axis('off')
    ax_title.text(0.5, 0.5, 'Finding 2: PRs Got Bigger',
                  fontsize=TITLE_FONT, fontweight='bold', ha='center', color=DARK)

    # Load and display the complexity time series
    ax_img = fig.add_axes([0.05, 0.15, 0.55, 0.65])
    try:
        img = mpimg.imread(RESULTS_DIR / 'ts_complexity.png')
        ax_img.imshow(img)
    except:
        ax_img.text(0.5, 0.5, '[Complexity Time Series]', ha='center', va='center')
    ax_img.axis('off')

    # Key stats on right
    ax_stats = fig.add_axes([0.62, 0.15, 0.35, 0.65])
    ax_stats.set_xlim(0, 1)
    ax_stats.set_ylim(0, 1)
    ax_stats.axis('off')

    ax_stats.text(0.5, 0.9, 'DiD Result', fontsize=22, fontweight='bold',
                  ha='center', color=SECONDARY)

    ax_stats.text(0.5, 0.7, '+10.03 lines', fontsize=36, fontweight='bold',
                  ha='center', color=ACCENT)
    ax_stats.text(0.5, 0.55, 'increase in median PR size', fontsize=16, ha='center')
    ax_stats.text(0.5, 0.42, 'p < 0.0001', fontsize=18, ha='center',
                  color=SECONDARY, fontweight='bold')

    ax_stats.text(0.5, 0.2, 'Developers use saved time to\ntackle larger changes,\nnot to produce more PRs',
                  fontsize=14, ha='center', va='center', linespacing=1.4, style='italic')

    # Footer
    ax_footer = fig.add_axes([0, 0, 1, 0.08])
    ax_footer.set_xlim(0, 16)
    ax_footer.set_ylim(0, 1)
    ax_footer.axis('off')
    ax_footer.text(15.5, 0.5, '5/8', fontsize=12, ha='right', color='gray')

    fig.savefig(SLIDES_DIR / 'slide_05_scope.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created slide 5: Scope Expansion")


def slide_6_did():
    """DiD evidence visualization."""
    fig = plt.figure(figsize=(16, 9))

    # Title
    ax_title = fig.add_axes([0, 0.85, 1, 0.15])
    ax_title.set_xlim(0, 1)
    ax_title.set_ylim(0, 1)
    ax_title.axis('off')
    ax_title.text(0.5, 0.5, 'The Difference-in-Differences Evidence',
                  fontsize=TITLE_FONT, fontweight='bold', ha='center', color=DARK)

    # Two charts side by side
    ax_left = fig.add_axes([0.05, 0.15, 0.42, 0.65])
    try:
        img1 = mpimg.imread(RESULTS_DIR / 'did_velocity.png')
        ax_left.imshow(img1)
    except:
        ax_left.text(0.5, 0.5, '[DiD Velocity]', ha='center', va='center')
    ax_left.axis('off')
    ax_left.set_title('Velocity (Lead Time)', fontsize=18, pad=10)

    ax_right = fig.add_axes([0.52, 0.15, 0.42, 0.65])
    try:
        img2 = mpimg.imread(RESULTS_DIR / 'did_complexity.png')
        ax_right.imshow(img2)
    except:
        ax_right.text(0.5, 0.5, '[DiD Complexity]', ha='center', va='center')
    ax_right.axis('off')
    ax_right.set_title('Scope (PR Size)', fontsize=18, pad=10)

    # Footer
    ax_footer = fig.add_axes([0, 0, 1, 0.1])
    ax_footer.set_xlim(0, 16)
    ax_footer.set_ylim(0, 1)
    ax_footer.axis('off')
    ax_footer.text(8, 0.6, 'Control group (Fortran, COBOL, Assembly) shows no change - confirming causal effect',
                   fontsize=14, ha='center', style='italic', color='gray')
    ax_footer.text(15.5, 0.2, '6/8', fontsize=12, ha='right', color='gray')

    fig.savefig(SLIDES_DIR / 'slide_06_did.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created slide 6: DiD Evidence")


def slide_7_mechanism():
    """The mechanism / explanation."""
    fig, ax = create_slide()

    # Title
    ax.text(8, 8.2, 'The Mechanism', fontsize=TITLE_FONT, fontweight='bold',
            ha='center', color=DARK)

    # Flow diagram
    # Box 1: AI Tools
    rect1 = mpatches.FancyBboxPatch((1, 4.5), 3.5, 2, boxstyle="round,pad=0.1",
                                     facecolor='#E3F2FD', edgecolor=PRIMARY, linewidth=2)
    ax.add_patch(rect1)
    ax.text(2.75, 5.5, 'AI Coding\nTools', fontsize=18, ha='center', va='center',
            fontweight='bold', color=PRIMARY, linespacing=1.3)

    # Arrow 1
    ax.annotate('', xy=(5, 5.5), xytext=(4.7, 5.5),
                arrowprops=dict(arrowstyle='->', color=DARK, lw=2))

    # Box 2: Faster Tasks
    rect2 = mpatches.FancyBboxPatch((5.2, 4.5), 3.5, 2, boxstyle="round,pad=0.1",
                                     facecolor='#E8F5E9', edgecolor='green', linewidth=2)
    ax.add_patch(rect2)
    ax.text(6.95, 5.5, 'Tasks\n30-50% Faster', fontsize=18, ha='center', va='center',
            fontweight='bold', color='green', linespacing=1.3)

    # Arrow 2
    ax.annotate('', xy=(9.2, 5.5), xytext=(8.9, 5.5),
                arrowprops=dict(arrowstyle='->', color=DARK, lw=2))

    # Box 3: Scope Expansion
    rect3 = mpatches.FancyBboxPatch((9.5, 4.5), 3.5, 2, boxstyle="round,pad=0.1",
                                     facecolor='#FFF3E0', edgecolor=ACCENT, linewidth=2)
    ax.add_patch(rect3)
    ax.text(11.25, 5.5, 'Scope\nExpansion', fontsize=18, ha='center', va='center',
            fontweight='bold', color=ACCENT, linespacing=1.3)

    # Arrow 3
    ax.annotate('', xy=(13.5, 5.5), xytext=(13.2, 5.5),
                arrowprops=dict(arrowstyle='->', color=DARK, lw=2))

    # Box 4: Same Output
    rect4 = mpatches.FancyBboxPatch((13.7, 4.5), 2, 2, boxstyle="round,pad=0.1",
                                     facecolor='#FCE4EC', edgecolor=SECONDARY, linewidth=2)
    ax.add_patch(rect4)
    ax.text(14.7, 5.5, 'Same\nOutput', fontsize=18, ha='center', va='center',
            fontweight='bold', color=SECONDARY, linespacing=1.3)

    # Explanation bullets
    bullets = [
        "Developers don't produce more PRs - they make bigger ones",
        "Time saved on coding is absorbed by increased ambition",
        "Quality/scope preferences adjust to available speed"
    ]
    for i, bullet in enumerate(bullets):
        ax.text(8, 2.8 - i*0.7, f"• {bullet}", fontsize=16, ha='center')

    add_footer(ax, 7)
    fig.savefig(SLIDES_DIR / 'slide_07_mechanism.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created slide 7: Mechanism")


def slide_8_conclusion():
    """Conclusion slide."""
    fig, ax = create_slide()

    # Main takeaway
    ax.text(8, 6.5, 'AI changes HOW we work,', fontsize=36, fontweight='bold',
            ha='center', color=DARK)
    ax.text(8, 5.3, 'not HOW MUCH we produce', fontsize=36, fontweight='bold',
            ha='center', color=PRIMARY)

    # Implications
    ax.text(8, 3.5, 'Implications:', fontsize=22, fontweight='bold',
            ha='center', color=SECONDARY)

    implications = [
        "Productivity metrics need to account for scope changes",
        "AI tools enable ambition more than efficiency",
        "Job-level output may not be the right measure of value"
    ]
    for i, imp in enumerate(implications):
        ax.text(8, 2.7 - i*0.6, f"• {imp}", fontsize=16, ha='center')

    # Meta note
    ax.text(8, 0.8, 'Research conducted with AI assistance - experiencing the paradox firsthand',
            fontsize=14, ha='center', color='gray', style='italic')

    add_footer(ax, 8)
    fig.savefig(SLIDES_DIR / 'slide_08_conclusion.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created slide 8: Conclusion")


def main():
    """Generate all slides."""
    print("=" * 50)
    print("GENERATING PRESENTATION SLIDES")
    print("=" * 50)

    slide_1_title()
    slide_2_puzzle()
    slide_3_method()
    slide_4_velocity()
    slide_5_scope()
    slide_6_did()
    slide_7_mechanism()
    slide_8_conclusion()

    print("\n" + "=" * 50)
    print(f"All slides saved to: {SLIDES_DIR}")
    print("=" * 50)


if __name__ == '__main__':
    main()
