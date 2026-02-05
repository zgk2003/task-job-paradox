"""
Configuration and Constants for Revised Empirical Strategy

This module defines:
- Language exposure groups (treatment vs control)
- Treatment date (ChatGPT launch)
- Metric definitions and expected directions
- Multi-granularity specifications
"""

from datetime import date
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# LANGUAGE GROUPS (DiD Treatment vs Control)
# =============================================================================

# High AI exposure: Languages where Copilot/LLMs perform well (61-80% accuracy)
HIGH_EXPOSURE_LANGUAGES: List[str] = [
    'Python',
    'JavaScript',
    'Java',
    'TypeScript'
]

# Low AI exposure: Languages where Copilot/LLMs perform poorly (~30% accuracy)
LOW_EXPOSURE_LANGUAGES: List[str] = [
    'Fortran',
    'COBOL',
    'Assembly',
    'Erlang',
    'Haskell'
]

ALL_LANGUAGES: List[str] = HIGH_EXPOSURE_LANGUAGES + LOW_EXPOSURE_LANGUAGES


# =============================================================================
# TREATMENT DATE
# =============================================================================

# ChatGPT public launch - sharp discontinuity for ITS
TREATMENT_DATE: date = date(2022, 11, 30)
TREATMENT_DATE_STR: str = TREATMENT_DATE.strftime('%Y-%m-%d')

# Copilot general availability (alternative treatment date)
COPILOT_GA_DATE: date = date(2022, 6, 21)


# =============================================================================
# METRIC DEFINITIONS
# =============================================================================

class MetricCategory(Enum):
    """Categories of metrics in the revised strategy."""
    VELOCITY = "velocity"
    THROUGHPUT = "throughput"
    BURSTINESS = "burstiness"
    SLACK = "slack"
    COMPLEXITY = "complexity"


@dataclass
class MetricDefinition:
    """Definition of a metric with its properties."""
    name: str
    category: MetricCategory
    description: str
    unit: str
    expected_direction: str  # "increase", "decrease", "flat"
    granularities: List[str]
    primary_granularity: str


# Core metrics for the revised strategy
METRIC_DEFINITIONS: Dict[str, MetricDefinition] = {
    # Velocity metrics (task speed)
    'pr_lead_time': MetricDefinition(
        name='PR Lead Time',
        category=MetricCategory.VELOCITY,
        description='Time from PR creation to merge',
        unit='hours',
        expected_direction='decrease',
        granularities=['hours', 'business_hours', 'days', 'business_days'],
        primary_granularity='hours'
    ),
    'time_to_first_review': MetricDefinition(
        name='Time to First Review',
        category=MetricCategory.VELOCITY,
        description='Time from PR creation to first review',
        unit='hours',
        expected_direction='decrease',
        granularities=['hours', 'business_hours', 'days'],
        primary_granularity='hours'
    ),

    # Throughput metrics (output volume)
    'prs_per_developer_month': MetricDefinition(
        name='PRs per Developer-Month',
        category=MetricCategory.THROUGHPUT,
        description='Count of merged PRs per active developer per month',
        unit='PRs',
        expected_direction='flat',  # The paradox!
        granularities=['weekly', 'monthly', 'quarterly'],
        primary_granularity='monthly'
    ),
    'prs_per_developer_week': MetricDefinition(
        name='PRs per Developer-Week',
        category=MetricCategory.THROUGHPUT,
        description='Count of merged PRs per active developer per week',
        unit='PRs',
        expected_direction='flat',
        granularities=['weekly'],
        primary_granularity='weekly'
    ),

    # Burstiness metrics (work pattern)
    'cv_daily_commits': MetricDefinition(
        name='CV of Daily Commits',
        category=MetricCategory.BURSTINESS,
        description='Coefficient of variation of daily commit counts per developer',
        unit='ratio',
        expected_direction='increase',
        granularities=['daily', 'weekly'],
        primary_granularity='daily'
    ),
    'cv_weekly_prs': MetricDefinition(
        name='CV of Weekly PRs',
        category=MetricCategory.BURSTINESS,
        description='Coefficient of variation of weekly PR counts per developer',
        unit='ratio',
        expected_direction='increase',
        granularities=['weekly'],
        primary_granularity='weekly'
    ),
    'active_days_ratio': MetricDefinition(
        name='Active Days Ratio',
        category=MetricCategory.BURSTINESS,
        description='Fraction of days with activity per developer',
        unit='ratio',
        expected_direction='decrease',
        granularities=['days_per_week', 'days_per_month'],
        primary_granularity='days_per_month'
    ),

    # Slack metrics (gaps between work)
    'inter_pr_gap': MetricDefinition(
        name='Inter-PR Gap',
        category=MetricCategory.SLACK,
        description='Time from PR merge to next PR creation (same author)',
        unit='hours',
        expected_direction='increase',
        granularities=['hours', 'business_hours', 'days', 'business_days'],
        primary_granularity='hours'
    ),

    # Complexity metrics (scope expansion)
    'pr_lines_changed': MetricDefinition(
        name='PR Lines Changed',
        category=MetricCategory.COMPLEXITY,
        description='Total lines added + deleted per PR',
        unit='lines',
        expected_direction='increase',
        granularities=['median', 'p75', 'p90', 'mean'],
        primary_granularity='median'
    ),
    'pr_files_changed': MetricDefinition(
        name='PR Files Changed',
        category=MetricCategory.COMPLEXITY,
        description='Number of files modified per PR',
        unit='files',
        expected_direction='increase',
        granularities=['median', 'p75', 'mean'],
        primary_granularity='median'
    ),
    'pr_commits': MetricDefinition(
        name='Commits per PR',
        category=MetricCategory.COMPLEXITY,
        description='Number of commits per PR',
        unit='commits',
        expected_direction='increase',
        granularities=['median', 'p75', 'mean'],
        primary_granularity='median'
    ),
}


# =============================================================================
# HYPOTHESES
# =============================================================================

@dataclass
class Hypothesis:
    """A testable hypothesis in the revised strategy."""
    id: str
    name: str
    description: str
    metric: str
    expected_sign: str  # "positive", "negative", "zero"
    mechanism: str


HYPOTHESES: List[Hypothesis] = [
    Hypothesis(
        id='H1',
        name='Velocity Improvement',
        description='LLM adoption decreases PR lead time',
        metric='pr_lead_time',
        expected_sign='negative',
        mechanism='AI-assisted coding speeds up individual tasks'
    ),
    Hypothesis(
        id='H2',
        name='Throughput Paradox',
        description='Despite velocity gains, throughput stays flat',
        metric='prs_per_developer_month',
        expected_sign='zero',
        mechanism='Saved time not reinvested in more PRs'
    ),
    Hypothesis(
        id='H3',
        name='Scope Expansion',
        description='PR complexity increases post-LLM',
        metric='pr_lines_changed',
        expected_sign='positive',
        mechanism='Developers tackle bigger PRs with AI help'
    ),
    Hypothesis(
        id='H4',
        name='Work Concentration',
        description='Active days ratio decreases post-LLM',
        metric='active_days_ratio',
        expected_sign='negative',
        mechanism='Work compressed into fewer, more intense sessions'
    ),
    Hypothesis(
        id='H5',
        name='Heterogeneous Effects',
        description='Top contributors show throughput gains, others flat',
        metric='prs_per_developer_month',
        expected_sign='positive for top 20%, zero for rest',
        mechanism='Heavy users adapt to and benefit from AI tools'
    ),
]


# =============================================================================
# ANALYSIS CONFIGURATION
# =============================================================================

@dataclass
class AnalysisConfig:
    """Configuration for running the analysis."""
    # Date range
    pre_period_start: date = date(2021, 1, 1)
    pre_period_end: date = date(2022, 11, 29)
    post_period_start: date = date(2022, 12, 1)
    post_period_end: date = date(2025, 6, 30)

    # Sample restrictions
    min_prs_per_repo: int = 10
    min_active_days_for_burstiness: int = 3
    min_gaps_for_slack: int = 2
    max_lead_time_hours: int = 720  # 30 days
    max_inter_pr_gap_hours: int = 720  # 30 days

    # Heterogeneity
    top_contributor_percentile: float = 0.80
    activity_quintiles: int = 5

    # BigQuery
    max_query_cost_usd: float = 5.0
    use_cache: bool = True


DEFAULT_CONFIG = AnalysisConfig()


# =============================================================================
# EXPECTED RESULTS (for validation)
# =============================================================================

EXPECTED_RESULTS = {
    'high_exposure': {
        'velocity_change': (-90, -70),  # -93% to -73%
        'throughput_change': (-5, 5),   # essentially flat
        'complexity_change': (50, 80),  # +64%
        'active_days_change': (-25, -15),  # -20%
    },
    'low_exposure': {
        'velocity_change': (-50, -30),  # smaller improvement
        'throughput_change': (-20, 0),  # may decrease
        'complexity_change': (0, 15),   # minimal change
    }
}
