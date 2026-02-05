"""
Revised Empirical Strategy: Peaks, Slack, and Scope Expansion

This package implements the revised empirical strategy for the Task-Job Paradox,
focusing on throughput, burstiness, slack, and scope expansion rather than
task vs job latency.

Key Question: Why don't LLM-driven velocity gains translate to proportional
throughput gains?

Answer: Developers use saved time for scope expansion (bigger PRs),
concentrated work patterns, and the gains are unequally distributed
(top 20% benefit most).

Modules:
- config: Constants, language groups, treatment dates
- queries: BigQuery queries for all metrics
- metrics: Metric extraction and computation
- statistical_analysis: ITS and DiD for causal identification
- visualizations: Publication-quality figures
- run_analysis: Main orchestration script
"""

from .config import (
    HIGH_EXPOSURE_LANGUAGES,
    LOW_EXPOSURE_LANGUAGES,
    ALL_LANGUAGES,
    TREATMENT_DATE,
    TREATMENT_DATE_STR,
    HYPOTHESES,
    METRIC_DEFINITIONS,
    DEFAULT_CONFIG,
)

from .metrics import (
    DataLoader,
    MetricsComputer,
    PeriodComparison,
    load_and_compute_all_metrics,
)

from .statistical_analysis import (
    InterruptedTimeSeries,
    DifferenceInDifferences,
    HeterogeneityAnalysis,
    StatisticalTests,
    run_full_statistical_analysis,
)

from .visualizations import (
    create_paradox_overview,
    create_summary_figure,
    create_all_figures,
)

from .run_analysis import run_full_analysis

__all__ = [
    # Config
    'HIGH_EXPOSURE_LANGUAGES',
    'LOW_EXPOSURE_LANGUAGES',
    'ALL_LANGUAGES',
    'TREATMENT_DATE',
    'TREATMENT_DATE_STR',
    'HYPOTHESES',
    'METRIC_DEFINITIONS',
    'DEFAULT_CONFIG',
    # Metrics
    'DataLoader',
    'MetricsComputer',
    'PeriodComparison',
    'load_and_compute_all_metrics',
    # Statistical Analysis
    'InterruptedTimeSeries',
    'DifferenceInDifferences',
    'HeterogeneityAnalysis',
    'StatisticalTests',
    'run_full_statistical_analysis',
    # Visualizations
    'create_paradox_overview',
    'create_summary_figure',
    'create_all_figures',
    # Main
    'run_full_analysis',
]
