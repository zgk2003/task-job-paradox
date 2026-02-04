"""
Task-Job Paradox Empirical Analysis Package

This package provides tools for analyzing the task-job productivity paradox
in AI-assisted software development.

Modules:
- data_model: Core data structures mirroring GitHub Archive
- data_simulator: Synthetic data generation for validation
- metrics: Task-level and job-level metric extraction
- statistical_analysis: ITS and DiD regression analysis
- visualizations: Publication-quality figure generation
- bigquery_client: Real GitHub Archive data via Google BigQuery
- run_analysis: Main analysis orchestration script
"""

from .data_model import (
    Repository, PullRequest, Commit, Review, CIRun, Release,
    Language, PRState, CIStatus,
    TREATMENT_START, PRE_PERIOD_START, POST_PERIOD_END
)

from .data_simulator import DataSimulator, SimulationConfig
from .metrics import MetricsExtractor, create_analysis_dataset
from .statistical_analysis import (
    InterruptedTimeSeriesAnalysis,
    DifferenceInDifferencesAnalysis,
    HeterogeneityAnalysis,
    run_full_analysis
)

# BigQuery client (optional - requires google-cloud-bigquery)
try:
    from .bigquery_client import (
        BigQueryGitHubClient,
        GitHubArchiveDataLoader,
        MonthlyMetricsQuery,
        create_bigquery_client,
        BIGQUERY_AVAILABLE
    )
except ImportError:
    BIGQUERY_AVAILABLE = False

__version__ = "0.1.0"
__author__ = "Task-Job Paradox Research Team"
