"""
Research Design Specification for Task-Job Paradox Study

This module documents the complete empirical strategy, including:
1. Metric hierarchy (task → PR → release)
2. Identification strategies (ITS, DiD, event study)
3. Control variables and fixed effects
4. Robustness checks

================================================================================
CONCEPTUAL FRAMEWORK
================================================================================

The Task-Job Spectrum:

    TASK ←――――――――――――――――――――――――――――――――――――――――――――――――→ JOB
    │                                                        │
    │  Review      CI Fix    PR Merge   Feature     Release  │
    │  Response    Time      Time       Delivery    Ship     │
    │  (hours)     (hours)   (days)     (days)      (weeks)  │
    │                                                        │
    │  Individual effort ←――――――――→ Coordination-heavy       │
    │  AI-augmentable ←――――――――――→ Human-bottlenecked        │

Key Insight:
- AI tools primarily accelerate INDIVIDUAL tasks (code writing, review response)
- But jobs require COORDINATION (reviews, approvals, integration, release management)
- Coordination doesn't scale with individual productivity
- Therefore: task gains don't translate to proportional job gains

================================================================================
METRIC HIERARCHY
================================================================================
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class MetricLevel(Enum):
    """Hierarchy of productivity metrics from task to job."""
    TASK = "task"           # Individual, execution-proximal activities
    INTERMEDIATE = "pr"     # Pull requests - bundles of tasks
    JOB = "release"         # Releases - ultimate deliverables


@dataclass
class MetricDefinition:
    """Definition of a productivity metric."""
    name: str
    level: MetricLevel
    description: str
    unit: str
    direction: str  # "lower_is_better" or "higher_is_better"
    bigquery_field: Optional[str] = None
    requires_join: bool = False


# Core metrics at each level
METRICS = {
    # TASK LEVEL - Individual, AI-augmentable activities
    "review_response_latency": MetricDefinition(
        name="Review Response Latency",
        level=MetricLevel.TASK,
        description="Time from code review feedback to author's revision commit",
        unit="hours",
        direction="lower_is_better",
        bigquery_field="avg_review_response_latency",
        requires_join=True  # Requires joining PullRequestReviewEvent with PullRequestEvent
    ),
    "ci_fix_latency": MetricDefinition(
        name="CI Fix Latency",
        level=MetricLevel.TASK,
        description="Time from CI failure to successful fix commit",
        unit="hours",
        direction="lower_is_better",
        requires_join=True  # Requires CheckRunEvent or WorkflowRunEvent
    ),
    "first_commit_to_pr": MetricDefinition(
        name="First Commit to PR Time",
        level=MetricLevel.TASK,
        description="Time from first commit to PR creation",
        unit="hours",
        direction="lower_is_better",
    ),

    # INTERMEDIATE LEVEL - PR as bundle of tasks
    "pr_lead_time": MetricDefinition(
        name="PR Lead Time",
        level=MetricLevel.INTERMEDIATE,
        description="Time from PR creation to merge",
        unit="hours",
        direction="lower_is_better",
        bigquery_field="avg_lead_time_hours",
    ),
    "pr_cycle_time": MetricDefinition(
        name="PR Cycle Time",
        level=MetricLevel.INTERMEDIATE,
        description="Time from first commit to PR merge",
        unit="hours",
        direction="lower_is_better",
    ),
    "commits_per_pr": MetricDefinition(
        name="Commits per PR",
        level=MetricLevel.INTERMEDIATE,
        description="Number of commits in a PR (iteration intensity)",
        unit="count",
        direction="neutral",  # More commits may indicate more iteration OR more complexity
        bigquery_field="avg_commits_per_pr",
    ),
    "review_rounds": MetricDefinition(
        name="Review Rounds",
        level=MetricLevel.INTERMEDIATE,
        description="Number of review cycles before approval",
        unit="count",
        direction="neutral",
        bigquery_field="avg_review_comments",  # Proxy
    ),

    # JOB LEVEL - Ultimate deliverables
    "release_lead_time": MetricDefinition(
        name="Release Lead Time",
        level=MetricLevel.JOB,
        description="Time from first commit in release to release publication",
        unit="days",
        direction="lower_is_better",
    ),
    "release_frequency": MetricDefinition(
        name="Release Frequency",
        level=MetricLevel.JOB,
        description="Number of releases per time period",
        unit="releases/month",
        direction="higher_is_better",
    ),
    "commit_to_release": MetricDefinition(
        name="Commit to Release Time",
        level=MetricLevel.JOB,
        description="Average time from commit merge to inclusion in a release",
        unit="days",
        direction="lower_is_better",
    ),
    "release_size": MetricDefinition(
        name="Release Size",
        level=MetricLevel.JOB,
        description="Number of commits/PRs included in each release",
        unit="count",
        direction="neutral",
    ),
}


"""
================================================================================
IDENTIFICATION STRATEGY
================================================================================

We employ two complementary identification strategies:

1. INTERRUPTED TIME SERIES (ITS)
   - Exploits the sharp discontinuity at AI tool adoption (Nov 2022)
   - Compares outcomes before/after within the same units
   - Controls for pre-existing trends

2. DIFFERENCE-IN-DIFFERENCES (DiD)
   - Exploits variation in AI tool effectiveness across languages
   - Treatment: High AI-exposure languages (Python, JS, Java, TS)
   - Control: Low AI-exposure languages (Fortran, COBOL, Assembly, Erlang, Haskell)
   - Requires parallel trends assumption

--------------------------------------------------------------------------------
ENHANCED DiD SPECIFICATION
--------------------------------------------------------------------------------

Basic DiD (current):
    Y_rt = β(Post × HighExposure) + γPost + δHighExposure + ε_rt

Enhanced DiD with controls and fixed effects:
    Y_rt = β(Post_t × HighExposure_r) + X_rt'Γ + α_r + γ_t + ε_rt

Where:
    Y_rt        = Outcome for repository r at time t
    Post_t      = 1 if t ≥ treatment date (Nov 30, 2022)
    HighExposure_r = 1 if primary language has high AI tool accuracy
    X_rt        = Vector of time-varying controls
    α_r         = Repository fixed effects
    γ_t         = Time (month) fixed effects
    ε_rt        = Error term, clustered at repository level

--------------------------------------------------------------------------------
EVENT STUDY SPECIFICATION (for parallel trends)
--------------------------------------------------------------------------------

    Y_rt = Σ_k β_k × 1[t=k] × HighExposure_r + X_rt'Γ + α_r + γ_t + ε_rt

Where k indexes months relative to treatment (k=0 is Nov 2022).
- β_k for k < 0 should be statistically indistinguishable from 0 (parallel trends)
- β_k for k ≥ 0 shows dynamic treatment effects

================================================================================
CONTROL VARIABLES
================================================================================
"""


@dataclass
class ControlVariable:
    """Definition of a control variable."""
    name: str
    description: str
    level: str  # "repository", "pr", "time"
    rationale: str
    bigquery_field: Optional[str] = None


CONTROLS = {
    # Repository-level controls (time-invariant or slow-moving)
    "repo_age": ControlVariable(
        name="Repository Age",
        description="Days since repository creation",
        level="repository",
        rationale="Older repos may have established workflows less affected by new tools",
    ),
    "repo_size": ControlVariable(
        name="Repository Size",
        description="Total lines of code or number of files",
        level="repository",
        rationale="Larger codebases may have different productivity dynamics",
    ),
    "num_contributors": ControlVariable(
        name="Number of Contributors",
        description="Unique contributors in observation window",
        level="repository",
        rationale="More contributors = more coordination overhead",
        bigquery_field="num_contributors",
    ),
    "star_count": ControlVariable(
        name="Star Count",
        description="GitHub stars (popularity proxy)",
        level="repository",
        rationale="Popular repos may attract different contributor types",
    ),
    "is_org_repo": ControlVariable(
        name="Organization Repository",
        description="Whether repo belongs to an organization vs personal",
        level="repository",
        rationale="Org repos may have different governance/review requirements",
    ),

    # PR-level controls (time-varying)
    "pr_size": ControlVariable(
        name="PR Size",
        description="Lines added + deleted",
        level="pr",
        rationale="Larger PRs take longer to review regardless of AI",
        bigquery_field="avg_code_churn",
    ),
    "files_changed": ControlVariable(
        name="Files Changed",
        description="Number of files modified in PR",
        level="pr",
        rationale="Cross-file changes require more careful review",
        bigquery_field="avg_changed_files",
    ),
    "is_first_time_contributor": ControlVariable(
        name="First-time Contributor",
        description="Whether PR author's first contribution to repo",
        level="pr",
        rationale="First-time contributors may receive more scrutiny",
    ),

    # Time controls
    "month_of_year": ControlVariable(
        name="Month of Year",
        description="Calendar month (1-12)",
        level="time",
        rationale="Seasonality effects (holidays, end-of-quarter rushes)",
    ),
    "day_of_week": ControlVariable(
        name="Day of Week",
        description="Day PR was created",
        level="time",
        rationale="Weekend vs weekday submission patterns",
    ),
}


"""
================================================================================
FIXED EFFECTS STRATEGY
================================================================================
"""


@dataclass
class FixedEffectSpec:
    """Specification for a fixed effect."""
    name: str
    description: str
    absorbs: str  # What variation it absorbs
    tradeoff: str  # What we lose by including it


FIXED_EFFECTS = {
    "repository_fe": FixedEffectSpec(
        name="Repository Fixed Effects (α_r)",
        description="Indicator for each repository",
        absorbs="All time-invariant repository characteristics (language, governance, "
                "domain, team culture, etc.)",
        tradeoff="Cannot estimate effects of time-invariant repo characteristics; "
                 "identification comes only from within-repo variation over time",
    ),
    "time_fe": FixedEffectSpec(
        name="Time Fixed Effects (γ_t)",
        description="Indicator for each month/quarter",
        absorbs="Common shocks affecting all repos (GitHub platform changes, "
                "economic conditions, pandemic effects, etc.)",
        tradeoff="Cannot separately identify aggregate time trends; "
                 "Post indicator is collinear if using month FE (use quarters or "
                 "exclude immediate post-treatment months)",
    ),
    "language_fe": FixedEffectSpec(
        name="Language Fixed Effects",
        description="Indicator for each programming language",
        absorbs="Language-specific productivity patterns (syntax complexity, "
                "ecosystem maturity, etc.)",
        tradeoff="With repo FE, only useful if repos change primary language "
                 "(rare); better absorbed into repo FE",
    ),
}


"""
================================================================================
ROBUSTNESS CHECKS
================================================================================
"""


@dataclass
class RobustnessCheck:
    """Definition of a robustness check."""
    name: str
    description: str
    purpose: str
    implementation: str


ROBUSTNESS_CHECKS = [
    RobustnessCheck(
        name="Pre-trends Test",
        description="Event study coefficients for pre-treatment periods",
        purpose="Verify parallel trends assumption for DiD validity",
        implementation="Estimate event study specification; test joint significance "
                      "of pre-treatment β_k coefficients (should not reject H0: β_k = 0 ∀ k < 0)",
    ),
    RobustnessCheck(
        name="Placebo Treatment Dates",
        description="Run DiD with fake treatment dates before actual treatment",
        purpose="Confirm no spurious effects from mis-specification",
        implementation="Re-estimate with treatment dates at 6, 12, 18 months before Nov 2022; "
                      "should find no significant effects",
    ),
    RobustnessCheck(
        name="Donut Hole Specification",
        description="Exclude observations immediately around treatment date",
        purpose="Account for anticipation effects and gradual adoption",
        implementation="Exclude Nov 2022 - Feb 2023 (adoption period); "
                      "results should be similar or stronger",
    ),
    RobustnessCheck(
        name="Alternative Control Groups",
        description="Use different definitions of low-exposure languages",
        purpose="Ensure results not driven by specific control group choice",
        implementation="Try: (1) Only Fortran/COBOL, (2) Add R/MATLAB, (3) Use Copilot accuracy scores",
    ),
    RobustnessCheck(
        name="Continuous Treatment Intensity",
        description="Use Copilot accuracy scores instead of binary high/low",
        purpose="Exploit continuous variation in AI tool effectiveness",
        implementation="Replace HighExposure with CopilotAccuracy ∈ [0,1]; "
                      "expect dose-response relationship",
    ),
    RobustnessCheck(
        name="Excluding Large Repos",
        description="Remove top 5% of repos by contributor count",
        purpose="Check if results driven by mega-projects with different dynamics",
        implementation="Re-estimate excluding repos with >100 contributors",
    ),
    RobustnessCheck(
        name="Triple Difference (DDD)",
        description="Add third difference: high vs low coordination intensity",
        purpose="Test mechanism hypothesis that coordination is the bottleneck",
        implementation="Y_rt = β(Post × HighExposure × HighCoordination) + ... "
                      "Expect β < 0 (paradox larger in coordination-heavy repos)",
    ),
    RobustnessCheck(
        name="Synthetic Control",
        description="Construct synthetic control repos for treated repos",
        purpose="Alternative to parallel trends assumption",
        implementation="For each high-exposure repo, construct weighted average of "
                      "low-exposure repos matching pre-treatment trends",
    ),
]


"""
================================================================================
STANDARD ERRORS
================================================================================

Clustering strategy:
- Cluster at REPOSITORY level (not PR level)
- Rationale: Serial correlation within repos over time
- PRs from same repo are not independent observations
- Avoids over-rejection of null hypothesis

For very large samples, consider:
- Two-way clustering (repo × time)
- Conley spatial HAC if geographic spillovers possible
"""


@dataclass
class ClusteringSpec:
    """Specification for standard error clustering."""
    level: str
    rationale: str
    stata_code: str
    python_code: str


CLUSTERING = ClusteringSpec(
    level="repository",
    rationale="Account for serial correlation within repositories over time. "
              "Multiple PRs from same repo are correlated; treating them as "
              "independent would understate standard errors.",
    stata_code="reghdfe Y Post#HighExposure X, absorb(repo_id month) cluster(repo_id)",
    python_code="smf.ols(formula, data).fit(cov_type='cluster', cov_kwds={'groups': df['repo_id']})",
)


"""
================================================================================
HYPOTHESES (Updated)
================================================================================
"""

HYPOTHESES = {
    "H1": {
        "statement": "Task-level metrics improve substantially after AI adoption",
        "expected_effect": "Review response latency decreases 30-50%",
        "metric": "review_response_latency",
        "level": MetricLevel.TASK,
    },
    "H2a": {
        "statement": "PR-level metrics improve moderately after AI adoption",
        "expected_effect": "PR lead time decreases 10-20%",
        "metric": "pr_lead_time",
        "level": MetricLevel.INTERMEDIATE,
    },
    "H2b": {
        "statement": "Release-level metrics improve minimally after AI adoption",
        "expected_effect": "Release lead time decreases 0-10%",
        "metric": "release_lead_time",
        "level": MetricLevel.JOB,
    },
    "H3": {
        "statement": "Effects are concentrated in high AI-exposure languages",
        "expected_effect": "DiD coefficient significant and negative for high exposure",
        "metric": "all",
        "level": "all",
    },
    "H4": {
        "statement": "Iteration intensity increases (time reallocation)",
        "expected_effect": "Commits per PR increases 15-30%",
        "metric": "commits_per_pr",
        "level": MetricLevel.INTERMEDIATE,
    },
    "H5": {
        "statement": "Task-job gap is larger in coordination-heavy projects",
        "expected_effect": "Triple-diff interaction term is significant",
        "metric": "all",
        "level": "all",
    },
    "H6": {
        "statement": "The paradox amplifies as we move up the metric hierarchy",
        "expected_effect": "Task improvement > PR improvement > Release improvement",
        "metric": "comparison",
        "level": "hierarchy",
    },
}


def get_full_model_specification() -> str:
    """Return the full econometric model specification as a string."""
    return """
================================================================================
FULL MODEL SPECIFICATION
================================================================================

MAIN DiD REGRESSION (with two-way fixed effects):

    Y_irt = β₁(Post_t × HighExposure_r)
          + β₂(Post_t × HighExposure_r × HighCoordination_r)  [H5 test]
          + Σⱼ γⱼ X_irt^j                                      [Controls]
          + α_r                                                 [Repo FE]
          + δ_t                                                 [Time FE]
          + ε_irt

Where:
    i = PR or release (unit of observation)
    r = repository
    t = time period (month)

    Y_irt ∈ {review_response_latency, pr_lead_time, release_lead_time, ...}

    X_irt = [log(pr_size), log(files_changed), is_first_contributor, ...]

Standard errors clustered at repository level.


EVENT STUDY SPECIFICATION (for parallel trends validation):

    Y_irt = Σₖ βₖ × 1[t=k] × HighExposure_r
          + Σⱼ γⱼ X_irt^j
          + α_r
          + δ_t
          + ε_irt

Where k ∈ {-24, -23, ..., -1, 0, 1, ..., 24} indexes months relative to treatment.
Normalize β₋₁ = 0 (reference period).

Test: H₀: β₋₂₄ = β₋₂₃ = ... = β₋₂ = 0 (joint F-test for parallel trends)


HIERARCHICAL COMPARISON (H6):

Run main regression separately for:
    1. Y = review_response_latency (task)
    2. Y = pr_lead_time (intermediate)
    3. Y = release_lead_time (job)

Compare β₁ across models:
    Expected: |β₁_task| > |β₁_pr| > |β₁_release|

================================================================================
"""


if __name__ == "__main__":
    print(get_full_model_specification())

    print("\n" + "="*70)
    print("METRIC HIERARCHY")
    print("="*70)
    for level in MetricLevel:
        print(f"\n{level.value.upper()} LEVEL:")
        for name, metric in METRICS.items():
            if metric.level == level:
                print(f"  - {metric.name}: {metric.description}")

    print("\n" + "="*70)
    print("CONTROL VARIABLES")
    print("="*70)
    for name, ctrl in CONTROLS.items():
        print(f"  - {ctrl.name} [{ctrl.level}]: {ctrl.rationale}")

    print("\n" + "="*70)
    print("ROBUSTNESS CHECKS")
    print("="*70)
    for i, check in enumerate(ROBUSTNESS_CHECKS, 1):
        print(f"  {i}. {check.name}: {check.purpose}")
