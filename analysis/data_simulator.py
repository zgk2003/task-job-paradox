"""
Data Simulation Framework for Task-Job Paradox Analysis

Generates synthetic GitHub-like data that embodies the theoretical patterns:
- Task-level metrics show significant improvement after AI adoption (~35%)
- Job-level metrics show minimal improvement (~8%)
- Effects are concentrated in high AI-exposure languages
- Mechanism metrics show increased iteration intensity

This simulation allows us to validate our analysis pipeline and
demonstrate expected patterns before applying to real GitHub Archive data.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple
import uuid
from dataclasses import dataclass

from .data_model import (
    Repository, PullRequest, Commit, Review, ReviewComment,
    CIRun, Release, Language, PRState, CIStatus,
    TREATMENT_START, PRE_PERIOD_START, POST_PERIOD_END
)


@dataclass
class SimulationConfig:
    """Configuration parameters for data simulation."""
    # Sample sizes
    num_repos: int = 500
    prs_per_repo_per_month: float = 8.0

    # Language distribution (matches real GitHub)
    language_weights: dict = None

    # Baseline metrics (pre-AI period, in hours)
    baseline_review_response_latency: float = 18.0  # hours
    baseline_ci_fix_latency: float = 4.0  # hours
    baseline_pr_lead_time: float = 72.0  # hours (3 days)
    baseline_commits_per_pr: float = 4.0

    # Treatment effects (AI adoption)
    # Task-level effects (large, as documented in literature)
    task_effect_high_exposure: float = -0.35  # 35% reduction
    task_effect_low_exposure: float = -0.10   # 10% reduction (some spillover)

    # Job-level effects (small - the paradox!)
    job_effect_high_exposure: float = -0.08   # Only 8% reduction
    job_effect_low_exposure: float = -0.03    # 3% reduction

    # Mechanism effects (iteration intensity increases)
    commits_per_pr_effect: float = 0.25  # 25% more commits per PR
    review_rounds_effect: float = 0.15   # 15% more review rounds

    # Noise parameters
    noise_scale: float = 0.3

    # Random seed for reproducibility
    seed: int = 42

    def __post_init__(self):
        if self.language_weights is None:
            self.language_weights = {
                # High exposure (70% of repos)
                Language.PYTHON: 0.25,
                Language.JAVASCRIPT: 0.25,
                Language.JAVA: 0.12,
                Language.TYPESCRIPT: 0.08,
                # Low exposure (30% of repos)
                Language.HASKELL: 0.08,
                Language.ERLANG: 0.07,
                Language.FORTRAN: 0.05,
                Language.COBOL: 0.05,
                Language.ASSEMBLY: 0.05,
            }


class DataSimulator:
    """Generates synthetic GitHub data for empirical analysis."""

    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        np.random.seed(self.config.seed)

    def _generate_id(self) -> str:
        return str(uuid.uuid4())[:8]

    def _is_post_treatment(self, date: datetime) -> bool:
        return date >= TREATMENT_START

    def _sample_language(self) -> Language:
        languages = list(self.config.language_weights.keys())
        weights = list(self.config.language_weights.values())
        return np.random.choice(languages, p=weights)

    def _apply_treatment_effect(
        self,
        baseline: float,
        date: datetime,
        high_exposure: bool,
        is_task_level: bool
    ) -> float:
        """Apply treatment effect based on timing and exposure level.

        Key insight: Task-level effects are large, job-level effects are small.
        This is the core of the paradox we're measuring.
        """
        if not self._is_post_treatment(date):
            # Pre-treatment period: baseline with noise
            noise = np.random.normal(0, self.config.noise_scale * baseline)
            return max(0.1, baseline + noise)

        # Post-treatment period: apply effect
        if is_task_level:
            effect = (self.config.task_effect_high_exposure if high_exposure
                     else self.config.task_effect_low_exposure)
        else:
            effect = (self.config.job_effect_high_exposure if high_exposure
                     else self.config.job_effect_low_exposure)

        # Gradual adoption: effect ramps up over 6 months
        months_since_treatment = (date - TREATMENT_START).days / 30
        adoption_factor = min(1.0, months_since_treatment / 6)
        effective_treatment = effect * adoption_factor

        treated_value = baseline * (1 + effective_treatment)
        noise = np.random.normal(0, self.config.noise_scale * treated_value)
        return max(0.1, treated_value + noise)

    def _generate_commits(
        self,
        pr_created: datetime,
        pr_merged: datetime,
        high_exposure: bool,
        num_reviews: int
    ) -> List[Commit]:
        """Generate commits for a PR."""
        # Number of commits increases after AI adoption (more iteration)
        base_commits = self.config.baseline_commits_per_pr
        if self._is_post_treatment(pr_created) and high_exposure:
            months_since = (pr_created - TREATMENT_START).days / 30
            adoption = min(1.0, months_since / 6)
            base_commits *= (1 + self.config.commits_per_pr_effect * adoption)

        num_commits = max(1, int(np.random.poisson(base_commits)))
        commits = []

        pr_duration = (pr_merged - pr_created).total_seconds()
        for i in range(num_commits):
            # Distribute commits over PR lifetime
            offset_fraction = (i + 1) / (num_commits + 1)
            offset_seconds = pr_duration * offset_fraction
            commit_time = pr_created + timedelta(seconds=offset_seconds)

            commits.append(Commit(
                sha=self._generate_id(),
                author="author",
                created_at=commit_time,
                message=f"Commit {i+1}",
                lines_added=int(np.random.exponential(50)),
                lines_deleted=int(np.random.exponential(20)),
            ))

        return sorted(commits, key=lambda c: c.created_at)

    def _generate_reviews(
        self,
        pr_created: datetime,
        pr_merged: datetime,
        commits: List[Commit],
        high_exposure: bool
    ) -> List[Review]:
        """Generate code reviews for a PR."""
        # Number of review rounds
        base_reviews = 2.0
        if self._is_post_treatment(pr_created) and high_exposure:
            months_since = (pr_created - TREATMENT_START).days / 30
            adoption = min(1.0, months_since / 6)
            base_reviews *= (1 + self.config.review_rounds_effect * adoption)

        num_reviews = max(1, int(np.random.poisson(base_reviews)))
        reviews = []

        for i in range(num_reviews):
            # Reviews happen after some commits
            review_after_commit_idx = min(i, len(commits) - 1)
            review_time = commits[review_after_commit_idx].created_at + timedelta(
                hours=np.random.exponential(12)
            )

            if review_time >= pr_merged:
                review_time = pr_merged - timedelta(hours=1)

            state = 'changes_requested' if i < num_reviews - 1 else 'approved'
            reviews.append(Review(
                review_id=self._generate_id(),
                reviewer=f"reviewer_{i}",
                submitted_at=review_time,
                state=state,
                comments=[]
            ))

        return sorted(reviews, key=lambda r: r.submitted_at)

    def _generate_ci_runs(
        self,
        commits: List[Commit],
        high_exposure: bool
    ) -> List[CIRun]:
        """Generate CI runs for commits."""
        ci_runs = []

        for commit in commits:
            # CI starts shortly after commit
            ci_start = commit.created_at + timedelta(minutes=np.random.exponential(5))
            ci_duration = timedelta(minutes=np.random.exponential(15))
            ci_end = ci_start + ci_duration

            # 30% chance of failure
            status = CIStatus.FAILURE if np.random.random() < 0.3 else CIStatus.SUCCESS

            ci_runs.append(CIRun(
                run_id=self._generate_id(),
                commit_sha=commit.sha,
                started_at=ci_start,
                completed_at=ci_end,
                status=status
            ))

        return ci_runs

    def _generate_pull_request(
        self,
        repo: Repository,
        created_at: datetime
    ) -> PullRequest:
        """Generate a complete pull request with all nested events."""
        high_exposure = repo.high_ai_exposure

        # PR lead time (job-level metric)
        lead_time_hours = self._apply_treatment_effect(
            self.config.baseline_pr_lead_time,
            created_at,
            high_exposure,
            is_task_level=False  # Job-level!
        )
        merged_at = created_at + timedelta(hours=lead_time_hours)

        # Ensure merged_at doesn't exceed our observation window
        if merged_at > POST_PERIOD_END:
            merged_at = POST_PERIOD_END

        pr = PullRequest(
            pr_id=self._generate_id(),
            repo_id=repo.repo_id,
            author="author",
            title=f"PR at {created_at.date()}",
            created_at=created_at,
            state=PRState.MERGED,
            merged_at=merged_at,
        )

        # Generate nested events
        pr.commits = self._generate_commits(
            created_at, merged_at, high_exposure, num_reviews=2
        )
        pr.reviews = self._generate_reviews(
            created_at, merged_at, pr.commits, high_exposure
        )
        pr.ci_runs = self._generate_ci_runs(pr.commits, high_exposure)

        # Now adjust review-response latency (task-level metric)
        # This requires regenerating commits to align with review timing
        pr = self._adjust_task_level_timing(pr, high_exposure)

        return pr

    def _adjust_task_level_timing(
        self,
        pr: PullRequest,
        high_exposure: bool
    ) -> PullRequest:
        """Adjust commit timing to reflect task-level treatment effects.

        Task-level metrics (review-response latency) should show larger
        improvement than job-level metrics. We achieve this by compressing
        the time between reviews and subsequent commits.
        """
        # For each review that requests changes, adjust the next commit timing
        reviews_requesting_changes = [
            r for r in pr.reviews if r.state == 'changes_requested'
        ]

        for review in reviews_requesting_changes:
            # Find next commit
            subsequent_commits = [
                c for c in pr.commits if c.created_at > review.submitted_at
            ]
            if not subsequent_commits:
                continue

            next_commit = min(subsequent_commits, key=lambda c: c.created_at)

            # Calculate new latency with task-level effect
            new_latency_hours = self._apply_treatment_effect(
                self.config.baseline_review_response_latency,
                pr.created_at,
                high_exposure,
                is_task_level=True  # Task-level!
            )

            # Update commit time
            next_commit.created_at = review.submitted_at + timedelta(
                hours=new_latency_hours
            )

        return pr

    def _generate_repository(self, repo_idx: int) -> Repository:
        """Generate a repository with its history of PRs."""
        language = self._sample_language()
        repo_created = PRE_PERIOD_START - timedelta(days=np.random.randint(0, 365))

        repo = Repository(
            repo_id=self._generate_id(),
            name=f"repo_{repo_idx}_{language.value}",
            primary_language=language,
            created_at=repo_created,
            num_contributors=int(np.random.exponential(8)) + 1,
        )

        # Generate PRs over the observation period
        current_date = PRE_PERIOD_START
        while current_date < POST_PERIOD_END:
            # PRs arrive according to Poisson process
            days_to_next = np.random.exponential(30 / self.config.prs_per_repo_per_month)
            current_date += timedelta(days=days_to_next)

            if current_date < POST_PERIOD_END:
                pr = self._generate_pull_request(repo, current_date)
                repo.pull_requests.append(pr)

        return repo

    def generate_dataset(self) -> List[Repository]:
        """Generate the complete simulated dataset."""
        print(f"Generating {self.config.num_repos} repositories...")
        repositories = []

        for i in range(self.config.num_repos):
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1} repositories...")
            repo = self._generate_repository(i)
            repositories.append(repo)

        # Summary statistics
        total_prs = sum(len(r.pull_requests) for r in repositories)
        high_exposure_repos = sum(1 for r in repositories if r.high_ai_exposure)

        print(f"\nDataset Summary:")
        print(f"  Total repositories: {len(repositories)}")
        print(f"  High AI exposure: {high_exposure_repos} ({100*high_exposure_repos/len(repositories):.1f}%)")
        print(f"  Total pull requests: {total_prs}")
        print(f"  Average PRs per repo: {total_prs/len(repositories):.1f}")

        return repositories


def simulate_and_save(output_path: str = None):
    """Generate and optionally save simulated dataset."""
    config = SimulationConfig(
        num_repos=500,
        seed=42
    )
    simulator = DataSimulator(config)
    repositories = simulator.generate_dataset()

    if output_path:
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(repositories, f)
        print(f"\nSaved dataset to {output_path}")

    return repositories


if __name__ == "__main__":
    repos = simulate_and_save("../data/simulated_data.pkl")
