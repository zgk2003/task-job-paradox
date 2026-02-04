"""
Data Model for Task-Job Paradox Empirical Analysis

This module defines the core data structures that mirror GitHub Archive data,
enabling measurement of both task-level and job-level productivity metrics.

Key Entities:
- Repository: Unit of analysis with language (AI exposure) attribute
- PullRequest: Job-level unit (broader work objective)
- Commit: Within-PR activity
- Review: Code review events with timestamps
- CIRun: Continuous integration test results
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional
from enum import Enum


class Language(Enum):
    """Programming languages categorized by AI tool exposure level.

    High exposure: 61-80% accuracy with GitHub Copilot
    Low exposure: ~30% accuracy with Copilot
    """
    # High AI exposure languages
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    TYPESCRIPT = "typescript"

    # Low AI exposure languages
    FORTRAN = "fortran"
    COBOL = "cobol"
    ASSEMBLY = "assembly"
    ERLANG = "erlang"
    HASKELL = "haskell"

    @property
    def high_exposure(self) -> bool:
        return self in {Language.PYTHON, Language.JAVASCRIPT,
                       Language.JAVA, Language.TYPESCRIPT}


class PRState(Enum):
    OPEN = "open"
    MERGED = "merged"
    CLOSED = "closed"


class CIStatus(Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass
class CIRun:
    """Continuous Integration run event.

    Used to measure CI-fix latency: time between test failure and fix.
    """
    run_id: str
    commit_sha: str
    started_at: datetime
    completed_at: datetime
    status: CIStatus

    @property
    def duration_seconds(self) -> float:
        return (self.completed_at - self.started_at).total_seconds()


@dataclass
class ReviewComment:
    """Individual review comment.

    Tracks the timestamp of feedback for review-response latency calculation.
    """
    comment_id: str
    reviewer: str
    created_at: datetime
    body: str
    requires_changes: bool = False


@dataclass
class Review:
    """Code review event on a pull request.

    A review round consists of reviewer feedback followed by author response.
    Key for measuring review-response latency.
    """
    review_id: str
    reviewer: str
    submitted_at: datetime
    state: str  # 'approved', 'changes_requested', 'commented'
    comments: List[ReviewComment] = field(default_factory=list)


@dataclass
class Commit:
    """Individual commit within a pull request.

    Tracks lines changed for code churn measurement.
    """
    sha: str
    author: str
    created_at: datetime
    message: str
    lines_added: int
    lines_deleted: int

    @property
    def lines_changed(self) -> int:
        return self.lines_added + self.lines_deleted


@dataclass
class PullRequest:
    """Pull Request - the primary job-level unit of analysis.

    Represents a broader work objective comprising multiple tasks:
    - Writing code (commits)
    - Responding to reviews (review rounds)
    - Fixing CI failures (CI runs)

    Job-level metrics measured here:
    - Lead time: created_at -> merged_at
    - Commits per PR (iteration intensity)
    - Review rounds
    - Code churn
    """
    pr_id: str
    repo_id: str
    author: str
    title: str
    created_at: datetime
    state: PRState
    merged_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    commits: List[Commit] = field(default_factory=list)
    reviews: List[Review] = field(default_factory=list)
    ci_runs: List[CIRun] = field(default_factory=list)

    @property
    def lead_time_hours(self) -> Optional[float]:
        """PR lead time: time from creation to merge (job-level metric)."""
        if self.merged_at:
            return (self.merged_at - self.created_at).total_seconds() / 3600
        return None

    @property
    def num_commits(self) -> int:
        """Number of commits in PR (iteration intensity metric)."""
        return len(self.commits)

    @property
    def num_review_rounds(self) -> int:
        """Number of review rounds (coordination metric)."""
        return len([r for r in self.reviews if r.state == 'changes_requested'])

    @property
    def total_code_churn(self) -> int:
        """Total lines changed across all commits (rework metric)."""
        return sum(c.lines_changed for c in self.commits)

    def review_response_latencies(self) -> List[float]:
        """Calculate review-response latency for each review round (task-level).

        Review-response latency = time between review feedback and next commit.
        This is a task-level metric: responding to a specific review.
        """
        latencies = []
        reviews_requesting_changes = [
            r for r in self.reviews if r.state == 'changes_requested'
        ]

        for review in reviews_requesting_changes:
            # Find the next commit after this review
            subsequent_commits = [
                c for c in self.commits if c.created_at > review.submitted_at
            ]
            if subsequent_commits:
                next_commit = min(subsequent_commits, key=lambda c: c.created_at)
                latency_hours = (next_commit.created_at - review.submitted_at).total_seconds() / 3600
                latencies.append(latency_hours)

        return latencies

    def ci_fix_latencies(self) -> List[float]:
        """Calculate CI-fix latency for each failure (task-level).

        CI-fix latency = time between CI failure and subsequent passing commit.
        This is a task-level metric: fixing a specific test failure.
        """
        latencies = []

        for i, run in enumerate(self.ci_runs):
            if run.status == CIStatus.FAILURE:
                # Find next successful run
                subsequent_successes = [
                    r for r in self.ci_runs[i+1:] if r.status == CIStatus.SUCCESS
                ]
                if subsequent_successes:
                    next_success = subsequent_successes[0]
                    latency_hours = (next_success.completed_at - run.completed_at).total_seconds() / 3600
                    latencies.append(latency_hours)

        return latencies


@dataclass
class Release:
    """Software release event.

    Used to measure release inclusion time: commit -> release (job-level).
    """
    release_id: str
    repo_id: str
    tag_name: str
    created_at: datetime
    published_at: datetime
    included_commits: List[str] = field(default_factory=list)  # SHAs


@dataclass
class Repository:
    """Repository - the unit of observation in our analysis.

    Key attributes:
    - Primary language determines AI exposure level
    - Contributor count indicates coordination complexity
    """
    repo_id: str
    name: str
    primary_language: Language
    created_at: datetime
    num_contributors: int
    pull_requests: List[PullRequest] = field(default_factory=list)
    releases: List[Release] = field(default_factory=list)

    @property
    def high_ai_exposure(self) -> bool:
        """Whether this repo has high AI tool exposure based on language."""
        return self.primary_language.high_exposure

    @property
    def high_coordination(self) -> bool:
        """Whether this repo has high coordination needs (H5 moderator)."""
        return self.num_contributors > 10


# Key dates for interrupted time series analysis
AI_ADOPTION_EVENTS = {
    'copilot_general': datetime(2022, 6, 21),   # GitHub Copilot general availability
    'chatgpt_launch': datetime(2022, 11, 30),   # ChatGPT public launch
}

# Treatment period starts after ChatGPT launch (more widespread adoption)
TREATMENT_START = datetime(2022, 11, 30)
PRE_PERIOD_START = datetime(2021, 1, 1)
POST_PERIOD_END = datetime(2025, 6, 30)
