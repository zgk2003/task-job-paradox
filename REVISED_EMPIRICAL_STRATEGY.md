# Revised Empirical Strategy: Peaks and Slack Hypothesis

## Conceptual Reframing

### The Problem with the Original Approach

The original "task vs job" framing measured latency at different pipeline stages:
- Task: Time to first review, review-to-merge time
- Job: Merge-to-release time

But "merge-to-release time" is still just measuring **latency of a single item through a pipeline stage**. It's structurally identical to other task metrics—not a true "job-level" measure.

### The Revised Hypothesis

**Pre-LLM**: Developer effort distributed relatively evenly over time → steady output flow

**Post-LLM**: Effort concentrates into intense bursts (productivity peaks) → but "saved time" becomes slack, not reinvested into more work

**Implication**: Total output over fixed calendar periods stays flat despite dramatic task velocity improvements.

This is about **temporal distribution of effort** and **slack absorption**, not task vs. job granularity.

### What "Job Productivity" Actually Means

A "job" isn't a bigger task. **Job productivity = cumulative output over a meaningful calendar period.**

The paradox isn't that downstream tasks are slow—it's that speeding up individual tasks doesn't produce proportionally more cumulative output.

---

## Selected Metrics (Feasible + Defensible + Coherent)

### Metric Set Overview

| # | Metric | Category | Story Role | Granularities |
|---|--------|----------|------------|---------------|
| 1 | PR Lead Time | Velocity | Confirms LLM speeds up tasks | Hours, business hours, days |
| 2 | Time to First Review | Velocity | Confirms LLM speeds up tasks | Hours, business hours, days |
| 3 | CV of Commits | Burstiness | Shows work became peaky | Hourly, daily, weekly, bi-weekly |
| 4 | Active Time Ratio | Burstiness | Shows effort concentration | Hours/day, days/week, days/month |
| 5 | Throughput | Throughput | Tests if speed → more output | Weekly, monthly, quarterly |
| 6 | Inter-PR Gap | Slack | Identifies slack periods | Hours, business hours, days |

**Principle**: Each metric measured at multiple temporal granularities. Consistent patterns across granularities strengthen findings.

### Metric Definitions

#### 1. PR Lead Time (Velocity)
- **Definition**: Time from PR creation to merge
- **Multi-granularity**:
  - Hours (raw) ← **primary**
  - Business hours (excluding nights/weekends)
  - Calendar days
  - Business days
- **Aggregation**: Median (primary), mean, 75th/90th percentile
- **Expected**: ↓ post-LLM (especially in treatment group)
- **Role**: Validates that LLM adoption actually sped up tasks

#### 2. Time to First Review (Velocity)
- **Definition**: Time from PR creation to first review comment
- **Multi-granularity**:
  - Hours (raw) ← **primary**
  - Business hours (excluding nights/weekends)
  - Calendar days
  - Business days
- **Aggregation**: Median (primary), mean, 75th/90th percentile
- **Expected**: ↓ post-LLM
- **Role**: Confirms task-level speedup at review stage

#### 3. Coefficient of Variation of Commits (Burstiness)
- **Definition**: σ(commits per unit) / μ(commits per unit) per developer
- **Multi-granularity**:
  - CV of hourly commits (within-day concentration)
  - CV of daily commits (day-to-day variation) ← **primary**
  - CV of weekly commits (week-to-week rhythm) ← **primary**
  - CV of bi-weekly commits (sprint-level variation)
- **Interpretation**: Higher CV = more variable/bursty work pattern
- **Expected**: ↑ post-LLM at all granularities (work becomes peaky)
- **Role**: Confirms productivity peaks exist

#### 4. Active Time Ratio (Burstiness)
- **Definition**: Active time units / total time units, per developer
- **Multi-granularity**:
  - Active hours per day (within-day spread)
  - Active days per week ← **primary**
  - Active days per month ← **primary**
  - Active weeks per quarter
- **Threshold variations**: ≥1 commit, ≥3 commits, ≥1 PR
- **Interpretation**: Lower ratio = more concentrated effort
- **Expected**: ↓ post-LLM if work concentrates into bursts
- **Role**: Alternative burstiness measure; confirms temporal concentration

#### 5. Throughput (Output Volume)
- **Definition**: Count of outputs per developer per time period
- **Multi-granularity**:
  - PRs per developer-week (short-term, noisy)
  - PRs per developer-month ← **primary**
  - PRs per developer-quarter (smoothed trend)
- **Alternative units**: Commits, issues closed, reviews completed
- **Expected**: ≈ flat or sub-linear growth relative to velocity gains
- **Role**: **Primary outcome**—tests whether speed translates to more output

#### 6. Inter-PR Gap (Slack)
- **Definition**: Time from PR merge to same author's next PR creation
- **Multi-granularity**:
  - Hours ← **primary** (most precise)
  - Business hours (work-time only)
  - Calendar days
  - Business days ← **primary** (most interpretable)
- **Gap definition variations**: Merge→PR, Merge→commit, Last commit→first commit
- **Aggregation**: Median (robust), Mean, 75th percentile
- **Expected**: ↑ post-LLM (saved time becomes slack)
- **Role**: Directly identifies where "saved time" goes

---

## Identification Strategy (Unchanged)

### Primary: Interrupted Time Series (ITS)

Exploits sharp discontinuity at AI tool adoption.

**Treatment date**: November 30, 2022 (ChatGPT public launch)

**Specification**:
```
Y_it = α + β₁(Post_t) + β₂(Time_t) + β₃(Post_t × Time_t) + ε_it
```

Where:
- Y_it = outcome metric for repo i at time t
- Post_t = 1 if t > Nov 30, 2022
- Time_t = linear time trend
- β₁ = level shift at treatment
- β₃ = slope change post-treatment

### Secondary: Difference-in-Differences (DiD)

Compares high vs. low AI-exposure languages.

**Treatment group**: Python, JavaScript, Java, TypeScript (61-80% Copilot accuracy)

**Control group**: Fortran, COBOL, Assembly, Erlang, Haskell (~30% accuracy)

**Specification**:
```
Y_it = α + β₁(Treat_i) + β₂(Post_t) + β₃(Treat_i × Post_t) + γ_i + δ_t + ε_it
```

Where:
- Treat_i = 1 if high AI-exposure language
- Post_t = 1 if post-treatment period
- β₃ = **DiD estimator** (causal effect of AI on treated)
- γ_i = repository fixed effects
- δ_t = time fixed effects

---

## Expected Results Pattern

If the "peaks and slack" hypothesis is true:

| Metric | Expected Change | Key Granularities | Interpretation |
|--------|-----------------|-------------------|----------------|
| PR Lead Time | ↓ 15-30% | Hours, business hours | LLM speeds up tasks ✓ |
| Time to First Review | ↓ 10-20% | Hours, business hours | LLM speeds up tasks ✓ |
| CV of Commits | ↑ 20-40% | Daily, weekly (both should show ↑) | Work became bursty ✓ |
| Active Time Ratio | ↓ 10-20% | Days/week, days/month | Effort concentrated ✓ |
| Throughput | ≈ flat (< 5%) | Monthly (primary), weekly/quarterly (confirm) | **Paradox confirmed** |
| Inter-PR Gap | ↑ 20-50% | Hours, business days (both should show ↑) | Slack absorbs saved time ✓ |

**Cross-granularity consistency**: The hypothesis is strongly supported if patterns hold across multiple granularities. Divergent patterns across granularities would suggest the phenomenon operates at specific time scales.

### The Coherent Story

1. **Velocity ↑**: LLMs genuinely speed up individual coding tasks (PR lead time, review time decrease)

2. **Burstiness ↑**: Work patterns shift from steady to peaky (CV increases, active days decrease)

3. **Throughput ≈ flat**: Despite faster tasks, developers don't produce more PRs per month

4. **Slack identified**: The "saved time" appears as longer gaps between PRs, not more PRs

**Conclusion**: LLM productivity gains are real at the task level but get absorbed as slack rather than translating to proportionally higher job-level output.

---

## Multi-Granularity Measurement Strategy

Each metric should be measured at multiple temporal granularities to ensure robustness and capture different phenomena. If the pattern holds across granularities, findings are more credible.

### Burstiness: CV of Commits

| Granularity | Definition | What It Captures |
|-------------|------------|------------------|
| **Hourly** | CV of commits per hour (working hours only) | Within-day concentration; focus blocks |
| **Daily** | CV of commits per day | Day-to-day variation; productive vs idle days |
| **Weekly** | CV of commits per week | Week-to-week rhythm; intense vs light weeks |
| **Bi-weekly** | CV of commits per 2-week sprint | Sprint-level variation (if applicable) |

**Primary**: Daily and Weekly (most interpretable for work patterns)
**Secondary**: Hourly (may be noisy), Bi-weekly (if sprint-based teams)

**Context considerations**:
- Hourly: Filter to working hours (9am-6pm local) or analyze 24h patterns separately
- Daily: Use weekdays only vs all days as separate analyses
- Weekly: Calendar weeks vs rolling 7-day windows

### Burstiness: Active Time Ratio

| Granularity | Definition | What It Captures |
|-------------|------------|------------------|
| **Active hours per day** | Hours with ≥1 commit / working hours | Within-day spread of effort |
| **Active days per week** | Days with ≥1 commit / weekdays | Weekly engagement pattern |
| **Active days per month** | Days with ≥1 commit / weekdays in month | Monthly engagement pattern |
| **Active weeks per quarter** | Weeks with ≥1 commit / weeks in quarter | Quarterly engagement pattern |

**Primary**: Active days per week, Active days per month
**Secondary**: Active hours per day (if timestamp precision allows)

**Threshold variations**: Test with different "active" thresholds:
- ≥1 commit (any activity)
- ≥3 commits (meaningful activity)
- ≥1 PR created or merged (PR-level activity)

### Throughput: Output Volume

| Granularity | Definition | What It Captures |
|-------------|------------|------------------|
| **PRs per developer-week** | Merged PRs / active devs / week | Short-term throughput fluctuation |
| **PRs per developer-month** | Merged PRs / active devs / month | Standard throughput measure |
| **PRs per developer-quarter** | Merged PRs / active devs / quarter | Smoothed throughput trend |
| **Commits per developer-week/month** | Commit count per dev per period | Finer-grained output measure |

**Primary**: PRs per developer-month (balances noise vs signal)
**Secondary**: Weekly (more observations, more noise), Quarterly (fewer observations, less noise)

**Alternative output units**:
- PRs merged
- Commits pushed
- Lines of code changed (requires additional data)
- Issues closed
- Reviews completed

### Slack: Inter-PR Gap

| Granularity | Definition | What It Captures |
|-------------|------------|------------------|
| **Hours** | Hours from PR merge to next PR creation | Fine-grained slack detection |
| **Business hours** | Same, excluding nights/weekends | Work-time slack only |
| **Days** | Calendar days between PRs | Coarser slack pattern |
| **Business days** | Weekdays between PRs | Work-day slack pattern |

**Primary**: Hours (most precise), Business days (most interpretable)
**Secondary**: Calendar days (simplest)

**Gap definition variations**:
- Merge → next PR creation (same author)
- Merge → next commit (same author, any repo)
- Merge → next commit (same author, same repo)
- Last commit on PR → first commit on next PR

### Velocity: Lead Times

| Granularity | Definition | What It Captures |
|-------------|------------|------------------|
| **Hours** | PR creation to merge in hours | Standard precision |
| **Business hours** | Same, excluding nights/weekends | "Work effort" time |
| **Days** | Calendar days | Coarser view |
| **Business days** | Weekdays only | Work calendar view |

**Aggregation variations**:
- Median (robust to outliers)
- Mean (sensitive to long tails)
- 75th/90th percentile (tail behavior)
- Trimmed mean (compromise)

### Summary: Recommended Multi-Granularity Matrix

| Metric | Primary | Secondary | Tertiary |
|--------|---------|-----------|----------|
| CV of commits | Daily, Weekly | Hourly | Bi-weekly |
| Active ratio | Days/week, Days/month | Hours/day | Weeks/quarter |
| Throughput | PRs/dev-month | PRs/dev-week | PRs/dev-quarter |
| Inter-PR gap | Hours, Business days | Calendar days | Business hours |
| Lead time | Hours | Business hours, Days | Percentile variants |

### Why Multiple Granularities Matter

1. **Robustness**: If burstiness ↑ at daily, weekly, AND bi-weekly levels, the finding is robust
2. **Different phenomena**: Hourly burstiness might capture "flow states"; weekly might capture "crunch weeks"
3. **Noise vs signal tradeoff**: Finer granularity = more observations but more noise
4. **Practical interpretation**: Some granularities map better to real work patterns (sprints, weeks)
5. **Publication strength**: Showing consistent patterns across granularities is more convincing

---

## Robustness Checks

### For Throughput Measures
1. **Alternative throughput metrics**: Commits per developer-month, issues closed per developer-month
2. **Weighted throughput**: PRs weighted by size (LOC) to ensure not just counting smaller PRs
3. **Exclude bot accounts**: Filter out automated PRs

### For Burstiness Measures
1. **Cross-granularity consistency**: Pattern should hold at daily, weekly, and bi-weekly levels (see Multi-Granularity section)
2. **Gini coefficient**: Alternative inequality measure for commit distribution
3. **Autocorrelation**: Test if productive days cluster (positive autocorrelation = bursty)
4. **Peak-to-median ratio**: Alternative measure of how extreme peaks are

### For Slack Measures
1. **Different gap definitions**: Merge-to-next-commit vs. merge-to-next-PR-creation
2. **Excluding weekends/holidays**: Business days only
3. **Censoring**: Handle developers who leave or become inactive

### General Robustness
1. **Placebo treatment dates**: Test for effects at fake treatment dates
2. **Pre-trends test**: Verify parallel trends assumption for DiD
3. **Donut specification**: Exclude months immediately around treatment date
4. **Heterogeneity**: By repo size, developer experience, language

---

## Data Requirements (GitHub Archive)

### Required Fields

| Field | Source Table | Used For |
|-------|--------------|----------|
| PR created_at | PullRequestEvent | Lead time, throughput |
| PR merged_at | PullRequestEvent | Lead time, throughput |
| PR author | PullRequestEvent | Per-developer metrics |
| Commit timestamp | PushEvent | Burstiness, active days |
| Commit author | PushEvent | Per-developer metrics |
| Review created_at | PullRequestReviewEvent | Time to first review |
| Repo language | Repository metadata | Treatment/control assignment |
| Repo id | All events | Fixed effects |

### Sample Size Considerations

For adequate statistical power:
- Minimum 1,000 repositories per language group
- Minimum 12 months pre and post treatment
- Minimum 10 active developers per repo for developer-level metrics

### Query Strategy

1. **Pre-period**: January 2021 - November 2022 (23 months)
2. **Post-period**: December 2022 - December 2024 (25 months)
3. **Aggregate to monthly panels** for ITS/DiD estimation

---

## Implementation Priority

### Phase 1: Core Paradox Test
1. PR Lead Time (velocity) - existing metric, validate
2. PRs per Developer-Month (throughput) - **new, primary outcome**
3. Basic ITS + DiD estimation

### Phase 2: Mechanism Evidence
4. CV of Weekly Commits (burstiness)
5. Inter-PR Gap (slack)
6. Mechanism analysis

### Phase 3: Robustness
7. Active Coding Days Ratio
8. All robustness checks
9. Heterogeneity analysis

---

## Comparison to Original Strategy

| Aspect | Original | Revised |
|--------|----------|---------|
| "Job" definition | Merge-to-release latency | Cumulative output per calendar period |
| Primary outcome | Release lead time | PRs per developer-month |
| Mechanism | Coordination overhead | Slack absorption |
| Key insight | Pipeline bottleneck shifted | Saved time not reinvested |
| Burstiness | Not measured | Central to hypothesis |
| Slack | Implicit | Explicitly measured |

The revised strategy provides a more direct test of the productivity paradox and identifies the specific mechanism (slack absorption) rather than just documenting that downstream stages are slow.
