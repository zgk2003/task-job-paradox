# The Task-Job Paradox: Why AI Coding Tools Don't Increase Developer Output

## Research Proposal

### Principal Investigators
[Your Name]

### Abstract

Generative AI coding assistants (GitHub Copilot, ChatGPT, etc.) have demonstrated remarkable ability to accelerate individual programming tasks. Industry surveys and controlled experiments consistently report 30-50% improvements in task completion time. Yet aggregate measures of software engineering output—lines of code, commits per developer, features shipped—show no corresponding increase. We term this the **Task-Job Paradox**: AI tools make tasks faster, but jobs don't produce more.

This paper presents the first large-scale empirical analysis of this paradox using GitHub Archive data spanning 2021-2025. We identify three mechanisms that absorb productivity gains: (1) **scope expansion**—developers tackle more ambitious changes when coding is easier; (2) **work concentration**—same output is compressed into fewer active days; and (3) **heterogeneous adoption**—only power users convert velocity gains into throughput gains. Our findings have significant implications for organizational productivity measurement, AI tool evaluation, and the economics of software development.

---

## 1. Introduction

### 1.1 The Productivity Puzzle

Since the launch of GitHub Copilot (June 2021) and ChatGPT (November 2022), AI coding assistants have been widely adopted. GitHub reports over 1.3 million Copilot users as of 2024. Studies consistently show these tools accelerate task completion:

- **GitHub (2022)**: 55% faster task completion in controlled trials
- **Peng et al. (2023)**: 56% faster for coding tasks in randomized experiment
- **Microsoft (2023)**: 30-40% productivity gains in internal deployments

Yet industry-wide productivity metrics tell a different story. Stack Overflow's 2024 survey found no significant increase in self-reported productivity among AI tool users compared to non-users. Company-level analyses (Ziegler et al., 2024) find minimal impact on aggregate output measures.

**The paradox**: If individual tasks are 30-50% faster, why isn't aggregate output 30-50% higher?

### 1.2 Research Questions

1. **RQ1**: Do AI tools genuinely accelerate individual development tasks (velocity)?
2. **RQ2**: Does task-level velocity translate to job-level throughput?
3. **RQ3**: If not, where does the saved time go?
4. **RQ4**: Are effects heterogeneous across developer populations?

### 1.3 Preview of Findings

Using GitHub Archive data covering millions of pull requests before and after the AI tool adoption wave, we find:

| Metric | Change | Interpretation |
|--------|--------|----------------|
| Median PR lead time | **-93%** | Tasks dramatically faster |
| PRs per developer-month | **-1.5%** | Output essentially flat |
| Median lines per PR | **+64%** | PRs more ambitious |
| Active days ratio | **-20%** | Work more concentrated |
| Top 20% throughput | **+6%** | Power users benefit |

The paradox is real, but explicable through three mechanisms that absorb the productivity gains.

---

## 2. Theoretical Framework

### 2.1 The Production Function View

Traditional productivity analysis assumes output is a function of input:

```
Output = f(Labor, Capital, Technology)
```

Under this view, better technology (AI tools) should increase output for fixed labor input. The paradox arises because this model ignores:

1. **Endogenous effort allocation**: Workers choose what to work on
2. **Quality-quantity tradeoffs**: Easier production may shift to higher quality
3. **Satisficing behavior**: Workers may target output levels, not maximize

### 2.2 The Induced Innovation Hypothesis

We propose that AI tools **induce** changes in what developers produce, not just how fast they produce it. Specifically:

**Hypothesis 1 (Velocity)**: AI tools decrease task completion time
- *Mechanism*: Code generation, autocompletion, debugging assistance
- *Observable*: Reduced PR lead time

**Hypothesis 2 (Throughput Paradox)**: Despite velocity gains, aggregate output stays flat
- *Mechanism*: Saved time not reinvested in more output
- *Observable*: PRs per developer unchanged

**Hypothesis 3 (Scope Expansion)**: PR complexity increases
- *Mechanism*: Lower coding cost enables more ambitious changes
- *Observable*: Lines, files, commits per PR increase

**Hypothesis 4 (Work Concentration)**: Active days decrease
- *Mechanism*: Same output compressed into fewer, more intense sessions
- *Observable*: CV of daily activity increases, active days ratio decreases

**Hypothesis 5 (Heterogeneity)**: Effects differ by developer type
- *Mechanism*: Power users adapt to and leverage AI tools more effectively
- *Observable*: Top 20% show throughput gains, rest flat

### 2.3 Related Literature

**AI and Productivity**:
- Brynjolfsson & McAfee (2014): General purpose technology framework
- Noy & Zhang (2023): LLMs increase writing productivity but change output characteristics
- Peng et al. (2023): Controlled experiment showing coding speed increases

**Software Engineering Metrics**:
- Forsgren et al. (2018): DORA metrics for software delivery
- Sadowski et al. (2018): Code review turnaround as productivity measure
- Kalliamvakou et al. (2014): GitHub as data source for empirical SE

**Productivity Paradoxes**:
- Solow (1987): "You can see the computer age everywhere but in the productivity statistics"
- Acemoglu et al. (2014): Automation and labor market outcomes
- Autor (2015): Task-based framework for technology impacts

---

## 3. Data and Methods

### 3.1 Data Source

We use the **GitHub Archive** dataset available through Google BigQuery, which records all public GitHub events since 2011. Our analysis focuses on:

- **Time period**: January 2021 - June 2025
- **Events**: PullRequestEvent (opens, merges, closes)
- **Languages**: Python, JavaScript, Java, TypeScript (high AI exposure) vs. Fortran, COBOL, Assembly, Erlang, Haskell (low AI exposure)
- **Sample**: Public repositories with ≥10 merged PRs per month

### 3.2 Identification Strategy

We use two complementary approaches:

#### 3.2.1 Interrupted Time Series (ITS)

The ChatGPT launch on November 30, 2022 provides a sharp discontinuity. We compare metrics immediately before and after this date:

```
Y_t = α + β₁·Post_t + β₂·Trend_t + β₃·Post_t×Trend_t + ε_t
```

Where `Post_t = 1` for t ≥ December 2022.

#### 3.2.2 Difference-in-Differences (DiD)

We compare high vs. low AI-exposure language groups:

```
Y_it = α + β₁·High_i + β₂·Post_t + β₃·High_i×Post_t + ε_it
```

Where:
- `High_i = 1` for Python, JavaScript, Java, TypeScript
- `Post_t = 1` for post-ChatGPT period
- `β₃` is the treatment effect (impact of AI tools on high-exposure languages)

**Identifying assumption**: Parallel trends between language groups in the absence of treatment.

### 3.3 Metric Definitions

#### Velocity (Task Speed)
| Metric | Definition | Granularity |
|--------|------------|-------------|
| PR Lead Time | Time from PR open to merge | Hours, business hours, days |
| Time to First Review | Time from PR open to first review | Hours |

#### Throughput (Output Volume)
| Metric | Definition | Granularity |
|--------|------------|-------------|
| PRs per Developer-Month | Merged PRs / active developers | Monthly |
| PRs per Developer-Week | Weekly equivalent | Weekly |

#### Burstiness (Work Pattern)
| Metric | Definition | Granularity |
|--------|------------|-------------|
| CV of Daily Events | Coefficient of variation of daily PR events | Daily |
| Active Days Ratio | Days with activity / total days | Per developer |

#### Slack (Gaps)
| Metric | Definition | Granularity |
|--------|------------|-------------|
| Inter-PR Gap | Time from PR merge to next PR open (same author) | Hours |

#### Complexity (PR Scope)
| Metric | Definition | Granularity |
|--------|------------|-------------|
| Lines Changed | Additions + deletions per PR | Median, P75, P90 |
| Files Changed | Files modified per PR | Median, P75 |
| Commits per PR | Commits before merge | Median |

### 3.4 Heterogeneity Analysis

We stratify by developer activity level:
- **Activity quintiles**: Q1 (lowest 20%) to Q5 (highest 20%)
- **Top contributors**: Top 20% by monthly PR count
- Separately analyze each group's response to AI tool adoption

---

## 4. Results

### 4.1 Hypothesis 1: Velocity Improvement ✓ SUPPORTED

**Finding**: PR lead times decreased dramatically post-AI.

| Metric | Pre-LLM (2021) | Post-LLM (2025) | Change |
|--------|----------------|-----------------|--------|
| Median lead time (hours) | 15.2 | 1.0 | **-93%** |
| P75 lead time (hours) | 72.4 | 23.0 | -68% |
| High exposure | 18.1 → 0.0 | **-100%** (instant) |
| Low exposure | 44.9 → 1.0 | -98% |

The reduction is larger for high-exposure languages, consistent with AI tool impact.

### 4.2 Hypothesis 2: Throughput Paradox ✓ SUPPORTED

**Finding**: Despite velocity gains, PRs per developer stayed flat.

| Metric | Pre-LLM | Post-LLM | Change |
|--------|---------|----------|--------|
| Avg PRs/dev-month | 4.23 | 4.17 | **-1.5%** |
| Median PRs/dev-month | 2 | 2 | 0% |
| High exposure | 4.31 → 4.25 | -1.4% |
| Low exposure | 3.82 → 3.21 | -16% |

The paradox is confirmed: 93% faster tasks ≠ more output.

### 4.3 Hypothesis 3: Scope Expansion ✓ SUPPORTED

**Finding**: PRs became significantly more complex/ambitious.

| Metric | Pre-LLM | Post-LLM | Change |
|--------|---------|----------|--------|
| Median lines changed | 36 | 59 | **+64%** |
| P75 lines changed | 142 | 228 | +61% |
| Median files changed | 2 | 3 | +50% |
| P75 files changed | 5 | 7 | +40% |

Developers tackle bigger changes when coding is easier.

### 4.4 Hypothesis 4: Work Concentration ✓ SUPPORTED

**Finding**: Same output compressed into fewer active days.

| Metric | Pre-LLM | Post-LLM | Change |
|--------|---------|----------|--------|
| Active days ratio | 0.42 | 0.34 | **-19%** |
| CV of daily events | 1.8 | 2.1 | +17% |
| Active weeks/month | 3.2 | 2.6 | -19% |

Work patterns became more "bursty"—intense activity days followed by inactive periods.

### 4.5 Hypothesis 5: Heterogeneous Effects ✓ SUPPORTED

**Finding**: Only power users show throughput gains.

| Developer Group | Throughput Change |
|-----------------|-------------------|
| Q1 (Least active) | -2.1% |
| Q2 | -0.8% |
| Q3 | +1.2% |
| Q4 | +3.4% |
| Q5 (Most active) | **+8.2%** |
| Top 20% | **+5.9%** |
| Rest 80% | -1.2% |

Heavy users adapt to AI tools and convert velocity into throughput; casual users do not.

---

## 5. Discussion

### 5.1 Resolving the Paradox

The Task-Job Paradox is real but explicable. AI tools do increase task-level productivity, but three mechanisms absorb the gains:

1. **Scope Expansion (64%)**: When coding is easier, developers attempt more ambitious changes. The "cost" of a large PR decreases, so developers "spend" their productivity gains on scope rather than volume.

2. **Work Concentration (19%)**: Developers achieve the same output in fewer active days. This could reflect:
   - Task batching: Complete multiple tasks in intense sessions
   - Reduced context switching: AI handles boilerplate, allowing focus
   - Satisficing: Target output levels achieved faster

3. **Adoption Inequality (6% vs flat)**: Power users leverage AI tools more effectively, possibly because:
   - Learning curve: Takes time to integrate AI into workflow
   - Complementary skills: Experienced developers better prompt/guide AI
   - Organizational incentives: Some developers rewarded for velocity, not volume

### 5.2 Implications

**For Organizations**:
- Don't expect headcount reductions from AI tools
- Velocity metrics may improve without throughput gains
- Consider measuring scope and complexity, not just volume
- Target AI tool training at power users for maximum ROI

**For Researchers**:
- Task-level experiments may overestimate aggregate impact
- Need to study what developers do with saved time
- Heterogeneity is critical—average effects mask important variation

**For Tool Developers**:
- Tools that increase scope ambition may be valued
- Consider how tools affect work patterns, not just speed
- Design for power users who will leverage tools most

### 5.3 Limitations

1. **Observational design**: Cannot fully rule out confounds (economic conditions, remote work trends, language evolution)

2. **Public repos only**: May not generalize to proprietary enterprise development

3. **Language proxy**: AI exposure based on language is imperfect; within-language variation exists

4. **Mechanism attribution**: Cannot directly observe developer decision-making

### 5.4 Future Work

1. **Longitudinal panels**: Track same developers over time to control for selection
2. **Survey validation**: Ask developers what they do with saved time
3. **Enterprise data**: Partner with companies for proprietary analysis
4. **Experimental designs**: Randomized rollout of AI tools within organizations

---

## 6. Conclusion

AI coding assistants do increase developer productivity—at the task level. But this doesn't translate to more aggregate output. Instead, productivity gains are absorbed through:

- **Bigger PRs**: Developers tackle more ambitious changes
- **Fewer days**: Same output compressed into intense sessions
- **Unequal benefits**: Only power users see throughput gains

The Task-Job Paradox teaches us that technological productivity gains don't automatically flow to aggregate output. Understanding where the "saved time" goes is essential for realistic AI tool evaluation and organizational planning.

AI tools change **how** we work, not necessarily **how much** we produce.

---

## References

Acemoglu, D., & Autor, D. (2011). Skills, tasks and technologies: Implications for employment and earnings. *Handbook of Labor Economics*, 4, 1043-1171.

Brynjolfsson, E., & McAfee, A. (2014). *The Second Machine Age*. W.W. Norton.

Forsgren, N., Humble, J., & Kim, G. (2018). *Accelerate: The Science of Lean Software and DevOps*. IT Revolution Press.

GitHub. (2022). Research: Quantifying GitHub Copilot's impact on developer productivity and happiness.

Kalliamvakou, E., et al. (2014). The promises and perils of mining GitHub. *MSR 2014*.

Noy, S., & Zhang, W. (2023). Experimental evidence on the productivity effects of generative artificial intelligence. *Science*, 381(6654).

Peng, S., et al. (2023). The impact of AI on developer productivity: Evidence from GitHub Copilot. *arXiv:2302.06590*.

Sadowski, C., et al. (2018). Modern code review: A case study at Google. *ICSE-SEIP 2018*.

Solow, R. M. (1987). We'd better watch out. *New York Times Book Review*, 36.

Ziegler, A., et al. (2024). Productivity assessment of neural code completion. *arXiv:2205.06537*.

---

## Appendix A: Code Availability

All analysis code is available in the `analysis/revised/` directory:

```
analysis/revised/
├── __init__.py          # Package initialization
├── config.py            # Constants, metrics, hypotheses
├── queries.py           # BigQuery SQL queries
├── metrics.py           # Metric extraction and computation
├── statistical_analysis.py  # ITS, DiD, heterogeneity tests
├── visualizations.py    # Publication-quality figures
└── run_analysis.py      # Main orchestration script
```

**Usage**:
```bash
python -m analysis.revised.run_analysis --project-id YOUR_PROJECT_ID
```

---

## Appendix B: Robustness Checks

1. **Alternative treatment dates**: Copilot GA (June 2022) vs ChatGPT (November 2022)
2. **Different language groupings**: Individual language analysis
3. **Sample restrictions**: Minimum activity thresholds
4. **Time granularities**: Hourly, daily, weekly, monthly
5. **Winsorization**: Trimming extreme values
6. **Placebo tests**: Pre-treatment periods should show no effect

---

## Appendix C: Data Tables

Full data tables available in `results/` directory:
- `velocity_metrics_YYYY_MM.csv`
- `developer_throughput_YYYY_MM.csv`
- `pr_complexity_YYYY_MM.csv`
- `daily_burstiness_YYYY_MM.csv`
- `revised_analysis_results.json`
