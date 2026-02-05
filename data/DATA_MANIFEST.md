# Data Manifest: Task-Job Paradox Panel Data

## Overview

This directory contains monthly panel data from GitHub Archive (2021-01 to 2025-06) for the Task-Job Paradox empirical analysis.

**Collection Date**: [To be filled on collection]
**BigQuery Project**: task-job-paradox-001
**Total Months**: 54 (Jan 2021 - Jun 2025)
**Treatment Date**: November 30, 2022 (ChatGPT launch)

## Directory Structure

```
data/
├── panel/
│   ├── velocity/           # PR lead times, review times
│   │   └── velocity_YYYY_MM.csv
│   ├── throughput/         # Developer-level PR counts
│   │   └── throughput_YYYY_MM.csv
│   ├── complexity/         # PR size metrics (lines, files, commits)
│   │   └── complexity_YYYY_MM.csv
│   ├── burstiness/         # Daily activity patterns
│   │   └── burstiness_YYYY_MM.csv
│   ├── slack/              # Inter-PR gaps
│   │   └── slack_YYYY_MM.csv
│   └── controls/           # Control variables
│       └── controls_YYYY_MM.csv
├── aggregated/             # Pre-aggregated monthly summaries
│   └── monthly_summary.csv
├── collection_log.json     # Tracks what's been collected
└── DATA_MANIFEST.md        # This file
```

## Data Schemas

### velocity_YYYY_MM.csv
Monthly aggregate velocity metrics by language exposure group.

| Column | Type | Description |
|--------|------|-------------|
| year_month | string | YYYY-MM format |
| high_exposure | bool | True for Python/JS/Java/TS |
| num_prs | int | Count of merged PRs |
| num_repos | int | Distinct repositories |
| num_authors | int | Distinct PR authors |
| avg_lead_time_hours | float | Mean time from open to merge |
| median_lead_time_hours | float | Median lead time |
| p75_lead_time_hours | float | 75th percentile |
| p90_lead_time_hours | float | 90th percentile |
| avg_time_to_review_hours | float | Mean time to first review |
| median_time_to_review_hours | float | Median time to first review |

### throughput_YYYY_MM.csv
Developer-level throughput for heterogeneity analysis.

| Column | Type | Description |
|--------|------|-------------|
| author | string | GitHub username (hashed) |
| high_exposure | bool | Primary language exposure |
| prs_merged_month | int | PRs merged this month |
| active_weeks | int | Weeks with ≥1 PR |
| cv_weekly_prs | float | CV of weekly PR counts |
| total_additions | int | Lines added |
| total_deletions | int | Lines deleted |

### complexity_YYYY_MM.csv
PR complexity metrics by exposure group.

| Column | Type | Description |
|--------|------|-------------|
| year_month | string | YYYY-MM format |
| high_exposure | bool | Language exposure |
| num_prs | int | Sample size |
| median_churn | float | Median lines changed |
| p75_churn | float | 75th percentile |
| p90_churn | float | 90th percentile |
| median_files | float | Median files changed |
| median_commits | float | Median commits per PR |

### burstiness_YYYY_MM.csv
Developer-level daily activity patterns.

| Column | Type | Description |
|--------|------|-------------|
| author | string | GitHub username |
| high_exposure | bool | Language exposure |
| active_days | int | Days with any PR activity |
| total_days_in_month | int | Calendar days in month |
| active_days_ratio | float | active_days / total_days |
| cv_daily_events | float | CV of daily event counts |
| total_events | int | Total PR events |

### slack_YYYY_MM.csv
Inter-PR gap analysis (with 3-month lookback for censoring fix).

| Column | Type | Description |
|--------|------|-------------|
| author | string | GitHub username |
| high_exposure | bool | Language exposure |
| num_gaps | int | Number of inter-PR gaps observed |
| avg_gap_hours | float | Mean gap duration |
| median_gap_hours | float | Median gap |
| p75_gap_hours | float | 75th percentile |
| right_censored | bool | True if last PR has no observed next |

### controls_YYYY_MM.csv
Control variables for regression analysis.

| Column | Type | Description |
|--------|------|-------------|
| year_month | string | YYYY-MM format |
| high_exposure | bool | Language exposure |
| avg_repo_age_days | float | Mean repo age |
| avg_repo_stars | float | Mean repo stars |
| avg_author_tenure_days | float | Mean author account age |
| pct_bot_prs | float | Percentage of bot-authored PRs |
| avg_team_size | float | Mean contributors per repo |
| weekend_pct | float | Percentage of weekend activity |

## Control Variables & Interaction Terms

### Recommended Controls
1. **Repo characteristics**: age, stars, team size (proxy for project maturity)
2. **Author characteristics**: tenure, activity level quintile
3. **Temporal**: month fixed effects, weekend/weekday
4. **Bot filtering**: exclude or control for bot PRs

### Interaction Terms to Test
1. **Language × Post-treatment**: Core DiD interaction
2. **Activity quintile × Post-treatment**: Heterogeneous effects
3. **Repo size × Post-treatment**: Do large projects respond differently?
4. **Author tenure × Post-treatment**: Do experienced devs adapt faster?

## Query Improvements (vs. v1)

1. **Multi-month windows for slack/review time**: Uses 3-month lookback to avoid right-censoring at month boundaries
2. **Actual days in month**: Burstiness uses EXTRACT(DAY FROM LAST_DAY()) instead of fixed 30
3. **Zero-week handling**: Throughput CV includes zero-activity weeks
4. **Bot filtering**: Optional flag to exclude [bot] authors

## Collection Status

| Metric | Months Collected | Last Updated |
|--------|------------------|--------------|
| velocity | 0/54 | - |
| throughput | 0/54 | - |
| complexity | 0/54 | - |
| burstiness | 0/54 | - |
| slack | 0/54 | - |
| controls | 0/54 | - |

## Estimated BigQuery Costs

- Velocity: ~$0.10/month × 54 = ~$5.40
- Throughput: ~$0.50/month × 54 = ~$27.00
- Complexity: ~$0.20/month × 54 = ~$10.80
- Burstiness: ~$0.30/month × 54 = ~$16.20
- Slack: ~$0.40/month × 54 = ~$21.60
- Controls: ~$0.15/month × 54 = ~$8.10

**Total estimated**: ~$89 (actual may vary based on data volume)

## Notes

- Data before 2015 may have quality issues in GitHub Archive
- Language detection uses repo primary language, not file-level
- "High exposure" = Python, JavaScript, Java, TypeScript
- "Low exposure" = Fortran, COBOL, Assembly, Erlang, Haskell
- Treatment date: 2022-11-30 (ChatGPT public launch)
