# Task-Job Paradox - Narration Script

Total duration: ~5 minutes (8 slides, ~35-40 seconds each)

---

## Slide 1: Title (35 seconds)

**The Task-Job Paradox: Why AI Coding Tools Speed Up Tasks But Not Total Output**

Welcome to this presentation on the Task-Job Paradox. This research examines a puzzling phenomenon in software development: while AI coding tools demonstrably make individual tasks faster, we don't see a corresponding increase in total developer output.

We analyzed five years of GitHub data spanning 2021 to 2025, using the launch of ChatGPT in November 2022 as a natural experiment to identify causal effects.

---

## Slide 2: The Puzzle (40 seconds)

**What's the paradox?**

On the left, we have task-level productivity. Studies consistently show that AI coding assistants like GitHub Copilot and ChatGPT make developers 30 to 50 percent faster on individual coding tasks. Code completion, bug fixes, writing tests - all get done more quickly.

On the right, we have job-level output. When we look at aggregate metrics like the number of pull requests merged or commits made, these numbers stay roughly the same.

This raises a fundamental question: where does the saved time go? Developers are clearly faster at individual tasks, yet total output isn't increasing proportionally.

---

## Slide 3: Our Approach (45 seconds)

**How we investigated this**

For data, we collected 53 months of GitHub Archive data - that's over 11 million developer-month observations. We tracked three key metrics: velocity or how fast PRs get completed, complexity meaning the size of pull requests, and throughput or the number of PRs per developer.

For causal identification, we used two established econometric methods. First, Difference-in-Differences, comparing languages with high AI exposure like Python and JavaScript against control languages like Fortran, COBOL, and Assembly that are largely unaffected by current AI tools.

Second, Interrupted Time Series analysis around the ChatGPT launch date. This design lets us attribute changes specifically to AI tools rather than general trends.

---

## Slide 4: Finding 1 - Tasks Got Faster (40 seconds)

**Our first key finding: velocity improved significantly**

The time series on the left shows the 75th percentile lead time - how long it takes for pull requests to get merged. You can see the trend before and after the ChatGPT launch.

Our Difference-in-Differences analysis found a reduction of 4.62 hours in P75 lead time for high-exposure languages compared to the control group. This effect is statistically significant with a p-value of 0.0145.

This confirms that AI tools genuinely accelerate task completion. High-exposure languages show faster PR completion specifically after ChatGPT launched, while control languages remain unchanged.

---

## Slide 5: Finding 2 - PRs Got Bigger (40 seconds)

**Our second key finding: scope expansion**

Here's where it gets interesting. The complexity time series shows median PR size - the number of lines changed per pull request.

Our DiD analysis reveals a statistically significant increase of over 10 lines per PR in high-exposure languages. The p-value is less than 0.0001 - this is a highly robust finding.

What this tells us is that developers aren't using saved time to produce more pull requests. Instead, they're tackling larger changes. The speed gains get absorbed into increased ambition.

---

## Slide 6: The DiD Evidence (35 seconds)

**Visual comparison of our DiD results**

These two charts show the Difference-in-Differences estimates side by side.

On the left, velocity: high-exposure languages got faster while the control group didn't change. On the right, complexity: high-exposure languages show larger PRs while controls remain stable.

The fact that our control group - Fortran, COBOL, and Assembly - shows no change is crucial. It confirms we're seeing a causal effect of AI tools, not just a general trend affecting all software development.

---

## Slide 7: The Mechanism (40 seconds)

**How does this work?**

The flow is straightforward. AI coding tools make individual tasks faster. But instead of producing more units of output, developers expand the scope of what they attempt.

The key insights are: First, developers don't produce more PRs - they make bigger ones. Second, time saved on coding is absorbed by increased ambition. Third, quality and scope preferences adjust dynamically to available speed.

Think of it like highway expansion. Adding lanes doesn't reduce commute times because more people start driving. Similarly, faster coding doesn't increase output because developers tackle more ambitious changes.

---

## Slide 8: Conclusion (35 seconds)

**AI changes HOW we work, not HOW MUCH we produce**

This is our main takeaway. AI tools are transforming the nature of software development, enabling developers to tackle more complex challenges - but this doesn't translate to more units of output.

The implications are significant. First, productivity metrics need to account for scope changes, not just count outputs. Second, AI tools enable ambition more than pure efficiency. Third, job-level output may not be the right measure of the value AI tools provide.

And a meta note: this research was itself conducted with significant AI assistance. We experienced the paradox firsthand - working faster but expanding scope rather than finishing sooner.

Thank you for your attention.

---

## Total Runtime Summary

| Slide | Topic | Duration |
|-------|-------|----------|
| 1 | Title | 35s |
| 2 | The Puzzle | 40s |
| 3 | Our Approach | 45s |
| 4 | Velocity | 40s |
| 5 | Scope | 40s |
| 6 | DiD Evidence | 35s |
| 7 | Mechanism | 40s |
| 8 | Conclusion | 35s |
| **Total** | | **~5 min 10 sec** |
