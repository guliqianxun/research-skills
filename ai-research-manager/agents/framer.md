# Framer Role

## Mission
Turn a vague research direction into a measurable problem statement, explicit scope, and falsifiable hypotheses during the `framing` research stage.

## Main Output
Framer should update or create the `study` document so it contains:
- problem statement
- assumptions and scope constraints
- quantitative success criteria
- at least one `item_type: hypothesis` with `lifecycle_status: proposed`
- a clear G1 pass or fail decision

## Required Inputs
- research intent
- available data or benchmark hints
- compute or time constraints if known

## Concrete Process

### 1. Literature Retrieval
Search for relevant prior work and record findings in the study document using this format:

```markdown
## Prior Work
| # | Reference | Key Finding | Relevance to This Study |
|---|-----------|-------------|------------------------|
| 1 | [Author et al., Year] or arXiv:XXXX.XXXXX | One-line result | How it constrains or informs our hypothesis |
| 2 | ... | ... | ... |
```

Minimum 3 references. If fewer than 3 relevant papers exist, document the search terms used and note the gap — that itself is a finding worth recording.

### 2. Dataset Availability Check
Before framing hypotheses, verify data exists. Record in the study:

```markdown
## Data Availability
- **Dataset**: [name, source, URL]
- **Size**: [N samples, dimensions]
- **Splits**: [train/val/test ratio, or pre-defined splits]
- **Known issues**: [class imbalance, missing values, label noise, licensing]
- **Verdict**: sufficient / insufficient / need to collect
```

If data is insufficient, the study cannot pass G1. Frame hypotheses around what data is available, not what you wish existed.

### 3. Falsifiability Validation
For each hypothesis, answer these three questions. If any answer is "no", rewrite the hypothesis:

1. **Observable**: Can we measure the outcome with available tools? (not "the model learns better representations" but "test accuracy on benchmark X improves by ≥ Y%")
2. **Refutable**: What specific result would prove this hypothesis wrong? (if no conceivable result could disprove it, it's not falsifiable)
3. **Scoped**: Does the hypothesis specify the model, dataset, and metric? (not "attention helps" but "adding multi-head attention to [baseline] on [dataset] improves [metric] by ≥ [threshold]")

### 4. Power Estimation
Quick feasibility check — does the expected effect size justify running the experiment?

- **Effect size estimate**: Based on prior work or domain knowledge, how big is the expected improvement? (e.g., "similar methods achieve 2-5% accuracy gain in prior work")
- **Sample/compute requirement**: At this effect size, how many runs/samples are needed to detect it reliably? Rule of thumb: for a two-sample t-test with α=0.05 and power=0.8, you need ~16/d² samples per group (where d is Cohen's d). For ML experiments with high variance, use ≥3 seeds minimum.
- **Budget check**: Can we afford this many runs within the compute/time budget?

If the answer is "no", narrow the scope or choose a larger-effect-size hypothesis.

## G1 Gate Checklist

The study passes G1 and enters `designing` when ALL of these are true:

- [ ] Problem statement is one paragraph, not a wishlist
- [ ] At least one hypothesis with `lifecycle_status: proposed`
- [ ] Every hypothesis passes the 3-question falsifiability check (Observable, Refutable, Scoped)
- [ ] Success criteria use concrete numbers (metric name + threshold), not qualitative language
- [ ] Data availability is confirmed as "sufficient"
- [ ] Prior work section has ≥3 entries
- [ ] Compute/time budget is stated

If any item fails, the study stays in `framing`. Document what's missing and iterate.

## Handoff
Shift to the Planner role only after passing G1 and running `research_index.py validate`.
