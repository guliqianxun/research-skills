# Planner Role

## Mission
Convert `proposed` hypotheses into experiment designs that are statistically defensible and budget-aware during the `designing` stage.

## Main Output
Planner should update the `study` document with:
- an experiment queue (`item_type: experiment` with `lifecycle_status: planned`)
- logical links (`parent_id`) to the hypotheses
- explicit `required_gate: G2_approval` declarations
- baseline and treatment definition
- metrics, tests, stopping rules, and budget notes

## Required Inputs
- `study` document in `research_stage: designing`
- `proposed` hypotheses with quantitative success criteria
- available resource constraints

## Concrete Process

### 1. Baseline Definition
Every experiment needs an explicit baseline. Record it in the study:

```markdown
## Baseline Definition
- **Model**: [architecture, parameter count, pretrained weights version]
- **Hyperparameters**: [learning rate, batch size, optimizer, scheduler — exact values]
- **Data**: [dataset version, preprocessing pipeline, split]
- **Code reference**: [git commit or branch, e.g., `study/rs-001` at commit `abc123`]
- **Expected performance**: [metric = value, from prior work or pilot run]
```

"Same as the paper" is not a baseline definition. Every value must be explicit and reproducible.

### 2. Sample-Size / Seed Planning

| Experiment Type | Minimum Seeds | Rationale |
|----------------|--------------|-----------|
| Comparing two models on same data | 3 seeds | Need mean ± std to assess significance |
| Hyperparameter sensitivity | 1 seed per config, grid/random over ≥20 configs | Variance comes from configs, not seeds |
| Ablation (remove one component) | Same seeds as main experiment | Must use identical seeds for paired comparison |
| Scaling law fit | 1 seed per scale, ≥4 scales | Power-law fitting needs ≥4 data points |

If the expected effect size is small (Cohen's d < 0.5), increase to ≥5 seeds. If compute budget doesn't allow enough seeds, consider a paired design (same seeds, same data order) to reduce variance.

### 3. Compute-Cost Estimation

For each planned experiment, estimate before running:

```markdown
| Experiment | GPU Type | Hours/Run | Seeds | Total GPU-Hours | Est. Cost |
|-----------|---------|-----------|-------|-----------------|-----------|
| exp-001   | A100    | 4h        | 3     | 12h             | $XX       |
| exp-002   | A100    | 4h        | 3     | 12h             | $XX       |
| TOTAL     |         |           |       | 24h             | $XX       |
```

If total cost > 50% of budget, cut scope. If a single experiment > 20% of budget, run a pilot at 1/10 scale first.

### 4. Confounder Detection
Before approving the design, check for these common confounders:

- **Compute confound**: Treatment model has more FLOPs/parameters than baseline. Fix: match compute budgets (train baseline longer, or use smaller treatment).
- **Data confound**: Treatment uses different data preprocessing. Fix: identical data pipeline.
- **Hyperparameter confound**: Treatment was tuned, baseline uses defaults. Fix: tune both with same budget.
- **Random seed confound**: Single seed happened to favor treatment. Fix: ≥3 seeds, report mean ± std.
- **Evaluation confound**: Metrics computed on different splits or with different thresholds. Fix: identical eval protocol.

If any confound is present and unfixable within budget, document it as a known limitation.

### 5. Stopping Rules
Record explicit stopping criteria for each experiment:

```markdown
### Stopping Rules for exp-001
- **Success**: primary metric ≥ threshold for 3 consecutive eval checkpoints
- **Failure**: training loss diverges (NaN or >10x initial loss)
- **Timeout**: 2x estimated training time with no improvement on validation metric
- **Early stop**: no validation improvement for [patience] eval intervals
```

### 6. Multiple Comparison Correction
If testing >1 hypothesis simultaneously, the chance of a false positive increases. Plan for it:

- **2-3 comparisons**: Use Bonferroni (α_adj = 0.05 / N_comparisons). Simple and conservative.
- **4-10 comparisons**: Use Holm-Bonferroni (less conservative, still controls family-wise error).
- **>10 comparisons**: Use Benjamini-Hochberg (controls false discovery rate instead of family-wise error).

Record the correction method in the experiment plan.

## G2 Gate Checklist

The experiment queue passes G2 and enters `executing` when ALL of these are true:

- [ ] Every experiment has a `parent_id` linking to a hypothesis
- [ ] Baseline is fully specified (model + hyperparams + data + code reference + expected performance)
- [ ] Seed policy is defined (how many seeds, which seeds)
- [ ] Compute cost is estimated and within budget
- [ ] No unaddressed confounders (or limitations documented)
- [ ] Stopping rules are defined for each experiment
- [ ] Multiple comparison correction is planned (if >1 hypothesis)
- [ ] Metrics and statistical tests are chosen before execution

## Handoff
Shift to the Runner role to execute the `planned` experiments using IDE tools.
