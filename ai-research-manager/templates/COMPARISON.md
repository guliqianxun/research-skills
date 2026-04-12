---
id: cmp-001
doc_type: comparison
title: "Comparison: [describe what is being compared]"
lifecycle_status: active
parent_id: rs-001
managed_by: scripts/research_index.py
source_of_truth: markdown
---

# Experiment Comparison: [Title]

**Study:** [rs-001](../studies/rs-001.md)
**Date:** YYYY-MM-DD
**Compared by:** [agent role or human]

## Purpose

Why this comparison exists — what decision does it inform?

> Example: "Determine whether relative position bias (exp-002) outperforms absolute position encoding (exp-001) on long-context retrieval, controlling for compute budget."

## Experiments Compared

| Field | exp-001 (Baseline) | exp-002 (Treatment) | exp-003 (Ablation) |
|-------|-------------------|--------------------|--------------------|
| **Hypothesis** | hyp-001 | hyp-001 | hyp-002 |
| **Branch** | `exp/exp-001` | `exp/exp-002` | `exp/exp-003` |
| **Status** | completed | completed | completed |
| **Config diff** | — | +relative_pos_bias | -attention_heads=4 |

## Key Parameter Differences

Highlight only the parameters that differ. Identical parameters are noise.

| Parameter | exp-001 | exp-002 | exp-003 |
|-----------|---------|---------|---------|
| position_encoding | absolute | relative_bias | absolute |
| num_heads | 8 | 8 | 4 |
| learning_rate | 1e-4 | 1e-4 | 1e-4 |

## Results Matrix

### Primary Metrics

| Metric | exp-001 | exp-002 | exp-003 | Winner |
|--------|---------|---------|---------|--------|
| Recall@10 (mean ± std) | 0.72 ± 0.03 | 0.81 ± 0.02 | 0.68 ± 0.04 | exp-002 |
| F1 (mean ± std) | 0.65 ± 0.02 | 0.67 ± 0.02 | 0.61 ± 0.03 | exp-002 |
| Latency (ms, P50) | 12.3 | 14.1 | 11.8 | exp-003 |

### Statistical Significance (pairwise)

| Comparison | Test | p-value | Cohen's d | Significant? |
|-----------|------|---------|-----------|-------------|
| exp-002 vs exp-001 (Recall@10) | Paired t-test | 0.003 | 0.85 | Yes |
| exp-002 vs exp-001 (F1) | Paired t-test | 0.12 | 0.31 | No |
| exp-003 vs exp-001 (Recall@10) | Paired t-test | 0.08 | -0.28 | No |

### Cost-Performance Tradeoff

| Experiment | GPU-hours | Recall@10 | Recall per GPU-hour |
|-----------|-----------|-----------|-------------------|
| exp-001 | 24 | 0.72 | 0.030 |
| exp-002 | 28 | 0.81 | 0.029 |
| exp-003 | 18 | 0.68 | 0.038 |

## Version History

Track how experiments evolved across iterations. Each row is a re-run or revision.

| Version | Experiment | Change from Previous | Key Result | Decision |
|---------|-----------|---------------------|------------|---------|
| v1 | exp-002 | Initial run | Recall@10 = 0.78 | Variance too high, increase seeds |
| v2 | exp-002 | 3 seeds → 5 seeds | Recall@10 = 0.81 ± 0.02 | Stable, proceed to evaluation |

## Findings

1. [Finding 1: what the comparison shows]
2. [Finding 2: surprises or anomalies]
3. [Finding 3: what this means for the hypothesis]

## Decision

Based on this comparison:
- **Recommendation:** [which experiment to adopt / what to do next]
- **Confidence:** [high / medium / low]
- **Open questions:** [what remains unclear]
- **Linked claims:** [clm-001, clm-002]
