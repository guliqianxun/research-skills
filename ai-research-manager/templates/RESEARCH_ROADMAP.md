---
id: research-roadmap-main
doc_type: roadmap
project: {Research Program}
lifecycle_status: active
index_file: docs/research/index.json
managed_by: scripts/research_index.py
source_of_truth: markdown
---

# {Research Program} - Research Roadmap

> This document tracks the active research portfolio, study priorities, and gate status.
>
> Study state machine: framing -> designing -> executing -> evaluating -> synthesizing -> closed
>
> Machine index: `docs/research/index.json`
> Query script: `scripts/research_index.py`

---

## Active Studies

| # | Study | Priority | Research Stage | Lifecycle Status | Study Doc | Notes |
|---|-------|----------|----------------|------------------|-----------|-------|
| 1 | {Study Title} | P1 | framing | active | [studies/{study-slug}.md](studies/{study-slug}.md) | id: rs-001 |

---

## Upcoming Studies

| # | Study | Priority | Planned Stage | Trigger | Notes |
|---|-------|----------|---------------|---------|-------|
| 1 | {Future Study} | P2 | framing | Need benchmark access | |

---

## Gate Review

| Study ID | G1 (Hypotheses formulated) | G2 (Experiment planned) | G3 (Evaluation passed) | Next Action |
|----------|----------------------------|-------------------------|------------------------|-------------|
| rs-001 | pending | pending | pending | Finalize measurable success criteria |

---

## Portfolio Risks

- Compute budget concentration on a single method family
- Missing external baseline replication for top-priority study
- Paper claims may outrun evidence if claim map is not maintained continuously
