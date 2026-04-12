---
id: claimmap-rs-001
doc_type: claim_map
title: Claim map for RS-001
lifecycle_status: active
parent_id: rs-001
managed_by: scripts/research_index.py
source_of_truth: markdown
---

# Claim Map for RS-001

## Core Claims

- Claim: {Main claim text} <!-- id: clm-001; item_type: claim; evidence_ids: exp-002, exp-003; claim_status: drafting -->
- Claim: {Secondary claim text} <!-- id: clm-002; item_type: claim; evidence_ids: exp-004; claim_status: drafting -->

## Evidence Bundles

| Claim ID | Supporting Experiments | Tables / Figures | Statistical Evidence | Risks |
|----------|------------------------|------------------|----------------------|------|
| clm-001 | exp-002, exp-003 | Fig.2, Tbl.1 | 95% CI excludes zero | Limited external baseline |
| clm-002 | exp-004 | Tbl.2 | Bootstrap CI stable | Small sample size |

## Missing Evidence

- clm-001 needs failure-case visualization
- clm-002 needs stronger baseline replication
