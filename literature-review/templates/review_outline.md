# <Title: what is reviewed, e.g. "CMIP6 simulation of X: a review">

<!-- Rules: references/writing-protocol.md. Every finding cites its ID in the same sentence; every number
     comes from claims.tsv; delete these comments when the section is written. -->

> **Scope**: <phenomenon, model generation/sub-projects, aspects covered (climatology, bias causes,
> generational change, projections)>.
> **Evidence base**: <N> core papers (<years>) from OpenAlex/Crossref/Semantic Scholar after
> deduplication, relevance scoring and screening; <F> read in full (★), <N−F> used through title and
> abstract (☆). Counts from `02_metadata/stats.json`. Citation format: `Author Year（ID）`.

## Abstract
<!-- 3–5 headline findings already written in the body, same numbers, same IDs. Nothing new. Confirmed by the user. -->

## 1. Introduction
<!-- Why the phenomenon matters and what changed in this model generation, each statement cited from the
     evidence base or an approved manual addition. Uncited scope sentences are marked no-cite. -->

## 2. Diagnostics and methods
<!-- The measurement paradigms used in the reviewed papers, with their parameters from the ledger, and the
     papers that compared them. No textbook defaults. -->

## 3. Present-day fidelity
### 3.1 <Region / hemisphere A> — by season
### 3.2 <Region / hemisphere B>
### 3.3 <Intensity / extremes>
### 3.4 <Consistency between metrics>
<!-- Per finding: models (N), period, season, reference data, metric. Disagreements reported side by side;
     a reason only if a source gives one. -->

## 4. Causes of systematic bias
### 4.1 <Mechanism 1>
### 4.2 <Mechanism 2>
### 4.3 <Resolution vs parameterisation>
<!-- Each mechanism names the experiment that supports it (AMIP vs coupled, sensitivity runs …). Without an
     attribution design: "associated with", or the authors' interpretation labelled as theirs. -->

## 5. Improvement over the previous generation; multi-model spread
<!-- Only comparisons the papers made; computed changes are derived claims labelled 本文据…计算. -->

## 6. Future projections
### 6.1 <Signal 1>
### 6.2 <Signal 2>
### 6.3 <Uncertain signals>
### 6.4 Forced signal vs internal variability
<!-- Scenario, period vs baseline, N models for every number. "Robust" only with >=3 agreeing studies and none
     disagreeing in the set. -->

## 7. Uncertainties, limitations and outlook
<!-- Only limitations and future work stated by the sources, each attributed and cited. No gaps, priorities
     or agenda of your own; no 本文认为. -->

## 8. Conclusions
<!-- Restates body findings with the same IDs. Nothing new. Confirmed by the user. -->

## References
<Paste `06_review/references_by_author.md`.>

## Appendix: method and reproducible materials
- Databases, search date, query (anchor × concepts), year window — from `01_retrieval/search_log.json`
- Funnel — from `02_metadata/stats.json`: raw → unique → near-duplicates merged → curated → tiers A/B/C →
  excluded by type → manual additions → selected → full text / abstract only
- Screening rubric — `02_metadata/screen/RUBRIC.md`; claims ledger — `06_review/claims.tsv`
- Files: `papers_selected.csv`, `core_papers.xlsx`, `04_notes/`, `05_figures/`, `scripts/litreview.py`
