# <Title: what is reviewed, e.g. "CMIP6 simulation of X: a review">

> **Scope**: <phenomenon, model generation/sub-projects, aspects covered (climatology, bias causes,
> generational change, projections)>.
> **Evidence base**: <N> core papers (<years>) from OpenAlex/Crossref/Semantic Scholar after
> deduplication, relevance scoring and screening; <F> read in full (★), <N−F> used through title and
> abstract (☆). Counts from `02_metadata/stats.json`. Citation format: `Author Year（ID）`.

## Abstract
<3–5 headline findings, each with a number and its source ID.>

## 1. Introduction
<Why the phenomenon matters; what changed in this model generation; which experiment families allow attribution.>

## 2. Diagnostics and methods
<Competing paradigms (e.g. Eulerian vs Lagrangian), their parameters, and how the choice changes conclusions.>

## 3. Present-day fidelity
### 3.1 <Region / hemisphere A> — by season
### 3.2 <Region / hemisphere B>
### 3.3 <Intensity / extremes>
### 3.4 <Consistency between metrics>

## 4. Causes of systematic bias
### 4.1 <Mechanism 1> — supporting experiment
### 4.2 <Mechanism 2>
### 4.3 <Resolution vs parameterisation>

## 5. Improvement over the previous generation; multi-model spread

## 6. Future projections
### 6.1 <Robust signal 1>
### 6.2 <Robust signal 2>
### 6.3 <Uncertain signals>
### 6.4 Forced signal vs internal variability

## 7. Uncertainties, limitations and outlook
<Compensating errors; resolution ≠ improvement; method dependence; data limitations; concrete next steps.>

## 8. Conclusions

## References
<Paste `06_review/references_by_author.md`.>

## Appendix: method and reproducible materials
- Databases, search date, query (anchor × concepts), year window — from `01_retrieval/search_log.json`
- Funnel — from `02_metadata/stats.json`: raw → unique → near-duplicates merged → curated → tiers A/B/C →
  excluded by type → selected → full text / abstract only
- Screening rubric — `02_metadata/screen/RUBRIC.md`
- Files: `papers_selected.csv`, `core_papers.xlsx`, `04_notes/`, `05_figures/`, `scripts/litreview.py`
