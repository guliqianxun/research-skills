# Reading notes, figures, writing and audit

## Reading notes (`04_notes/<ID>.md`)

Write one note per full-text paper from `templates/paper_note.md` before drafting anything. The note is
the only place numbers enter the review from, so record them in a form you can cite:

- **Setup**: models (count and which), experiments (historical, AMIP, SSPs, HighResMIP…), period, reference
  data (reanalysis), season, region, resolution groups.
- **Diagnostics with parameters**: e.g. "2–6-day band-pass MSLP variance", "TRACK on T42 ξ850, lifetime ≥48 h,
  displacement ≥1000 km". Parameter differences explain disagreements between papers.
- **Findings**: one bullet per finding, each with a number and its location: `RMSE 0.389 hPa (Table 2, p. 9)`.
  Separate *what they measured* from *what they conclude*.
- **Figures worth showing**: figure number, page PNG, one-line takeaway.
- **Mechanism claims**, labelled by the evidence the paper gives: experiment (AMIP vs coupled, sensitivity runs),
  regression across models, or interpretation.
- **Limitations** the authors state, and ones you notice.
- **Feeds**: which review section(s) this paper supports.

Read figures, not only text: open the page PNG, and check that the number in the text matches the plot
(axes, units, season). Note disagreements.

## Choosing figures for the review

- 4–8 figures for a 40–60 paper review. Each must carry an argument the text makes, e.g. the canonical bias
  map, a generational comparison, a mechanism experiment, a projection.
- Prefer multi-model or multi-generation figures over single-model case studies.
- Caption template:
  `图 N　<what is plotted, period, season, reference>. <one-sentence takeaway>. 来源：<Author> (<Year>)，<ID> 图 <n>。`
- Respect the license recorded in `download_log.csv`. CC-BY figures can be reproduced with attribution;
  for others, check the publisher's terms or describe the figure instead of embedding it.
- Copy chosen PNGs into `06_review/figures/` and crop to the figure panel.

## Structure of the review

Organise by question, not paper by paper. For a model-evaluation topic, the outline in
`templates/review_outline.md` works well:

1. Scope box: what is and isn't covered; evidence base (counts from `stats.json`, ★/☆ convention, citation format).
2. Abstract: the 3–5 headline findings with numbers.
3. Introduction: why the phenomenon matters; what the new model generation changed; which experiment
   families (AMIP, HighResMIP…) make attribution possible.
4. Diagnostics and methods: the competing measurement paradigms and why the choice changes conclusions.
5. Present-day fidelity, by region/hemisphere and season.
6. Causes of bias, organised by mechanism, each with the experiment that supports it.
7. Change relative to the previous generation and multi-model spread.
8. Future projections: robust vs uncertain signals, forced vs internal variability.
9. Uncertainties, limitations and outlook: compensating errors, resolution vs tuning, method dependence.
10. Conclusions.
11. References (from `references_by_author.md`) and a reproducibility appendix.

## Writing rules

- **Synthesise.** A paragraph should combine ≥2 sources or state why only one exists. Where papers
  disagree, say so and explain the cause (method, period, model subset) if the notes show it.
- **Hedge by evidence level.** Full-text findings (★) can be stated with numbers and figure references.
  Abstract-only findings are reported as the authors' summary claims ("X et al. report…").
- **Citation format**: `Priestley & Catto 2022（★P002）`, `Yang et al. 2021（P020）`. IDs make every claim
  traceable to the CSV, the note and the figure.
- **Grey literature**: mark preprints `（预印本）`/`(preprint)` at first citation; prefer the published version.
  Conference abstracts are excluded by default; if one is kept for a unique result, label it and don't
  build a conclusion on it alone.
- **Numbers**: give units, period, season and reference dataset with every quantitative claim.
- **Counts**: copy them from `stats.json` and rerun `check` after any change to the selected set.

## Reproducibility appendix (always include)

- Databases and search date (`01_retrieval/search_log.json`), anchor and concept list, year window.
- Funnel: raw records → unique → near-duplicates merged → curated → tiers A/B/C → excluded by type → selected
  → full text / abstract only (all from `stats.json`).
- Screening rubric (`screen/RUBRIC.md`) and who screened.
- File map: CSVs, notes, figures, scripts.

## Audit

Run `check` and fix every error. Then read the draft once more against this list:

- [ ] Every ★ paper has a note, and every number attributed to it is in the note.
- [ ] No sentence relies only on a conference abstract.
- [ ] Each figure caption names its source figure and the claim it supports.
- [ ] Counts in the scope box, method text and reference header agree with `stats.json`.
- [ ] Disagreements between papers are reported, not averaged away.
- [ ] Limitations section names compensating errors and method dependence where the notes show them.

## Export

```bash
pandoc 06_review/review.md -o 06_review/review.docx --resource-path=06_review
```

For journal styling (fonts, heading sizes, Chinese typesetting), generate a reference document with the
`latex-to-word` skill (`make_reference_doc.py`) and pass `--reference-doc=<ref.docx>`.
