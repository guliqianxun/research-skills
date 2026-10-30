# Reading notes, figures, structure and export

## Reading notes (`04_notes/<ID>.md`)

Write one note per full-text paper from `templates/paper_note.md` before drafting anything. The note is
the only place numbers enter the review from, so record them in a form you can cite:

- **Setup**: models (count and which), experiments (historical, AMIP, SSPs, HighResMIP…), period, reference
  data (reanalysis), season, region, resolution groups.
- **Diagnostics with parameters**: e.g. "2–6-day band-pass MSLP variance", "TRACK on T42 ξ850, lifetime ≥48 h,
  displacement ≥1000 km". When papers disagree, the review can state that their parameters differ; it names
  the difference as the cause only if a source does.
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
9. Uncertainties, limitations and outlook, as stated by the sources (e.g. compensating errors, resolution vs
   tuning, method dependence), each attributed.
10. Conclusions.
11. References (from `references_by_author.md`) and a reproducibility appendix.

## Writing rules and audit

See `references/writing-protocol.md`: claims ledger, sentence and section rules, references outside the
evidence base, and the audit loop. Follow it for every draft.

## Reproducibility appendix (always include)

- Databases and search date (`01_retrieval/search_log.json`), anchor and concept list, year window.
- Funnel: raw records → unique → near-duplicates merged → curated → tiers A/B/C → excluded by type → selected
  → full text / abstract only (all from `stats.json`).
- Screening rubric (`screen/RUBRIC.md`) and who screened.
- File map: CSVs, notes, figures, scripts.

## Export

```bash
pandoc 06_review/review.md -o 06_review/review.docx --resource-path=06_review
```

For journal styling (fonts, heading sizes, Chinese typesetting), generate a reference document with the
`latex-to-word` skill (`make_reference_doc.py`) and pass `--reference-doc=<ref.docx>`.
