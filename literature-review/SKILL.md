---
name: literature-review
description: >-
  Run a reproducible, auditable literature review: multi-database retrieval (OpenAlex, Crossref,
  Semantic Scholar), deduplication, keyword relevance scoring, title/abstract screening into A/B/C
  tiers, core-set selection, open-access PDF retrieval, full-text and figure extraction, structured
  reading notes, and a written review whose every citation, ★ full-text claim, count and figure is
  checked against the evidence files. Use when the user asks for a literature review, survey, state
  of the art, systematic review, 文献综述, 文献调研, 研究进展, or a core-paper list on a research
  topic, or wants to update or audit an existing review built this way. Worked example: CMIP6
  simulation of storm tracks and synoptic transient activity.
---

# Literature Review

Turn a research question into a reviewed, traceable body of evidence and a written review. The method
is a funnel: **retrieve broadly → filter by rules → screen by judgement → read deeply → write → audit**.
Every stage writes files; nothing important lives only in the conversation.

All script paths below are relative to this skill's folder. Call the script by its full path, and
run it on a project folder outside the skill:

```bash
python <skill-dir>/scripts/litreview.py <stage> --project <project-dir>
```

Dependencies: Python 3.9+, `pandas`, `requests`; `pymupdf` for `extract`; `openpyxl` for the `.xlsx`
core list. Set `CONTACT_EMAIL` before `search`/`download`. Optional keys go in the environment
(`OPENALEX_API_KEY`, `S2_API_KEY`), never in files.

## 0. Scope before searching

Settle these with the user first; they become `topic.json` (`init` creates it from
`templates/topic.example.json`):

| Decision | Field | Guidance |
|---|---|---|
| Anchor term that every paper must mention | `anchor`, `anchor_regex` | The dataset/model/method that defines the field, e.g. `CMIP6` |
| Concept phrasings to search | `concepts` | 10–20 phrasings incl. hyphen variants and plurals (`storm track`, `storm-track`) |
| What makes a paper on-topic | `core_groups` | One regex per concept family; a paper needs the anchor + ≥1 group |
| Axes the review will be organised by | `context_tags` | e.g. bias, vs-previous-generation, resolution, future scenario, hemisphere |
| Time window, target size | `year_min`, `select` | Core set of ~40–60 is readable in depth; >80 is a bibliography, not a review |
| Grey literature | `select.exclude_types` | Default excludes conference abstracts; preprints are kept but labelled |
| Review language, deliverable | `language` | Markdown first; Word via pandoc at the end |

## 1. Pipeline

| Stage | Command | Produces | Gate before moving on |
|---|---|---|---|
| Search | `search` | `01_retrieval/*.json`, `search_log.json` (date, counts) | Every concept returned results; zero-hit concepts are rephrased |
| Curate | `curate` | `papers_raw.csv`, `papers_curated.csv`, `stats.json` | Curated set is screenable (~150–250); adjust `curated_min_score` if not |
| Screen | `batches`, then **you** tier each batch | `screen/batch_NN.txt` → `screen/tiers.tsv` | Every curated ID has a tier (`select` refuses otherwise) |
| Select | `select` | `papers_selected.csv` | Type breakdown reviewed; preprints checked for published versions |
| Fetch | `download` | `03_pdfs/<ID>.pdf`, `download_log.csv`, `manual_fetch.md` | Give `manual_fetch.md` to the user for library access |
| Extract | `extract` | `04_notes/<ID>.fulltext.txt`, `05_figures/*.png`, `figures_index.csv` | — |
| Read | **you**, per full-text paper | `04_notes/<ID>.md` from `templates/paper_note.md` | Every full-text paper has a note |
| Bibliography | `biblio` | `06_review/references*.md`, `abstracts_digest.md`, `core_papers.csv/.xlsx`, final `stats.json` | — |
| Ledger | **you** | `06_review/claims.tsv` from notes and abstracts | Outline + headline findings confirmed by the user |
| Write | **you** | `06_review/<title>.md` by `references/writing-protocol.md` | — |
| Audit | `check` | errors / warnings, exit 1 on errors | Zero errors; warnings resolved; user reviews abstract, conclusions, `本文认为` |

Rerunning is safe: paper IDs (`P001`…) are stable across reruns, so tiers and notes stay attached.
Details, file columns and tuning advice: `references/pipeline.md`.

## 2. Screening (your judgement, not keywords)

Keyword scores only rank; screening decides. For each `screen/batch_NN.txt`, read title + abstract
and append one line per paper to `screen/tiers.tsv`, following `screen/RUBRIC.md`:

```text
P001	A	evaluates NH storm-track bias across CMIP3/5/6
```

- **A** — directly answers the review question. **B** — relevant but secondary. **C** — off-topic.
- Tier on content, not on publication type; `select` applies the type policy afterwards.
- Reasons ≤15 words and specific. If you can delegate, give one batch to each sub-agent with the rubric.
- Re-screen ~10% of papers blind; if more than a few tiers flip, sharpen the rubric and redo.

## 3. Deep reading

For each paper with a PDF, read `04_notes/<ID>.fulltext.txt` and look at its figure pages
(`05_figures/<ID>_pNN.png`, listed in `figures_index.csv`). Write `04_notes/<ID>.md` from
`templates/paper_note.md`: data and models used, diagnostics with their parameters, key quantitative
findings **with figure/table/page**, mechanisms claimed, limitations, and which review section it feeds.
Papers without full text are used through `02_metadata/abstracts_digest.md` only.

## 4. Writing (strict protocol)

Writing is the step where an AI is most tempted to fill gaps from general knowledge. Don't. **Read
`references/writing-protocol.md` before drafting and follow it step by step.** In short:

1. **Claims ledger first.** Record every finding you will use in `06_review/claims.tsv`
   (`templates/claims.tsv`): paper ID, `fulltext` / `abstract` / `derived`, location (Fig./Table/page),
   faithful statement with its qualifiers and hedging, and the numbers. Nothing outside the ledger goes in.
2. **Plan from the ledger and get the user's confirmation** of the outline and headline findings.
3. **Draft by the sentence rules**:
   - Cite in the sentence that states the finding; copy numbers exactly, with units and qualifiers.
   - Wording must match evidence strength (one study ≠ "研究表明"; consensus needs ≥2 cited studies).
   - Causal verbs need an attribution design; `显著/significant` needs a reported test.
   - Keep the source's hedging; stay inside each study's scope; no promotional words.
4. **Abstract and conclusions add nothing new.** Outlook contains only gaps stated by sources or visible
   in the evidence base, labelled `本文认为`.
5. **Never cite from memory.** References outside the set go through `manual_additions.csv` after the
   user approves them.
6. **Audit**: run `check` to zero errors, resolve every warning, trace each sentence to a claim_id. The
   user then reviews the abstract, conclusions and every `本文认为` statement.

Structure: `templates/review_outline.md`. Notes, figures and export: `references/reading-and-writing.md`.
Export with `pandoc review.md -o review.docx --resource-path=06_review`; for journal styling build a reference
doc with the `latex-to-word` skill's `make_reference_doc.py`.

## Hard rules

1. **No secrets in files.** API keys and e-mail come from environment variables.
2. **Open access only.** Download what Unpaywall / OpenAlex / Semantic Scholar / the publisher list as
   open; send the rest to the user via `manual_fetch.md`. Do not spoof browser user agents or bypass paywalls.
3. **★ means read.** A ★ claim requires a full-text PDF *and* a note. A conference abstract is never ★.
4. **Nothing without a ledger claim.** Every finding and number in the review traces to a row in
   `claims.tsv` (from a note or an abstract), and every count to `stats.json`. Computed numbers are labelled.
5. **No citations from memory.** Only papers in the selected set or user-approved manual additions.
6. **Preserve the record.** Keep raw retrieval JSON, `search_log.json`, `tiers.tsv` and the download log;
   they are the review's method section.
7. **Never renumber IDs by hand.** IDs link tiers, notes, figures and citations.

## Files to read on demand

| File | When |
|---|---|
| `references/pipeline.md` | Running or tuning any script stage; file formats |
| `references/writing-protocol.md` | **Before and while drafting**: claims ledger, sentence and section rules, audit |
| `references/reading-and-writing.md` | Writing reading notes, choosing figures, review structure, export |
| `references/failure-modes.md` | Something looks wrong: symptom → cause → fix |
| `references/example-cmip6-transient.md` | Worked example; also a primer on transient-activity diagnostics for CMIP6 reviews |
| `templates/topic.example.json` | Starting config (the CMIP6 example) |
| `templates/paper_note.md`, `templates/claims.tsv`, `templates/review_outline.md` | Reading, ledger, writing |
