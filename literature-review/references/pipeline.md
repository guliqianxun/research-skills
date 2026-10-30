# Pipeline reference

Project layout (one folder per review, outside the skill):

```text
<project>/
  topic.json                 configuration (no secrets)
  01_retrieval/              raw API responses per source + search_log.json
  02_metadata/               papers_raw.csv, papers_curated.csv, papers_selected.csv,
                             screen/ (RUBRIC.md, batch_NN.txt, tiers.tsv),
                             download_log.csv, manual_fetch.md, abstracts_digest.md, stats.json
  03_pdfs/<ID>.pdf           open-access full texts (+ files the user fetched by hand)
  04_notes/<ID>.fulltext.txt extracted text with "=== page N ===" markers
  04_notes/<ID>.md           your structured reading note
  05_figures/<ID>_pNN.png    rendered figure pages + figures_index.csv
  06_review/                 review draft, figures/ you embed, references*.md
  core_papers.csv/.xlsx      shareable core-paper list
```

## search

- Query = `anchor + concept`, one query per concept per source. OpenAlex searches title + abstract with
  cursor paging; Crossref and Semantic Scholar are relevance-ranked, so their cap (`max_per_query`) is
  the main recall knob: raise it if late pages still look relevant.
- `crossref_types` defaults to journal articles. Add `posted-content` to include preprints from Crossref.
- `search_log.json` records date, anchor, year limit and per-concept counts; quote it in the method appendix.
- A concept returning 0 results everywhere is usually a phrasing problem (try the noun form, British/American
  spelling, or drop the anchor for that concept and rely on `anchor_regex` at scoring time).

## curate

Order of operations, with the count recorded in `stats.json` at each step:

1. **Load and clean**: HTML entities are unescaped repeatedly before tags are stripped; Crossref and
   OpenAlex abstracts are often double-escaped JATS (`&amp;lt;p&amp;gt;`).
2. **Junk filter**: replies, referee comments, errata, editorials, titles ≤15 characters, retracted works.
3. **Exact dedupe** by DOI, else by normalised title. The record with the longest abstract wins; OA URL,
   OA flag, max citation count and the list of sources are pooled across duplicates.
4. **Publication type** (`pub_type`): `journal`, `preprint`, `conference_abstract`, `thesis`, `other`, from
   DOI patterns, venue and source type hints. Add project-specific rules in `pub_type_rules`
   (`{"thesis": "10\\.26686/"}`); they are checked before the defaults.
5. **Near-duplicates**: same normalised title (`Part II` = `Part 2`, accents folded, CJK kept). The best
   version is kept (journal > preprint > thesis > other > abstract, then citations); the others are
   listed in `other_versions`, and their open URLs in `alt_urls` so `download` can fall back to them.
6. **Score**: `anchor` ×3 + each matched `core_groups` family ×2 + each `context_tags` axis ×1 (weights are
   configurable). `theme_tags` lists the matched families and axes.
7. **Stable IDs**: IDs from an existing `papers_raw.csv` are reused by DOI/title; new papers get the next
   free number. This is what keeps `tiers.tsv` and notes valid when you rerun.
8. `papers_curated.csv` = anchor present, ≥1 core family, score ≥ `curated_min_score`.

Tuning: look at the printed theme-tag distribution. If one core family dominates through a loose regex
(e.g. `\bjet\b` catching every circulation paper), tighten it or move it to `context_tags`.

### papers_raw.csv / papers_curated.csv columns

`id, title, authors (first 12, "; "-separated), year, venue, doi, pub_type, cited_by_count, is_oa, oa_url,
alt_urls, other_versions, sources, relevance_score, theme_tags, _core, abstract`

## batches and screening

`batches` writes only papers that have no tier yet, continuing the batch numbering, plus `RUBRIC.md` built
from `screen.rubric`. Each entry shows ID, title, year, citations, OA, type, tags and up to 1500 characters
of abstract. `tiers.tsv` is plain text, tab-separated, one line per paper:

```text
<ID>\t<A|B|C>\t<reason>
```

Lines that don't match this shape are ignored, so a header or notes line does no harm. The last line for an
ID wins, which lets you re-tier by appending.

## select

1. Every curated paper must have a tier (`--allow-untiered` treats missing ones as C. Use it only for tests).
2. Drop types in `select.exclude_types` (default: conference abstracts). The count of A/B papers dropped this
   way is printed and stored as `excluded_by_type_AB`; mention it in the method appendix.
3. Take all A papers by (score, citations); if fewer than `target_min`, top up with the best B; cap at `max`.
4. Preprints are kept and listed: search for a published version (journal DOI, same title) and swap it in
   by adding the journal record, or note that none exists yet.

## download

Candidate order per paper: OpenAlex `oa_url` → Unpaywall PDF locations → Unpaywall landing pages →
Semantic Scholar `openAccessPdf` → publisher patterns for OA articles (Copernicus `.pdf`, MDPI `/pdf`, Wiley
`pdfdirect`, Springer `content/pdf`) → open versions of the same work (`alt_urls`). An HTML response is
searched once for `citation_pdf_url` or a PDF link. A file counts only if it starts with `%PDF-` and is
≥20 KB. The log records source URL and the Unpaywall license.

Expect 30–50% success for a geoscience topic. Paywalled journals (AMS, AGU, Wiley, Elsevier, Springer) are
the usual failures. `manual_fetch.md` is a checklist for the user; files they save as `03_pdfs/<ID>.pdf`
are picked up by `extract` and `biblio` automatically.

## extract

- Full text with page markers, so notes can cite `p. 7`.
- A page is rendered (`dpi`, default 130) when a line *starts* with a caption label (`Fig. 3`, `Figure 3`,
  `图3`) and the page has a raster image or a vector drawing. This avoids pages that only mention a figure.
- `figures_index.csv`: `id, page, figure, caption (first 200 chars), file`. Use it to find the figure that
  supports a claim, then look at the PNG.
- Rendered pages are whole pages. Crop before embedding in a deliverable if the page has other content.

## biblio

- `has_fulltext` = valid PDF and not a conference abstract; written back to `papers_selected.csv`.
- `references.md` (by ID, for tracing) and `references_by_author.md` (numbered, by first-author surname, for
  the review) with ★/☆ and type labels; counts in headings are computed.
- `abstracts_digest.md`: abstracts of the papers without full text, highest score first.
- `core_papers.csv/.xlsx`: shareable list; headers in Chinese when `language` is `zh`.
- `stats.json`: the whole funnel. Every count in the review's method text must come from here.
- Warns about full-text papers that have no `04_notes/<ID>.md` yet.

## check

Errors (exit code 1):
- an ID cited in the review that is not in the selected set
- `★` on a paper without full text, or on a conference abstract
- an embedded image path that does not exist

Warnings:
- selected papers never cited
- full-text papers without notes
- cited preprints/theses/abstracts (label them)
- a number followed by `篇`/`papers`/`articles`/`studies` that matches no value in `stats.json`
