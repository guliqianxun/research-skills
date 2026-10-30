# Failure modes

Most rows come from the first run of this method (CMIP6 transient-activity review). Each fix is
already built into `scripts/litreview.py` unless the row says otherwise.

| Symptom | Cause | Fix |
|---|---|---|
| API key visible in a script or repo | Key hard-coded in the search script | Keys only in `OPENALEX_API_KEY` / `S2_API_KEY`; rotate any key that was ever committed or shared |
| ~25% of the "core" set are EGU/EMS/OOS abstracts (DOIs like `10.5194/egusphere-egu24-…`, `10.5194/ems2024-…`) | Relevance scoring and screening judged content only; type was never checked | `pub_type` classification in `curate`; `select.exclude_types` drops conference abstracts by default |
| A conference abstract marked ★ "full text read" | The 1-page abstract PDF downloaded fine and was counted as full text | `has_fulltext` excludes conference abstracts; `check` flags ★ on them |
| A paper cited with ★ but no PDF exists | ★ typed by hand while drafting | `check` compares ★ IDs with `has_fulltext` |
| Counts disagree between sections ("41 papers without full text" in one file, 40 in another) | Counts typed into headings and prose by hand | All counts computed into `stats.json`; `check` warns on counts not found there |
| Screening batches show `&amp;lt;p&amp;gt;` in abstracts | Abstracts double-escaped; tags stripped before unescaping | Unescape repeatedly, then strip tags |
| Rerunning `curate` breaks `tiers.tsv` | IDs were assigned by sort order, so they change when scores or inputs change | IDs reused from the previous `papers_raw.csv` by DOI/title |
| The same study screened 2–3 times (preprint, abstract, journal version) | Near-duplicate removal ran after screening | Near-duplicates merged in `curate`; open versions kept as `alt_urls` |
| Non-English titles vanish during dedupe | Title key `[^a-z0-9]` deletes CJK and accented letters, so the key is empty | Unicode-aware key (`[\W_]+`, accents folded) |
| Unscreened papers silently treated as C | Missing tiers filled with C | `select` refuses until every curated ID has a tier |
| Crossref / Semantic Scholar contribute little | One page of 60 results per query | Cursor/offset paging up to `max_per_query` |
| Most PDFs fail for AMS/AGU/Wiley journals | Paywalls; blocked automated access | Use OA routes only; give `manual_fetch.md` to the user for library download. Don't spoof a browser UA. |
| Figure pages that contain no figure | Any page mentioning "figure" was rendered | Only pages with a line *starting* with a caption label, plus a graphic |
| Review numbers can't be traced back | Numbers came from memory of reading, not from notes | Notes with figure/table/page for every number (`templates/paper_note.md`) |
| Preprints cited although a journal version exists | No published-version check | `select` lists selected preprints; search by title and swap in the journal record |
| Theme tag "hemisphere" on unrelated papers | `SH\b`/`NH\b` with case-insensitive matching also hits words ending in "sh"/"nh" (e.g. "fresh") | Keep it as a low-weight context tag, or use `\bSH\b` with a case-sensitive pattern via `(?-i:...)` |
| Concept dominates the tag distribution | Over-broad regex (`\bjet\b`) | Tighten it or move it from `core_groups` to `context_tags` |
| A number in the review has no source (e.g. "cyclone count −5% under SSP5-8.5" with no citation) | Written from memory or from a paper read earlier | Claims ledger; `check` fails numbers that are not in the ledger claims of the cited papers |
| "研究表明 / 多项研究" backed by one paper | Wording stronger than the evidence | Evidence-strength table in `writing-protocol.md`; `check` fails consensus wording with <2 citations |
| A classic reference cited from memory (wrong year, wrong title, or nonexistent) | Citation outside the evidence base | Manual additions with a verified DOI after user approval; `check` fails author–year citations without an ID |
| Conclusions introduce a new paper or number | Summary written independently of the body | `check` fails IDs in abstract/conclusions that the body never cites |
| "may" in the source became "会/将" in the review | Hedging lost in paraphrase or translation | Ledger statements keep the source's hedging; sentence rule 8 |
| A percentage change appears that no paper reports | Reviewer computed it silently | Derived claims in the ledger; `check` requires the `本文据…计算` label |
| Paper IDs change after adding one record, orphaning tiers | Unstable choice of the representative in near-duplicate groups | Deterministic sort; IDs matched by DOI, title key and other versions |
