---
name: latex-to-word
description: Convert LaTeX manuscripts (especially ctex / ctexart Chinese academic papers) to editable Word (.docx) while preserving cross-references for figures, tables, equations, and bibliography numbering, and while matching a target journal template's styling. Use this skill whenever the user wants to export a .tex file to .docx, convert a LaTeX paper to Word, run pandoc tex→docx, produce a Word version of a manuscript for a Chinese journal (气象与环境学报 etc.), or fix a broken LaTeX→Word export (missing figure list, broken cross-refs, wrong numbering, ctex layout commands leaking through, equation refs not clickable, template styling not applied). Trigger even when the user just says "turn this paper into Word" or "I need a .docx of this manuscript" without mentioning pandoc.
---

# LaTeX → Word

A naive `pandoc manuscript.tex -o manuscript.docx` almost never produces a usable Word file: figure/table cross-reference lists are empty, equation numbers disappear, bibliography numbering is wrong, ctex-specific layout commands (`\zihao{}`, `\kaishu`, `\hangindent=...`) leak through as literal text, and the styling drifts from the target template. This skill implements a four-stage pipeline (ctex-aware preprocess → pandoc with reference doc → SEQ-field postprocess → equation + style postprocess) that fixes all four problems at once.

## 1. Pick the right path

Ask the user these three questions before running anything — the answers determine which path to use:

1. Does the output need to be **editable** in Word, or just **look like the PDF**?
2. Do cross-references to figures/tables/equations/bibliography need to work inside Word?
3. Should it match an existing Word template / house style (e.g. 气象与环境学报)?

| Path | When | Command |
|---|---|---|
| **Editable** (main) | Editable deliverable; cross-refs must work; template styling matters | `scripts/latex_to_docx.py` |
| **Review** | Quick internal read-through of text only; not a deliverable | `pandoc src.tex -o out.docx` (plain) |
| **Fidelity** | Must be `.docx` but visually identical to PDF; editing not needed | Compile to PDF, then rasterize pages into a .docx (not in this skill — tell the user to use `pdftoppm` + python-docx if they need this) |

Default to **Editable** unless the user explicitly asks for one of the others.

## 2. Why this takes four stages

LaTeX and Word have different reference systems, and ctex adds Chinese-academic layout commands pandoc can't parse:

- `\cite{key}` — LaTeX computes the number at compile time. Word expects an actual numbered list that cross-reference dialogs can see.
- `\ref{fig:x}` / `\ref{tab:x}` — In Word, cross-references require both a **bookmark** on the caption **and** a `SEQ Figure` / `SEQ Table` field in the caption paragraph. Pandoc creates the bookmark but not the SEQ field.
- `\eqref{eq:y}` — Pandoc silently drops equation numbering. Cross-refs need us to (a) number each display equation with a `SEQ Equation` field, (b) bookmark the number, (c) emit `REF` fields in the body.
- `\zihao{}`, `\kaishu`, `\bfseries`, `\hangindent=...`, `\noindent`, `\hfill`, `\vspace{}`, `\rule{}` — ctex-only layout commands; pandoc either ignores them or leaks them as literal text (`=2.4em 摘要：` is the classic symptom).

So the pipeline:

1. **ctex-aware preprocess** — strip ctex layout commands; split the first `\begin{center}...\end{center}` (title/author/affiliation) into separate paragraphs so pandoc emits distinct `<w:p>`s; split the `{\zihao{6} ... \\ ... }` author-footnote block; convert `\hfill` into an ideographic-space gap; resolve `\cite` and `\ref{sec:…}` statically; rewrite `\begin{thebibliography}` to `enumerate`; convert `\eqref{eq:…}` to a `ZZEQREF_…_ZZ` marker for the equation postprocess; record ordered equation labels in a sidecar JSON.
2. **Pandoc** with a `--reference-doc` for styling. The reference doc must define every style name pandoc actually emits — not just `Normal`/`Heading N`. See §4.
3. **SEQ postprocess** — walk `word/document.xml`, find paragraphs with `ImageCaption` / `TableCaption` styles, inject `SEQ Figure` / `SEQ Table` field runs. Makes Word's Insert → Cross-reference dialog list figures and tables.
4. **Styles + equation postprocess** — (a) reassign `pStyle` on the leading paragraphs so the title/author/affiliation/abstract/keywords/classification/author-footnote land on the correct Word styles; (b) number every display-math paragraph with a `SEQ Equation` field behind a right-aligned tab, bookmark the number as `eq:<label>`, and replace every `ZZEQREF_…_ZZ` in the body with a Word `REF` field pointing to that bookmark.

If any stage is skipped, something breaks. See `references/failure-modes.md` for specific symptoms.

## 3. Running the editable pipeline

```bash
python scripts/latex_to_docx.py <input.tex> <output.docx> [--reference-doc <ref.docx>]
```

The script does all four stages in one call. It requires `pandoc` on PATH and Python 3.9+ with `python-docx` (only needed for `make_reference_doc.py`).  If `--reference-doc` is omitted, pandoc's default styling is used — readable, but won't match a journal template.

`--keep-intermediate` keeps the preprocessed `.tex` and the `.meta.json` sidecar (contains the ordered equation-label list the equation postprocess consumes) for debugging.

### Generating a reference doc

If the user needs template styling but doesn't have a reference.docx yet:

```bash
python scripts/make_reference_doc.py <output.docx> [--preset qihuanbao]
```

The default preset (`qihuanbao` = 《气象与环境学报》) defines:

- **Title** 2号黑体 (22pt bold 黑体), centered, 段前 30 磅 / 段后 10 磅
- **Author** 小4号楷体 (12pt 楷体), centered
- **Affiliation** 小5号楷体 (9pt 楷体), centered
- **Abstract / Keywords** 小5号宋体 (9pt) with left/right indent
- **ClassificationCode** 小5号, for 中图分类号 / 文献标识码 / doi lines
- **AuthorFootnote** 六号宋体 (7.5pt)
- **Heading 1** 小4号宋体加粗 (12pt bold 宋体)
- **Heading 2/3** 5号宋体加粗 (10.5pt bold)
- **Normal / FirstParagraph / BodyText / Compact** 5号宋体 (10.5pt), first-line indent 2 字符, 1.5 倍行距
- **ImageCaption / TableCaption / Caption / CaptionedFigure** 小5号宋体加粗 (9pt bold), centered
- **Bibliography** 小5号

All paragraph styles clear `asciiTheme` / `eastAsiaTheme` attributes so pandoc's heading runs can't hijack the font via the Office theme.  On top of that, **after `doc.save()`** the script rewrites two deeper layers that would otherwise leak theme fonts in:

- `word/theme/theme1.xml` — `minorFont` / `majorFont` are set to the preset fonts (Times New Roman + 宋体 for body, + 黑体 for headings) instead of the default Cambria / Calibri with an empty East-Asian slot.  Without this, Word shows body text as "Cambria (Body)" and Chinese characters as "MS Mincho (Body Asian)" even though the paragraph styles themselves are clean.
- `word/styles.xml` `<w:docDefaults>` — the root of the font-inheritance chain.  Its `<w:rFonts>` is rewritten from `asciiTheme="minorHAnsi" eastAsiaTheme="minorEastAsia"` to explicit `ascii="Times New Roman" eastAsia="宋体"`.

See `_patch_theme_and_defaults` in `make_reference_doc.py` and failure-modes §10b for the full story.

To target a different journal, edit the `QIHUANBAO` dict in `make_reference_doc.py` (or add a new preset dict and register it under `PRESETS`). Font sizes live in the `SZ` table at the top of the file.

### If Word has the output file open

`pandoc` will fail with "permission denied". The pipeline script detects this and writes to `<name>_latest.docx` instead, so progress isn't lost. Tell the user to close Word and rerun, or rename the fallback file.

## 4. Verifying the output

Don't just "open it in Word and it looks OK." Check each object class:

- **Title block** — title, authors, affiliation should be on three distinct paragraphs, each with its own style (Title / Author / Affiliation). If they're merged, something in the ctex preprocess didn't fire — look at the intermediate `.tex` (use `--keep-intermediate`) to see whether the `\begin{center}` handler ran.
- **`=2.4em` / `hangindent` / `zihao` literals** — should be absent. Any appearance means a ctex command leaked through; add the missing command to `CTEX_BARE` / `CTEX_CMD_WITH_ARG` in `latex_to_docx.py`.
- **Figures/Tables** — Insert → Cross-reference → Reference type: *Figure* / *Table*. The list should be non-empty. If empty, SEQ postprocess didn't run or the caption style name doesn't match `ImageCaption` / `TableCaption`.
- **Equations** — every display equation should have a right-aligned `(1)` / `(2)` / … at its right margin **on first open, without pressing F9** (we bake the correct number into every SEQ field's cached display value because Word doesn't auto-update SEQ fields on open — see failure-modes §5b). Insert → Cross-reference → Reference type: *Equation* should list them. Body references (previously `\eqref`) should be live REF fields — right-click one and *Toggle Field Codes* should show ` REF eq:… \h `.
- **Bibliography** — open Insert → Cross-reference → Reference type: *Numbered item*. Bibliography entries should appear. If empty, the `\begin{thebibliography}` → `enumerate` rewrite didn't happen.
- **Fonts** — click into a Chinese body paragraph; Word's font picker should show 宋体 (it may be labelled "宋体 (Body Asian)" if the theme slot is being used — that's fine and expected).  If it shows `Cambria` or `MS Mincho`, the theme + docDefaults patch didn't land — see failure-modes §10b.
- **Paragraph spacing** — body paragraphs should feel evenly spaced.  If you notice one tight / one loose alternating every other paragraph, it's almost certainly the `BodyText` vs `FirstParagraph` `w:after` mismatch — see failure-modes §5c.
- **Styling** — compare heading font/size, body indent, and caption placement against the target template.

## 5. When something breaks

See `references/failure-modes.md` — symptom → root cause → fix table for the common failure modes.

## 6. Porting to a new project or journal

The things that vary by project:

1. **Reference doc preset.** `make_reference_doc.py`'s `QIHUANBAO` dict is one preset; add new presets to target other journals. Most changes boil down to font sizes and family; rarely do you need to add a new *style* because the set covered above matches what pandoc emits.
2. **Caption style names.** Pandoc uses `ImageCaption` / `TableCaption` by default. If the reference doc renames them, the SEQ postprocess won't find captions — grep `word/document.xml` for the actual style name and update the constants at the top of `postprocess_docx_seq_fields.py`.
3. **Figure/table caption prefix.** The SEQ postprocess expects Chinese `图N` / `表N` prefixes on caption text. For English manuscripts, change the regex in `postprocess_docx_seq_fields.py` from `[图表]` to `(Figure|Table)` and update the leading character passed to `seq_field_xml`.
4. **Title-block patterns.** `postprocess_docx_styles.py::STYLE_PATTERNS` is a regex list matched against the first ~25 paragraphs. Add your template's classification-code / funding / correspondence patterns there.

Everything else — citation ordering, equation resolution, bibliography rewrite, equation cross-refs — is project-agnostic.

## 7. File layout

```
scripts/
  latex_to_docx.py                    # pipeline entry — preprocess + pandoc + chained postprocess
  make_reference_doc.py               # generates --reference-doc with journal-preset styles
  postprocess_docx_seq_fields.py      # SEQ Figure/Table on captions
  postprocess_docx_styles.py          # reassign pStyle on leading title/abstract/footnote paragraphs
  postprocess_docx_equations.py       # number display equations; wire REF fields for ZZEQREF markers
references/
  failure-modes.md                    # symptom → cause → fix table
```
