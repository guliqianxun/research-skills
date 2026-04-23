# Failure modes

When a LaTeX→Word export looks broken, don't ask "why is the export broken." Ask "which *object class* is broken." The root cause is almost always in one specific stage of the pipeline.

## Symptom → cause → fix

### 1. Insert → Cross-reference → Figure / Table list is empty

- **Symptom:** Captions look right in the body, clicking a `\ref{fig:x}` link may even jump correctly — but the cross-reference dialog's Figure/Table lists are empty.
- **Cause:** The docx has bookmarks but no `SEQ Figure` / `SEQ Table` fields. Pandoc doesn't emit these.
- **Fix:** Run `scripts/postprocess_docx_seq_fields.py <out.docx>`. If the postprocess ran but reports "no Caption paragraphs modified," the caption style name in your reference doc isn't `ImageCaption` / `TableCaption` — grep `word/document.xml` for the actual style name and update the regexes at the top of the script.

### 2. Figure references became plain text like "see Figure 3"

- **Symptom:** No clickable links on figure/table references.
- **Cause:** The preprocessing stage statically replaced `\ref{fig:…}` with numbers before pandoc saw it. Pandoc then has no label to wire to a bookmark.
- **Fix:** Confirm `replace_refs` in `latex_to_docx.py` passes `fig:` / `tab:` through untouched. Also confirm `preprocess` doesn't strip `\label{fig:…}` / `\label{tab:…}` (only `sec:` labels should be stripped).

### 3. Bibliography appears numbered on screen but cross-ref dialog doesn't see it

- **Symptom:** Body shows `[1] [2]` correctly, but Insert → Cross-reference → Numbered Item doesn't list the bibliography entries.
- **Cause:** The `\begin{thebibliography}` block wasn't rewritten to an `enumerate`. Word's cross-reference dialog only sees list-structured paragraphs.
- **Fix:** Confirm `rewrite_bibliography` in `latex_to_docx.py` actually matched the bib block. If the manuscript uses BibTeX (`.bib` files) instead of `thebibliography`, run `pandoc-citeproc` or `pandoc --citeproc` first to render the bibliography inline, then the rewrite step becomes unnecessary (but the cross-ref dialog still won't see it — in that case use `--csl` with a numeric style and accept the limitation).

### 4. Literal `[eq:foo]` appears in the output

- **Symptom:** Text like `see equation [eq:convergence]` instead of `see equation (3)`.
- **Cause:** The label isn't in the label map — usually a typo or a label defined inside an unsupported math environment. The equation postprocess uses the label map; anything missing becomes bracket-literal as a debug aid.
- **Fix:** Check spelling of the `\label{}`, and confirm the enclosing environment is in `BEGIN_MATH_RE` in `latex_to_docx.py`. Add it to the regex if missing.

### 5. `ZZEQREF_eq:xxx_ZZ` markers visible in the output

- **Symptom:** Literal `ZZEQREF_eq:xxx_ZZ` strings in the final docx.
- **Cause:** The equation postprocess (`postprocess_docx_equations.py`) didn't run, or the metadata JSON sidecar wasn't passed to it, or the marker text was split across pandoc-generated runs by an inline formatting boundary. The marker contains only ASCII letters/digits/underscore/colon so pandoc shouldn't split it, but custom inline markup near it could.
- **Fix:** Confirm `latex_to_docx.py` is calling all three postprocess scripts and passing the `.meta.json` path to the equations one. If markers are still appearing, run the pandoc step with `--keep-intermediate` and grep the preprocessed tex — if the marker shows up inside a `\textbf{}` or other inline group, lift it out.

### 5b. Every equation displays as "(1)" (or every REF displays as "(1)") until the user presses F9

- **Symptom:** Equations show "(1), (1), (1), ..." down the page on first open, and `式 (1)` in the body is always "(1)" regardless of which equation is referenced.  After the user clicks anywhere in the field and presses F9 — or Ctrl-A, F9 — the numbers correct themselves to "(1), (2), (3), ...".
- **Cause:** Word does not auto-update SEQ / REF fields on file-open; it displays the cached result stored between `<w:fldChar w:fldCharType="separate"/>` and the matching `end` fldChar.  If every SEQ field ships with the hard-coded cache "1", every field reads "(1)" until F9 forces recomputation.
- **Fix:** In `postprocess_docx_equations.py`, `_seq_equation_runs(num)` and `_ref_field_runs(bookmark, cached_num)` both bake the known-correct number into the cached `<w:t>` run.  If you add another field type, do the same — Word's "Update fields on open" setting is per-user and unreliable; don't rely on it.

### 5c. Every body paragraph has different space-below from the one above (alternating loose/tight feel)

- **Symptom:** Paragraphs feel inconsistently spaced — one has a clear gap below, the next runs tight against the one that follows.  The problem appears to come and go "every other paragraph."
- **Cause:** pandoc assigns `FirstParagraph` style to the first paragraph after each heading and `BodyText` to subsequent ones.  If those two styles have different `w:spacing` (specifically, different `w:before` / `w:after`), the rhythm breaks.  python-docx's built-in `BodyText` starts with `w:after="120"` and `FirstParagraph` has no spacing element at all; if the preset code doesn't explicitly zero both of them, the old `w:after="120"` persists on BodyText and you get 6pt of extra gap on every-other paragraph.
- **Fix:** In `make_reference_doc.py`'s preset, set `space_before_pt=0` and `space_after_pt=0` on `Normal`, `FirstParagraph`, `BodyText`, `Compact`.  Do it explicitly — don't rely on the absence of a key defaulting the attribute away, because the built-in style is not empty to begin with.

### 6. Equation numbers appear in the body but cross-refs return the equation itself (not "(1)")

- **Symptom:** `REF eq:uv_transform \h` in a body field, but double-click or F9 yields the full equation math as text.
- **Cause:** The bookmark `eq:xxx` encloses the math runs instead of the SEQ number runs.
- **Fix:** In `postprocess_docx_equations.py::number_equations`, make sure the insertion order is `[math] [tab] [bookmarkStart] [SEQ runs] [bookmarkEnd]` — NOT `[bookmarkStart] [math] [bookmarkEnd] [SEQ]`.

### 7. Opening paragraph is "多源风廓线… XXX XXX … (1. XXX …)" all glued together

- **Symptom:** Title, authors, affiliations on a single line, no style differentiation.
- **Cause:** The ctex preprocess didn't split the first `\begin{center}…\end{center}` block into separate paragraphs. Either the block is nested inside something unexpected, or the `\\[Xpt]` separators are written in a way `BACKSLASH_LINE_BREAK` doesn't match (e.g. `\newline` instead of `\\`).
- **Fix:** Confirm `_CENTER_RE` matches your document's title block (it's non-greedy and requires a literal `\begin{center}`; if the source uses `\begin{titlepage}` or similar, add an alternate pattern). If `\newline` is used instead of `\\[...]`, extend `BACKSLASH_LINE_BREAK` to match it.

### 8. Literal `=2.4em` / `=4.6em` at the start of the abstract or keywords

- **Symptom:** Text like `=2.4em 摘要：针对单站多源……`.
- **Cause:** `\hangindent=2.4em` or similar leaked through because the preprocess didn't strip it. `\hangindent` is in `CTEX_ASSIGN`; if you still see the symptom, the assignment form in your source differs (e.g. `\setlength{\hangindent}{2.4em}` — pandoc handles `\setlength` *differently*).
- **Fix:** Add the exact command name to `CTEX_ASSIGN` if you see a new form. Or add the whole `\setlength{...}{...}` pattern to `CTEX_CMD_WITH_ARG` (which already covers `\setlength`).

### 8b. `中图分类号：P412.25文献标识码：A` — fields glued with no spacing

- **Symptom:** On the 中图分类号 line (or any line where the tex source uses `\hfill` to spread segments across the page width), the segments collapse to zero whitespace.
- **Cause:** `\hfill` is a horizontal "fill to page width" command that pandoc doesn't translate.  If the preprocess silently drops it, neighbours touch.  If we replace it with a single regular space, it's still visually too tight for the original intent (the template shows generous spread between 中图分类号 / 文献标识码 / doi).
- **Fix:** `CTEX_HFILL` in `latex_to_docx.py` maps `\hfill`/`\hfil`/`\vfill`/`\vfil` to `　　` (two U+3000 ideographic spaces).  That lands visually between "tight" and "spread full width" and matches how a human would retype the line in Word.  Adjust the replacement string if the template wants a different look.

### 8c. Author-footnote block collapses to one paragraph

- **Symptom:** The `\{\zihao\{6\} 收稿日期…\\\\ 资助项目…\\\\ 作者简介…\\\\ 通信作者…\}` block renders as one long paragraph instead of four.
- **Cause:** Inside that block `\\\\` is meant as a paragraph break, but a global `\\\\` → paragraph substitution would destroy tabular syntax elsewhere (see §12).  A naive non-greedy `\{\s*\\zihao\{6\}\s*(.*?)\s*\}` regex matches the *first* closing brace, which lands inside the first `\\textbf{…}` group — missing the real block end.
- **Fix:** `preprocess_ctex` uses a depth-aware scanner (`find_footnote_block`) to locate the matching `}` by tracking brace depth, then splits the interior on `\\\\`.  If you see the symptom, confirm `find_footnote_block` is actually being called and returning a non-None hit; if the block's opening form differs (`{\zihao{-5} …}` or similar), broaden the opening regex to match.

### 9. "Permission denied" / "being used by another process"

- **Cause:** Word has the output file open.
- **Fix:** The pipeline auto-writes to `<name>_latest.docx` in this case. Close Word and either rerun or rename the fallback.

### 10. Styles don't match the target template

- **Cause:** No `--reference-doc` was passed; the reference doc doesn't define the styles pandoc emits; or pandoc's heading styles inherited the Office theme font because the reference doc left `asciiTheme` / `eastAsiaTheme` attributes in place.
- **Fix:** Generate a reference doc with `scripts/make_reference_doc.py --preset qihuanbao` (or write a preset for your journal). Confirm that the reference doc's `FirstParagraph`, `BodyText`, `Compact`, `ImageCaption`, `TableCaption`, `CaptionedFigure`, `Title`, `Author`, `Affiliation`, `Abstract`, `Keywords`, `ClassificationCode`, `AuthorFootnote`, `Heading 1/2/3/4`, `Bibliography` all exist AND have `w:rFonts` elements with `asciiTheme` / `eastAsiaTheme` attributes *absent* — otherwise Word uses the theme font.

### 10b. Word's font panel shows "Cambria (Body)" and "MS Mincho (Body Asian)" instead of Times New Roman / 宋体

- **Symptom:** Even though every paragraph-style's `w:rFonts` in styles.xml is clean (explicit ascii/eastAsia, no *Theme attrs), the rendered document and Word's font picker still label body text as `Cambria (Body)` and Chinese text as `MS Mincho (Body Asian)`.
- **Cause:** Two deeper layers still reference the theme. (a) `word/styles.xml`'s `<w:docDefaults><w:rPrDefault>` ships with `<w:rFonts asciiTheme="minorHAnsi" eastAsiaTheme="minorEastAsia" …/>`, which is the root of the font-inheritance chain. (b) `word/theme/theme1.xml` defines `<a:minorFont>` / `<a:majorFont>` with `<a:latin typeface="Cambria"/>` and an **empty** `<a:ea typeface=""/>` — so anything that falls through to theme resolves to Cambria (Latin) and a system default (often MS Mincho on Windows) for East Asian.
- **Fix:** After `doc.save()`, rewrite both: set theme's minor / major slots to explicit typefaces (`Times New Roman` / `宋体` / `黑体`), and replace docDefaults' rFonts with an explicit-font form. `make_reference_doc.py` does this in `_patch_theme_and_defaults`; if you hand-craft a reference doc, apply the same patches.

### 11. Chinese characters render as boxes / tofu

- **Cause:** Reference doc lacks East-Asian font mapping, or pandoc stripped mixed-font runs.
- **Fix:** Confirm `w:eastAsia` is set on `w:rFonts` in the reference doc's styles. `make_reference_doc.py` does this; handmade reference docs often miss it.

### 12. All tables disappeared from the output

- **Symptom:** Tables that are present in the tex source don't show up in the docx at all.
- **Cause:** The ctex preprocess converted tabular row separators (`\\`) to paragraph breaks, so pandoc no longer saw valid tabular syntax.
- **Fix:** Don't run a blanket `\\\\` → paragraph-break substitution outside the first `\begin{center}` title block and the `{\zihao{6} … }` footnote block. Both of those are handled by targeted regex in `preprocess_ctex()`; everything else must preserve `\\` so tables and multi-line math environments still parse.

## Known limitations (not fixed)

### L1. Bilingual table captions lose the `\\` line break between Chinese and English

- **Symptom:** A tex caption like `\caption{表 1 …指标 \\ Key technical specifications…}` renders in Word as `表1 …指标Key technical specifications…` — Chinese and English glued together with no line break.  Bilingual **figure** captions do keep the break; only table captions are affected.
- **Cause:** pandoc treats `\\` inside a `\caption{}` differently for figure vs. table environments.  In a figure caption pandoc preserves the break (emits `<w:br/>`); in a table caption it collapses the break.  The SEQ postprocess doesn't try to restore the break because it would need to split the second half of the caption off from the first.
- **Workaround:** Manually add a line break in Word after the Chinese portion, or write the caption without `\\`: put the English on the next `\caption` argument via a separate mechanism (pandoc doesn't support that either, so in practice hand-edit in Word).  A proper fix would detect `中文…English` transitions inside `TableCaption` paragraphs and inject `<w:br/>` — patches welcome.

### L2. `\paragraph{…}` becomes `Heading 4`

- **Symptom:** A paragraph-lead-in command like `\paragraph{数据可得性}` renders in Word as a Heading 4 style with its own line, not as a bold lead-in integrated with the following paragraph.
- **Cause:** pandoc maps `\paragraph` → Heading 4 by default.  The QIHUANBAO preset configures Heading 4 sensibly (5号宋体加粗) so this is readable, but it's not how the template intends a lead-in.
- **Workaround:** In the tex source, rewrite `\paragraph{X}Y` as `\textbf{X}Y` if you want it inlined.  Or hand-edit in Word.
