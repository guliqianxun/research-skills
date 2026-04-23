"""
LaTeX -> editable Word pipeline.

Stages:
  1. Preprocess the .tex:
       - Strip ctex-only layout commands that pandoc can't parse
         (\\zihao, \\kaishu, \\bfseries, \\hangindent, \\hangafter,
         \\noindent, \\hfill, \\vspace, \\rule) and split the
         title/author/affiliation \\begin{center} block into separate
         paragraphs so pandoc emits distinct paragraphs for each.
       - Number citations by first-occurrence order.
       - Rewrite \\begin{thebibliography} as an enumerate (so Word sees
         a numbered list).
       - Resolve \\ref{eq:...} / \\ref{sec:...} and \\eqref{eq:...} —
         but keep eqref as a unique marker (ZZEQREF_<label>_ZZ) so
         the equation postprocess can wire a Word REF field.
       - Leave \\ref{fig:...} / \\ref{tab:...} alone so pandoc
         generates Word bookmarks.
       - Record ordered list of equation labels for the postprocess.
  2. Run pandoc with --reference-doc for styling.
  3. Postprocess the docx:
       - SEQ Figure / SEQ Table fields on captions (for Word's
         cross-reference dialog).
       - Reassign paragraph styles for the title block, abstract,
         keywords, classification code, and author footnote.
       - Number display equations with SEQ Equation fields and
         replace ZZEQREF markers with REF fields.

Usage:
    python latex_to_docx.py INPUT.tex OUTPUT.docx [--reference-doc REF.docx]

Requires: pandoc on PATH, Python 3.9+.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# ctex / ctexart layout-command stripper
# ---------------------------------------------------------------------------
# pandoc's LaTeX reader doesn't understand the ctex font-sizing and
# paragraph-layout commands — if we leave them in, the values leak
# through as literal text (e.g. "=2.4em 摘要：") and the title
# block collapses into a single paragraph.

# Commands that take a single {arg} and must be dropped along with the arg
# (the args are pure layout values, not body content).
CTEX_CMD_WITH_ARG = re.compile(
    r"\\(?:vspace\*?|rule|makebox|parbox|setlength|kaishu@|hspace\*?)\s*\{[^}]*\}(?:\s*\{[^}]*\})?"
)

# Commands that take a numeric/dimension assignment and must be dropped
# including the assignment target.  Example: `\\hangindent=2.4em`.
CTEX_ASSIGN = re.compile(
    r"\\(?:hangindent|hangafter|parindent|parskip|leftskip|rightskip|baselineskip)\s*=\s*-?\d*\.?\d+(?:pt|em|ex|cm|mm|in|sp)?"
)

# Font-sizing / face-switch commands that take no argument.  We strip
# them whole; any text that used to be wrapped by the group falls back
# to the enclosing paragraph style.  hfill / vfill aren't here — they
# convey horizontal spacing which pandoc discards, so we convert them
# to plain spaces ourselves below rather than dropping them silently.
CTEX_BARE = re.compile(
    r"\\(?:zihao\{-?\d\}|kaishu|heiti|fangsong|songti|bfseries|itshape|"
    r"rmfamily|sffamily|ttfamily|normalfont|small|footnotesize|scriptsize|"
    r"tiny|normalsize|large|Large|LARGE|huge|Huge|noindent|centering|"
    r"raggedright|raggedleft)\b"
)

# Horizontal fill commands — replace with a visible gap so segments
# like "中图分类号：X\\hfill文献标识码：Y" don't collapse to "X文献标识码：Y".
CTEX_HFILL = re.compile(r"\\(?:hfill|hfil|vfill|vfil)\b")

# `\\[10pt]` / `\\[2pt]` — LaTeX line breaks with explicit spacing.  We
# want pandoc to treat each such break as a paragraph break (blank
# line), so the title/author/affiliation become distinct paragraphs.
BACKSLASH_LINE_BREAK = re.compile(r"\\\\\s*(?:\[[^\]]*\])?")


def _strip_layout(text: str) -> str:
    """Drop ctex layout-only commands that pandoc can't consume."""
    text = CTEX_CMD_WITH_ARG.sub("", text)
    text = CTEX_ASSIGN.sub("", text)
    text = CTEX_HFILL.sub("　　", text)  # 2 ideographic spaces
    text = CTEX_BARE.sub("", text)
    return text


_CENTER_RE = re.compile(r"\\begin\{center\}(.*?)\\end\{center\}", re.DOTALL)


def preprocess_ctex(text: str) -> str:
    """Strip ctex-only layout commands and reformat the title block.

    Splits the first \\begin{center}...\\end{center} block (which in
    ctexart usually wraps title+authors+affiliation) into separate
    paragraphs so pandoc emits distinct <w:p> elements for each line —
    the postprocess can then assign Title / Author / Affiliation
    styles one paragraph at a time.
    """
    def rewrite_center(m, _state={"done": False}):
        # Only rewrite the first \begin{center} at document start.  The
        # tail of the paper has an English title block that pandoc
        # handles the same way, but the first-pass style reassign hits
        # only the first few paragraphs, so later \begin{center}s are
        # left alone.
        if _state["done"]:
            return m.group(0)
        _state["done"] = True
        inner = m.group(1)
        # Split on \\ (with or without [Xpt]) to get each visual line.
        parts = [p.strip() for p in BACKSLASH_LINE_BREAK.split(inner)]
        parts = [p for p in parts if p.strip()]
        # Strip ctex commands from each piece.  Remove surrounding
        # braces left over after stripping font-size commands.
        cleaned = []
        for p in parts:
            p = _strip_layout(p)
            # Collapse {...} wrappers that held only font/face commands
            # and now contain just text.
            p = re.sub(r"\{\s*", "", p)
            p = re.sub(r"\s*\}", "", p)
            p = p.strip()
            if p:
                cleaned.append(p)
        # Emit as blank-line-separated paragraphs.  Wrap in blank lines
        # at the top and bottom so pandoc closes the surrounding
        # paragraph cleanly.
        return "\n\n" + "\n\n".join(cleaned) + "\n\n"

    text = _CENTER_RE.sub(rewrite_center, text, count=1)

    # Author-footnote block: `{\zihao{6} line1 \\ line2 \\ ... }`.
    # Inside this (and only this) group, `\\` denotes a paragraph break
    # — if we leave them in, all the footnote lines collapse into one
    # paragraph; if we ran a global `\\\\` → paragraph-break we'd break
    # tables elsewhere.  The footnote body contains nested `\textbf{…}`
    # groups, so a naive ``(.*?)`` match stops at the first ``}``; we
    # do a depth-aware scan instead.
    def find_footnote_block(src):
        m = re.search(r"\{\s*\\zihao\{6\}\s*", src)
        if not m:
            return None
        depth = 1
        i = m.end()
        while i < len(src) and depth > 0:
            c = src[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return m.start(), i + 1, src[m.end():i]
            i += 1
        return None

    hit = find_footnote_block(text)
    if hit is not None:
        start, end, body = hit
        parts = [p.strip() for p in BACKSLASH_LINE_BREAK.split(body)]
        parts = [p for p in parts if p.strip()]
        replacement = "\n\n" + "\n\n".join(parts) + "\n\n"
        text = text[:start] + replacement + text[end:]

    # Strip ctex font/face/layout commands everywhere.  We deliberately
    # do NOT do a global `\\\\` → paragraph-break substitution here:
    # outside the title / footnote blocks handled above, `\\\\` is
    # overwhelmingly a tabular row separator or an align/cases line
    # break, and pandoc needs it intact to parse the environment.
    text = _strip_layout(text)

    return text


# ---------------------------------------------------------------------------
# Preprocessing — citation / label / bibliography rewrites
# ---------------------------------------------------------------------------

BIB_BLOCK_RE = re.compile(
    r"\\begin\{thebibliography\}\{[^}]*\}([\s\S]*?)\\end\{thebibliography\}"
)
BIBITEM_RE = re.compile(r"^\\bibitem\{([^}]+)\}\s*(.*)$")
CITE_RE = re.compile(r"\\cite\{([^}]+)\}")
LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
BEGIN_MATH_RE = re.compile(r"\\begin\{(equation|align|gather|multline|flalign|alignat)\*?\}")
END_MATH_RE = re.compile(r"\\end\{(equation|align|gather|multline|flalign|alignat)\*?\}")


def parse_bibliography(text):
    m = BIB_BLOCK_RE.search(text)
    if not m:
        return None, []
    body = m.group(1)
    entries = []
    current = None
    for raw in body.splitlines():
        line = raw.strip()
        bm = BIBITEM_RE.match(line)
        if bm:
            if current:
                current["text"] = current["text"].strip()
                entries.append(current)
            current = {"key": bm.group(1).strip(), "text": (bm.group(2) or "").strip()}
            continue
        if current:
            current["text"] = (current["text"] + " " + line).strip()
    if current:
        current["text"] = current["text"].strip()
        entries.append(current)
    return m.group(0), entries


def build_citation_order(text, bib_entries):
    order = {}
    idx = 1
    for m in CITE_RE.finditer(text):
        for key in (k.strip() for k in m.group(1).split(",")):
            if key and key not in order:
                order[key] = idx
                idx += 1
    for entry in bib_entries:
        if entry["key"] not in order:
            order[entry["key"]] = idx
            idx += 1
    return order


def format_citation_numbers(nums):
    unique = sorted(set(nums))
    if not unique:
        return ""
    parts = []
    start = prev = unique[0]
    for cur in unique[1:]:
        if cur == prev + 1:
            prev = cur
            continue
        parts.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = cur
    parts.append(f"{start}" if start == prev else f"{start}-{prev}")
    return f"[{','.join(parts)}]"


def build_label_map(text):
    """Assign numbers to labels by walking the tex.  Returns the number
    map as well as an ordered list of equation labels (needed so the
    postprocess can attach SEQ Equation + bookmark to the i-th display
    equation paragraph in the docx)."""
    labels = {}
    eq_order = []  # equation labels, in the order they appear
    section = subsection = subsubsection = 0
    figure = table = equation = 0
    current_float = None
    in_math = False
    pending_section_number = None

    for line in text.splitlines():
        if re.search(r"\\section\{", line):
            section += 1
            subsection = subsubsection = 0
            pending_section_number = f"{section}"
        if re.search(r"\\subsection\{", line):
            subsection += 1
            subsubsection = 0
            pending_section_number = f"{section}.{subsection}"
        if re.search(r"\\subsubsection\{", line):
            subsubsection += 1
            pending_section_number = f"{section}.{subsection}.{subsubsection}"
        if re.search(r"\\begin\{figure\}", line):
            figure += 1
            current_float = ("fig", str(figure))
        if re.search(r"\\begin\{table\}", line):
            table += 1
            current_float = ("tab", str(table))
        if BEGIN_MATH_RE.search(line):
            in_math = True

        for m in LABEL_RE.finditer(line):
            label = m.group(1)
            if current_float:
                labels[label] = current_float[1]
                continue
            if in_math or label.startswith("eq:"):
                equation += 1
                labels[label] = str(equation)
                if label not in eq_order:
                    eq_order.append(label)
                continue
            if pending_section_number:
                labels[label] = pending_section_number
                pending_section_number = None

        if re.search(r"\\end\{figure\}", line) and current_float and current_float[0] == "fig":
            current_float = None
        if re.search(r"\\end\{table\}", line) and current_float and current_float[0] == "tab":
            current_float = None
        if END_MATH_RE.search(line):
            in_math = False

    return labels, eq_order


def replace_citations(text, order):
    def repl(m):
        keys = [k.strip() for k in m.group(1).split(",") if k.strip()]
        nums, unresolved = [], []
        for k in keys:
            if k in order:
                nums.append(order[k])
            else:
                unresolved.append(k)
        if unresolved and not nums:
            return f"[{','.join(unresolved)}]"
        if unresolved:
            return f"{format_citation_numbers(nums)}[{','.join(unresolved)}]"
        return format_citation_numbers(nums)

    return CITE_RE.sub(repl, text)


def rewrite_bibliography(text, bib_block, bib_entries, order):
    if not bib_block or not bib_entries:
        return text
    ordered = sorted(bib_entries, key=lambda e: order.get(e["key"], 10**9))
    lines = ["\\section*{References}", "\\begin{enumerate}"]
    lines.extend(f"\\item {e['text']}" for e in ordered)
    lines.append("\\end{enumerate}")
    return text.replace(bib_block, "\n".join(lines))


def replace_refs(text, labels):
    # \eqref{eq:X} -> a unique marker (ZZEQREF_<label>_ZZ).  The
    # equation postprocess finds these in body text and turns them
    # into Word REF fields pointing at the bookmark injected on the
    # corresponding display equation.  Leaving them as static "(n)"
    # would mean the user can't re-flow equations without hand-
    # editing every reference.
    def eqref(m):
        label = m.group(1)
        if label in labels:
            return f"ZZEQREF_{label}_ZZ"
        return f"[{label}]"

    text = re.sub(r"\\eqref\{([^}]+)\}", eqref, text)

    # \ref{...} — only static-resolve eq: / sec:; pass fig:/tab: through to pandoc
    def ref(m):
        full = m.group(0)
        prefix = m.group(1) or ""
        rest = m.group(2)
        label = prefix + rest
        if not prefix:
            if label.startswith(("fig:", "tab:")):
                return full
        if label.startswith("eq:") and label in labels:
            # Same treatment as \eqref, without the parentheses — the
            # postprocess-produced REF field also has no parens.
            return f"ZZEQREF_{label}_ZZ"
        val = labels.get(label)
        return val if val is not None else f"[{label}]"

    text = re.sub(r"\\ref\{(eq:|sec:)?([^}]*)\}", ref, text)
    return text


def preprocess(text):
    # Strip ctex layout first — otherwise the cite/ref regexes may
    # trip on braces inside font-command groups.
    text = preprocess_ctex(text)

    bib_block, bib_entries = parse_bibliography(text)
    text_no_bib = text.replace(bib_block, "") if bib_block else text
    labels, eq_order = build_label_map(text)
    order = build_citation_order(text_no_bib, bib_entries)

    out = text
    out = replace_citations(out, order)
    out = replace_refs(out, labels)
    out = rewrite_bibliography(out, bib_block, bib_entries, order)
    # Strip section labels (already resolved); keep eq:/fig:/tab:.
    # eq: labels stay inside display-math envs so the postprocess can
    # either detect them there, or fall back to document-order matching
    # using the eq_order list (which is what we actually use).
    out = re.sub(r"\\label\{sec:[^}]+\}", "", out)
    # Inline-math fragments pandoc handles poorly — pre-convert to
    # plain Unicode so the body text reads cleanly.
    out = out.replace("m·s$^{-1}$", "m·s⁻¹")
    out = out.replace("$\\leq$", "≤")
    out = out.replace("$\\geq$", "≥")
    out = out.replace("$^\\circ$", "°")

    return out, {"equation_labels": eq_order}


# ---------------------------------------------------------------------------
# Pandoc + postprocess orchestration
# ---------------------------------------------------------------------------

def run_pandoc(tex_path: Path, out_path: Path, reference_doc: Path | None):
    args = [
        "pandoc",
        str(tex_path),
        "--from=latex",
        "--to=docx",
        "--standalone",
        "--wrap=none",
        f"--resource-path={tex_path.parent}",
        f"--output={out_path}",
    ]
    if reference_doc:
        args.append(f"--reference-doc={reference_doc}")
    subprocess.run(args, check=True)


def postprocess(docx_path: Path, metadata_path: Path):
    """Run the three docx postprocess steps in order:
      - SEQ Figure / SEQ Table field injection (needed first so later
        steps can assume captions carry SEQ fields).
      - First-paragraph style reassignment (title/author/abstract/etc.).
      - Equation numbering and cross-reference wiring.
    """
    for script_name in (
        "postprocess_docx_seq_fields.py",
        "postprocess_docx_styles.py",
        "postprocess_docx_equations.py",
    ):
        script = SCRIPT_DIR / script_name
        cmd = [sys.executable, str(script), str(docx_path)]
        if script_name == "postprocess_docx_equations.py":
            cmd.append(str(metadata_path))
        subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, help="Input .tex file")
    ap.add_argument("output", type=Path, help="Output .docx file")
    ap.add_argument("--reference-doc", type=Path, default=None,
                    help="Pandoc reference.docx for template styling")
    ap.add_argument("--keep-intermediate", action="store_true",
                    help="Keep the preprocessed .tex + metadata for debugging")
    args = ap.parse_args()

    src: Path = args.input
    out: Path = args.output
    tmp = src.with_name(f"_{src.stem}_word_editable.tex")
    meta = src.with_name(f"_{src.stem}_word_editable.meta.json")

    text = src.read_text(encoding="utf-8")
    processed, metadata = preprocess(text)
    tmp.write_text(processed, encoding="utf-8")
    meta.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")

    target = out
    try:
        run_pandoc(tmp, target, args.reference_doc)
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or b"").decode("utf-8", "ignore") if isinstance(e.stderr, bytes) else (e.stderr or "")
        if re.search(r"permission denied|being used by another process", stderr, re.I):
            fallback = out.with_name(f"{out.stem}_latest{out.suffix}")
            print(f"Output locked, writing fallback: {fallback}", file=sys.stderr)
            run_pandoc(tmp, fallback, args.reference_doc)
            target = fallback
        else:
            raise

    postprocess(target, meta)

    if not args.keep_intermediate:
        for f in (tmp, meta):
            try:
                os.unlink(f)
            except OSError:
                pass

    print(f"Done: {target}")


if __name__ == "__main__":
    main()
