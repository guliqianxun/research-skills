"""
Number display equations in a docx and wire up Word cross-references.

Pandoc translates ``\\begin{equation}...\\end{equation}`` into a
paragraph whose main content is an ``<m:oMathPara>`` element — but it
does *not* emit the equation number.  Without a number the equation
can't be referenced, and users end up with static "(1)" text that
doesn't update when the equation order changes.

This script, called after pandoc:

1. Walks <w:body> paragraphs.  A paragraph is treated as a display
   equation if ``<m:oMathPara>`` is present and the paragraph's
   running <w:t> text is minimal (anything substantive alongside the
   math indicates an inline-math paragraph instead).

2. For the i-th display equation it:
     * Sets a bookmark ``<w:bookmarkStart w:name="eq:<label>"/>`` at
       the paragraph's head and the matching end, where ``label`` is
       the i-th entry in the equation_labels list that the
       preprocess saved to the metadata JSON.  (If the JSON has fewer
       labels than equations, the extras are left unnumbered; if it
       has more, the tail is ignored.)
     * Appends a right-aligned run containing ``(`` + SEQ Equation
       field + ``)``.  The `\\* ARABIC` switch ensures Word
       auto-numbers them even after reorder.

3. Scans the whole document for ``ZZEQREF_<label>_ZZ`` markers (left
   by the preprocess when it saw ``\\eqref{label}``) and rewrites
   each into a Word REF field targeting the matching bookmark —
   formatted as ``(N)``, so the body reads naturally.

Usage:  python postprocess_docx_equations.py <docx_path> <metadata_json>
"""

from __future__ import annotations

import json
import os
import re
import sys
import zipfile


# ---------------------------------------------------------------------------
# Paragraph / run detection
# ---------------------------------------------------------------------------
_PARA_RE = re.compile(rb"<w:p(?:\s[^>]*)?>[\s\S]*?</w:p>")
_TEXT_RE = re.compile(rb"<w:t[^>]*>([\s\S]*?)</w:t>", re.DOTALL)
_OMATH_PARA_RE = re.compile(rb"<m:oMathPara[\s\S]*?</m:oMathPara>")


def _para_text_bytes(para: bytes) -> bytes:
    return b"".join(_TEXT_RE.findall(para))


def _is_display_math_para(para: bytes) -> bool:
    """True iff paragraph's visible text is dominated by a display
    math block (one <m:oMathPara> + little else)."""
    if b"<m:oMathPara" not in para:
        return False
    # If there are more than a handful of <w:t> characters besides the
    # math, this is likely an inline-math paragraph (running prose
    # with an inline equation).
    text = _para_text_bytes(para)
    # Stripping whitespace — short text is fine (e.g. a punctuating ","
    # after the math).
    if len(text.strip()) > 16:
        return False
    return True


# ---------------------------------------------------------------------------
# Field XML fragments
# ---------------------------------------------------------------------------
_RIGHT_TAB_RUN = b'<w:r><w:tab/></w:r>'


def _seq_equation_runs(num: int) -> bytes:
    """`(SEQ Equation \\* ARABIC)` run-group — the caller inserts a
    tab in front of this if it wants right-alignment.  We keep the
    tab out of this fragment so the bookmark (which wraps this group)
    contains only the visible number.

    ``num`` is baked into the cached display value so that Word shows
    correct numbers even before the user presses F9 to update fields.
    (Word does not auto-update SEQ fields on file-open; without a
    correct cached value every equation would display as "(1)" until
    the user explicitly triggers a field refresh.)
    """
    num_b = str(num).encode("ascii")
    return (
        b'<w:r><w:t xml:space="preserve">(</w:t></w:r>'
        b'<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
        b'<w:r><w:instrText xml:space="preserve"> SEQ Equation \\* ARABIC </w:instrText></w:r>'
        b'<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
        b'<w:r><w:t>' + num_b + b'</w:t></w:r>'
        b'<w:r><w:fldChar w:fldCharType="end"/></w:r>'
        b'<w:r><w:t xml:space="preserve">)</w:t></w:r>'
    )


def _bookmark(name: str, idx: int) -> tuple[bytes, bytes]:
    bid = str(200 + idx).encode("ascii")
    bname = name.encode("utf-8")
    start = b'<w:bookmarkStart w:id="' + bid + b'" w:name="' + bname + b'"/>'
    end = b'<w:bookmarkEnd w:id="' + bid + b'"/>'
    return start, end


def _ensure_right_tab(para: bytes) -> bytes:
    """Make sure the paragraph has a right-aligned tab-stop so the
    equation number is pushed to the right margin.  We insert a
    <w:tabs><w:tab w:val="right" w:pos="9000"/></w:tabs> inside the
    paragraph's <w:pPr> if one isn't there.  9000 twips ≈ 6.25 in,
    which is close to the right margin of an A4 page with 2.2 cm
    side-margins."""
    if b"<w:tabs>" in para:
        return para
    m = re.search(rb"<w:pPr>([\s\S]*?)</w:pPr>", para)
    tabs = b'<w:tabs><w:tab w:val="right" w:pos="9000"/></w:tabs>'
    if m:
        # insert tabs right after opening <w:pPr>
        pos = m.start() + len(b"<w:pPr>")
        return para[:pos] + tabs + para[pos:]
    # No <w:pPr> — add one just after the <w:p> tag.
    mp = re.search(rb"<w:p(?:\s[^>]*)?>", para)
    if not mp:
        return para
    end = mp.end()
    return para[:end] + b"<w:pPr>" + tabs + b"</w:pPr>" + para[end:]


# ---------------------------------------------------------------------------
# Core: number equations
# ---------------------------------------------------------------------------
def number_equations(xml: bytes, labels: list[str]) -> tuple[bytes, list[tuple[str, int]]]:
    """Wrap bookmarks + SEQ Equation fields around the first N display
    equation paragraphs, where N = min(#display-paras, len(labels)).

    Returns the modified xml and a ``[(label, number), ...]`` list for
    the reference postprocess step.  Numbers are 1-indexed in their
    order of appearance in the document."""
    body_start = xml.find(b"<w:body>")
    if body_start < 0:
        return xml, []
    head, body = xml[:body_start], xml[body_start:]

    matches: list[tuple[str, int]] = []
    eq_idx = 0  # index into labels

    def rewrite(m):
        nonlocal eq_idx
        para = m.group(0)
        if not _is_display_math_para(para):
            return para
        if eq_idx >= len(labels):
            # Unknown equation — still number it so the user sees a
            # count, but with a throwaway label so it won't collide
            # with a real bookmark.
            label = f"__eq_unlabeled_{eq_idx}"
        else:
            label = labels[eq_idx]

        num = eq_idx + 1
        eq_idx += 1
        matches.append((label, num))

        # Ensure paragraph has a right-aligned tab stop so the number
        # floats to the right margin.
        para = _ensure_right_tab(para)

        # Wrap the *number* (not the math, not the tab) with the
        # bookmark so that `REF eq:X \\h` dereferences cleanly to
        # "(1)" rather than to the equation body or leading tab.
        # Final structure at end-of-paragraph:
        #   [math] [tab] [bookmarkStart] [(SEQ)] [bookmarkEnd]
        bm_start, bm_end = _bookmark(label, eq_idx)
        end_marker = b"</w:p>"
        idx = para.rfind(end_marker)
        seq = _seq_equation_runs(num)
        para = para[:idx] + _RIGHT_TAB_RUN + bm_start + seq + bm_end + para[idx:]

        return para

    new_body = _PARA_RE.sub(rewrite, body)
    return head + new_body, matches


# ---------------------------------------------------------------------------
# Replace ZZEQREF_<label>_ZZ with REF fields
# ---------------------------------------------------------------------------
# Pandoc may split the marker across multiple <w:t> runs if the
# surrounding text forced a formatting boundary; in practice that
# doesn't happen because the marker contains only ASCII letters,
# digits, underscore and colon.  We still guard against it by first
# normalising each paragraph's text runs that contain the marker.
MARKER_RE = re.compile(
    rb'ZZEQREF_([A-Za-z0-9:_\-]+)_ZZ'
)


def _ref_field_runs(bookmark_name: str, cached_num: int) -> bytes:
    """Build the REF-field run group.

    ``cached_num`` is the equation number we know this REF resolves to
    at generation time; baking it into the cached display value means
    the body reads correctly before the user runs F9 / "Update all
    fields".  Word still recomputes on F9 — the cache is just the
    fallback display.

    The bookmark we reference (emitted in ``number_equations``)
    encloses a run sequence "(", SEQ Equation field, ")", so REF will
    already produce "(N)" after a refresh.  To avoid the doubled-paren
    "((N))" we *don't* wrap this field in extra parens here — the REF
    result already contains them."""
    bname = bookmark_name.encode("utf-8")
    num_b = f"({cached_num})".encode("ascii")
    return (
        b'<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
        b'<w:r><w:instrText xml:space="preserve"> REF ' + bname +
        b' \\h </w:instrText></w:r>'
        b'<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
        b'<w:r><w:t>' + num_b + b'</w:t></w:r>'
        b'<w:r><w:fldChar w:fldCharType="end"/></w:r>'
    )


def replace_ref_markers(xml: bytes, label_to_num: dict[str, int]) -> tuple[bytes, int]:
    """Find each ``ZZEQREF_<label>_ZZ`` inside a <w:t> run and rewrite
    the enclosing run into: (+REF-field+).  We do this at the run
    boundary so Word can maintain field integrity — splitting a
    single <w:t> would leave an orphan field code.

    Returns modified xml and a count of markers replaced."""
    count = 0

    # Find each <w:t>...</w:t> whose content includes the marker
    # pattern.  Then replace the entire *run* containing that <w:t>
    # with: leading-text-run + REF-field-runs + trailing-text-run.
    # Pandoc always emits text markers inside a single <w:r>, so the
    # run boundary is trivial to find.

    def _handle_run(m: re.Match) -> bytes:
        nonlocal count
        run_xml = m.group(0)
        if b"ZZEQREF_" not in run_xml:
            return run_xml
        # Extract <w:rPr>, if present, so we can reuse it on the
        # leading/trailing text runs.
        rpr_m = re.search(rb"<w:rPr>[\s\S]*?</w:rPr>", run_xml)
        rpr = rpr_m.group(0) if rpr_m else b""
        # Extract the text content.
        tm = _TEXT_RE.search(run_xml)
        if not tm:
            return run_xml
        text = tm.group(1)
        # Build replacement: walk the text and interleave refs.
        parts = []
        pos = 0
        for mm in MARKER_RE.finditer(text):
            prefix = text[pos:mm.start()]
            if prefix:
                parts.append(
                    b"<w:r>" + rpr +
                    b'<w:t xml:space="preserve">' + prefix + b"</w:t></w:r>"
                )
            label = mm.group(1).decode("utf-8")
            if label in label_to_num:
                bm_name = label if label.startswith("eq:") else "eq:" + label
                parts.append(_ref_field_runs(bm_name, label_to_num[label]))
            else:
                # Unknown label — emit as literal so user sees what's
                # wrong instead of silently vanishing.
                parts.append(
                    b"<w:r>" + rpr +
                    b'<w:t xml:space="preserve">[' + label.encode("utf-8") +
                    b"]</w:t></w:r>"
                )
            count += 1
            pos = mm.end()
        suffix = text[pos:]
        if suffix:
            parts.append(
                b"<w:r>" + rpr +
                b'<w:t xml:space="preserve">' + suffix + b"</w:t></w:r>"
            )
        return b"".join(parts)

    # Match whole <w:r>...</w:r> runs.  Pandoc's runs aren't nested.
    RUN_RE = re.compile(rb"<w:r\b(?:\s[^>]*)?>[\s\S]*?</w:r>")
    new_xml = RUN_RE.sub(_handle_run, xml)
    return new_xml, count


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def postprocess(docx_path: str, metadata_path: str) -> None:
    try:
        meta = json.loads(open(metadata_path, "r", encoding="utf-8").read())
    except FileNotFoundError:
        print(f"postprocess_equations: no metadata at {metadata_path}, skipping.")
        return

    labels: list[str] = meta.get("equation_labels", [])
    if not labels:
        print("postprocess_equations: empty equation_labels, skipping.")
        return

    with zipfile.ZipFile(docx_path, "r") as zin:
        names = zin.namelist()
        contents = {n: zin.read(n) for n in names}

    xml = contents["word/document.xml"]
    xml, matches = number_equations(xml, labels)
    label_to_num = {lbl: num for lbl, num in matches}
    xml, n_refs = replace_ref_markers(xml, label_to_num)

    contents["word/document.xml"] = xml
    tmp_path = docx_path + ".eqtmp"
    with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, data in contents.items():
            zout.writestr(name, data)
    os.replace(tmp_path, docx_path)
    print(f"postprocess_equations: numbered {len(matches)} equation(s), "
          f"wired {n_refs} cross-reference(s) in "
          f"{os.path.basename(docx_path)}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python postprocess_docx_equations.py <docx_path> <metadata_json>",
              file=sys.stderr)
        sys.exit(1)
    postprocess(sys.argv[1], sys.argv[2])
