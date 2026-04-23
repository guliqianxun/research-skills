"""
Reassign paragraph styles on the first ~20 paragraphs of a docx so
that the ctex title/author/affiliation/abstract/keywords block gets
the right Word styles.

The ctex preprocess step in ``latex_to_docx.py`` splits
``\\begin{center}...\\end{center}`` into one paragraph per visual
line, so pandoc emits 3+ distinct <w:p> elements for title / author /
affiliation.  It can't, however, tell Word *which* style each one
should use — they all land in ``FirstParagraph`` (pandoc's default
for the first paragraph after a structural break).  This script walks
the opening paragraphs and rewrites ``<w:pStyle>`` to match content
patterns that are characteristic of a 《气象与环境学报》-style head
matter (and similar Chinese academic templates).

Usage:  python postprocess_docx_styles.py <docx_path>
"""

from __future__ import annotations

import os
import re
import sys
import zipfile


# How many leading paragraphs we consider.  Past the opening matter
# (~15 short paragraphs at most), everything is body content we
# shouldn't touch.
SCAN_LIMIT = 25


# Pattern → target style.  Patterns are matched against the *plain
# text* of the paragraph (after flattening <w:t> runs), in order, and
# each paragraph is rewritten at most once.  Patterns are tried in the
# order below; the first match wins.
#
# The patterns are deliberately permissive on whitespace — pandoc can
# insert U+00A0 (NBSP), narrow-no-break-space, etc. between the label
# character and the content.
STYLE_PATTERNS = [
    # 摘要：...  (any whitespace/NBSP between "摘" and "要")
    (re.compile(r"^\s*摘\s*要\s*[：:]"), "Abstract"),
    (re.compile(r"^\s*Abstract\s*[:：]", re.IGNORECASE), "Abstract"),
    # 关键词：...
    (re.compile(r"^\s*关\s*键\s*词\s*[：:]"), "Keywords"),
    (re.compile(r"^\s*Key\s*words?\s*[:：]", re.IGNORECASE), "Keywords"),
    # 中图分类号/文献标识码/doi
    (re.compile(r"^\s*中图分类号"), "ClassificationCode"),
    # 收稿日期/资助项目/作者简介/通信作者 (the footnote block)
    (re.compile(r"^\s*(?:收稿日期|资助项目|作者简介|通信作者)"), "AuthorFootnote"),
    # Affiliation line — starts with "（1." or "(1." — Chinese paper
    # affiliation block typically does.
    (re.compile(r"^\s*[（(]\s*1\s*[.．]"), "Affiliation"),
]


# Paragraph / run parsing
_PARA_RE = re.compile(rb"<w:p[ >][\s\S]*?</w:p>")
_TEXT_RE = re.compile(rb"<w:t[^>]*>([\s\S]*?)</w:t>", re.DOTALL)
_PPR_OPEN_RE = re.compile(rb"<w:pPr>")
_PPR_FULL_RE = re.compile(rb"<w:pPr>[\s\S]*?</w:pPr>")
_PSTYLE_RE = re.compile(rb'<w:pStyle w:val="[^"]+"\s*/>')


def _para_text(para: bytes) -> str:
    parts = _TEXT_RE.findall(para)
    return "".join(p.decode("utf-8", "replace") for p in parts)


def _set_pstyle(para: bytes, style_id: str) -> bytes:
    """Return paragraph with its pStyle rewritten (or inserted)."""
    new_pstyle = f'<w:pStyle w:val="{style_id}" />'.encode("utf-8")

    # Case 1: paragraph already has a <w:pStyle>.  Replace it.
    if _PSTYLE_RE.search(para):
        return _PSTYLE_RE.sub(new_pstyle, para, count=1)

    # Case 2: paragraph has <w:pPr> but no <w:pStyle>.  Insert at head.
    m = _PPR_OPEN_RE.search(para)
    if m:
        end = m.end()
        return para[:end] + new_pstyle + para[end:]

    # Case 3: no <w:pPr>.  Insert one right after the opening <w:p...>.
    m = re.search(rb"<w:p(?:\s[^>]*)?>", para)
    if not m:
        return para
    end = m.end()
    insertion = b"<w:pPr>" + new_pstyle + b"</w:pPr>"
    return para[:end] + insertion + para[end:]


def _classify(text: str) -> str | None:
    for pat, style in STYLE_PATTERNS:
        if pat.search(text):
            return style
    return None


def process_document_xml(xml: bytes) -> tuple[bytes, int]:
    """Rewrite pStyle on qualifying leading paragraphs.  Returns the
    new xml and a count of modifications."""
    # Find the body boundary; we only rewrite paragraphs inside <w:body>.
    body_start = xml.find(b"<w:body>")
    if body_start < 0:
        return xml, 0
    head, body = xml[:body_start], xml[body_start:]

    scanned = 0
    modifications = 0
    # Title / Author / Affiliation heuristic for the first three
    # non-empty paragraphs of the body.  We only assign by position if
    # they don't match a more specific pattern (摘要/关键词/etc.).
    positional_styles = ["Title", "Author", "Affiliation"]
    positional_idx = 0

    modified_paragraphs = []

    def replace_para(m):
        nonlocal scanned, modifications, positional_idx
        para = m.group(0)
        if scanned >= SCAN_LIMIT:
            return para
        scanned += 1
        text = _para_text(para).strip()
        if not text:
            return para

        # Stop positional title/author/affiliation assignment the
        # moment we hit a heading paragraph (pStyle = Heading N) — the
        # body proper has started.
        pstyle = _PSTYLE_RE.search(para)
        if pstyle:
            val = pstyle.group(0).decode("utf-8", "replace")
            if "Heading" in val:
                # Flush the positional counter so nothing after a
                # heading gets miscategorised as Affiliation etc.
                positional_idx = len(positional_styles)
                return para

        style = _classify(text)
        if style is None and positional_idx < len(positional_styles):
            style = positional_styles[positional_idx]
            positional_idx += 1
        if style is None:
            return para

        modifications += 1
        return _set_pstyle(para, style)

    new_body = _PARA_RE.sub(replace_para, body)
    return head + new_body, modifications


def postprocess(docx_path: str) -> None:
    with zipfile.ZipFile(docx_path, "r") as zin:
        names = zin.namelist()
        contents = {n: zin.read(n) for n in names}

    xml = contents["word/document.xml"]
    new_xml, n = process_document_xml(xml)

    if n == 0:
        print("postprocess_styles: no leading paragraphs reassigned.")
        return

    contents["word/document.xml"] = new_xml
    tmp_path = docx_path + ".stytmp"
    with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, data in contents.items():
            zout.writestr(name, data)
    os.replace(tmp_path, docx_path)
    print(f"postprocess_styles: reassigned {n} leading paragraph(s) "
          f"in {os.path.basename(docx_path)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python postprocess_docx_styles.py <docx_path>",
              file=sys.stderr)
        sys.exit(1)
    postprocess(sys.argv[1])
