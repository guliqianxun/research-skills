"""
Generate a pandoc --reference-doc that matches a journal style spec.

Covers every paragraph style pandoc actually emits for tex->docx
(FirstParagraph / BodyText / Compact / ImageCaption / TableCaption /
CaptionedFigure / Heading N / Title ...) plus the extra custom-style
slots (Author / Affiliation / Abstract / Keywords /
ClassificationCode / AuthorFootnote) used by the ctex post-process
step. Every rFonts element is written with explicit ascii/hAnsi/
eastAsia fonts AND with the asciiTheme/eastAsiaTheme attributes
cleared, so that pandoc's heading runs won't silently fall back to
the Office theme and render in Calibri.

Default preset targets 《气象与环境学报》 (2号黑体 title, 小4 宋体加粗
一级标题, 5号宋体 body, 1.5 倍行距, etc.).  Edit ``PRESETS`` or pass
``--preset`` to target a different journal.

Usage:
    python make_reference_doc.py OUTPUT.docx [--preset qihuanbao]

Requires: python-docx (`pip install python-docx`).
"""

from __future__ import annotations

import argparse
import copy
import os
import re
import zipfile
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import nsmap, qn
from docx.shared import Cm, Pt


# ---------------------------------------------------------------------------
# Font size constants (Chinese typographic "号" → points)
# ---------------------------------------------------------------------------
# 二号=22, 小二=18, 三号=16, 小三=15, 四号=14, 小四=12,
# 五号=10.5, 小五=9, 六号=7.5, 小六=6.5, 七号=5.5
SZ = {
    "2": 22, "-2": 18,
    "3": 16, "-3": 15,
    "4": 14, "-4": 12,
    "5": 10.5, "-5": 9,
    "6": 7.5,
}


# ---------------------------------------------------------------------------
# Style spec — one preset for 《气象与环境学报》.  Each entry is a dict
# consumed by configure_style().  `name` is the style name docx uses
# internally; `base_on` / `next_style` chain makes them show up in the
# Word UI sensibly.
# ---------------------------------------------------------------------------
QIHUANBAO = {
    # Font family defaults
    "font_cn": "宋体",
    "font_cn_kaishu": "楷体",
    "font_cn_heiti": "黑体",
    "font_en": "Times New Roman",

    # A4 layout
    "page": {"width_cm": 21.0, "height_cm": 29.7, "margin_cm": 2.2},

    # Paragraph styles — keyed by docx style id.  Values feed configure_style.
    # size_pt is the point size (number, not string).  `en_font` / `cn_font`
    # override the preset defaults when set.
    "styles": {
        # Body styles — every variant pandoc emits.  Pandoc uses
        # FirstParagraph for the first paragraph after a heading and
        # BodyText for all continuation paragraphs; since the 《气象与
        # 环境学报》 template gives them identical treatment (5号宋体,
        # 首行缩进 2 字符, 1.5 倍行距, 无段前段后), we explicitly zero
        # space_before/after on all of them — otherwise python-docx's
        # built-in BodyText keeps its default w:after="120" and the
        # document appears to have inconsistent spacing on every other
        # paragraph (FirstParagraph tight, BodyText loose).
        "Normal": dict(size_pt=SZ["5"], first_line_indent_pt=SZ["5"]*2,
                       line_spacing=1.5,
                       space_before_pt=0, space_after_pt=0),
        "FirstParagraph": dict(base="Normal",  # first body para after heading
                               size_pt=SZ["5"], first_line_indent_pt=SZ["5"]*2,
                               line_spacing=1.5,
                               space_before_pt=0, space_after_pt=0),
        "BodyText": dict(base="Normal", size_pt=SZ["5"],
                         first_line_indent_pt=SZ["5"]*2, line_spacing=1.5,
                         space_before_pt=0, space_after_pt=0),
        "Compact": dict(base="Normal", size_pt=SZ["5"], line_spacing=1.5,
                        space_before_pt=0, space_after_pt=0),

        # Heading 1 — 小四号宋体加粗, number 小四 Times Roman 加粗
        "Heading1": dict(size_pt=SZ["-4"], bold=True,
                         space_before_pt=12, space_after_pt=6,
                         line_spacing=1.5, alignment="left"),
        # Heading 2 — 5号宋体加粗
        "Heading2": dict(size_pt=SZ["5"], bold=True,
                         space_before_pt=10, space_after_pt=5,
                         line_spacing=1.5, alignment="left"),
        # Heading 3 — 5号宋体加粗 (template shows both 标题加粗 and plain;
        # keep bold so it reads as a subheading)
        "Heading3": dict(size_pt=SZ["5"], bold=True,
                         space_before_pt=8, space_after_pt=4,
                         line_spacing=1.5, alignment="left"),
        # Heading 4 — for \paragraph{...}; keep small bold inline-ish look
        "Heading4": dict(size_pt=SZ["5"], bold=True,
                         space_before_pt=6, space_after_pt=3,
                         line_spacing=1.5, alignment="left"),

        # Title block styles — custom; post-process reassigns first paragraphs
        # Title — 2号黑体, 段前30磅 段后10磅, 1.5倍行距, centered
        "Title": dict(cn_font="黑体",
                      size_pt=SZ["2"], bold=True,
                      space_before_pt=30, space_after_pt=10,
                      line_spacing=1.5, alignment="center"),
        # Author — 小4号楷体, centered
        "Author": dict(cn_font="楷体",
                       size_pt=SZ["-4"], alignment="center",
                       line_spacing=1.5, space_after_pt=2),
        # Affiliation — 小5号楷体, centered
        "Affiliation": dict(cn_font="楷体",
                            size_pt=SZ["-5"], alignment="center",
                            line_spacing=1.5, space_after_pt=4),
        # Abstract — 小5号宋体, left/right indent 2 字符, 1.5 spacing
        "Abstract": dict(size_pt=SZ["-5"], line_spacing=1.5,
                         left_indent_pt=SZ["-5"]*2, right_indent_pt=SZ["-5"]*2,
                         space_after_pt=2),
        # Keywords — similar to abstract, slightly wider indent
        "Keywords": dict(size_pt=SZ["-5"], line_spacing=1.5,
                         left_indent_pt=SZ["-5"]*4, right_indent_pt=SZ["-5"]*2,
                         space_after_pt=2),
        # ClassificationCode (中图分类号/文献标识码/doi)
        "ClassificationCode": dict(size_pt=SZ["-5"], line_spacing=1.5,
                                   space_after_pt=4),
        # AuthorFootnote — 六号宋体
        "AuthorFootnote": dict(size_pt=SZ["6"], line_spacing=1.5),

        # Caption styles
        "Caption": dict(size_pt=SZ["-5"], bold=True, alignment="center",
                        space_before_pt=4, space_after_pt=4, line_spacing=1.25),
        "ImageCaption": dict(size_pt=SZ["-5"], bold=True, alignment="center",
                             space_before_pt=4, space_after_pt=4,
                             line_spacing=1.25),
        "TableCaption": dict(size_pt=SZ["-5"], bold=True, alignment="center",
                             space_before_pt=4, space_after_pt=4,
                             line_spacing=1.25),
        "CaptionedFigure": dict(alignment="center", line_spacing=1.0,
                                space_after_pt=4),

        # Bibliography — 小5号
        "Bibliography": dict(size_pt=SZ["-5"], line_spacing=1.5),
    },
}

PRESETS = {"qihuanbao": QIHUANBAO}


# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------
def _ensure(el, tag):
    """Return (and insert if missing) a direct child element."""
    child = el.find(qn(tag))
    if child is None:
        child = OxmlElement(tag)
        el.append(child)
    return child


def _set_rfonts(rpr, ascii_font, cn_font, en_font=None):
    """Overwrite w:rFonts with explicit fonts AND clear asciiTheme/
    eastAsiaTheme so theme-based inheritance can't hijack the font.
    pandoc heading styles set ``asciiTheme="majorHAnsi"`` which in Word
    wins over the explicit ``w:ascii`` attribute; we must remove it."""
    # Drop any pre-existing rFonts
    for existing in rpr.findall(qn("w:rFonts")):
        rpr.remove(existing)
    rFonts = OxmlElement("w:rFonts")
    rFonts.set(qn("w:ascii"), en_font or ascii_font)
    rFonts.set(qn("w:hAnsi"), en_font or ascii_font)
    rFonts.set(qn("w:eastAsia"), cn_font)
    rFonts.set(qn("w:cs"), en_font or ascii_font)
    # Theme attributes are deliberately absent — see docstring.
    # Insert at head so it precedes any sibling run-properties.
    rpr.insert(0, rFonts)


def _set_bool(rpr, tag, on):
    """Set/unset an on/off element like <w:b/>."""
    existing = rpr.find(qn(tag))
    if on:
        if existing is None:
            rpr.append(OxmlElement(tag))
    else:
        if existing is not None:
            rpr.remove(existing)


def configure_style(doc, style_name, spec, preset):
    """Apply a spec dict to an existing Word style (creating it if needed)."""
    styles = doc.styles
    # If the style doesn't exist yet, add it as a paragraph style
    if style_name not in [s.name for s in styles]:
        try:
            styles.add_style(style_name, 1)  # WD_STYLE_TYPE.PARAGRAPH = 1
        except Exception:
            pass
    # Walk by underlying XML to avoid python-docx name/id mismatches
    style = None
    for s in styles:
        # python-docx style.name is the display name; style.style_id is the id
        if s.name == style_name or getattr(s, "style_id", None) == style_name:
            style = s
            break
    if style is None:
        return

    en_font = spec.get("en_font", preset["font_en"])
    cn_font = spec.get("cn_font", preset["font_cn"])

    rpr = style.element.get_or_add_rPr()
    _set_rfonts(rpr, ascii_font=en_font, cn_font=cn_font, en_font=en_font)

    # Size — half-points
    size_pt = spec.get("size_pt")
    if size_pt is not None:
        sz = _ensure(rpr, "w:sz")
        sz.set(qn("w:val"), str(int(round(size_pt * 2))))
        szCs = _ensure(rpr, "w:szCs")
        szCs.set(qn("w:val"), str(int(round(size_pt * 2))))

    _set_bool(rpr, "w:b", spec.get("bold", False))
    _set_bool(rpr, "w:bCs", spec.get("bold", False))
    _set_bool(rpr, "w:i", spec.get("italic", False))

    # Paragraph properties
    pPr = style.element.get_or_add_pPr()

    # Alignment
    alignment = spec.get("alignment")
    if alignment:
        jc = _ensure(pPr, "w:jc")
        jc.set(qn("w:val"),
               {"left": "left", "center": "center",
                "right": "right", "both": "both"}.get(alignment, "left"))

    # Spacing (before/after in 20ths of a point; line in 240 = single)
    spacing = pPr.find(qn("w:spacing"))
    if spacing is None:
        spacing = OxmlElement("w:spacing")
        pPr.append(spacing)
    sb = spec.get("space_before_pt")
    sa = spec.get("space_after_pt")
    ls = spec.get("line_spacing")
    if sb is not None:
        spacing.set(qn("w:before"), str(int(round(sb * 20))))
    if sa is not None:
        spacing.set(qn("w:after"), str(int(round(sa * 20))))
    if ls is not None:
        # line_spacing 1.5 → w:line=360, w:lineRule=auto
        spacing.set(qn("w:line"), str(int(round(ls * 240))))
        spacing.set(qn("w:lineRule"), "auto")

    # Indent (first line / left / right, in 20ths of a point)
    ind = pPr.find(qn("w:ind"))
    if ind is None:
        ind = OxmlElement("w:ind")
        pPr.append(ind)
    fli = spec.get("first_line_indent_pt")
    li = spec.get("left_indent_pt")
    ri = spec.get("right_indent_pt")
    if fli is not None:
        ind.set(qn("w:firstLine"), str(int(round(fli * 20))))
    if li is not None:
        ind.set(qn("w:left"), str(int(round(li * 20))))
        ind.set(qn("w:start"), str(int(round(li * 20))))  # w:start alias
    if ri is not None:
        ind.set(qn("w:right"), str(int(round(ri * 20))))
        ind.set(qn("w:end"), str(int(round(ri * 20))))


# Mapping from style-id in spec to actual docx style *names*.  python-docx
# exposes heading styles as "Heading 1" (with space) but their id is
# "Heading1" (no space); same for other built-ins.
STYLE_ID_TO_NAME = {
    "Heading1": "Heading 1",
    "Heading2": "Heading 2",
    "Heading3": "Heading 3",
    "Heading4": "Heading 4",
    "Title": "Title",
    "Caption": "Caption",
}


def create_custom_style(doc, style_id, display_name=None):
    """Create a new paragraph style with a specific XML id.  We manipulate
    styles.xml directly because python-docx's add_style picks the id for you."""
    from docx.enum.style import WD_STYLE_TYPE
    display_name = display_name or style_id
    styles = doc.styles
    # If already present (by name or id), return
    for s in styles:
        if getattr(s, "style_id", None) == style_id or s.name == display_name:
            return s
    style = styles.add_style(display_name, WD_STYLE_TYPE.PARAGRAPH)
    style.element.set(qn("w:styleId"), style_id)
    # Set the name element inside the style
    name_el = style.element.find(qn("w:name"))
    if name_el is not None:
        name_el.set(qn("w:val"), display_name)
    return style


def build(preset, output_path):
    doc = Document()

    # Page setup
    page = preset["page"]
    for section in doc.sections:
        section.page_height = Cm(page["height_cm"])
        section.page_width = Cm(page["width_cm"])
        section.top_margin = section.bottom_margin = \
            section.left_margin = section.right_margin = Cm(page["margin_cm"])

    # Create all custom styles that aren't built-in
    custom_style_ids = [
        "Author", "Affiliation", "Abstract", "Keywords",
        "ClassificationCode", "AuthorFootnote",
        "FirstParagraph", "BodyText", "Compact",
        "ImageCaption", "TableCaption", "CaptionedFigure",
        "Bibliography",
    ]
    for sid in custom_style_ids:
        create_custom_style(doc, sid, display_name=sid)

    # Configure every style listed in the preset
    for style_id, spec in preset["styles"].items():
        display = STYLE_ID_TO_NAME.get(style_id, style_id)
        configure_style(doc, display, spec, preset)

    # Seed one paragraph per style so pandoc can see the style mapping.
    # pandoc picks up styles by name; seeding helps Word resolve the style
    # reference when opening the doc.
    seed_pairs = [
        ("Title", "Reference title"),
        ("Author", "Reference author"),
        ("Affiliation", "Reference affiliation"),
        ("Abstract", "Reference abstract"),
        ("Keywords", "Reference keywords"),
        ("ClassificationCode", "Reference classification"),
        ("AuthorFootnote", "Reference footnote"),
        ("Heading 1", "Reference Heading 1"),
        ("Heading 2", "Reference Heading 2"),
        ("Heading 3", "Reference Heading 3"),
        ("Heading 4", "Reference Heading 4"),
        ("FirstParagraph", "Reference first paragraph."),
        ("BodyText", "Reference body text."),
        ("Compact", "Reference compact text."),
        ("ImageCaption", "Reference image caption."),
        ("TableCaption", "Reference table caption."),
        ("CaptionedFigure", "Reference captioned figure."),
        ("Caption", "Reference caption."),
        ("Bibliography", "Reference bibliography entry."),
    ]
    for style_name, seed_text in seed_pairs:
        try:
            doc.add_paragraph(seed_text, style=style_name)
        except Exception:
            # If style lookup fails (e.g. Word uses a slightly different
            # display name internally), skip — the style block is still
            # present in styles.xml, which is what pandoc checks.
            continue

    doc.save(str(output_path))
    _patch_theme_and_defaults(
        output_path,
        latin_font=preset["font_en"],
        ea_body=preset["font_cn"],
        ea_heading=preset.get("font_cn_heiti") or preset["font_cn"],
    )


# ---------------------------------------------------------------------------
# Post-save patch: theme1.xml + docDefaults
# ---------------------------------------------------------------------------
# Word's font resolution walks: run rFonts → paragraph-style rFonts →
# docDefaults.rPrDefault rFonts → theme1.xml font scheme.  Even when a
# paragraph style sets explicit ``w:ascii="Times New Roman"``, Word's UI
# (and some merge paths) can still reach theme-based attributes if
# docDefaults carries them and the theme's Latin slot is Cambria or its
# East-Asian slot is empty.  That's what causes users to see text
# labelled "Cambria (Body)" or "MS Mincho (Body Asian)" even though the
# style itself is clean.
#
# We patch both layers after python-docx has saved the reference doc:
#   (a) theme1.xml minorFont/majorFont → point Latin and East-Asian
#       slots at the preset's actual fonts.  This also makes Word's
#       font picker UI read "Times New Roman (Body)" / "宋体 (Body
#       Asian)" — matches the intent.
#   (b) docDefaults.rPrDefault rFonts → replace the Theme-only form
#       with an explicit ascii/hAnsi/eastAsia/cs form.

_FONT_SCHEME_BLOCK_RE = re.compile(
    rb'(<a:(?P<slot>major|minor)Font>)[\s\S]*?(</a:(?P=slot)Font>)'
)


def _rewrite_font_scheme(theme_xml: bytes, slot: str,
                         latin: str, ea: str) -> bytes:
    """Rewrite <a:majorFont>…</a:majorFont> or <a:minorFont>…</a:minorFont>
    in theme1.xml so that the Latin and East-Asian typefaces are set
    to what we actually want."""
    replacement_body = (
        f'<a:latin typeface="{latin}"/>'
        f'<a:ea typeface="{ea}"/>'
        f'<a:cs typeface=""/>'
        f'<a:font script="Hans" typeface="{ea}"/>'
    ).encode("utf-8")

    def repl(m):
        if m.group("slot").decode() != slot:
            return m.group(0)
        return m.group(1) + replacement_body + m.group(3)

    return _FONT_SCHEME_BLOCK_RE.sub(repl, theme_xml)


_DOC_DEFAULTS_RFONTS_RE = re.compile(
    rb'<w:docDefaults>[\s\S]*?<w:rPrDefault>[\s\S]*?<w:rPr>[\s\S]*?'
    rb'(<w:rFonts\b[^/]*/>)'
    rb'[\s\S]*?</w:rPr>[\s\S]*?</w:rPrDefault>[\s\S]*?</w:docDefaults>'
)


def _rewrite_doc_defaults(styles_xml: bytes,
                          latin: str, ea: str) -> bytes:
    """Replace the ``<w:rFonts …/>`` inside docDefaults.rPrDefault with
    an explicit-font form so inheritance never bottoms out in a theme
    reference."""
    new_rfonts = (
        f'<w:rFonts w:ascii="{latin}" w:cs="{latin}" '
        f'w:eastAsia="{ea}" w:hAnsi="{latin}"/>'
    ).encode("utf-8")

    def repl(m):
        full = m.group(0)
        old_rf = m.group(1)
        return full.replace(old_rf, new_rfonts, 1)

    return _DOC_DEFAULTS_RFONTS_RE.sub(repl, styles_xml, count=1)


def _patch_theme_and_defaults(docx_path: Path,
                              latin_font: str,
                              ea_body: str,
                              ea_heading: str) -> None:
    path = str(docx_path)
    with zipfile.ZipFile(path, "r") as zin:
        names = zin.namelist()
        contents = {n: zin.read(n) for n in names}

    # Patch theme1.xml if present
    theme_key = next((n for n in names if n.endswith("theme/theme1.xml")), None)
    if theme_key:
        t = contents[theme_key]
        t = _rewrite_font_scheme(t, "minor", latin_font, ea_body)
        t = _rewrite_font_scheme(t, "major", latin_font, ea_heading)
        contents[theme_key] = t

    # Patch docDefaults
    s = contents["word/styles.xml"]
    contents["word/styles.xml"] = _rewrite_doc_defaults(s, latin_font, ea_body)

    tmp = path + ".patch"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, data in contents.items():
            zout.writestr(name, data)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output", type=Path, help="Output reference .docx")
    ap.add_argument("--preset", default="qihuanbao",
                    choices=sorted(PRESETS.keys()),
                    help="Journal style preset (default: qihuanbao)")
    args = ap.parse_args()

    preset = PRESETS[args.preset]
    build(preset, args.output)
    print(f"Wrote {args.output} (preset={args.preset})")


if __name__ == "__main__":
    main()
