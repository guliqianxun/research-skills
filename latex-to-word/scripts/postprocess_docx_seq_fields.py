"""
Post-process a DOCX file to inject SEQ Figure / SEQ Table fields into
Caption-style paragraphs, so that Word's Insert > Cross-reference dialog
lists figures and tables correctly.

Usage:  python postprocess_docx_seq_fields.py <docx_path>
"""

import sys
import re
import os
import zipfile
import io

# ---------------------------------------------------------------------------
# SEQ field XML fragments
# ---------------------------------------------------------------------------
def seq_field_xml(seq_name: str, num: int, char: str) -> str:
    """Return the XML runs that represent:  <char><SEQ seq_name>num"""
    return (
        f'<w:r><w:t xml:space="preserve">{char}</w:t></w:r>'
        f'<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
        f'<w:r><w:instrText xml:space="preserve"> SEQ {seq_name} \\* ARABIC </w:instrText></w:r>'
        f'<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
        f'<w:r><w:t>{num}</w:t></w:r>'
        f'<w:r><w:fldChar w:fldCharType="end"/></w:r>'
    )


# ---------------------------------------------------------------------------
# Helpers to extract plain text from an XML run cluster
# ---------------------------------------------------------------------------
_TEXT_RE = re.compile(r'<w:t[^>]*>(.*?)</w:t>', re.DOTALL)

def para_text(para_xml: str) -> str:
    return ''.join(_TEXT_RE.findall(para_xml))


# ---------------------------------------------------------------------------
# Core injection logic
# ---------------------------------------------------------------------------
# Matches the FIRST <w:r>...</w:r> block that contains a <w:t> with text
_FIRST_RUN_RE = re.compile(r'(<w:r\b[^>]*>(?:(?!<w:r\b).)*?<w:t[^>]*>)(.*?)(</w:t>.*?</w:r>)', re.DOTALL)


def inject_seq_into_para(para_xml: str, seq_name: str, num: int, lead_char: str) -> str:
    """
    Replace the leading character+number in the first text run of a Caption
    paragraph with proper SEQ field XML.

    E.g. for a figure caption whose text run starts with "图1　...",
    we split into:  SEQ_FIELD_RUNS  +  run containing "　..."
    """
    # Find the first run that carries actual text
    m = _FIRST_RUN_RE.search(para_xml)
    if not m:
        return para_xml

    run_prefix = m.group(1)   # opening tags up to and including <w:t ...>
    run_text   = m.group(2)   # raw text content
    run_suffix = m.group(3)   # </w:t>...</w:r>

    # The text should start with  图N  or  表N  (N = 1-2 digit number)
    # We accept a full-width space (U+3000), ideographic space, or \quad-like spaces after the number
    pattern = re.compile(r'^([图表])(\d{1,2})([\s\u3000\u0020]*)(.*)', re.DOTALL)
    tm = pattern.match(run_text)
    if not tm:
        return para_xml  # unexpected format, leave untouched

    after_num = tm.group(3) + tm.group(4)   # spaces + rest of caption

    # Build replacement: SEQ fields + remaining text run
    seq_runs = seq_field_xml(seq_name, num, lead_char)
    remaining_run = f'{run_prefix}{after_num}{run_suffix}'

    original_full_run = m.group(0)
    replacement = seq_runs + remaining_run

    return para_xml.replace(original_full_run, replacement, 1)


def inject_seq_table_into_para(para_xml: str, num: int) -> str:
    """
    Table captions in this document do NOT have a leading 表N prefix,
    so we just prepend SEQ Table field runs before any existing content.
    """
    # Find insertion point: right after </w:pPr> (end of paragraph properties)
    insert_after = re.search(r'</w:pPr>', para_xml)
    if not insert_after:
        return para_xml

    pos = insert_after.end()
    seq_runs = seq_field_xml('Table', num, '表')
    # Also add a separator space run
    sep_run = '<w:r><w:t xml:space="preserve"> </w:t></w:r>'
    return para_xml[:pos] + seq_runs + sep_run + para_xml[pos:]


# ---------------------------------------------------------------------------
# Main paragraph scanner
# ---------------------------------------------------------------------------
_PARA_RE = re.compile(r'<w:p[ >][\s\S]*?</w:p>')
# Pandoc uses these style names for figure/table captions
_IMAGE_CAPTION_RE = re.compile(r'w:val="ImageCaption"')
_TABLE_CAPTION_RE = re.compile(r'w:val="TableCaption"')

# LaTeX source has captions like "图1　" for figures; tables have no number prefix.
_FIG_LEAD_RE = re.compile(r'^[图]\d')
_TAB_LEAD_RE = re.compile(r'^[表]\d')


def process_document_xml(xml: str) -> str:
    fig_counter = 0
    tab_counter = 0

    def process_para(m: re.Match) -> str:
        nonlocal fig_counter, tab_counter
        para = m.group(0)
        if _IMAGE_CAPTION_RE.search(para):
            fig_counter += 1
            return inject_seq_into_para(para, 'Figure', fig_counter, '图')
        elif _TABLE_CAPTION_RE.search(para):
            tab_counter += 1
            text = para_text(para)
            if _TAB_LEAD_RE.match(text):
                return inject_seq_into_para(para, 'Table', tab_counter, '表')
            else:
                # Table caption without leading 表N — prepend SEQ field
                return inject_seq_table_into_para(para, tab_counter)
        return para  # not a caption paragraph

    return _PARA_RE.sub(process_para, xml)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def postprocess(docx_path: str) -> None:
    with zipfile.ZipFile(docx_path, 'r') as zin:
        names = zin.namelist()
        file_contents: dict[str, bytes] = {n: zin.read(n) for n in names}

    xml = file_contents['word/document.xml'].decode('utf-8')
    modified = process_document_xml(xml)

    if modified == xml:
        print('postprocess: no Caption paragraphs modified, skipping rewrite.')
        return

    file_contents['word/document.xml'] = modified.encode('utf-8')

    tmp_path = docx_path + '.seqtmp'
    with zipfile.ZipFile(tmp_path, 'w', zipfile.ZIP_DEFLATED) as zout:
        for name, data in file_contents.items():
            zout.writestr(name, data)

    os.replace(tmp_path, docx_path)
    print(f'postprocess: SEQ fields injected into {os.path.basename(docx_path)}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python postprocess_docx_seq_fields.py <docx_path>', file=sys.stderr)
        sys.exit(1)
    postprocess(sys.argv[1])
