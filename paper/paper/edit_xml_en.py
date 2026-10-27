"""English manuscript post-processing: author block, caption styling, keepNext."""

import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
doc = root / "word" / "document.xml"
s = doc.read_text(encoding="utf-8")

# ---------- 1. author block after the title ----------
title_key = "Design and Implementation of a Win-Probability"
author_block = (
    '<w:p><w:pPr><w:spacing w:before="120" w:after="0"/><w:ind w:firstLine="0"/>'
    '<w:jc w:val="center"/></w:pPr>'
    '<w:r><w:rPr><w:sz w:val="26"/></w:rPr><w:t>Zixuan Tang</w:t></w:r></w:p>'
    '<w:p><w:pPr><w:spacing w:before="0" w:after="240"/><w:ind w:firstLine="0"/>'
    '<w:jc w:val="center"/></w:pPr>'
    '<w:r><w:rPr><w:sz w:val="22"/><w:color w:val="444444"/></w:rPr>'
    '<w:t>Southwest Petroleum University, Chengdu, China</w:t></w:r></w:p>'
)

if "Zixuan Tang" not in s:
    idx = s.find(title_key)
    para_end = s.find("</w:p>", idx)
    s = s[:para_end + 6] + author_block + s[para_end + 6:]
    print("author block inserted")

# ---------- 2. centre caption paragraphs ----------
def style_captions(xml: str) -> str:
    count = 0
    for m in list(re.finditer(r"<w:p>(?:(?!</w:p>).)*?</w:p>", xml, re.S)):
        para = m.group(0)
        text = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", para, re.S))
        if not re.match(r"^(Figure|Table) \d", text):
            continue
        if "<w:drawing>" in para:
            continue
        new = para
        if "<w:pPr>" in new:
            new = new.replace("<w:pPr>", '<w:pPr><w:spacing w:before="60" w:after="60"/><w:ind w:firstLine="0"/><w:jc w:val="center"/>', 1)
            # drop duplicated spacing/ind if present
            new = re.sub(r"(<w:spacing w:before=\"60\" w:after=\"60\"/><w:ind w:firstLine=\"0\"/><w:jc w:val=\"center\"/>)(?=.*?</w:pPr>)", r"\1", new, count=1)
        else:
            new = new.replace("<w:p>", '<w:p><w:pPr><w:spacing w:before="60" w:after="60"/><w:ind w:firstLine="0"/><w:jc w:val="center"/></w:pPr>', 1)
        xml = xml.replace(para, new, 1)
        count += 1
    print("captions styled:", count)
    return xml


s = style_captions(s)

# ---------- 3. keepNext for image paragraphs ----------
def add_keepnext(xml: str) -> str:
    out = []
    count = 0
    for m in re.finditer(r"<w:p>(?:(?!</w:p>).)*?<w:drawing>.*?</w:p>", xml, re.S):
        para = m.group(0)
        if "<w:keepNext" in para:
            continue
        if "<w:pPr>" in para:
            new = para.replace("<w:pPr>", "<w:pPr><w:keepNext/>", 1)
        else:
            new = para.replace("<w:p>", "<w:p><w:pPr><w:keepNext/></w:pPr>", 1)
        out.append((m.start(), m.end(), new))
        count += 1
    for start, end, new in reversed(out):
        xml = xml[:start] + new + xml[end:]
    print("keepNext:", count)
    return xml


s = add_keepnext(s)
# ---------- 4. repeat table header rows ----------
def header_rows(xml: str) -> str:
    count = 0
    for m in re.finditer(r"<w:tbl>.*?</w:tbl>", xml, re.S):
        tbl = m.group(0)
        first = re.search(r"<w:tr[ >].*?</w:tr>", tbl, re.S)
        if not first:
            continue
        row = first.group(0)
        if "w:tblHeader" in row:
            continue
        if "<w:trPr>" in row:
            new_row = row.replace("<w:trPr>", "<w:trPr><w:tblHeader/>", 1)
        else:
            new_row = re.sub(r"(<w:tr[^>]*>)", r"\1<w:trPr><w:tblHeader/></w:trPr>", row, count=1)
        xml = xml.replace(row, new_row, 1)
        count += 1
    print("table headers marked:", count)
    return xml


s = header_rows(s)

# ---------- 5. enlarge figures to 15 cm width ----------
def scale_images(xml: str) -> str:
    target = 5400000  # 15 cm in EMU
    count = 0
    def repl(m):
        nonlocal count
        cx, cy = int(m.group(1)), int(m.group(2))
        if cx <= 0:
            return m.group(0)
        k = target / cx
        count += 1
        return f'<wp:extent cx="{target}" cy="{int(cy * k)}"/>'
    xml = re.sub(r'<wp:extent cx="(\d+)" cy="(\d+)"/>', repl, xml)
    xml = re.sub(r'<a:ext cx="(\d+)" cy="(\d+)"/>',
                 lambda m: f'<a:ext cx="{target}" cy="{int(int(m.group(2)) * target / max(int(m.group(1)), 1))}"/>',
                 xml)
    print("images scaled:", count)
    return xml


s = scale_images(s)
doc.write_text(s, encoding="utf-8")
print("done")
