import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])  # 解包目录

doc = root / "word" / "document.xml"
s = doc.read_text(encoding="utf-8")

brk = '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'

# 1) 在真正的 Heading1「四、方法」前加分页（避开目录里的同名条目）
pat = re.compile(
    r"<w:p>(?:(?!</w:p>).)*?<w:pStyle w:val=\"Heading1\"/>"
    r"(?:(?!</w:p>).)*?四、方法.*?</w:p>", re.S)
m = pat.search(s)
if m and "w:br w:type=\"page\"" not in s[max(0, m.start() - 300):m.start()]:
    s = s[:m.start()] + brk + s[m.start():]
    print("page break inserted before chapter 4")
else:
    print("chapter 4 break skipped")

# 2) 目录标题
title = ('<w:p><w:pPr><w:spacing w:before="0" w:after="120"/><w:ind w:firstLine="0"/>'
         '<w:jc w:val="center"/></w:pPr>'
         '<w:r><w:rPr><w:rFonts w:ascii="Cambria Math" w:eastAsia="SimHei" w:hAnsi="Cambria Math"/>'
         '<w:b/><w:sz w:val="32"/></w:rPr><w:t>目  录</w:t></w:r></w:p>')
idx = s.find("<w:sdt>")
if idx != -1 and "目  录" not in s[max(0, idx - 600):idx]:
    s = s[:idx] + title + s[idx:]
    print("toc title inserted")

doc.write_text(s, encoding="utf-8")

st = root / "word" / "settings.xml"
t = st.read_text(encoding="utf-8")
if "updateFields" not in t:
    mm = re.search(r"<w:compat[ >/]", t)
    pos = mm.start() if mm else t.rindex("</w:settings>")
    st.write_text(t[:pos] + '<w:updateFields w:val="true"/>' + t[pos:], encoding="utf-8")
    print("updateFields added")


# 3) 图片段落加 keepNext，保证图与题注不分页
def add_keepnext(xml: str) -> str:
    out = []
    last = 0
    count = 0
    for m in re.finditer(r"<w:p>(?:(?!</w:p>).)*?<w:drawing>.*?</w:p>", xml, re.S):
        para = m.group(0)
        if "<w:keepNext" in para:
            continue
        if "<w:pStyle" in para:
            new = re.sub(r"(<w:pStyle[^/]*/>)", r"\1<w:keepNext/>", para, count=1)
        elif "<w:pPr>" in para:
            new = para.replace("<w:pPr>", "<w:pPr><w:keepNext/>", 1)
        else:
            new = para.replace("<w:p>", "<w:p><w:pPr><w:keepNext/></w:pPr>", 1)
        out.append((m.start(), m.end(), new))
        count += 1
    for start, end, new in reversed(out):
        xml = xml[:start] + new + xml[end:]
    print("keepNext added to", count, "image paragraphs")
    return xml

doc.write_text(add_keepnext(doc.read_text(encoding="utf-8")), encoding="utf-8")
