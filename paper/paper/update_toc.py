"""用 LibreOffice UNO 更新目录页码并导出 docx/pdf。"""

import sys
import time

import uno
from com.sun.star.beans import PropertyValue

SRC = r"C:\Users\46889\Desktop\csgo预测模型\paper\CS2胜率预测系统技术论文.docx"
DOCX = r"C:\Users\46889\Desktop\csgo预测模型\paper\out_updated.docx"
PDF = r"C:\Users\46889\Desktop\csgo预测模型\paper\CS2胜率预测系统技术论文.pdf"


def prop(name, value):
    p = PropertyValue()
    p.Name = name
    p.Value = value
    return p


local = uno.getComponentContext()
resolver = local.ServiceManager.createInstanceWithContext(
    "com.sun.star.bridge.UnoUrlResolver", local)
ctx = None
for _ in range(40):
    try:
        ctx = resolver.resolve(
            "uno:socket,host=127.0.0.1,port=2002;urp;StarOffice.ComponentContext")
        break
    except Exception:
        time.sleep(1)
if ctx is None:
    raise SystemExit("cannot connect to soffice")

desktop = ctx.ServiceManager.createInstanceWithContext(
    "com.sun.star.frame.Desktop", ctx)
doc = desktop.loadComponentFromURL(uno.systemPathToFileUrl(SRC), "_blank", 0,
                                   (prop("Hidden", True),))

idx = doc.getDocumentIndexes()
count = idx.getCount()
print("indexes:", count)
for i in range(count):
    idx.getByIndex(i).update()
doc.refresh()

doc.storeToURL(uno.systemPathToFileUrl(DOCX), (prop("FilterName", "MS Word 2007 XML"),))
doc.storeToURL(uno.systemPathToFileUrl(PDF), (prop("FilterName", "writer_pdf_Export"),))
doc.close(True)
print("done")
