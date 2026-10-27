"""用 LibreOffice UNO 更新目录页码并导出 docx/pdf（路径由参数给出）。"""

import sys
import time

import uno
from com.sun.star.beans import PropertyValue

SRC, DOCX, PDF = sys.argv[1], sys.argv[2], sys.argv[3]


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
for i in range(idx.getCount()):
    idx.getByIndex(i).update()
doc.refresh()
doc.storeToURL(uno.systemPathToFileUrl(DOCX), (prop("FilterName", "MS Word 2007 XML"),))
doc.storeToURL(uno.systemPathToFileUrl(PDF), (prop("FilterName", "writer_pdf_Export"),))
doc.close(True)
print("updated:", DOCX)
