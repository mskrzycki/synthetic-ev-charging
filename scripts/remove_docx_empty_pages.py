from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree


NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def paragraph_text(p: etree._Element) -> str:
    return "".join(p.xpath(".//w:t/text()", namespaces=NS)).strip()


def has_only_empty_runs(p: etree._Element) -> bool:
    if paragraph_text(p):
        return False
    # Keep paragraphs that contain drawings, tables anchors, fields, or explicit breaks.
    protected = p.xpath(".//w:drawing | .//w:pict | .//w:object | .//w:fldChar | .//w:br", namespaces=NS)
    return not protected


def clean_docx(src: Path, dst: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with ZipFile(src) as zin:
            zin.extractall(tmp_path)

        doc_path = tmp_path / "word" / "document.xml"
        parser = etree.XMLParser(remove_blank_text=False)
        root = etree.parse(str(doc_path), parser)

        removed_page_break_before = 0
        removed_empty = 0

        for p in root.xpath(".//w:body/w:p", namespaces=NS):
            text = paragraph_text(p)
            # Remove forced page starts from reviewer-comment headings so entries flow naturally.
            if text.startswith("Reviewer ") and " - Comment " in text:
                for node in p.xpath("./w:pPr/w:pageBreakBefore", namespaces=NS):
                    node.getparent().remove(node)
                    removed_page_break_before += 1

        body = root.xpath(".//w:body", namespaces=NS)[0]
        children = list(body)
        # Remove consecutive empty paragraphs and trailing empty paragraphs before section properties.
        previous_empty = False
        for child in children:
            if child.tag != f"{{{NS['w']}}}p":
                previous_empty = False
                continue
            empty = has_only_empty_runs(child)
            if empty and previous_empty:
                body.remove(child)
                removed_empty += 1
                continue
            previous_empty = empty

        for child in list(body):
            if child.tag == f"{{{NS['w']}}}sectPr":
                break
            last_p = child
        # Walk backwards and remove empty paragraphs immediately before final sectPr.
        for child in reversed(list(body)):
            if child.tag == f"{{{NS['w']}}}sectPr":
                continue
            if child.tag == f"{{{NS['w']}}}p" and has_only_empty_runs(child):
                body.remove(child)
                removed_empty += 1
                continue
            break

        root.write(str(doc_path), xml_declaration=True, encoding="UTF-8", standalone=True)

        if dst.exists():
            dst.unlink()
        with ZipFile(dst, "w", ZIP_DEFLATED) as zout:
            for path in tmp_path.rglob("*"):
                if path.is_file():
                    zout.write(path, path.relative_to(tmp_path).as_posix())

    print(f"Removed pageBreakBefore markers: {removed_page_break_before}")
    print(f"Removed empty paragraphs: {removed_empty}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: remove_docx_empty_pages.py input.docx output.docx")
    clean_docx(Path(sys.argv[1]), Path(sys.argv[2]))
