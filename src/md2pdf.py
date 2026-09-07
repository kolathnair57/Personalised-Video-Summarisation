
import re
import sys
from pathlib import Path

from fpdf import FPDF

# Core PDF fonts are latin-1 only; map the unicode the notes use.
SUBS = {
    "\u2014": "--", "\u2013": "-", "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"', "\u00b1": "+/-", "\u2248": "~",
    "\u2192": "->", "\u2265": ">=", "\u2264": "<=", "\u00d7": "x",
    "\u2705": "[OK]", "\u274c": "[X]", "\u26a0": "[!]", "\ufe0f": "",
    "\u03c4": "tau", "\u03c1": "rho", "\u03bb": "lambda", "\u03b3": "gamma",
    "\u03b2": "beta", "\u00b7": "-", "\u2026": "...", "\u2011": "-",
    "\u00a0": " ", "\u2212": "-", "\u201a": ",", "\u2022": "-",
}


def clean(s):
    for a, b in SUBS.items():
        s = s.replace(a, b)
    return s.encode("latin-1", "replace").decode("latin-1")


def strip_inline(s):
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"\*(.+?)\*", r"\1", s)
    s = re.sub(r"`(.+?)`", r"\1", s)
    s = re.sub(r"\[(.+?)\]\((.+?)\)", r"\1", s)
    # A **bold** or `code` span that wraps across source lines leaves an unmatched
    # marker the regexes above cannot pair up; drop the stragglers.
    s = s.replace("**", "").replace("`", "")
    return s


class Doc(FPDF):
    def __init__(self, title, subtitle):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.title_text = title
        self.subtitle = subtitle
        self.set_auto_page_break(True, margin=18)
        self.set_margins(16, 16, 16)

    def mc(self, h, txt, x=None, **kw):
        """multi_cell that always starts from a known x.

        `multi_cell(0, ...)` measures its width from the CURRENT x, so after a centred
        or indented cell the next call can be left with no horizontal space and raises
        'Not enough horizontal space to render a single character'.
        """
        self.set_x(self.l_margin if x is None else x)
        return self.multi_cell(0, h, txt, **kw)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(120)
        self.cell(0, 6, clean(self.title_text), align="L")
        self.set_x(-60)
        self.cell(0, 6, clean(self.subtitle), align="R")
        self.ln(7)
        self.set_draw_color(210)
        self.line(16, self.get_y(), 194, self.get_y())
        self.ln(3)
        self.set_text_color(0)

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-14)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(130)
        self.cell(0, 6, f"{self.page_no()}", align="C")
        self.set_text_color(0)


def render(md_path, pdf_path):
    lines = Path(md_path).read_text(encoding="utf-8").splitlines()

    title, author, date = "Project Log", "", ""
    meta = []
    while lines and lines[0].startswith("%"):
        meta.append(lines.pop(0)[1:].strip())
    if meta:
        title = meta[0]
        author = meta[1] if len(meta) > 1 else ""
        date = meta[2] if len(meta) > 2 else ""

    pdf = Doc(title, date or author)

    # ---- title page ----
    pdf.add_page()
    pdf.ln(70)
    pdf.set_font("Helvetica", "B", 22)
    pdf.mc(10, clean(title), align="C")
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 12)
    if author:
        pdf.mc(7, clean(author), align="C")
    if date:
        pdf.set_font("Helvetica", "", 10.5)
        pdf.set_text_color(90)
        pdf.mc(7, clean(date), align="C")
        pdf.set_text_color(0)

    i = 0
    n = len(lines)
    first_h1 = True
    while i < n:
        raw = lines[i]
        line = raw.rstrip()

        if line.strip() == "\\newpage":
            pdf.add_page(); i += 1; continue

        # fenced code
        if line.startswith("```"):
            i += 1
            buf = []
            while i < n and not lines[i].startswith("```"):
                buf.append(lines[i]); i += 1
            i += 1
            pdf.ln(1.5)
            pdf.set_font("Courier", "", 7.6)
            pdf.set_fill_color(244, 244, 246)
            for b in buf:
                pdf.mc(3.9, clean(b if b.strip() else " "), fill=True)
            pdf.set_fill_color(255, 255, 255)
            pdf.ln(2)
            continue

        # tables
        if line.startswith("|") and i + 1 < n and re.match(r"^\|[\s:\-\|]+\|$", lines[i + 1].strip()):
            def cells(r):
                return [clean(strip_inline(c.strip())) for c in r.strip().strip("|").split("|")]
            head = cells(line)
            i += 2
            body = []
            while i < n and lines[i].strip().startswith("|"):
                body.append(cells(lines[i])); i += 1
            ncol = len(head)
            body = [r + [""] * (ncol - len(r)) if len(r) < ncol else r[:ncol] for r in body]
            widths = []
            for c in range(ncol):
                col = [head[c]] + [r[c] for r in body]
                widths.append(max(3, max(len(x) for x in col)))
            tot = sum(widths)
            avail = 178
            widths = [max(14, w / tot * avail) for w in widths]
            if pdf.get_y() > 240:
                pdf.add_page()
            pdf.ln(1)
            pdf.set_font("Helvetica", "", 7.8)
            with pdf.table(col_widths=widths, text_align="LEFT",
                           borders_layout="SINGLE_TOP_LINE",
                           line_height=4.4, padding=1.1) as t:
                r = t.row()
                for h in head:
                    r.cell(h)
                for brow in body:
                    r = t.row()
                    for cvl in brow:
                        r.cell(cvl)
            pdf.ln(2.5)
            continue

        # headings
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            lvl, txt = len(m.group(1)), clean(strip_inline(m.group(2)))
            if lvl == 1:
                if not first_h1:
                    pdf.add_page()
                else:
                    pdf.add_page(); first_h1 = False
                pdf.set_font("Helvetica", "B", 16)
                pdf.set_text_color(20, 40, 90)
                pdf.mc(8.5, txt)
                pdf.set_text_color(0)
                pdf.ln(2)
            elif lvl == 2:
                if pdf.get_y() > 250:
                    pdf.add_page()
                pdf.ln(2.5)
                pdf.set_font("Helvetica", "B", 11.5)
                pdf.set_text_color(30, 55, 110)
                pdf.mc(6, txt)
                pdf.set_text_color(0)
                pdf.ln(1)
            else:
                pdf.ln(1.5)
                pdf.set_font("Helvetica", "B", 9.8)
                pdf.mc(5, txt)
                pdf.ln(0.5)
            i += 1
            continue

        # horizontal rule
        if re.match(r"^-{3,}$", line.strip()):
            pdf.ln(1.5)
            pdf.set_draw_color(200)
            pdf.line(16, pdf.get_y(), 194, pdf.get_y())
            pdf.ln(3)
            i += 1
            continue

        # blockquote
        if line.startswith(">"):
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(70)
            pdf.mc(4.8, clean(strip_inline(line.lstrip("> ").strip())))
            pdf.set_text_color(0)
            pdf.ln(1)
            i += 1
            continue

        # lists
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)
        if m:
            indent = len(m.group(1)) // 2
            pdf.set_font("Helvetica", "", 9)
            bullet = "-" if m.group(2) in "-*" else m.group(2)
            pdf.mc(4.6, f"{bullet} {clean(strip_inline(m.group(3)))}",
                   x=min(16 + 4 + indent * 4, 150))
            i += 1
            continue

        if not line.strip():
            pdf.ln(2)
            i += 1
            continue

        para = [line.strip()]
        j = i + 1
        while j < n:
            nxt = lines[j]
            if (not nxt.strip() or nxt.startswith(("#", "|", "```", ">"))
                    or re.match(r"^(\s*)([-*]|\d+\.)\s+", nxt)
                    or re.match(r"^-{3,}$", nxt.strip())
                    or nxt.strip() == "\\newpage"):
                break
            para.append(nxt.strip())
            j += 1
        pdf.set_font("Helvetica", "", 9)
        pdf.mc(4.7, clean(strip_inline(" ".join(para))))
        i = j

    pdf.output(pdf_path)
    return pdf.page_no()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("markdown", help="input .md file")
    ap.add_argument("pdf", help="output .pdf file")
    a = ap.parse_args()
    print(f"wrote {a.pdf} ({render(a.markdown, a.pdf)} pages)")
