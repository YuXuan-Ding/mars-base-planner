"""Render a review's Markdown as a PDF (fpdf2 + markdown).

Uses DejaVu Sans when it's installed, so symbols like >=, x and en dashes print as written;
otherwise falls back to the built-in Helvetica and swaps those characters for plain ones.
"""
from __future__ import annotations

import re
from pathlib import Path

import markdown
from fpdf import FPDF, FontFace

FONT_DIRS = ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu", "/Library/Fonts", "C:/Windows/Fonts"]
PLAIN = {"–": "-", "—": "-", "≥": ">=", "≤": "<=", "×": "x", "≈": "~", "→": "->", "’": "'", "‘": "'",
         "“": '"', "”": '"', "…": "...", "°": " deg", "±": "+/-", "·": "-", " ": " "}


def _font(pdf: FPDF) -> str:
    for d in FONT_DIRS:
        reg, bold = Path(d, "DejaVuSans.ttf"), Path(d, "DejaVuSans-Bold.ttf")
        if reg.exists() and bold.exists():
            pdf.add_font("DejaVu", "", str(reg))
            pdf.add_font("DejaVu", "B", str(bold))
            # no italic file needed: map italics to the regular face
            pdf.add_font("DejaVu", "I", str(reg))
            pdf.add_font("DejaVu", "BI", str(bold))
            return "DejaVu"
    return "Helvetica"


class _Doc(FPDF):
    title_text = ""

    def footer(self):
        self.set_y(-12)
        self.set_font(self.font_family, size=8)
        self.set_text_color(130)
        self.cell(0, 6, f"{self.title_text}   ·   page {self.page_no()}/{{nb}}"
                  if self.font_family == "dejavu" else f"{self.title_text} - page {self.page_no()}/{{nb}}", align="C")


def render(md: str, title: str = "Hazard review") -> bytes:
    pdf = _Doc(format="A4")
    pdf.set_margins(18, 16, 18)
    pdf.set_auto_page_break(True, margin=18)
    family = _font(pdf)
    if family == "Helvetica":
        md = "".join(PLAIN.get(ch, ch) for ch in md).encode("latin-1", "replace").decode("latin-1")
    pdf.title_text = title
    pdf.set_title(title)
    pdf.set_creator("Mars base planner")
    pdf.add_page()
    pdf.set_font(family, size=10)
    html = markdown.markdown(_fix_lists(md), extensions=["tables", "sane_lists"])
    html = re.sub(r"<table>.*?</table>", _size_table, html, flags=re.S)
    pdf.write_html(html, font_family=family, table_line_separators=True, tag_styles={
        "h1": FontFace(color=(25, 45, 70), size_pt=17, emphasis="B"),
        "h2": FontFace(color=(25, 45, 70), size_pt=13.5, emphasis="B"),
        "h3": FontFace(color=(60, 75, 95), size_pt=11.5, emphasis="B"),
    })
    return bytes(pdf.output())


def _fix_lists(md: str) -> str:
    """Markdown needs a blank line before a list; Claude often writes '**Label**' straight above one."""
    out, prev = [], ""
    for line in md.splitlines():
        is_item = bool(re.match(r"\s*([-*+]|\d+\.)\s", line))
        prev_item = bool(re.match(r"\s*([-*+]|\d+\.)\s", prev))
        if is_item and prev.strip() and not prev_item:
            out.append("")
        out.append(line)
        prev = line
    return "\n".join(out)


def _size_table(m: re.Match) -> str:
    """Column widths in proportion to their longest word and typical text, left-aligned cells."""
    t = m.group(0)
    rows = re.findall(r"<tr>(.*?)</tr>", t, flags=re.S)
    cells = [re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", r, flags=re.S) for r in rows]
    n = max((len(c) for c in cells), default=0)
    if not n:
        return t
    strip = lambda h: re.sub(r"<[^>]+>", "", h)
    need = []
    for k in range(n):
        col = [strip(c[k]) for c in cells if k < len(c)]
        head_word = max((len(w) for w in col[0].split()), default=1)     # header row is bold, so wider
        word = max((len(w) for v in col[1:] for w in v.split()), default=1)
        avg = sum(len(v) for v in col) / len(col)
        need.append(max(head_word * 1.7, word * 1.3, 0.6 * min(avg, 40), 4))
    tot = sum(need)
    widths = [max(6, round(100 * x / tot)) for x in need]
    widths[-1] = max(6, 100 - sum(widths[:-1]))
    head = iter(widths)
    t = re.sub(r"<th(?=[ >])", lambda _: f'<th width="{next(head, 10)}%" align="left"', t, count=n)
    t = re.sub(r"<td(?=[ >])", '<td align="left"', t)
    return t.replace("<table>", '<table border="1" width="100%">', 1)
