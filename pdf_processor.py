"""Extraction de texte depuis un PDF et reconstruction d'un PDF anonymisé."""
import io

import pdfplumber
from fpdf import FPDF


def extract_pages(pdf_bytes: bytes) -> list[str]:
    """Retourne la liste du texte de chaque page."""
    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return pages


_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_FONT_BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def build_pdf(pages_text: list[str]) -> bytes:
    """Reconstruit un PDF à partir d'une liste de textes (une entrée par page)."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_font("DejaVu", fname=_FONT_PATH)
    pdf.add_font("DejaVu", style="B", fname=_FONT_BOLD_PATH)

    for i, text in enumerate(pages_text):
        pdf.add_page()
        pdf.set_font("DejaVu", size=11)
        pdf.multi_cell(0, 7, text)

    return bytes(pdf.output())
