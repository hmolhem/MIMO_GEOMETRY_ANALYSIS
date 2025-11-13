"""Extract text from the Weight-Constrained Sparse Arrays reference PDF.

Usage (PowerShell):
    python analysis_scripts/extract_wcsa_pdf.py \
        --pdf 01-Weight_Constrained_Sparse.pdf \
        --out results/wcsa_text.txt

If --out is omitted, defaults to results/wcsa_text.txt

Dependencies: PyPDF2 (declared in requirements.txt)
"""
from __future__ import annotations
import argparse
import sys
import os
from pathlib import Path

try:
    from PyPDF2 import PdfReader  # type: ignore
except ImportError as e:  # pragma: no cover
    print("[ERROR] PyPDF2 not installed. Please install requirements.", file=sys.stderr)
    raise


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception as ex:  # robust to occasional parsing glitches
            txt = f"\n[PAGE {i+1} PARSE ERROR]: {ex}\n"
        parts.append(f"\n\n===== PAGE {i+1} =====\n\n{txt}\n")
    return "".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract text from WCSA PDF")
    parser.add_argument('--pdf', type=str, default='01-Weight_Constrained_Sparse.pdf', help='Path to PDF')
    parser.add_argument('--out', type=str, default='results/wcsa_text.txt', help='Output text file')
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"[ERROR] PDF not found: {pdf_path}", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Reading PDF: {pdf_path}")
    text = extract_text(pdf_path)

    # Basic normalization: strip trailing spaces, ensure UTF-8 encodable
    cleaned = '\n'.join(line.rstrip() for line in text.splitlines())

    out_path.write_text(cleaned, encoding='utf-8')
    print(f"[INFO] Wrote extracted text to: {out_path} ({out_path.stat().st_size} bytes)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
