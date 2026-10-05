"""A/B: does feeding the original PDF alongside the MinerU markdown help?

Identical prompt, identical model, identical schema. The only difference is
whether `pdf_url_key` rides along, so any delta is attributable to the PDF.

Targets the fields most likely to benefit from seeing the real page rather than
MinerU's HTML-table rendering:
  is_signed          -- today inferred from <img> placeholders, which is
                        indirect: an <img> in a signature cell could be anything
  inventors          -- the TRANSPOSED table (people are columns)
  funding_sources    -- selection shows as a tick word next to a menu item

Run: uv run python test-invention-disclosure/ab_multimodal.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

import docetl

HERE = os.path.dirname(os.path.abspath(__file__))
MD_DIR = os.path.join(HERE, "Converted-MD-Hybrid-Engine")
PDF_DIR = os.path.join(HERE, "samples_pdf")

MODEL = os.environ.get("IDF_MODEL", "gemini/gemini-2.5-flash")
docetl.default_model = MODEL

PROMPT = """
Read this National University of Singapore INVENTION DISCLOSURE FORM and
extract the signature status, the inventors, and the funding sources.

MARKDOWN RENDERING OF THE FORM:
{{ input.body }}

- is_signed: "yes" ONLY if the inventor table's "Signature" row actually holds
  a signature -- a hand-written mark, scrawled initials, or a typed name in the
  signature cell. A blank signature row -> "no". Do NOT infer signing from
  inventor names being present elsewhere, from the "/pdf/ electronic signature"
  boilerplate clause, or from the filename.
- n_signatures: how many inventor signature cells are actually filled.
- inventors: the "III-A. NUS INVENTORS" / "III-B. NON-NUS INVENTORS" tables are
  TRANSPOSED -- inventors are COLUMNS ("Inventor 1".."Inventor 5") and
  attributes are ROWS. Read DOWN each column to assemble one inventor. Skip
  columns that are entirely blank.
- funding_sources: the form prints a MENU of ~20 numbered agencies. Capture
  ONLY the agencies actually SELECTED or filled in, never the menu itself.
  Selection shows either as the agency copied into "a. Funding source (from
  above)" or as a tick/mark beside a menu item.

If the document is not an invention disclosure form at all, return empty
values.
""".strip()

SCHEMA = {
    "schema": {
        "is_signed": "enum[yes, no, unknown]",
        "n_signatures": "integer",
        "inventors": (
            "list[{family_name: string, given_name: string, "
            "department: string, contribution_pct: string, is_nus: string}]"
        ),
        "funding_sources": "list[{agency: string, grant_no: string}]",
    }
}


def rows() -> list[dict]:
    out = []
    for md in sorted(glob.glob(os.path.join(MD_DIR, "*.md"))):
        stem = os.path.splitext(os.path.basename(md))[0]
        pdf = os.path.join(PDF_DIR, f"{stem}.pdf")
        if not os.path.exists(pdf):
            continue
        with open(md, encoding="utf-8") as f:
            out.append({"stem": stem, "body": f.read(), "pdf_path": pdf})
    return out


def run(label: str, with_pdf: bool) -> tuple[list[dict], float]:
    frame = docetl.from_list(rows())
    kwargs = {"pdf_url_key": "pdf_path"} if with_pdf else {}
    frame = frame.map(name=label, prompt=PROMPT, output=SCHEMA,
                      skip_on_error=True, **kwargs)
    data = frame.collect()
    return data, frame.total_cost


def main() -> int:
    if not os.path.isdir(PDF_DIR):
        print(f"missing {PDF_DIR} -- run the docx->pdf conversion first", file=sys.stderr)
        return 1

    md_only, cost_a = run("md_only", with_pdf=False)
    both, cost_b = run("md_plus_pdf", with_pdf=True)

    by_stem = {d["stem"]: d for d in both}
    print(f"\n{'='*86}")
    print(f"{'文档':<30} {'':<4} {'is_signed':>10} {'#签名':>7} {'#发明人':>8} {'#资助':>7}")
    print("=" * 86)
    for a in md_only:
        b = by_stem.get(a["stem"], {})
        for lab, d in (("md", a), ("+pdf", b)):
            print(f"{a['stem'][:30]:<30} {lab:<4} "
                  f"{str(d.get('is_signed')):>10} {str(d.get('n_signatures')):>7} "
                  f"{len(d.get('inventors') or []):>8} {len(d.get('funding_sources') or []):>7}")
        print("-" * 86)

    out = os.path.join(HERE, "ab_multimodal_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"md_only": md_only, "md_plus_pdf": both,
                   "cost_md_only": cost_a, "cost_md_plus_pdf": cost_b},
                  f, ensure_ascii=False, indent=1, default=str)
    print(f"\n成本  仅 md: ${cost_a:.4f}    md+pdf: ${cost_b:.4f}"
          f"   ({cost_b/max(cost_a,1e-9):.1f}×)")
    print(f"明细写入 {os.path.basename(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
