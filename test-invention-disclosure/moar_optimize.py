"""Let MOAR search the pipeline space, scored by grounding rate.

Manual tuning got this pipeline from 84.2% to 94.4% grounding, and then hit a
wall: two hand-designed cost optimisations both made things worse (section
routing doubled cost, a cheaper model lost 4 points of coverage). That is the
point at which an optimiser earns its keep -- it searches rewrites nobody
thought to try, and scores them against the same objective.

The objective is `grounding`: the fraction of extracted text that occurs
verbatim in the source document. It is computed mechanically, with no LLM
judging, so the search signal cannot drift the way an LLM-judge signal can.
Coverage is reported alongside it as a guard -- a pipeline that extracts
nothing would otherwise score a perfect 1.0.

COST: MOAR re-runs the whole pipeline once per iteration. At ~$0.43 a run,
`--iterations 5` is roughly $2-4 including the agent's own calls. Start small.

Run:
    export GEMINI_API_KEY=...
    uv run python test-invention-disclosure/moar_optimize.py --iterations 5
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import docetl

HERE = os.path.dirname(os.path.abspath(__file__))
MD_DIR = os.path.join(HERE, "Converted-MD-Hybrid-Engine")
WORK = os.path.join(HERE, "moar_work")

sys.path.insert(0, HERE)
from docetl_extract import build, strip_boilerplate  # noqa: E402


# MOAR auto-detects models from the API keys present, and on this account that
# detection returns names Google has retired: gemini-2.5-pro (which it picks as
# the AGENT model) and gemini-2.5-flash-lite both 404. Every agent call then
# fails, no rewrite is ever proposed, and the search still reports success with
# a single "frontier" point at zero cost. Pin a verified list instead.
WORKING_MODELS = [
    "gemini/gemini-2.5-flash",
    "gemini/gemini-flash-latest",
    "gemini/gemini-flash-lite-latest",
]
AGENT_MODEL = "gemini/gemini-2.5-flash"


def norm(s) -> str:
    return re.sub(r"\s+", "", s or "")


def text_leaves(v) -> list[str]:
    out = []
    if isinstance(v, str):
        out.append(v)
    elif isinstance(v, list):
        for i in v:
            out.extend(text_leaves(i))
    elif isinstance(v, dict):
        for i in v.values():
            out.extend(text_leaves(i))
    return [t for t in out if isinstance(t, str) and len(t.strip()) >= 25]


# Fields that merely echo the input back; scoring them would reward a pipeline
# for copying the document into its own output.
_ECHO = {"body", "text", "path", "stem", "orig_len", "body_len", "pdf_path",
         "sec_head", "sec_inventors", "sec_details"}


def make_eval(sources: dict[str, str]):
    """-> eval_fn(results_path) -> {'grounding', 'coverage', 'score'}."""

    def evaluate(results_path: str) -> dict:
        with open(results_path, encoding="utf-8") as f:
            rows = json.load(f)

        ok = tot = filled = fields = 0
        for row in rows:
            src = sources.get(row.get("stem", ""), "")
            if not src:
                continue
            for key, val in row.items():
                if key in _ECHO or key.startswith(("sec_", "q_")):
                    continue
                fields += 1
                if val not in (None, "", [], {}):
                    filled += 1
                for leaf in text_leaves(val):
                    tot += 1
                    if norm(leaf) in src:
                        ok += 1

        grounding = ok / tot if tot else 0.0
        coverage = filled / fields if fields else 0.0
        # Guard the degenerate optimum: a pipeline that extracts nothing is
        # trivially 100% grounded, so weight grounding by how much it found.
        return {
            "grounding": round(grounding, 4),
            "coverage": round(coverage, 4),
            "score": round(grounding * min(coverage / 0.55, 1.0), 4),
        }

    return evaluate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=5,
                    help="MOAR search iterations; each one re-runs the pipeline (~$0.43)")
    ap.add_argument("--docs", type=int, default=0, help="limit documents (0 = all)")
    ap.add_argument("--models", default=",".join(WORKING_MODELS),
                    help="comma-separated models to search over")
    ap.add_argument("--agent-model", default=AGENT_MODEL)
    args = ap.parse_args()

    os.makedirs(WORK, exist_ok=True)
    docetl.default_model = os.environ.get("IDF_MODEL", "gemini/gemini-2.5-flash")

    rows, sources = [], {}
    for path in sorted(glob.glob(os.path.join(MD_DIR, "*.md")))[: args.docs or None]:
        stem = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        # Preprocess up front -- see build(preprocess=False).
        row = {"stem": stem, "path": path, "text": raw}
        row.update(strip_boilerplate(row))
        row.pop("text", None)
        rows.append(row)
        sources[stem] = norm(raw)

    # MOAR serializes the pipeline to YAML, so the dataset has to be a file.
    ds = os.path.join(WORK, "docs.json")
    with open(ds, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)

    frame = build(docetl.read_json(ds), preprocess=False)
    evaluate = make_eval(sources)

    print(f"{len(rows)} documents, {args.iterations} iterations")
    print("baseline:", evaluate_baseline(frame, evaluate))

    optimized = frame.optimize(
        eval_fn=evaluate,
        metric_key="score",
        max_iterations=args.iterations,
        models=[m.strip() for m in args.models.split(",") if m.strip()],
        agent_model=args.agent_model,
        save_dir=os.path.join(WORK, "results"),
    )

    res = getattr(frame, "search_results", None) or getattr(optimized, "search_results", None)
    if res is not None:
        try:
            print("\nPareto frontier:")
            print(res.to_df().to_string())
        except Exception:
            print(res)
    print(f"\n结果目录 {os.path.join(WORK, 'results')}")
    return 0


def evaluate_baseline(frame, evaluate) -> dict:
    out = os.path.join(WORK, "baseline.json")
    frame.write_json(out)
    return evaluate(out)


if __name__ == "__main__":
    sys.exit(main())
