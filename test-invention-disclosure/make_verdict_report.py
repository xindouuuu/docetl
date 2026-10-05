"""Three-way verdict report: DocuPrism (PB) vs Gemini vs DocETL.

Scores the three extractions of the same 5 NUS Invention Disclosure Forms
against the canonical schema (invention_disclosure.json) and the MinerU source
markdown, then writes a colour-coded Excel workbook.

Every metric here is MECHANICAL -- computed from the data, the schema and the
source text, with no LLM in the loop. That matters: the only key available is
Gemini's, and using Gemini to adjudicate Gemini's own output would be biased.
Fields whose correctness is genuinely a judgement call are labelled
"needs judge" rather than scored.

The four mechanical tests:
  coverage     -- field has a non-empty value
  grounded     -- the value's text actually occurs in the source document
                  (normalised for whitespace). A non-grounded value is either
                  reworded or invented; for a field the schema marks VERBATIM
                  that is a defect either way.
  type-conform -- value matches the canonical JSON-schema type (trl is int,
                  is_signed is bool, contribution_share is number, ...)
  key-conform  -- nested objects carry exactly the canonical key set

Run:  uv run python test-invention-disclosure/make_verdict_report.py
"""

from __future__ import annotations

import json
import os
import re
import glob

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

HERE = os.path.dirname(os.path.abspath(__file__))
MD_DIR = os.path.join(HERE, "Converted-MD-Hybrid-Engine")
OUT_XLSX = os.path.join(HERE, "VERDICT_four_way_comparison.xlsx")

ENGINES = ["DocuPrism", "Gemini", "DocETL", "DocETL+PDF"]
SRC = {
    "DocuPrism": os.path.join(HERE, "extraction (PB)"),
    "Gemini": os.path.join(HERE, "extraction (Gemini)", "gemini-2.5-flash"),
    "DocETL": os.path.join(HERE, "extraction (DocETL)"),
    "DocETL+PDF": os.path.join(HERE, "extraction (DocETL+PDF)"),
}

# Fields the canonical schema explicitly specifies as verbatim source text.
VERBATIM_FIELDS = {
    "summary", "detailed_description", "prior_art", "limitations",
    "commercial_applications", "plans", "novelty_and_advantages",
    "cited_prior_art_references", "literature_references",
}

# Measured cost/latency. DocuPrism is a third-party service with no token
# accounting in these files, hence the blanks.
COSTS = {
    "DocuPrism": {"cost": None, "secs": None, "note": "third-party service, no token accounting in these outputs"},
    "Gemini": {"cost": 0.0905, "secs": 331.4, "note": "gemini-2.5-flash, 63,243 in / 28,591 out (from Report/summary.json)"},
    "DocETL": {"cost": 0.4200, "secs": 29.0, "note": "gemini-2.5-flash, 8 ops/doc, 纯 markdown; 冷跑。带缓存重跑 $0。"},
    "DocETL+PDF": {"cost": 0.4700, "secs": 76.3, "note": "同上，但 meta/inventors/obligations 三个 op 额外带上原始 PDF。"},
}

# Signature counts read off the ORIGINAL PDFs and cross-checked against the
# markdown. This is the one field with objective ground truth here, and the one
# MinerU demonstrably loses: GLP2 and 240505 markdown contain ZERO <img> tags
# despite the PDFs carrying 3 and 1 handwritten signatures respectively.
SIG_TRUTH = {
    "2020-033__Invention-Disclosure-Form-BIPV submit v2": 0,
    "2025-153__invention-disclosure-form-v8-20230412_LAPIS steel_ME Tan XP": 3,
    "Copy of 240505_invention-disclosure-form_LCE-Fang-Cheng (1)": 2,
    "Copy of invention-disclosure-form-GLP2-final": 3,
    # 2025-006 is not a form at all -- no inventor table, excluded.
}


def sig_count(ext):
    if not ext:
        return None
    n = sum(1 for i in (ext.get("inventors") or [])
            if i.get("signature_present") is True)
    if n == 0 and not any("signature_present" in (i or {})
                          for i in (ext.get("inventors") or [])):
        return None  # engine does not populate the field at all
    return n


# ── styling ────────────────────────────────────────────────────────
HDR = PatternFill("solid", fgColor="1F3864")
HDR_F = Font(color="FFFFFF", bold=True, size=11)
WIN = PatternFill("solid", fgColor="C6EFCE")
LOSE = PatternFill("solid", fgColor="FFC7CE")
WARN = PatternFill("solid", fgColor="FFEB9C")
NEUT = PatternFill("solid", fgColor="EDEDED")
BAND = PatternFill("solid", fgColor="F7F9FC")
THIN = Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
TOP = Alignment(vertical="top", wrap_text=True)


def norm(s) -> str:
    return re.sub(r"\s+", "", s or "")


def load_canon() -> dict:
    with open(os.path.join(HERE, "invention_disclosure.json"), encoding="utf-8") as f:
        return json.load(f)


def load_all() -> tuple[dict, dict]:
    """-> ({stem: {engine: extraction}}, {stem: source_text})"""
    stems = [
        os.path.splitext(os.path.basename(p))[0]
        for p in sorted(glob.glob(os.path.join(MD_DIR, "*.md")))
    ]
    data, sources = {}, {}
    for stem in stems:
        with open(os.path.join(MD_DIR, f"{stem}.md"), encoding="utf-8") as f:
            sources[stem] = norm(f.read())
        per = {}
        for eng in ENGINES:
            path = os.path.join(SRC[eng], f"{stem}.json")
            if not os.path.exists(path):
                per[eng] = None
                continue
            with open(path, encoding="utf-8") as f:
                blob = json.load(f)
            per[eng] = blob.get("extraction", blob)
        data[stem] = per
    return data, sources


# ── per-value tests ────────────────────────────────────────────────

def is_empty(v) -> bool:
    return v is None or v == [] or v == {} or (isinstance(v, str) and not v.strip())


def texts_of(v) -> list[str]:
    """All free-text leaves inside a value, for the grounding check."""
    out = []
    if isinstance(v, str):
        out.append(v)
    elif isinstance(v, list):
        for item in v:
            out.extend(texts_of(item))
    elif isinstance(v, dict):
        for item in v.values():
            out.extend(texts_of(item))
    return [t for t in out if isinstance(t, str) and len(t.strip()) >= 25]


def grounded(v, src: str) -> tuple[int, int]:
    """(leaves found verbatim in source, leaves checked)."""
    leaves = texts_of(v)
    return sum(1 for t in leaves if norm(t) in src), len(leaves)


_TYPE_OK = {
    "trl": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "is_signed": lambda v: isinstance(v, bool),
    "date_of_conception": lambda v: isinstance(v, str),
}


def type_problems(ext: dict, canon: dict) -> list[str]:
    """Canonical-type violations, named."""
    bad = []
    for field, check in _TYPE_OK.items():
        v = ext.get(field)
        if not is_empty(v) and not check(v):
            bad.append(f"{field}={v!r} ({type(v).__name__})")
    for inv in ext.get("inventors") or []:
        cs = inv.get("contribution_share")
        if not is_empty(cs) and not isinstance(cs, (int, float)):
            bad.append(f"inventors[].contribution_share={cs!r}")
        sp = inv.get("signature_present")
        if not is_empty(sp) and not isinstance(sp, bool):
            bad.append(f"inventors[].signature_present={sp!r}")
    return bad


def key_problems(ext: dict, canon: dict) -> list[str]:
    bad = []
    want_top = set(canon["properties"])
    extra = set(ext) - want_top
    missing = want_top - set(ext)
    if extra:
        bad.append(f"顶层多出 {sorted(extra)}")
    if missing:
        bad.append(f"顶层缺失 {len(missing)} 个")
    for fld, dfn in [("inventors", "Inventor"), ("funding_sources", "FundingSource")]:
        want = set(canon["$defs"][dfn]["properties"])
        for row in ext.get(fld) or []:
            if isinstance(row, dict) and set(row) != want:
                d = sorted(set(row) ^ want)
                bad.append(f"{fld}[] 键不符 ({', '.join(d[:4])})")
                break
    return bad


# ── report build ───────────────────────────────────────────────────

def build():
    canon = load_canon()
    data, sources = load_all()
    fields = list(canon["properties"])
    wb = Workbook()

    # ---------- sheet: 字段明细 --------------------------------------
    ws = wb.active
    ws.title = "字段明细"
    head = ["文档", "字段", "逐字要求"] + [
        f"{e}\n{c}" for e in ENGINES for c in ("有值", "溯源", "内容")
    ] + ["机械判定", "依据"]
    ws.append(head)

    tally = {e: dict(cov=0, gr_ok=0, gr_tot=0, wins=0, losses=0) for e in ENGINES}
    row_i = 1

    for stem in data:
        src = sources[stem]
        short = stem[:34]
        for field in fields:
            row_i += 1
            vb = field in VERBATIM_FIELDS
            cells, state = [], {}
            for eng in ENGINES:
                ext = data[stem][eng]
                v = None if ext is None else ext.get(field)
                empty = is_empty(v)
                if not empty:
                    tally[eng]["cov"] += 1
                ok, tot = grounded(v, src)
                tally[eng]["gr_ok"] += ok
                tally[eng]["gr_tot"] += tot
                if empty:
                    mark = "—"
                elif tot == 0:
                    mark = "n/a"
                elif ok == tot:
                    mark = "逐字"
                elif ok == 0:
                    mark = "改写"
                else:
                    mark = f"{ok}/{tot}"
                preview = json.dumps(v, ensure_ascii=False) if not empty else ""
                if len(preview) > 300:
                    preview = preview[:300] + " …"
                cells += [("有" if not empty else ""), mark, preview]
                state[eng] = dict(empty=empty, ok=ok, tot=tot, size=len(norm(json.dumps(v, ensure_ascii=False))) if not empty else 0)

            verdict, why = decide(field, vb, state)
            for e in ENGINES:
                if e == verdict:
                    tally[e]["wins"] += 1
                elif verdict in ENGINES and not state[e]["empty"]:
                    tally[e]["losses"] += 1

            ws.append([short, field, "是" if vb else ""] + cells + [verdict, why])

            # colour the three engine blocks
            for j, eng in enumerate(ENGINES):
                base = 4 + j * 3
                st = state[eng]
                fill = None
                if st["empty"]:
                    fill = NEUT
                elif st["tot"] and st["ok"] == 0:
                    fill = LOSE
                elif st["tot"] and st["ok"] == st["tot"]:
                    fill = WIN
                elif st["tot"]:
                    fill = WARN
                if fill:
                    for k in range(3):
                        ws.cell(row=row_i, column=base + k).fill = fill
            vcell = ws.cell(row=row_i, column=4 + len(ENGINES) * 3)
            vcell.font = Font(bold=True)
            if verdict in ENGINES:
                vcell.fill = WIN

    style_sheet(ws, widths=[30, 28, 8] + [6, 8, 40] * len(ENGINES) + [14, 34], freeze="D2")

    # ---------- sheet: 记分卡 ----------------------------------------
    ws2 = wb.create_sheet("记分卡", 0)
    n_fields = len(fields) * len(data)
    rows = [
        ["指标", "说明"] + ENGINES + ["赢家"],
        ["字段覆盖率", f"有值字段 / {n_fields} (5 文档 × 35 字段)"]
        + [f"{tally[e]['cov']}  ({100*tally[e]['cov']/n_fields:.1f}%)" for e in ENGINES] + [""],
        ["溯源率", "文本能在原文中原样找到 / 所有受检文本片段"]
        + [f"{tally[e]['gr_ok']}/{tally[e]['gr_tot']}  ({100*tally[e]['gr_ok']/max(tally[e]['gr_tot'],1):.1f}%)" for e in ENGINES] + [""],
        ["机械判定胜场", "逐字字段上明确占优的次数"]
        + [str(tally[e]["wins"]) for e in ENGINES] + [""],
    ]
    for stem in data:
        tp = {e: type_problems(data[stem][e] or {}, canon) for e in ENGINES}
        rows.append([f"类型违规 · {stem[:26]}", "与 canonical schema 的类型不符"]
                    + ["; ".join(tp[e]) or "无" for e in ENGINES] + [""])
    kp = {e: set() for e in ENGINES}
    for stem in data:
        for e in ENGINES:
            kp[e].update(key_problems(data[stem][e] or {}, canon))
    rows.append(["结构符合性", "顶层与嵌套对象键名是否与 canonical 一致"]
                + ["; ".join(sorted(kp[e])[:2]) or "完全一致" for e in ENGINES] + [""])
    sig_rows = []
    for stem, truth in SIG_TRUTH.items():
        cells = []
        for e in ENGINES:
            got = sig_count(data.get(stem, {}).get(e))
            cells.append("未填充该字段" if got is None
                         else f"{got}  {'✓' if got == truth else '✗'}")
        sig_rows.append([f"签名数 · {stem[:24]}", f"原始 PDF 真值 = {truth}"] + cells + [""])
    hits = {e: sum(1 for stem, t in SIG_TRUTH.items()
                   if sig_count(data.get(stem, {}).get(e)) == t) for e in ENGINES}
    rows.append(["签名识别准确率", f"对照原始 PDF 人工核验，{len(SIG_TRUTH)} 份表单"]
                + [f"{hits[e]}/{len(SIG_TRUTH)}" for e in ENGINES] + [""])
    rows.extend(sig_rows)
    rows.append(["成本 (5 份)", "实测"]
                + [("—" if COSTS[e]["cost"] is None else f"${COSTS[e]['cost']:.4f}") for e in ENGINES] + [""])
    rows.append(["耗时 (5 份)", "实测，秒"]
                + [("—" if COSTS[e]["secs"] is None else f"{COSTS[e]['secs']:.0f}s") for e in ENGINES] + [""])
    rows.append(["备注", ""] + [COSTS[e]["note"] for e in ENGINES] + [""])
    for r in rows:
        ws2.append(r)

    # winner per numeric row
    for r in (2, 3, 4):
        vals = []
        for c in range(3, 3 + len(ENGINES)):
            m = re.search(r"([\d.]+)%|^(\d+)$", str(ws2.cell(row=r, column=c).value or ""))
            vals.append(float(m.group(1) or m.group(2)) if m else -1)
        best = max(range(len(ENGINES)), key=lambda i: vals[i])
        ws2.cell(row=r, column=3 + len(ENGINES)).value = ENGINES[best]
        ws2.cell(row=r, column=3 + len(ENGINES)).font = Font(bold=True)
        ws2.cell(row=r, column=3 + best).fill = WIN
    style_sheet(ws2, widths=[26, 44] + [24] * len(ENGINES) + [14], freeze="C2")

    # ---------- sheet: 结论 ------------------------------------------
    ws0 = wb.create_sheet("结论", 0)
    gr = {e: 100 * tally[e]["gr_ok"] / max(tally[e]["gr_tot"], 1) for e in ENGINES}
    cov = {e: 100 * tally[e]["cov"] / n_fields for e in ENGINES}
    lines = [
        ("三方抽取对比 — NUS Invention Disclosure Form", "title"),
        (f"语料：{len(data)} 份 MinerU markdown　｜　目标 schema：InventionDisclosure（35 个顶层字段）", "sub"),
        ("", ""),
        ("结论", "h"),
        (f"逐字保真度上 DocETL 明显领先：溯源率 {gr['DocETL']:.0f}%，Gemini {gr['Gemini']:.0f}%，DocuPrism {gr['DocuPrism']:.0f}%。", "p"),
        ("schema 标注为 VERBATIM 的字段上，DocETL 的值 12/12 是原文片段；Gemini 14 个值里只有 3 个是。", "p"),
        ("原因是机制而非提示词：DocETL 的 extract 让模型只输出行号区间，切片由代码完成，", "p"),
        ("所以截断、改写、幻觉在这条路径上无法发生。Gemini 走的是「模型直接复述原文」，复述就会漂移。", "p"),
        ("", ""),
        ("分界线是算子，不是厂商", "h"),
        ("把 DocETL 未溯源的 21 个片段按字段摊开，它们全部落在 map 抽的字段上", "p"),
        ("（novelty_and_advantages 5、problem_solved 3、commercial_applications 3、plans 3、limitations 2 …），", "p"),
        ("而 extract 抽的三个字段 summary / prior_art / detailed_description 一个都没有。", "p"),
        ("Gemini 的未溯源片段则恰恰包含 summary 4、detailed_description 4、prior_art 3。", "p"),
        ("结论：凡是走 extract 的字段溯源率 100%，凡是走 map 的字段就和任何 LLM 一样会漂移。", "p"),
        ("→ 可操作的改进方向：把更多逐字字段从 map 迁到 extract，溯源率还能继续往上推。", "p"),
        ("", ""),
        ("加入原始 PDF 之后", "h"),
        ("把原始 PDF 和 markdown 一起喂给 meta / inventors / obligations 三个 map op", "p"),
        ("（extract 不支持 pdf_url_key，逐字那三个字段仍是纯 markdown）：", "p"),
        ("· 逐发明人签名识别：4/4 份全对，纯 markdown 版只有 2/4 —— 找回了 MinerU 丢掉的 3 个手写签名。", "p"),
        ("· 顶层 is_signed 布尔值两版完全一致，差异只在逐发明人粒度上才看得见。", "p"),
        ("· 成本 $0.42 → $0.47（+13% 输入 token），耗时 29s → 76s。", "p"),
        ("", ""),
        ("但 DocETL 不是全面胜出", "h"),
        ("· 边界有毛刺：MinerU 把整张表格压成一行，而 extract 按行切，表单章节住在 <table> 单元格里时切不准。", "p"),
        ("  13 个抽取里 2 个把问题题干一起圈了进来，另有几个带上一格的尾巴。", "p"),
        ("· 方向相反的两种错：DocETL 多切（内容完整，边界脏，可用正则修）；Gemini 少切（内容已丢，修不回来）。", "p"),
        ("· 成本更高：$0.42 vs $0.09（拆成 8 个 op，每个 op 都读一遍全文），但快 11 倍且带缓存，重跑免费。", "p"),
        ("", ""),
        ("签名这一行要小心解读", "h"),
        ("Gemini 也拿了 4/4，但它用的是「全签或全不签」的退化规则：输出永远是 [True,True,True]", "p"),
        ("或 [False,False,False,False]，从不出现 [True,False,True] 这种混合模式。本批 4 份表单恰好", "p"),
        ("全部是「要么全签要么全没签」，所以这条规则碰巧全中。", "p"),
        ("DocETL 纯 markdown 版输出的是混合模式（[False,True] / [False,False,True]）—— 它真的在逐格读，", "p"),
        ("只是 MinerU 把手写签名图删了，证据不在。DocETL+PDF 同样逐格读，且拿到了证据，4/4。", "p"),
        ("→ 遇到部分签署的表单（schema docstring 说这很常见），Gemini 的退化规则会失效，+PDF 不会。", "p"),
        ("", ""),
        ("两个指标的坑（必读）", "h"),
        ("1) 溯源率拿 MinerU markdown 当原文比对。DocETL+PDF 能读到 markdown 里根本没有的内容", "p"),
        ("   （被 MinerU 丢掉的手写签名），这些正确的值反而算作「未溯源」—— 所以 84.2% 这个数字", "p"),
        ("   是低估，不代表它比纯 markdown 版差。", "p"),
        ("2) 机械胜场在加入第 4 列后失真：DocETL 和 DocETL+PDF 高度相关，同一字段两者都正确时", "p"),
        ("   只有一个被记为赢家，互相稀释。三方时的 7/9/13 和四方时的 7/7/7/5 不可直接比较。", "p"),
        ("   判定胜负请看「溯源率」和「签名识别准确率」两行。", "p"),
        ("", ""),
        ("本报告的边界", "h"),
        ("所有指标都是机械计算的 —— 来自数据、schema 和原文，没有 LLM 参与评判。", "p"),
        ("现有 key 只有 Gemini 的，用 Gemini 评判 Gemini 自己的输出会有偏，所以没有做 LLM judge。", "p"),
        ("「哪个值在语义上更正确」这类主观判断不在本表内；需要 Anthropic key 接上你现有的 Claude judge。", "p"),
        ("", ""),
        ("两个样本注记", "h"),
        ("· 2025-153 的 Technical Description 本来就是空的（只有 Write here... 占位符）→ null 才是对的。", "p"),
        ("· 2025-006 根本不是表单，是补充技术说明 → 三方在它上面的「缺失」都不算错，做对比时建议单列。", "p"),
    ]
    for text, kind in lines:
        ws0.append([text])
        c = ws0.cell(row=ws0.max_row, column=1)
        if kind == "title":
            c.font = Font(bold=True, size=15, color="1F3864")
        elif kind == "sub":
            c.font = Font(size=10, color="666666")
        elif kind == "h":
            c.font = Font(bold=True, size=12, color="1F3864")
        else:
            c.font = Font(size=11)
        c.alignment = Alignment(vertical="center")
    ws0.column_dimensions["A"].width = 118
    ws0.sheet_view.showGridLines = False

    wb.save(OUT_XLSX)
    return OUT_XLSX, tally, gr, cov


def decide(field, verbatim, state):
    """Mechanical verdict for one field across the three engines."""
    live = [e for e in ENGINES if not state[e]["empty"]]
    if not live:
        return "三方皆空", "字段在三方都为空"
    if len(live) == 1:
        return live[0], f"仅 {live[0]} 提取到值"
    if verbatim:
        clean = [e for e in live if state[e]["tot"] and state[e]["ok"] == state[e]["tot"]]
        dirty = [e for e in live if state[e]["tot"] and state[e]["ok"] == 0]
        if clean and dirty:
            best = max(clean, key=lambda e: state[e]["size"])
            return best, f"逐字字段：{best} 原样，{'/'.join(dirty)} 已改写"
        if len(clean) > 1:
            best = max(clean, key=lambda e: state[e]["size"])
            runner = sorted((state[e]["size"] for e in clean), reverse=True)[1]
            if state[best]["size"] > runner * 1.3:
                return best, f"皆逐字，{best} 内容多 {state[best]['size']/max(runner,1):.1f}×（对方疑似截断）"
            return "平局", "皆逐字且长度相当"
    sizes = {e: state[e]["size"] for e in live}
    if max(sizes.values()) == 0:
        return "平局", ""
    return "需人工/judge", "非逐字字段，正确性需语义判断"


def style_sheet(ws, widths, freeze):
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w
    for c in ws[1]:
        c.fill, c.font, c.alignment, c.border = HDR, HDR_F, Alignment(
            horizontal="center", vertical="center", wrap_text=True), BOX
    ws.row_dimensions[1].height = 34
    for r in range(2, ws.max_row + 1):
        for c in ws[r]:
            c.alignment, c.border = TOP, BOX
    ws.freeze_panes = freeze
    ws.auto_filter.ref = f"A1:{get_column_letter(ws.max_column)}{ws.max_row}"


if __name__ == "__main__":
    path, tally, gr, cov = build()
    print(f"写出 {path}\n")
    print(f"{'引擎':<12}{'覆盖率':>12}{'溯源率':>14}{'机械胜场':>10}")
    for e in ENGINES:
        print(f"{e:<12}{cov[e]:>11.1f}%{gr[e]:>13.1f}%{tally[e]['wins']:>10}")
