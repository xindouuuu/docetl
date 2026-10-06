"""DocETL extraction pipeline for NUS Invention Disclosure Forms.

Targets the canonical `InventionDisclosure` schema (invention_disclosure.py /
invention_disclosure.json) from the MinerU markdown in Converted-MD-Hybrid-Engine/,
and writes per-document JSON in the same envelope as the existing
`extraction (Gemini)/` outputs so the benchmark harness can consume it unchanged.

Design notes (why this is not one big 35-field LLM call):

1. DocETL's output-schema notation (`convert_val`) supports only ONE level of
   object nesting -- `list[{name: {family_name: string}}]` raises ValueError.
   Inventor.name is a nested PersonName, so inventors are extracted FLAT here
   and re-nested in `assemble()`.
2. Every field in a DocETL schema is required (no `| None`). Since most form
   fields are legitimately blank, each op returns a sentinel ("" / "unknown")
   which `assemble()` maps back to null. Forcing the model to emit a value for
   a blank field is exactly how hallucinations get in.
3. The schema is split by FORM SECTION into focused ops. DocETL's own guidance
   is that output quality degrades as the schema grows; the 35-field target is
   well past that point.
4. The long narrative fields are pulled with `extract`, not `map`: the model
   emits line RANGES and code slices the original text, so these fields cannot
   be truncated or paraphrased. The Gemini baseline's error profile on this
   same corpus was truncated=4, content_error=3, hallucination=1 -- all three
   are failure modes `extract` removes by construction.

Run:
    export GEMINI_API_KEY=...        # or OPENAI_API_KEY / ANTHROPIC_API_KEY
    uv run python test-invention-disclosure/docetl_extract.py
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
import time

import docetl

HERE = os.path.dirname(os.path.abspath(__file__))
MD_DIR = os.path.join(HERE, "Converted-MD-Hybrid-Engine")
PDF_DIR = os.path.join(HERE, "samples_pdf")
# Feed the ORIGINAL pdf alongside the markdown on the ops that read layout
# (signatures, the transposed inventor table, ticked funding menu items).
# MinerU drops embedded images entirely on some documents -- 2 of these 5
# have zero <img> in their markdown despite having handwritten signatures.
USE_PDF = os.environ.get("IDF_USE_PDF", "0") == "1"
OUT_DIR = os.path.join(
    HERE, "extraction (DocETL+PDF)" if USE_PDF else "extraction (DocETL)"
)

MODEL = os.environ.get("IDF_MODEL", "gemini/gemini-2.5-flash")
RUN = os.environ.get("IDF_RUN", "docetl-run1")

docetl.default_model = MODEL
docetl.rate_limits = {
    "llm_call": [{"count": 60, "per": 1, "unit": "minute"}],
    "llm_tokens": [{"count": 900_000, "per": 1, "unit": "minute"}],
}

# Shared grounding rule, lifted verbatim from the schema module docstring so the
# extractor and the schema agree on what "absent" means.
# The three extract ops were poisoned by an upstream bug (quoted schema keys
# in convert_val) whose cache key does not change when the bug is fixed.
BYPASS_EXTRACT = os.environ.get("IDF_BYPASS_EXTRACT", "0") == "1"

# Two ideas MOAR found that hand-tuning missed (its pipeline_gemini_9 scored
# 97.1% grounding at the baseline's 53.7% coverage, where the baseline grounded
# 95.1%): a verbatim compression pass in front of everything else, and pinning
# temperature to 0.
COMPRESS = os.environ.get("IDF_COMPRESS", "0") == "1"
TEMP0 = os.environ.get("IDF_TEMP0", "0") == "1"
LLM_KW = {"temperature": 0} if TEMP0 else {}

GROUNDING = """
GROUNDING RULE (applies to every field): populate a field ONLY from text
physically present in THIS form. If a section is blank, filled with
"NA"/"N.A."/"Write here..."/placeholder dots, or absent entirely, return the
empty string "" (or [] for a list) -- never infer a value from the invention
title, from world knowledge, or from a look-alike section. Returning "" is
always correct when the form does not state the value.
""".strip()


# ── 1. load ────────────────────────────────────────────────────────

def load_docs() -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(MD_DIR, "*.md"))):
        with open(path, encoding="utf-8") as f:
            stem = os.path.splitext(os.path.basename(path))[0]
            row = {"stem": stem, "path": path, "text": f.read()}
            pdf = os.path.join(PDF_DIR, f"{stem}.pdf")
            if USE_PDF:
                if not os.path.exists(pdf):
                    raise FileNotFoundError(
                        f"IDF_USE_PDF=1 but no original PDF for {stem!r} in {PDF_DIR}"
                    )
                row["pdf_path"] = pdf
            rows.append(row)
    return rows


# `pdf_url_key` lives on map/parallel_map only -- `extract` has no such
# parameter, so the verbatim ops stay markdown-only by construction.
def _vision(**kw):
    return {"pdf_url_key": "pdf_path", "skip_on_error": True, **kw} if USE_PDF else kw


def _llm(**kw):
    """Shared per-op LLM settings. temperature=0 makes reruns comparable."""
    return {"litellm_completion_kwargs": LLM_KW, **kw} if LLM_KW else kw


# ── 2. preprocess (deterministic, no LLM) ──────────────────────────

# The consent/assignment legalese is reprinted identically under III-A and
# III-B and carries zero extractable information. Dropping it cuts input
# tokens on every downstream op.
_BOILERPLATE = [
    r"Each undersigned Inventor acknowledges and agrees.*?original signature for all purposes\.?",
    r"I/We hereby fully consent to NUS collecting.*?submitted in this [Ff]orm\.",
    r"I/We, \(an\) inventor\(s\) who is \(are\) not.*?relevant Grant or research agreement\.",
    r"I/We, \(an\) inventor\(s\) who is \(are\).*?as may be updated from time to time\.",
]


def explode_tables(t: str) -> str:
    """Put each table row on its own line.

    `extract` works at LINE granularity, but MinerU emits a whole <table> as a
    single line -- on these forms that is 26-43% of the characters packed into
    8-10 lines, with the longest line running to ~3,900 characters. A form
    section living inside one <td> therefore cannot be cut at its boundary, and
    the extraction picks up the neighbouring question's text.

    Breaking after each </tr> gives one logical row per line. Measured on the
    five forms: longest line 3,976 -> 1,827 chars.
    """
    t = re.sub(r"(<table[^>]*>)", r"\n\1\n", t)
    t = re.sub(r"(</tr>)", r"\1\n", t)
    t = re.sub(r"(</table>)", r"\n\1\n", t)
    return re.sub(r"\n{3,}", "\n\n", t)


# ── section routing: MEASURED, REJECTED ───────────────────────────
#
# Kept for the record because the result is counter-intuitive. Feeding each op
# only its own slice of the form cuts input characters by 41%, and it still
# made the pipeline WORSE: $0.59 -> $1.25 for the same 5 documents, with
# coverage down 54.3% -> 53.1%.
#
# The cause is the thinking model. Isolating one question strips the context
# it uses to locate that question's boundaries, so it deliberates instead.
# Measured on the "Plans" question of 2025-153, same prompt and model:
#
#     slice   (1,452 chars)   1,041 prompt tok    725 completion tok   1,279 chars out
#     section (10,907 chars)  5,571 prompt tok    240 completion tok   1,426 chars out
#
# Output tokens cost ~8x input on Gemini, so 3x the reasoning erases the input
# saving -- and the narrow slice extracted LESS text. Cost on this workload is
# not driven by how much you feed the model but by how hard it has to think;
# `split_sections` is therefore unused. A cheaper non-reasoning model for the
# mechanical ops is the lever that would actually work.
#
# ── original note ────────────────────────────────────────────────
#
# Every op used to receive the whole document, so input tokens scaled with
# op count (12 LLM ops x ~7k tokens). The form is rigidly structured, so the
# slice each op needs can be cut deterministically. Splitting only at the
# Roman-numeral sections saves 22%; splitting Section IV per NUMBERED QUESTION
# is what actually pays, because 9 of the 12 ops live inside Section IV.
#
# Every slice falls back to a wider one when its marker is absent, so a form
# generation this does not recognise degrades to the old behaviour (more
# expensive, same result) instead of extracting from an empty string.

_Q_TITLES = [
    ("q_overview", r"Overview"),
    ("q_novelty", r"Novelty\s*&\s*Unobviousness"),
    ("q_limitations", r"Limitations"),
    ("q_prior_art", r"Prior\s*Art"),
    ("q_commercial_apps", r"Commercial\s*Applications"),
    ("q_commercial_interest", r"Commercial\s*Interest"),
    ("q_plans", r"Plans"),
    ("q_trl", r"Technology\s*Readiness\s*Level"),
    ("q_material", r"Material"),
    ("q_technical", r"Technical\s*Description"),
]
_Q_PAT = re.compile(
    r"(?:^|<td[^>]*>|\n)\s*(\d{1,2})\s*\.\s*("
    + "|".join(p for _, p in _Q_TITLES) + r")\b",
    re.I | re.M,
)


def split_sections(body: str) -> dict:
    inv = re.search(r"III[-\s]?A\.?\s*NUS\s+INVENTORS", body, re.I)
    det = re.search(r"(?:IV\.\s*)?DETAILS\s+OF\s+THE\s+INVENTION", body, re.I)
    i = inv.start() if inv else None
    d = det.start() if det and (not i or det.start() > i) else None

    out = {
        "sec_head": body[:i] if i else body,
        "sec_inventors": body[i:d] if i and d else (body[i:] if i else body),
        "sec_details": body[d:] if d else body,
    }

    hits = [(m.start(), m.group(2)) for m in _Q_PAT.finditer(out["sec_details"])]
    found = {}
    for n, (pos, title) in enumerate(hits):
        end = hits[n + 1][0] if n + 1 < len(hits) else len(out["sec_details"])
        key = next(k for k, pat in _Q_TITLES if re.fullmatch(pat, title, re.I))
        found.setdefault(key, out["sec_details"][pos:end])
    for key, _ in _Q_TITLES:
        out[key] = found.get(key) or out["sec_details"]
    return out


def strip_boilerplate(doc: dict) -> dict:
    t = doc["text"]
    for pattern in _BOILERPLATE:
        t = re.sub(pattern, "", t, flags=re.S)
    t = explode_tables(t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = t.strip()
    out = {"body": t, "orig_len": len(doc["text"]), "body_len": len(t)}
    if "pdf_path" in doc:
        out["pdf_path"] = doc["pdf_path"]
    return out


def adopt_compressed(doc: dict) -> dict:
    """Swap the compressed text in as `body`, unless it looks lossy.

    A compression that returns less than half the document is more likely to
    have dropped a filled answer than to have found that much boilerplate, so
    fall back to the full text. `_compressed_to` records what happened.
    """
    full = doc["body"]
    comp = doc.get("body_extracted_compress") or ""
    use = comp if len(comp) >= 0.5 * len(full) else full
    out = dict(doc)
    out["body"] = use
    out["_compressed_to"] = round(len(use) / max(len(full), 1), 3)
    return out


# ── 3. extraction ops ──────────────────────────────────────────────

def build(frame: docetl.Frame, *, preprocess: bool = True) -> docetl.Frame:
    """Assemble the op DAG.

    ``preprocess=False`` omits the code_map and expects `body` to already be
    on each row. MOAR serializes the pipeline to YAML to search over it, and a
    `code` op holding a Python function object cannot round-trip through
    yaml.safe_load -- so the deterministic preprocessing is applied up front
    for optimizer runs. Nothing is lost: there is nothing in a fixed regex for
    an optimizer to improve.
    """
    if preprocess:
        frame = frame.code_map(name="preprocess", code=strip_boilerplate)

    if COMPRESS:
        # One verbatim compression pass shared by every downstream op, rather
        # than 12 per-op slices. The per-op slicing experiment failed because
        # a narrow fragment strips the context the model needs to locate
        # boundaries; a single coherent-but-shorter document does not.
        # `extract` keeps this lossless-by-construction: the model emits line
        # ranges and code slices them, so compression cannot reword anything.
        frame = frame.extract(
            name="compress",
            prompt="""Extract every part of this Invention Disclosure Form that
carries filled-in content, and omit the blank template scaffolding.

KEEP: the form's own title/instruction headers that identify its generation
(the "For ILO/TTI use only" box, the submission mailbox); the Title of the
Invention; Section I grants and contractual obligations; Section II public
disclosure; the III-A and III-B inventor tables IN FULL including the
Signature row; and every answer body in Section IV -- Overview, Novelty &
Unobviousness with its table, Limitations, Prior Art, Commercial
Applications, Commercial Interest, Plans, Technology Readiness Level,
Material, and the complete Technical Description with its sub-headings,
figure captions, tables, equations, reference list and any proposed claims.

OMIT: the multi-paragraph consent and assignment legalese, and the printed
menu of ~20 numbered funding agencies (keep only the lines showing which
agency was actually selected).

When in doubt, KEEP the text. Omitting a filled answer is far worse than
keeping some boilerplate.""",
            document_keys=["body"],
            skip_on_error=True,
            **_llm(),
            bypass_cache=BYPASS_EXTRACT,
        )
        frame = frame.code_map(name="use_compressed", code=adopt_compressed)

    # --- 3a. form identity / metadata -------------------------------
    frame = frame.map(
        name="meta",
        **_vision(), **_llm(),
        prompt=GROUNDING
        + """

You are reading a National University of Singapore INVENTION DISCLOSURE FORM
converted to markdown. Extract the form's identity and metadata.

{{ input.body }}

Field notes:
- case_reference: ONLY from the "For ILO use only. ID:" / "For TTI use only. ID:"
  box. This is office-filled and almost always BLANK on the applicant's copy ->
  "" . Do NOT substitute an ID from the filename.
- managing_office: "ILO" on v3-v7 forms, "TTI" on v8/2025 forms. Read from the
  "For ... use only" box and body references.
- submission_email: the mailbox the instructions direct inventors to --
  "ilobox3@nus.edu.sg" (v3-v7) or "Patent-Admin@nus.edu.sg" (v8).
- form_version: "v3-era/ILO" (questions 16-25, single contribution row, no
  Chinese-name field), "v5" or "v7" (Chinese-name field + 15a/15b split,
  questions 17-26, ILO), "v8/TTI" (TTI office, Patent-Admin email).
- invention_type: "general", "software", or "reagent" -- which form THIS is,
  not which ones the instructions mention. Default "general".
- is_signed: "yes" ONLY if the inventor table's "Signature" row actually holds
  a signature -- an <img> placeholder, a scrawled initial, or a typed name in
  the signature cell. A blank signature row -> "no". Do NOT infer signing from
  inventor names being present, from the "/pdf/ electronic signature"
  boilerplate, or from the filename.
- date_of_submission: the form has NO submission-date field. Only fill if a
  date is actually written on the form; otherwise "".
- trl_raw: the Technology Readiness Level answer EXACTLY as written
  ("TRL6", "4-5", "TRL 6-7", "In between Level 4 and Level 5", or prose with
  no number). Do not normalise -- that happens downstream.
- document_is_a_form: "no" if this document is not actually an IDF at all
  (e.g. a supplementary technical write-up with no form sections).""",
        output={
            "schema": {
                "case_reference": "string",
                "managing_office": "string",
                "submission_email": "string",
                "date_of_submission": "string",
                "title": "string",
                "form_version": "string",
                "invention_type": "string",
                "is_signed": "enum[yes, no, unknown]",
                "trl_raw": "string",
                "document_is_a_form": "enum[yes, no]",
            }
        },
    )

    # --- 3b. inventors (transposed table) ---------------------------
    frame = frame.map(
        name="inventors",
        **_vision(), **_llm(),
        prompt=GROUNDING
        + """

Extract every inventor from the "III-A. NUS INVENTORS" and "III-B. NON-NUS
INVENTORS" tables of this disclosure form.

{{ input.body }}

CRITICAL -- these tables are TRANSPOSED. Inventors are COLUMNS ("Inventor 1"
.. "Inventor 5") and attributes are ROWS (Family Name, Given Name, Full Name in
Chinese Characters, Citizenship, Staff or Student No. / Organization, Faculty &
Department, Contact Phone, Email, Mailing Address, 15a Percentage Contribution,
15b Aspect of Contribution, Signature). Read DOWN each column to assemble ONE
inventor record.

Rules:
- is_nus = "yes" for people in III-A, "no" for people in III-B.
- SKIP inventor columns that are entirely blank -- do not emit empty
  "Inventor 4"/"Inventor 5" records.
- Keep an inventor even if most cells are blank; put "" in the blank cells
  rather than dropping the person.
- Some forms duplicate the SAME people into both III-A and III-B. Dedupe to one
  record each (keep the III-A version).
- Quirk: the "Staff or Student No." cell sometimes holds a ROLE word
  ("Student", "Research Scientist") instead of a number. Put it in `role` and
  leave `staff_student_no` empty.
- Quirk: "Citizenship" sometimes holds a residency status ("Permanent Resident
  (PR)", "Singapore Citizen (SC)") rather than a nationality. Keep verbatim.
- contribution_pct: verbatim as written ("35", "80%").
- signature_present: "yes" if THIS inventor's signature cell holds an <img>
  placeholder, initials, or a typed name; "no" if that cell is blank.
- If the document has no inventor table at all, return [].""",
        output={
            "schema": {
                "inventors_flat": (
                    "list[{family_name: string, given_name: string, "
                    "name_cn: string, citizenship: string, "
                    "staff_student_no: string, organization: string, "
                    "department: string, role: string, phone: string, "
                    "email: string, address: string, contribution_pct: string, "
                    "contribution_aspect: string, is_nus: string, "
                    "signature_present: string}]"
                )
            }
        },
    )

    # --- 3c. grants, materials, disclosure, commercial interest -----
    frame = frame.map(
        name="obligations",
        **_vision(), **_llm(),
        prompt=GROUNDING
        + """

Extract Section I (GRANTS AND CONTRACTUAL OBLIGATIONS), Section II (PUBLIC
DISCLOSURE), the Material table and the Commercial Interest table.

{{ input.body }}

funding_sources -- from items 2, 2.1, 2.2. The form prints a MENU of ~20-22
numbered agencies (A*STAR, MOE Tier 1/2/3, NRF programmes, NUSRI-*, SHARE,
NUS-Tianjin Joint School, "Other Agency"). Capture ONLY agencies actually
SELECTED or filled in, NEVER the menu itself. Selection shows either as the
agency copied into "a. Funding source (from above)" or as a tick word like
"YES" beside a menu item. An out-of-menu agency typed into "a.1 If Other
Agency, please specify" goes in `agency_other` (real examples: "Wellcome Leap",
"National Robotics Programme Office", "NRF-NSFC Joint Research Grant Call").
Grant numbers look like "R221000023112", "R-712000064-272", "USS-IF-2018-1".
SKIP entries that are only "Write here..." / "NA" / dots. [] if none declared.

materials -- rows of the "Material" table (Q24 v3-era / Q25 v5+), columns
Material / Source / Material Transfer Agreement. A row that is only "NA" or
blank is not a real material -- skip it. [] if the table is empty.

commercial_interest -- rows of the "Commercial Interest" table (Q21 v3-era /
Q22 v5+), columns Company / Contact Person / Email. [] if empty or "NA".

research_agreement -- item 3 "If the invention is the result of a signed
research agreement": a. Title, b. ILO/TTI Reference Number, c. Collaborator(s)'
organisation(s). All "" if the block shows only "NA"/"Write here..."/blanks.

contractual_obligations -- other REAL, FILLED commitments stated in Section I
(MTAs, options/licences already granted, consulting agreements, third-party
rights). Do NOT restate the templated section legalese. "" if none.

public_disclosure -- the two-row table, Q4 and Q5.
- has_been_disclosed: "yes" ONLY if Q4 describes a REAL PAST disclosure with
  when/where. "No"/"NA"/blank -> "no". A future plan stated under Q5 does NOT
  make this yes. Q4 blank or unclear -> "unknown".
- disclosure_detail: Q4 free text if a real disclosure is described, else "".
- planned_disclosure: Q5 free text (e.g. "Feb 2019", "Journal paper
  publication this year"), else "".""",
        output={
            "schema": {
                "funding_sources": (
                    "list[{agency: string, agency_other: string, "
                    "grant_no: string, project_title: string}]"
                ),
                "materials": "list[{name: string, source: string, mta_reference: string}]",
                "commercial_interest": (
                    "list[{company: string, contact: string, email: string}]"
                ),
                "ra_title": "string",
                "ra_internal_reference": "string",
                "ra_collaborator_orgs": "string",
                "contractual_obligations": "string",
                "pd_has_been_disclosed": "enum[yes, no, unknown]",
                "pd_disclosure_detail": "string",
                "pd_planned_disclosure": "string",
            }
        },
    )

    # --- 3d. references / claims / approvals ------------------------
    frame = frame.map(
        name="refs",
        **_llm(),
        prompt=GROUNDING
        + """

Extract citations, draft claims and study approvals from this disclosure form.

{{ input.body }}

cited_prior_art_references -- patent numbers and named products/papers that the
"Prior Art" section (or Novelty / Technical Description) cites as EXISTING
technology. One verbatim string each, identifier plus its title if given.
Examples of the shape: "EP2559664A1", "US20140131285A1", "WO 2010065957 A2",
"CN112624455A", "Abbott FreeStyle Libre", "Dexcom G7". [] if none are named.
NEVER add a plausible-looking patent number from world knowledge -- only
strings physically written in this form.

literature_references -- the invention's OWN numbered bibliography, usually a
"References" list at the end of the Technical Description, e.g.
"[1] C.Y. Tang, ... (2018).". One verbatim string per reference. This is
DISTINCT from cited_prior_art_references. [] if none.

proposed_claims -- ONLY if the Technical Description contains an explicit
"Proposed Claims:" numbered list. The printed claim number is the ground truth
for where one claim ends and the next begins. Capture each claim's full
verbatim text including "comprising:" sub-parts and (a)/(b) elements. This
block is ABSENT on the vast majority of forms -> []. Do NOT synthesise claims
from the narrative or from the Novelty table.

study_approvals -- regulatory / ethics / clinical-study approvals stated in the
narrative (usually under "Plans" or the TRL answer), e.g. "NHG DSRB number
2024-4624 ... with Tan Tock Seng Hospital ... 20 diabetic patients". [] if the
form describes no such study. Do NOT infer a study from a bare mention of
"patients" or "clinical".

field_of_invention -- technical field/keywords ONLY if the Overview states them
explicitly (e.g. "the field of advanced high-performance steels and additive
manufacturing"). [] if not explicitly stated.""",
        output={
            "schema": {
                "cited_prior_art_references": "list[string]",
                "literature_references": "list[string]",
                "proposed_claims": "list[{number: integer, text: string}]",
                "study_approvals": (
                    "list[{approval_reference: string, board_or_body: string, "
                    "collaborating_institution: string, details: string}]"
                ),
                "field_of_invention": "list[string]",
            }
        },
    )

    # --- 3e. narrative fields -----------------------------------------
    # Short/structured answers go through map. The three long ones that the
    # Gemini baseline truncated go through `extract`, which emits line ranges
    # rather than text -- so the stored value is a slice of the source and
    # cannot be shortened or reworded by the model.
    frame = frame.map(
        name="narrative",
        **_llm(),
        prompt=GROUNDING
        + """

Extract the remaining narrative answers from Section IV, DETAILS OF THE
INVENTION. All four of these are normally ABSENT on this form -- the form has
no dedicated box for any of them -- so "" is the expected answer unless the
text really does state them separately.

{{ input.body }}

- background: the problem/context motivating the invention, ONLY if stated
  separately from the Overview. On this form it is usually embedded inside the
  Overview -> "" .
- drawings_description: only if a DEDICATED drawings/figures section exists. On
  this form figures are inline in the Technical Description, so expect "".
- modifications: other modes of practice / modifications, only if a section
  states them. No dedicated box on this form -> usually "".
- industrial_applications: specific industrial application areas ONLY if listed
  separately from Overview / Commercial Applications -> usually "".""",
        output={
            "schema": {
                "background": "string",
                "drawings_description": "string",
                "modifications": "string",
                "industrial_applications": "string",
            }
        },
    )

    # Verbatim long-form fields. One extract op per field; each writes
    # body_extracted_<name>.
    frame = frame.extract(
        name="summary",
        prompt="""Extract the complete "Overview" answer -- the question that asks
for a summary or general description of the invention including its field of
application (numbered 16 on v3-era forms, 17 on v5+ forms).

Extract the full answer body ONLY: start at the first line of the inventor's
prose answer and end at the last line before the next numbered question
("Novelty & Unobviousness"). Do NOT include the question text itself, and do
not include the following question. If the answer is blank or only
"Write here...", extract nothing.""",
        document_keys=["body"],
        skip_on_error=True,
        **_llm(),
        bypass_cache=BYPASS_EXTRACT,
    )

    frame = frame.extract(
        name="prior_art",
        prompt="""Extract the complete "Prior Art" answer -- the question asking
for other existing technologies or literature that closely resemble this
invention and how this invention differentiates from them (numbered 19 on
v3-era forms, 20 on v5+ forms).

Extract the full answer body ONLY, including any lettered/numbered sub-items
a), b), c) and the patent numbers or papers cited inside it. Stop before the
next numbered question. NOTE: the oldest v3-era forms have no Prior Art
question at all -- if there is none, extract nothing.""",
        document_keys=["body"],
        skip_on_error=True,
        **_llm(),
        bypass_cache=BYPASS_EXTRACT,
    )

    frame = frame.extract(
        name="novelty",
        prompt="""Extract the complete "Novelty & Unobviousness" answer -- the
question listing the features that make this invention a substantial
improvement over existing technology, with the unique benefits or advantages
each feature provides (numbered 17 on v3-era forms, 18 on v5+ forms).

This answer is a two-column table: Feature | Benefit/Advantage. Extract the
ENTIRE table including its header row and every feature/benefit pair, so the
pairing is preserved. Start at the table and stop before the next numbered
question ("Limitations"). Do not include the question text itself. If the
table is empty or only placeholder text, extract nothing.""",
        document_keys=["body"],
        skip_on_error=True,
        **_llm(),
        bypass_cache=BYPASS_EXTRACT,
    )

    frame = frame.extract(
        name="limitations",
        prompt="""Extract the complete "Limitations" answer -- the question asking
for the limitations of this invention in terms of scalability, speed, power
consumption, efficiency, use of exotic compounds and so on (numbered 18 on
v3-era forms, 19 on v5+ forms).

Extract the full answer body only, stopping before the next numbered question
("Prior Art"). Do not include the question text itself. If the answer is
blank, "NA", or only placeholder text, extract nothing.""",
        document_keys=["body"],
        skip_on_error=True,
        **_llm(),
        bypass_cache=BYPASS_EXTRACT,
    )

    frame = frame.extract(
        name="commercial",
        prompt="""Extract the complete "Commercial Applications" answer -- the
question asking what critical commercial problems this invention solves
(numbered 20 on v3-era forms, 21 on v5+ forms).

This is DISTINCT from "Commercial Interest", which is a table of interested
companies -- do not extract that one. Extract the full answer body only,
stopping before the next numbered question. Do not include the question text
itself. If blank or "NA", extract nothing.""",
        document_keys=["body"],
        skip_on_error=True,
        **_llm(),
        bypass_cache=BYPASS_EXTRACT,
    )

    frame = frame.extract(
        name="plans",
        prompt="""Extract the complete "Plans" answer -- the question asking about
plans regarding this invention: who is working on it, what further research is
planned, what grants are being applied for, the commercialisation path
(numbered 22 on v3-era forms, 23 on v5+ forms).

Extract the full answer body only, stopping before the next numbered question
("Technology Readiness Level"). Do not include the question text itself. If
blank or "NA", extract nothing.""",
        document_keys=["body"],
        skip_on_error=True,
        **_llm(),
        bypass_cache=BYPASS_EXTRACT,
    )

    frame = frame.extract(
        name="technical",
        prompt="""Extract the complete "Technical Description" answer -- the final
and longest question, asking for full technical details roughly corresponding
to the Methodology / Results / Analysis / Conclusions of a paper (numbered 25
on v3-era forms, 26 on v5+ forms).

Extract the ENTIRE answer body from its first line to its last, including: all
sub-headings, inline figure image placeholders and their "Figure N." captions,
tables, LaTeX equations, quantitative results, any numbered reference list at
the end, and any "Proposed Claims:" block. Do not stop early -- this answer
often runs for many pages. Do not include the question text itself.""",
        document_keys=["body"],
        skip_on_error=True,
        **_llm(),
        bypass_cache=BYPASS_EXTRACT,
    )

    return frame


# ── 4. assemble into the canonical schema ──────────────────────────

_NULLISH = {"", "na", "n.a.", "n/a", "none", "nil", "write here", "write here…",
            "write here...", "unknown", "not applicable", "-", "--", "..."}


# An `extract` on an unfilled section returns the form's own placeholder text
# ("Write here...Please include again the Title of the Invention..."). That is
# an empty answer, not content.
_PLACEHOLDER = re.compile(
    r"write here|please include again the title|should be distinguishable in black",
    re.I,
)


# `extract` cuts at LINE boundaries, so a range that starts on the line holding
# the question header carries that header into the answer. Exploding the tables
# took this from 15% of extractions to 9%; the rest is deterministic text, so
# strip it in code rather than asking the prompt again.
_QUESTION_LEAD = re.compile(
    r"""^\s*
    (?:\#+\s*)?                                   # stray markdown heading
    (?:DETAILS\ OF\ THE\ INVENTION\s*)?           # section banner
    (?:</?t[dr][^>]*>|</?table[^>]*>|\s)*     # table scaffolding
    (?:\d{1,2}\.\s*)?                             # question number
    (?:Overview|Novelty\s*&\s*Unobviousness|Novelty|Limitations|
       Prior\s*Art|Commercial\s*Applications|Plans|Technical\s*Description|
       Material|Technology\s*Readiness\s*Level)
    \s*[-–—]\s*[^:]{0,400}?(?::|etc\.)\s*             # template prompt, to ":" or "etc."
    """,
    re.X | re.I,
)


def _trim_lead(s: str) -> str:
    """Drop a leading question header that the line range swept in."""
    for _ in range(2):  # banner + question can both be present
        new = _QUESTION_LEAD.sub("", s, count=1).lstrip()
        if new == s:
            break
        s = new
    # A line range can also start PART-WAY through the question text, leaving a
    # lowercase fragment ending in the question's colon ("commercial problems
    # this invention solves: ..."). Requiring a lowercase start and no sentence
    # punctuation before the colon keeps real answers (which start with a
    # capital, or carry no colon at all) untouched.
    s = re.sub(r"^[a-z][^.!?\n]{0,150}?:\s+", "", s)
    # leftover table scaffolding at either end
    s = re.sub(r"^(?:</?t[dr][^>]*>|</?table[^>]*>|\s)+", "", s)
    return re.sub(r"(?:</?t[dr][^>]*>|</?table[^>]*>|\s)+$", "", s).strip()


def _n(v):
    """Sentinel -> None. Lists/dicts pass through, empty ones become []/None."""
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        if s.lower().strip(" .\u2026") in _NULLISH:
            return None
        # Mostly-placeholder text (the real answer was never filled in).
        if len(s) < 400 and _PLACEHOLDER.search(s):
            return None
        return s
    return v


def _tri(v):
    """enum[yes,no,unknown] -> True/False/None."""
    s = (v or "").strip().lower()
    return True if s == "yes" else False if s == "no" else None


def _trl(raw):
    """'TRL3-4' / 'TRL 6-7' / '4-5' / 'In between Level 4 and Level 5' -> int.

    The canonical schema types trl as `int | None` and specifies the lower
    bound of a range. The Gemini baseline returned the raw string 'TRL3-4' /
    'TRL 4' on 2 of 5 documents, violating the type; doing it in code makes
    that class of error impossible.
    """
    s = (raw or "").strip()
    if not s:
        return None
    # A standalone 1-9 digit: "TRL3-4" -> [3, 4], "In between Level 4 and
    # Level 5" -> [4, 5]. \b would miss the 3 in "TRL3" (glued to letters).
    nums = [int(x) for x in re.findall(r"(?<!\d)([1-9])(?!\d)", s)]
    return min(nums) if nums else None


def _pct(v):
    """'80%' / '35' / '' -> float | None. Canonical contribution_share is a number."""
    s = (v or "").strip().rstrip("%").strip()
    try:
        return float(s)
    except ValueError:
        return None


def assemble(doc: dict) -> dict:
    inventors = []
    for raw in doc.get("inventors_flat") or []:
        fam, giv = _n(raw.get("family_name")), _n(raw.get("given_name"))
        full = " ".join(p for p in (giv, fam) if p) or None
        inventors.append(
            {
                "name": {
                    "family_name": fam,
                    "given_name": giv,
                    "name_cn": _n(raw.get("name_cn")),
                    "full_name_raw": full,
                },
                "citizenship": _n(raw.get("citizenship")),
                "residence": None,
                "affiliation": _n(raw.get("organization"))
                or ("NUS" if _tri(raw.get("is_nus")) else None),
                "is_nus": _tri(raw.get("is_nus")),
                "staff_student_no": _n(raw.get("staff_student_no")),
                "department": _n(raw.get("department")),
                "role": _n(raw.get("role")),
                "address": _n(raw.get("address")),
                "email": _n(raw.get("email")),
                "contribution_share": _pct(raw.get("contribution_pct")),
                "contribution_aspect": _n(raw.get("contribution_aspect")),
                "funding_admin_contact": None,
                "signature_present": _tri(raw.get("signature_present")),
            }
        )

    def rows(key, mapper):
        out = [mapper(r) for r in (doc.get(key) or [])]
        return [r for r in out if any(v is not None for v in r.values())]

    ra = {
        "title": _n(doc.get("ra_title")),
        "internal_reference": _n(doc.get("ra_internal_reference")),
        "collaborator_orgs": _n(doc.get("ra_collaborator_orgs")),
        "details": None,
    }

    return {
        "case_reference": _n(doc.get("case_reference")),
        "managing_office": _n(doc.get("managing_office")),
        "submission_email": _n(doc.get("submission_email")),
        "date_of_submission": _n(doc.get("date_of_submission")),
        "title": _n(doc.get("title")),
        "field_of_invention": [x for x in (doc.get("field_of_invention") or []) if _n(x)],
        "summary": _n(_trim_lead(doc.get("body_extracted_summary") or "")),
        "background": _n(doc.get("background")),
        "detailed_description": _n(_trim_lead(doc.get("body_extracted_technical") or "")),
        "novelty_and_advantages": _n(_trim_lead(doc.get("body_extracted_novelty") or "")),
        "problem_solved": _n(_trim_lead(doc.get("body_extracted_commercial") or "")),
        "commercial_applications": _n(_trim_lead(doc.get("body_extracted_commercial") or "")),
        "industrial_applications": _n(doc.get("industrial_applications")),
        "limitations": _n(_trim_lead(doc.get("body_extracted_limitations") or "")),
        "prior_art": _n(_trim_lead(doc.get("body_extracted_prior_art") or "")),
        "cited_prior_art_references": [
            x for x in (doc.get("cited_prior_art_references") or []) if _n(x)
        ],
        "modifications": _n(doc.get("modifications")),
        "plans": _n(_trim_lead(doc.get("body_extracted_plans") or "")),
        "drawings_description": _n(doc.get("drawings_description")),
        "literature_references": [
            x for x in (doc.get("literature_references") or []) if _n(x)
        ],
        "proposed_claims": rows(
            "proposed_claims",
            lambda r: {"number": r.get("number"), "text": _n(r.get("text"))},
        ),
        "inventors": inventors or None,
        "funding_sources": rows(
            "funding_sources",
            lambda r: {
                "agency": _n(r.get("agency")),
                "agency_other": _n(r.get("agency_other")),
                "grant_no": _n(r.get("grant_no")),
                "wbs_number": None,
                "project_title": _n(r.get("project_title")),
                "program_manager": None,
                "collaborators": None,
                "formal_agreement_signed": None,
                "used_university_resources": None,
            },
        ),
        "materials": rows(
            "materials",
            lambda r: {
                "name": _n(r.get("name")),
                "source": _n(r.get("source")),
                "mta_reference": _n(r.get("mta_reference")),
            },
        ),
        "research_agreement": ra if any(ra.values()) else None,
        "contractual_obligations": _n(doc.get("contractual_obligations")),
        "commercial_interest": rows(
            "commercial_interest",
            lambda r: {
                "company": _n(r.get("company")),
                "contact": _n(r.get("contact")),
                "email": _n(r.get("email")),
                "notes": None,
            },
        ),
        "date_of_conception": None,
        "public_disclosure": {
            "has_been_disclosed": _tri(doc.get("pd_has_been_disclosed")),
            "disclosure_detail": _n(doc.get("pd_disclosure_detail")),
            "planned_disclosure": _n(doc.get("pd_planned_disclosure")),
        },
        "trl": _trl(doc.get("trl_raw")),
        "study_approvals": rows(
            "study_approvals",
            lambda r: {
                "approval_reference": _n(r.get("approval_reference")),
                "board_or_body": _n(r.get("board_or_body")),
                "collaborating_institution": _n(r.get("collaborating_institution")),
                "details": _n(r.get("details")),
            },
        ),
        "software_details": None,
        "invention_type": _n(doc.get("invention_type")) or "general",
        "form_version": _n(doc.get("form_version")),
        "is_signed": _tri(doc.get("is_signed")),
    }


# ── 5. run ─────────────────────────────────────────────────────────

def main() -> int:
    rows = load_docs()
    if not rows:
        print(f"No markdown found in {MD_DIR}", file=sys.stderr)
        return 1
    print(f"{len(rows)} documents from {os.path.basename(MD_DIR)}")

    frame = build(docetl.from_list(rows))

    if "--explain" in sys.argv:
        print(frame.explain())
        return 0

    t0 = time.time()
    results = frame.collect()
    elapsed = time.time() - t0

    os.makedirs(OUT_DIR, exist_ok=True)
    for doc in results:
        payload = {
            "model": MODEL,
            "doctype": "invention_disclosure",
            "run": RUN,
            "stem": doc["stem"],
            "usage": frame.token_usage,
            "seconds": round(elapsed / len(results), 1),
            "extraction": assemble(doc),
        }
        out = os.path.join(OUT_DIR, f"{doc['stem']}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        print(f"  wrote {os.path.basename(out)}")

    print(f"\n{len(results)} docs in {elapsed:.1f}s   cost ${frame.total_cost:.4f}")
    print(f"tokens: {frame.token_usage}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
