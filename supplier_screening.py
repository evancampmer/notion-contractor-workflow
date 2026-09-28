import os
import io
import re
import json
import logging
import tempfile
import threading
import time
from datetime import datetime

import numpy as np
import requests
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from flask import Flask, request
from openai import OpenAI
from docx import Document
from docx.shared import Pt, RGBColor
from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession

# =========================
# CONFIG + LOGGING
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()

# =========================
# CLIENTS
# =========================

slack_app = App(
    token=os.getenv("SLACK_BOT_TOKEN"),
    signing_secret=os.getenv("SLACK_SIGNING_SECRET")
)

openrouter = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY_ALINA")
)

MODEL = "anthropic/claude-sonnet-4-5"

SCOPES = ["https://www.googleapis.com/auth/drive"]
service_account_info = json.loads(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"))
credentials = service_account.Credentials.from_service_account_info(
    service_account_info, scopes=SCOPES
)
drive_session = AuthorizedSession(credentials)

SLACK_CHANNEL_ID = "C0C2QRTGAUV"
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
OUTPUT_FOLDER = "output_docs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

TOP_K = 30          # profiles to retrieve per pass via vector search before scoring
PROFILE_CHARS = 3000  # max chars per profile sent to the scoring LLM

# =========================
# PROMPTS
# =========================

CRITERIA_SYSTEM_PROMPT = """You are a business analyst at a consulting and staffing firm. Convert a project description into a focused screening rubric.

Output exactly two things with NO extra text:

1. A 1-2 sentence project summary.

2. A pipe-delimited criteria table:
| Category | Criterion | Importance | Evidence standard |

Rules:
- Category is "Technical" or "Contextual"
- 4-7 criteria total — only the most critical ones
- Criterion names: 2-5 words, plain English, no abbreviations or underscores (e.g. "510(k) submission experience", "FDA regulatory strategy", "submission program leadership")
- Importance: 1-5 (5 = must-have)
- Evidence standard: a SHORT phrase describing what to look for, plain English (e.g. "Explicit 510(k) filing work", "Direct FDA device regulatory experience"). NOT a list of quoted keywords.
- No commentary, no headers, no explanation outside these two items"""

SCORING_PROMPT = """You are a supplier screening agent at a consulting and staffing firm.

Score EVERY supplier profile against the criteria. Return ONLY a pipe-delimited table — no preamble, no commentary, no extra text.

Table columns (use exactly these headers):
| Rank / Name | Total Score | Technical / Contextual Subtotals | Each Criterion Score (0-5) | Strengths Summary | Weakness Summary | Overall Summary on Fit |

Column definitions — follow exactly:
- Rank / Name: rank number + full name, e.g. "1. Jane Smith"
- Total Score: a SINGLE integer 0-100. This is the overall weighted score normalized to 100. Example: "44". Never use a slash or fraction here.
- Technical / Contextual Subtotals: TWO numbers separated by " / ", each independently normalized to 100. Example: "20 / 77". This column only appears AFTER Total Score.
- Each Criterion Score (0-5): list each criterion by its plain-English name followed by its 0-5 score, e.g. "510(k) submission experience: 0; FDA regulatory strategy: 2; Medical device sector: 3". Use plain English names exactly as written in the criteria table — no abbreviations, no underscores.
- Score each criterion 0-5 based ONLY on explicit text evidence in the profile — no assumptions
- Strengths Summary: 1-2 sentences citing specific evidence; enclose 1-2 key phrases in **double asterisks**
- Weakness Summary: 1-2 sentences on specific gaps, plain text
- Overall Summary on Fit: 1-2 sentences; first sentence is a clear plain-English verdict (e.g. "Best available adjacent candidate, but not a verified 510(k) lead.")
- Score ALL suppliers; assign 0 on any criterion with no relevant evidence
- Sort rows by Total Score descending"""

SYNTHESIS_PROMPT = """You are a supplier screening agent summarizing a completed supplier screen.

Write exactly three labeled sections. Use **double asterisks** around key terms inline where indicated. No other text outside these sections.

HEADLINE: [One short noun phrase — the overall finding. Format: "Screening result: [specific finding]". Example: "Screening result: no verified 510(k) filing specialist identified"]

OPENING: [One paragraph. Describe what was screened and the key conclusion. Bold 2-4 key criteria names and the core conclusion phrase using **double asterisks**. Style: "I screened the supplier profiles for explicit evidence of **X**, **Y**, and **Z**. [Finding sentence with **bold conclusion**]."]

RECOMMENDATION: [2 paragraphs separated by a blank line.
Para 1: Overall action recommendation. Bold key candidate names and key roles or caveats with **double asterisks**.
Para 2: What the ideal required profile should explicitly demonstrate. Bold 2-3 key requirements with **double asterisks**.]"""

# =========================
# GOOGLE DRIVE
# =========================

def get_supplier_docs():
    """Download all DOCX files from the Google Drive folder and extract text."""
    supplier_docs = []

    list_resp = drive_session.get(
        "https://www.googleapis.com/drive/v3/files",
        params={
            "q": (
                f"'{GOOGLE_DRIVE_FOLDER_ID}' in parents "
                f"and mimeType='application/vnd.openxmlformats-officedocument.wordprocessingml.document' "
                f"and trashed=false"
            ),
            "fields": "files(id, name)",
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
            "corpora": "allDrives",
        }
    )
    list_resp.raise_for_status()
    files = list_resp.json().get("files", [])
    logging.info(f"Found {len(files)} supplier docs in Drive")

    for file in files:
        try:
            dl_resp = drive_session.get(
                f"https://www.googleapis.com/drive/v3/files/{file['id']}",
                params={"alt": "media", "supportsAllDrives": "true"}
            )
            dl_resp.raise_for_status()

            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
                tmp.write(dl_resp.content)
                tmp_path = tmp.name

            doc = Document(tmp_path)
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())

            supplier_docs.append({
                "name": file["name"].replace(".docx", "").replace("_", " "),
                "text": text
            })

            os.unlink(tmp_path)

        except Exception as e:
            logging.warning(f"Failed to read {file['name']}: {e}")

    return supplier_docs


# =========================
# VECTOR INDEX
# =========================

EMBEDDINGS_FILENAME = "supplier_embeddings.json"
REFRESH_SECRET = os.getenv("REFRESH_SECRET", "")

_supplier_index = None   # {"docs": list, "vectors": np.ndarray}
_index_lock = threading.Lock()
_index_ready = threading.Event()
_embed_model = None
_embed_lock = threading.Lock()


def _get_embed_model():
    """Lazy-load the fastembed model (thread-safe)."""
    global _embed_model
    if _embed_model is None:
        with _embed_lock:
            if _embed_model is None:
                from fastembed import TextEmbedding
                logging.info("Loading fastembed model (BAAI/bge-small-en-v1.5)...")
                _embed_model = TextEmbedding("BAAI/bge-large-en-v1.5")
                logging.info("Embedding model loaded")
    return _embed_model


def _find_drive_file_id(filename):
    """Return the Drive file ID for a file in the supplier folder, or None."""
    resp = drive_session.get(
        "https://www.googleapis.com/drive/v3/files",
        params={
            "q": f"name='{filename}' and '{GOOGLE_DRIVE_FOLDER_ID}' in parents and trashed=false",
            "fields": "files(id)",
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
            "corpora": "allDrives",
        }
    )
    files = resp.json().get("files", [])
    return files[0]["id"] if files else None


def _load_embeddings_from_drive():
    """Load pre-computed embeddings JSON from Drive. Returns index dict or None."""
    try:
        file_id = _find_drive_file_id(EMBEDDINGS_FILENAME)
        if not file_id:
            return None
        dl = drive_session.get(
            f"https://www.googleapis.com/drive/v3/files/{file_id}",
            params={"alt": "media", "supportsAllDrives": "true"}
        )
        dl.raise_for_status()
        data = dl.json()
        profiles = data.get("profiles", [])
        if not profiles:
            return None
        docs = [{"name": p["name"], "text": p["text"]} for p in profiles]
        vectors = np.array([p["vector"] for p in profiles], dtype=np.float32)
        logging.info(f"Loaded {len(docs)} pre-computed embeddings from Drive (version {data.get('version', 'unknown')})")
        return {"docs": docs, "vectors": vectors}
    except Exception as e:
        logging.warning(f"Could not load embeddings from Drive: {e}")
        return None


def _build_index_from_scratch():
    """Compute embeddings from Drive DOCXs. Fallback when no pre-computed JSON exists."""
    docs = get_supplier_docs()
    if not docs:
        return None
    model = _get_embed_model()
    texts = [d["text"][:PROFILE_CHARS] for d in docs]
    logging.info(f"Computing embeddings for {len(docs)} profiles (first-time setup)...")
    vectors = np.array(list(model.embed(texts)), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-10)
    logging.info(f"Embeddings computed for {len(docs)} profiles")
    return {"docs": docs, "vectors": vectors}


def _build_index():
    """Build the in-memory index on startup. Tries Drive JSON first, falls back to computing."""
    global _supplier_index
    try:
        index = _load_embeddings_from_drive()
        if index is None:
            logging.warning("No pre-computed embeddings found — computing from scratch (this takes ~60s)")
            index = _build_index_from_scratch()
        if index:
            with _index_lock:
                _supplier_index = index
            logging.info(f"Vector index ready: {len(index['docs'])} profiles")
        else:
            logging.error("Index build produced no results")
    except Exception as e:
        logging.error(f"Index build failed: {e}", exc_info=True)
    finally:
        _index_ready.set()


def retrieve_top_k(query_text, k=TOP_K):
    """Return the k most semantically similar supplier profiles to the query."""
    model = _get_embed_model()
    q_vec = np.array(list(model.embed([query_text]))[0], dtype=np.float32)
    q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-10)

    with _index_lock:
        index = _supplier_index

    scores = index["vectors"] @ q_vec
    top_indices = np.argsort(scores)[::-1][:k]
    top_docs = [index["docs"][i] for i in top_indices]
    logging.info(f"Vector search: top {len(top_docs)} profiles (scores {scores[top_indices[0]]:.3f} — {scores[top_indices[-1]]:.3f})")
    return top_docs


def retrieve_two_pass(message_text, criteria_text, k=TOP_K):
    """Two-pass retrieval: narrow (message+criteria) + broad (criterion names only).
    Catches both exact-match profiles and adjacent-skill profiles like Milan Babic.
    Returns deduplicated merged list, pass1 results first."""

    # Pass 1 — narrow: full message + criteria (exact terminology match)
    query1 = f"{message_text}\n\n{criteria_text}"
    pass1 = retrieve_top_k(query1, k=k)

    # Pass 2 — broad: just the criterion names, no evidence standards
    # Extracts e.g. "510(k) submission experience; FDA regulatory strategy; ..."
    criterion_names = []
    for line in criteria_text.split("\n"):
        if "|" not in line:
            continue
        if re.match(r"^\|[-|:\s]+\|$", line.strip()):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) >= 2 and parts[1].lower() not in ("criterion", ""):
            criterion_names.append(parts[1])
    broad_query = ("adjacent experience and background in: " + "; ".join(criterion_names)
                   if criterion_names else message_text)
    pass2 = retrieve_top_k(broad_query, k=k)

    # Merge, deduplicate by name (pass1 order preserved first)
    seen = set()
    merged = []
    for doc in pass1 + pass2:
        if doc["name"] not in seen:
            seen.add(doc["name"])
            merged.append(doc)

    logging.info(f"Two-pass retrieval: pass1={len(pass1)}, pass2={len(pass2)}, merged={len(merged)} unique profiles")
    return merged


# Start building the index in the background immediately on startup
threading.Thread(target=_build_index, daemon=True).start()


# =========================
# TEXT HELPERS
# =========================

def strip_mention(text):
    """Remove Slack @mention tags like <@U0C407F3AKT> from the message."""
    return re.sub(r"<@[A-Z0-9]+>\s*", "", text).strip()


def _strip_md(text):
    """Strip markdown bold/italic markers."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    return text.strip()


def _parse_md_table(text):
    """Extract rows from a markdown pipe-delimited table."""
    rows = []
    for line in text.split("\n"):
        line = line.strip()
        if not line.startswith("|"):
            continue
        if re.match(r"^\|[-|:\s]+\|$", line):
            continue
        cells = [_strip_md(c.strip()) for c in line.strip("|").split("|")]
        if any(cells):
            rows.append(cells)
    return rows


def _extract_score(row):
    try:
        return int(re.search(r"\d+", row[1]).group())
    except Exception:
        return 0


def _is_header_row(row):
    return any(h in row[0].lower() for h in ["name", "rank", "supplier"])


# =========================
# AI CALLS
# =========================

def create_supplier_criteria(message_text):
    logging.info("Creating supplier criteria...")
    response = openrouter.chat.completions.create(
        model=MODEL,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": CRITERIA_SYSTEM_PROMPT},
            {"role": "user", "content": f"Create the screening criteria for this project brief:\n\n{message_text}"}
        ]
    )
    return response.choices[0].message.content


def score_suppliers(criteria, supplier_docs):
    """Score the pre-filtered supplier docs in a single LLM call."""
    supplier_text = "\n\n---\n\n".join(
        f"SUPPLIER: {doc['name']}\n\n{doc['text'][:PROFILE_CHARS]}"
        for doc in supplier_docs
    )
    logging.info(f"Scoring {len(supplier_docs)} profiles in one call...")
    response = openrouter.chat.completions.create(
        model=MODEL,
        max_tokens=6000,
        messages=[
            {"role": "system", "content": SCORING_PROMPT},
            {"role": "user", "content": f"CRITERIA:\n{criteria}\n\nSUPPLIER PROFILES:\n{supplier_text}"}
        ]
    )
    return response.choices[0].message.content


def synthesize(criteria, top_rows):
    """Generate HEADLINE/OPENING/RECOMMENDATION from scored candidates."""
    candidate_summary = "\n".join(
        f"{row[0]}: Total={row[1]}, Subtotals={row[2] if len(row) > 2 else 'N/A'}, "
        f"Strengths: {row[4] if len(row) > 4 else 'N/A'}, "
        f"Gaps: {row[5] if len(row) > 5 else 'N/A'}"
        for row in top_rows
    )
    response = openrouter.chat.completions.create(
        model=MODEL,
        max_tokens=2000,
        messages=[
            {"role": "system", "content": SYNTHESIS_PROMPT},
            {"role": "user", "content": f"CRITERIA:\n{criteria}\n\nSCORED CANDIDATES (ranked):\n{candidate_summary}"}
        ]
    )
    return response.choices[0].message.content


def rank_suppliers(criteria, supplier_docs):
    """Full pipeline: score docs, sort, synthesize, return structured result."""
    raw = score_suppliers(criteria, supplier_docs)
    rows = _parse_md_table(raw)

    header_row = None
    data_rows = []
    for row in rows:
        if _is_header_row(row):
            if header_row is None:
                header_row = row
        else:
            data_rows.append(row)

    data_rows.sort(key=_extract_score, reverse=True)
    top_rows = data_rows[:15]
    logging.info(f"Scored {len(data_rows)} candidates, kept top {len(top_rows)}")

    # Build references: clean display names for candidates with score > 0
    # Fallback to top 3 if none scored above 0
    scored = [r for r in top_rows if _extract_score(r) > 0]
    ref_rows = scored[:5] if scored else top_rows[:3]
    references = []
    for row in ref_rows:
        name_raw = re.sub(r"^\d+\.\s*", "", row[0]).strip()
        # Clean up filename artifacts (e.g. "coming soon", underscores)
        name_clean = re.sub(r"\s*(coming soon|tbd|pending)\s*", "", name_raw, flags=re.IGNORECASE).strip()
        references.append(name_clean)

    synthesis = synthesize(criteria, top_rows)
    logging.info("Synthesis complete")

    return {
        "synthesis": synthesis,
        "header_row": header_row,
        "ranked_rows": top_rows,
        "references": references
    }


# =========================
# DOCX GENERATION
# =========================

def _add_rich_paragraph(doc, text, style="Normal"):
    """Add a paragraph with **bold** markers parsed into bold runs."""
    para = doc.add_paragraph(style=style)
    parts = re.split(r"(\*\*[^*]+\*\*)", text)
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            run = para.add_run(part[2:-2])
            run.bold = True
        elif part:
            para.add_run(part)
    return para


def _set_cell_rich(cell, text):
    """Set cell content with **bold** markers parsed into bold runs."""
    cell.text = ""
    para = cell.paragraphs[0]
    parts = re.split(r"(\*\*[^*]+\*\*)", str(text))
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            run = para.add_run(part[2:-2])
            run.bold = True
        elif part:
            para.add_run(part)


def _bold_criterion_numbers(cell, text):
    """Set criterion scores cell with score numbers bolded."""
    cell.text = ""
    para = cell.paragraphs[0]
    parts = re.split(r"(\b\d+\b)", str(text))
    for part in parts:
        run = para.add_run(part)
        if re.match(r"^\d+$", part):
            run.bold = True


def _bold_first_sentence(cell, text):
    """Set cell content with the first sentence bolded."""
    cell.text = ""
    para = cell.paragraphs[0]
    m = re.match(r"^([^.!?]+[.!?])\s*(.*)", str(text), re.DOTALL)
    if m:
        run = para.add_run(m.group(1))
        run.bold = True
        if m.group(2):
            para.add_run(" " + m.group(2))
    else:
        run = para.add_run(str(text))
        run.bold = True


def _add_criteria_table(doc, rows):
    """Add the 4-column criteria table (all plain text)."""
    if not rows:
        return
    num_cols = 4
    table = doc.add_table(rows=len(rows), cols=num_cols)
    table.style = "Table Grid"
    for r_idx, row in enumerate(rows):
        row = list(row) + [""] * (num_cols - len(row))
        row = row[:num_cols]
        for c_idx, cell_text in enumerate(row):
            table.cell(r_idx, c_idx).text = cell_text
    doc.add_paragraph()


def _add_ranking_table(doc, header_row, data_rows):
    """Add the 7-column ranking table with Cassidy-style per-column bold formatting."""
    if not data_rows:
        return
    num_cols = 7
    all_rows = ([header_row] if header_row else []) + data_rows
    table = doc.add_table(rows=len(all_rows), cols=num_cols)
    table.style = "Table Grid"

    for r_idx, row in enumerate(all_rows):
        row = list(row) + [""] * (num_cols - len(row))
        row = row[:num_cols]
        is_header = (r_idx == 0 and header_row is not None)

        for c_idx, cell_text in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            if is_header:
                cell.text = cell_text  # header: plain text
            else:
                if c_idx in (0, 1, 2):
                    # Rank/Name, Total Score, Subtotals: entire content bold
                    cell.text = ""
                    run = cell.paragraphs[0].add_run(cell_text)
                    run.bold = True
                elif c_idx == 3:
                    # Criterion scores: bold just the numbers
                    _bold_criterion_numbers(cell, cell_text)
                elif c_idx == 4:
                    # Strengths: **markers** from LLM
                    _set_cell_rich(cell, cell_text)
                elif c_idx == 5:
                    # Weakness: plain
                    cell.text = cell_text
                elif c_idx == 6:
                    # Overall Summary on Fit: first sentence bold
                    _bold_first_sentence(cell, cell_text)

    doc.add_paragraph()


def _parse_synthesis(synthesis_text):
    """Parse HEADLINE/OPENING/RECOMMENDATION sections from synthesis output.
    Handles bold markers, markdown headers, and varied label formatting."""
    result = {"headline": "", "opening": "", "recommendation": ""}
    current = None
    lines = {"headline": [], "opening": [], "recommendation": []}

    for line in synthesis_text.split("\n"):
        # Strip markdown/bold markers for label detection only
        clean = re.sub(r"[*#_]+", "", line).strip()
        upper = clean.upper()

        if re.match(r"^HEADLINE\s*:", upper):
            current = "headline"
            rest = re.sub(r"(?i)^headline\s*:\s*", "", clean).strip()
            if rest:
                lines["headline"].append(rest)
        elif re.match(r"^OPENING\s*:", upper):
            current = "opening"
            rest = re.sub(r"(?i)^opening\s*:\s*", "", clean).strip()
            if rest:
                lines["opening"].append(rest)
        elif re.match(r"^RECOMMENDATION\s*:", upper):
            current = "recommendation"
            rest = re.sub(r"(?i)^recommendation\s*:\s*", "", clean).strip()
            if rest:
                lines["recommendation"].append(rest)
        elif current:
            lines[current].append(line)

    result["headline"] = "\n".join(lines["headline"]).strip()
    result["opening"] = "\n".join(lines["opening"]).strip()
    result["recommendation"] = "\n".join(lines["recommendation"]).strip()
    return result


def generate_recommendations_docx(criteria, rankings, original_message):
    doc = Document()

    synthesis_text = rankings.get("synthesis", "")
    sections = _parse_synthesis(synthesis_text)
    logging.info(f"Synthesis sections: { {k: len(v) for k, v in sections.items()} }")

    # H2: headline from synthesis, fallback to "Screening result: [message]"
    headline = sections.get("headline", "").strip()
    if not headline:
        fallback = next((l.strip() for l in original_message.split("\n") if l.strip()), "Supplier Screening")
        if len(fallback) > 70:
            fallback = fallback[:67] + "..."
        headline = f"Screening result: {fallback}"
    doc.add_heading(headline, level=2)

    # Opening paragraph with inline bold
    opening = sections.get("opening", "").strip()
    if opening:
        _add_rich_paragraph(doc, opening)

    # H3: Scoring criteria and weights
    doc.add_heading("Scoring criteria and weights", level=3)
    doc.add_paragraph()

    # Calculation line: "Calculation:" bold + rest plain
    calc_para = doc.add_paragraph()
    calc_para.add_run("Calculation:").bold = True
    calc_para.add_run(
        " each criterion is scored 0–5, multiplied by its importance; "
        "total is normalized to 100. Technical and Contextual subtotals are "
        "also independently normalized to 100."
    )
    doc.add_paragraph()

    # Criteria table (4 cols, all plain)
    table_lines = [l for l in criteria.split("\n") if "|" in l]
    criteria_rows = _parse_md_table("\n".join(table_lines))
    if criteria_rows:
        _add_criteria_table(doc, criteria_rows)

    # H3: Recommendation
    doc.add_heading("Recommendation", level=3)

    recommendation = sections.get("recommendation", "").strip()
    if recommendation:
        for para_text in recommendation.split("\n\n"):
            para_text = para_text.strip()
            if para_text:
                _add_rich_paragraph(doc, para_text)

    # References: bold label + List Paragraph per file
    references = rankings.get("references", [])
    if references:
        ref_para = doc.add_paragraph()
        ref_para.add_run("References:").bold = True
        for ref in references:
            doc.add_paragraph(ref, style="List Paragraph")

    # Ranking table (7 cols, per-column bold)
    header_row = rankings.get("header_row")
    ranked_rows = rankings.get("ranked_rows", [])
    if ranked_rows:
        _add_ranking_table(doc, header_row, ranked_rows)

    filename = os.path.join(OUTPUT_FOLDER, "Supplier_Recommendations.docx")
    doc.save(filename)
    return filename


# =========================
# PIPELINE
# =========================

def process_message(message_text, channel_id, thread_ts, client):
    try:
        criteria = create_supplier_criteria(message_text)
        logging.info("Criteria created")

        # Wait up to 3 min for the vector index (only matters on first boot)
        if not _index_ready.is_set():
            logging.info("Waiting for supplier index to finish loading...")
            client.chat_postMessage(
                channel=channel_id, thread_ts=thread_ts,
                text="Still loading the supplier database — first run takes an extra minute. Hang tight! 🔄"
            )
            _index_ready.wait(timeout=180)

        if _supplier_index and len(_supplier_index["docs"]) > 0:
            # Two-pass retrieval: narrow (message+criteria) + broad (criterion names)
            supplier_docs = retrieve_two_pass(message_text, criteria, k=TOP_K)
        else:
            # Fallback: score all docs the slow way
            logging.warning("Vector index unavailable — falling back to full scan")
            supplier_docs = get_supplier_docs()

        if not supplier_docs:
            client.chat_postMessage(
                channel=channel_id, thread_ts=thread_ts,
                text="No supplier profiles found in the Google Drive folder."
            )
            return

        rankings = rank_suppliers(criteria, supplier_docs)
        logging.info("Rankings complete")

        docx_path = generate_recommendations_docx(criteria, rankings, message_text)

        client.files_upload_v2(
            channel=channel_id,
            thread_ts=thread_ts,
            file=docx_path,
            filename="Supplier_Recommendations.docx",
            initial_comment="Here are your supplier recommendations! Searched all profiles, scored the best matches. 📋"
        )

        logging.info("Supplier recommendations sent to Slack")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        client.chat_postMessage(
            channel=channel_id, thread_ts=thread_ts,
            text=f"Something went wrong: {str(e)}"
        )


# =========================
# SLACK EVENT HANDLER
# =========================

@slack_app.event("message")
def handle_message(event, client):
    if event.get("bot_id") or event.get("subtype"):
        return

    channel_id = event.get("channel")
    if channel_id != SLACK_CHANNEL_ID:
        return

    raw_text = event.get("text", "").strip()
    message_text = strip_mention(raw_text)

    if not message_text:
        return

    thread_ts = event.get("ts")
    logging.info(f"Received message: {message_text[:100]}...")

    client.chat_postMessage(
        channel=channel_id,
        thread_ts=thread_ts,
        text="Got it! Analyzing your request and scanning the supplier database... this may take a minute. 🔍"
    )

    thread = threading.Thread(
        target=process_message,
        args=(message_text, channel_id, thread_ts, client)
    )
    thread.daemon = True
    thread.start()


# =========================
# FLASK SERVER
# =========================

flask_app = Flask(__name__)
handler = SlackRequestHandler(slack_app)


@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    return handler.handle(request)


@flask_app.route("/health", methods=["GET"])
def health():
    index_status = "ready" if (_supplier_index and len(_supplier_index["docs"]) > 0) else "loading"
    count = len(_supplier_index["docs"]) if _supplier_index else 0
    return {"status": "ok", "index": index_status, "profiles": count}, 200


@flask_app.route("/index-check", methods=["GET"])
def index_check():
    """Diagnostic endpoint — shows index health and a sample of loaded profiles."""
    if not _supplier_index or len(_supplier_index["docs"]) == 0:
        return {
            "status": "empty",
            "profiles_loaded": 0,
            "message": "Index is empty or still loading. Try again in 60-120 seconds."
        }, 200

    docs = _supplier_index["docs"]
    vectors = _supplier_index["vectors"]

    # Check vectors are non-zero
    zero_count = int((np.linalg.norm(vectors, axis=1) < 0.01).sum())
    sample = [d["name"] for d in docs[:10]]

    return {
        "status": "ok",
        "profiles_loaded": len(docs),
        "zero_vectors": zero_count,
        "embedding_dimensions": int(vectors.shape[1]),
        "sample_profiles": sample,
        "message": "Index looks healthy." if zero_count == 0 else f"WARNING: {zero_count} profiles have zero vectors — embeddings may be corrupted."
    }, 200


@flask_app.route("/refresh-index", methods=["POST"])
def refresh_index():
    """Called by the Notion pipeline after uploading new DOCXs.
    Reloads the pre-computed embeddings JSON from Drive into memory.
    Requires Authorization: Bearer <REFRESH_SECRET> header.
    """
    auth = request.headers.get("Authorization", "")
    if REFRESH_SECRET and auth != f"Bearer {REFRESH_SECRET}":
        return {"error": "Unauthorized"}, 401

    def _refresh():
        global _supplier_index
        try:
            index = _load_embeddings_from_drive()
            if index:
                with _index_lock:
                    _supplier_index = index
                logging.info(f"Index refreshed: {len(index['docs'])} profiles loaded from Drive")
            else:
                logging.warning("Refresh called but no embeddings file found in Drive")
        except Exception as e:
            logging.error(f"Index refresh failed: {e}", exc_info=True)

    threading.Thread(target=_refresh, daemon=True).start()
    return {"status": "refreshing"}, 202


if __name__ == "__main__":
    flask_app.run(port=int(os.getenv("PORT", 3000)))