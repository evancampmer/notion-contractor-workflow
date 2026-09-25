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

TOP_K = 20          # profiles to retrieve via vector search before scoring
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
- Use SHORT criterion names (2-4 words max, e.g. "GTM execution", "PMO cadence", "PE value creation")
- Importance: 1-5 (5 = must-have)
- Evidence standard: specific words/phrases to look for in a profile
- No commentary, no headers, no explanation outside these two items"""

SCORING_PROMPT = """You are a supplier screening agent at a consulting and staffing firm.

Score EVERY supplier profile against the criteria. Return ONLY a pipe-delimited table — no other text, no preamble, no commentary.

Table columns (use exactly these headers):
| Name | Total Score /100 | Technical /100 | Contextual /100 | Each Criterion Score (0-5) | Strengths Summary | Weakness Summary |

Scoring rules:
- Score each criterion 0-5 based ONLY on explicit text evidence in the profile — no assumptions
- Multiply each score by its importance weight, sum all weighted scores, normalize to 100
- Technical subtotal and Contextual subtotal each normalized to 100 independently
- "Each Criterion Score": inline list using the short criterion names from the criteria table, e.g. "510k sub 5; FDA reg 4; Tech writing 2"
- Score ALL suppliers; give 0 if no relevant evidence exists
- Sort by Total Score descending
- Each cell: 1-2 sentences max"""

SYNTHESIS_PROMPT = """You are a supplier screening agent summarizing final candidate scores.

You will receive the top-scored candidates. Write exactly three sections:

SECTION 1 — HEADLINE
One sentence: overall finding (e.g. "Three strong candidates identified; no exact FDA regulatory specialist found in database")

SECTION 2 — SCORING METHOD
2-3 sentences: how technical vs contextual were weighted, which criteria drove the most differentiation, and the evidence standard applied.

SECTION 3 — RECOMMENDATION
One bullet per recommended candidate (top 5 maximum). Each bullet: name, action ("Advance as primary" or "Advance as alternative"), 1-2 sentences on fit and key gap. End with one sentence on any universal gap across all candidates.

No other text outside these three sections."""

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
                _embed_model = TextEmbedding("BAAI/bge-small-en-v1.5")
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
    """Run synthesis pass over top-scored candidates."""
    top_summary = "\n".join(
        f"{row[0]}: Total {row[1]}, "
        f"Tech {row[2] if len(row) > 2 else 'N/A'}, "
        f"Contextual {row[3] if len(row) > 3 else 'N/A'}, "
        f"Strengths: {row[5] if len(row) > 5 else 'N/A'}"
        for row in top_rows
    )
    response = openrouter.chat.completions.create(
        model=MODEL,
        max_tokens=2000,
        messages=[
            {"role": "system", "content": SYNTHESIS_PROMPT},
            {"role": "user", "content": f"CRITERIA:\n{criteria}\n\nTOP CANDIDATES:\n{top_summary}"}
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

    # Keep candidates scoring >= 70, but always include at least top 3
    MIN_SCORE = 70
    qualified = [r for r in data_rows if _extract_score(r) >= MIN_SCORE]
    if len(qualified) < 3:
        qualified = data_rows[:3]  # fallback: top 3 regardless of score
    top_rows = qualified[:15]
    logging.info(f"Scored {len(data_rows)} candidates: {len(qualified)} scored >= {MIN_SCORE}, kept {len(top_rows)}")

    synthesis = synthesize(criteria, top_rows)
    logging.info("Synthesis complete")

    return {
        "synthesis": synthesis,
        "header_row": header_row,
        "ranked_rows": top_rows
    }


# =========================
# DOCX GENERATION
# =========================

def _add_table_to_doc(doc, rows):
    if not rows:
        return
    num_cols = len(rows[0])
    table = doc.add_table(rows=len(rows), cols=num_cols)
    table.style = "Table Grid"
    for r_idx, row in enumerate(rows):
        while len(row) < num_cols:
            row.append("")
        row = row[:num_cols]
        for c_idx, cell_text in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = cell_text
            para = cell.paragraphs[0]
            if r_idx == 0:
                for run in para.runs:
                    run.bold = True
    doc.add_paragraph()


def _parse_rankings(rankings_text):
    sections = {1: [], 2: [], 3: [], 4: []}
    current = None
    for line in rankings_text.split("\n"):
        clean = re.sub(r"[#\*\_]+", "", line).strip()
        if re.search(r"SECTION\s*1", clean, re.IGNORECASE):
            current = 1; continue
        if re.search(r"SECTION\s*2", clean, re.IGNORECASE):
            current = 2; continue
        if re.search(r"SECTION\s*3", clean, re.IGNORECASE):
            current = 3; continue
        if re.search(r"SECTION\s*4", clean, re.IGNORECASE):
            current = 4; continue
        if current is not None:
            sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def generate_recommendations_docx(criteria, rankings, original_message):
    doc = Document()

    title_line = next((l.strip() for l in original_message.split("\n") if l.strip()), "Supplier Screening")
    if len(title_line) > 80:
        title_line = title_line[:77] + "..."

    doc.add_heading(f"Supplier screening results — {title_line}", level=2)
    doc.add_heading("Scoring method", level=3)

    criteria_lines = criteria.split("\n")
    summary_parts, table_lines = [], []
    in_table = False
    for line in criteria_lines:
        if "|" in line:
            in_table = True
        (table_lines if in_table else summary_parts).append(line)

    summary_text = " ".join(l.strip() for l in summary_parts if l.strip())
    if summary_text:
        doc.add_paragraph(summary_text)

    criteria_rows = _parse_md_table("\n".join(table_lines))
    if criteria_rows:
        _add_table_to_doc(doc, criteria_rows)

    synthesis_text = rankings.get("synthesis", "")
    sections = _parse_rankings(synthesis_text)
    logging.info(f"Synthesis sections lengths: { {k: len(v) for k, v in sections.items()} }")

    if sections[1]:
        doc.add_paragraph(_strip_md(sections[1]))
    if sections[2]:
        doc.add_paragraph(_strip_md(sections[2]))

    doc.add_heading("Recommendation", level=3)
    for line in sections[3].split("\n"):
        s = line.strip()
        if not s:
            continue
        if re.match(r"^[\*\-]\s", s):
            content = _strip_md(re.sub(r"^[\*\-]\s+", "", s))
            doc.add_paragraph(content, style="List Paragraph")
        else:
            doc.add_paragraph(_strip_md(s))

    header_row = rankings.get("header_row")
    ranked_rows = rankings.get("ranked_rows", [])
    if header_row and ranked_rows:
        _add_table_to_doc(doc, [header_row] + ranked_rows)
    elif ranked_rows:
        _add_table_to_doc(doc, ranked_rows)

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
            # Vector retrieval: combine brief + criteria for best semantic match
            query = f"{message_text}\n\n{criteria}"
            supplier_docs = retrieve_top_k(query, k=TOP_K)
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