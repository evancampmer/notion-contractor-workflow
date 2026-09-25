import os
import io
import re
import json
import logging
import tempfile
import threading
from datetime import datetime

import requests
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from flask import Flask, request
from openai import OpenAI
from docx import Document
from fpdf import FPDF
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

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
    api_key=os.getenv("OPENROUTER_API_KEY")
)

MODEL = "anthropic/claude-sonnet-4-5"

SCOPES = ["https://www.googleapis.com/auth/drive"]
service_account_info = json.loads(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"))
credentials = service_account.Credentials.from_service_account_info(
    service_account_info, scopes=SCOPES
)
drive_service = build("drive", "v3", credentials=credentials)

SLACK_CHANNEL_ID = "C0C2QRTGAUV"
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
OUTPUT_FOLDER = "output_docs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

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

SCREENING_SYSTEM_PROMPT = """You are a supplier screening agent at a consulting and staffing firm.

You will receive a criteria table and supplier profiles.

Output exactly four sections:

SECTION 1 — HEADLINE
One sentence: overall finding (e.g. "Two strong candidates identified; no exact sector match found")

SECTION 2 — SCORING METHOD
2-3 sentences explaining: how you weighted technical vs contextual (e.g. 40/60), which criteria drove the most differentiation, and the evidence standard you applied.

SECTION 3 — RECOMMENDATION
Bullet points (one per recommended candidate). Each bullet: name, action ("Advance as primary", "Advance as alternative"), and 1-2 sentences on why and any gap. End with one sentence on any universal gap across all candidates.

SECTION 4 — RANKED TABLE
| Rank / Name | Total Score /100 | Technical /100 | Contextual /100 | Each Criterion Score (0-5) | Strengths Summary | Weakness Summary | Overall Fit |

Scoring rules:
- Score each criterion 0-5 based ONLY on explicit text evidence in the profile — no assumptions
- Multiply each score by importance weight, sum, normalize to 100
- Technical and Contextual subtotals each normalized to 100 independently
- "Each Criterion Score" column: write as inline list, e.g. "GTM 5; PMO 4; PE/consulting 3; Industrial 2"
- Include only the top 10-15 candidates with meaningful scores (above ~30/100); skip candidates with no relevant evidence
- Sort by Total Score descending
- Each table cell: 1-3 sentences max, concise

No commentary outside these four sections. Use plain pipe-delimited markdown table format."""

# =========================
# GOOGLE DRIVE
# =========================

def get_supplier_docs():
    """Download all DOCX files from the Google Drive folder and extract text."""
    supplier_docs = []

    query = (
        f"'{GOOGLE_DRIVE_FOLDER_ID}' in parents "
        f"and mimeType='application/vnd.openxmlformats-officedocument.wordprocessingml.document' "
        f"and trashed=false"
    )

    results = drive_service.files().list(
        q=query,
        fields="files(id, name)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
        corpora="allDrives"
    ).execute()

    files = results.get("files", [])
    logging.info(f"Found {len(files)} supplier docs in Drive")

    for file in files:
        try:
            req = drive_service.files().get_media(
                fileId=file["id"],
                supportsAllDrives=True
            )
            buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(buffer, req)
            done = False
            while not done:
                _, done = downloader.next_chunk()

            buffer.seek(0)
            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
                tmp.write(buffer.read())
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
# TEXT HELPERS
# =========================

def strip_mention(text):
    """Remove Slack @mention tags like <@U0C407F3AKT> from the message."""
    return re.sub(r"<@[A-Z0-9]+>\s*", "", text).strip()


def _safe(text):
    """Sanitize text for fpdf latin-1 core fonts."""
    replacements = {
        "•": "-", "’": "'", "‘": "'",
        "“": '"', "”": '"', "–": "-",
        "—": "--", "…": "...", "→": "->",
        "✓": "v", "✔": "v", "✘": "x",
        "é": "e", "ó": "o", "á": "a",
    }
    for ch, rep in replacements.items():
        text = text.replace(ch, rep)
    # Strip remaining bold/italic markers
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _strip_bold(text):
    return re.sub(r"\*\*(.+?)\*\*", r"\1", re.sub(r"\*(.+?)\*", r"\1", text))


# =========================
# AI CALLS
# =========================

def create_supplier_criteria(message_text):
    logging.info("Creating supplier criteria...")
    response = openrouter.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": CRITERIA_SYSTEM_PROMPT},
            {"role": "user", "content": (
                "Create the screening criteria for this project brief:\n\n"
                f"{message_text}"
            )}
        ]
    )
    return response.choices[0].message.content


def rank_suppliers(criteria, supplier_docs):
    logging.info(f"Ranking {len(supplier_docs)} suppliers...")
    supplier_text = "\n\n---\n\n".join(
        f"SUPPLIER: {doc['name']}\n\n{doc['text']}"
        for doc in supplier_docs
    )
    response = openrouter.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SCREENING_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"CRITERIA:\n{criteria}\n\n"
                f"SUPPLIER PROFILES:\n{supplier_text}"
            )}
        ]
    )
    return response.choices[0].message.content


# =========================
# PDF GENERATION
# =========================

def _parse_table(lines):
    """Parse markdown table lines into a list of row lists."""
    rows = []
    for line in lines:
        if re.match(r"^\s*\|[-|:\s]+\|\s*$", line):
            continue  # separator row
        cells = [_safe(_strip_bold(c.strip())) for c in line.strip().strip("|").split("|")]
        if any(cells):
            rows.append(cells)
    return rows


def _render_table(pdf, rows):
    """Render a table with shaded header row."""
    if not rows:
        return
    num_cols = len(rows[0])
    page_w = pdf.w - pdf.l_margin - pdf.r_margin
    col_w = page_w / num_cols
    font_size = 8 if num_cols > 5 else 9

    for r_idx, row in enumerate(rows):
        # Pad/trim row to expected columns
        while len(row) < num_cols:
            row.append("")
        row = row[:num_cols]

        # Measure tallest cell in this row
        pdf.set_font("Helvetica", "B" if r_idx == 0 else "", font_size)
        cell_heights = []
        for cell in row:
            lines_needed = max(1, len(cell) // max(1, int(col_w / (font_size * 0.5))) + 1)
            cell_heights.append(lines_needed * (font_size * 0.5 + 1))
        row_h = max(cell_heights)
        row_h = max(row_h, font_size + 2)

        # Page break check
        if pdf.get_y() + row_h > pdf.h - pdf.b_margin - 10:
            pdf.add_page()

        y0 = pdf.get_y()
        if r_idx == 0:
            pdf.set_fill_color(210, 210, 210)
        elif r_idx % 2 == 0:
            pdf.set_fill_color(245, 245, 245)
        else:
            pdf.set_fill_color(255, 255, 255)

        for c_idx, cell in enumerate(row):
            pdf.set_xy(pdf.l_margin + c_idx * col_w, y0)
            pdf.set_font("Helvetica", "B" if r_idx == 0 else "", font_size)
            pdf.multi_cell(col_w, font_size + 2, cell, border=1,
                           fill=True, max_line_height=font_size + 1)

        pdf.set_y(y0 + row_h)

    pdf.ln(4)


def render_markdown(pdf, text):
    """Render markdown text into the PDF with proper formatting."""
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Collect table block
        if "|" in stripped and i + 1 < len(lines) and re.match(r"^\s*\|[-|:\s]+\|\s*$", lines[i + 1]):
            table_lines = []
            while i < len(lines) and "|" in lines[i]:
                table_lines.append(lines[i])
                i += 1
            _render_table(pdf, _parse_table(table_lines))
            continue

        # Headings
        if stripped.startswith("### "):
            pdf.set_font("Helvetica", "B", 11)
            pdf.multi_cell(0, 6, _safe(stripped[4:]))
            pdf.ln(1)
        elif stripped.startswith("## "):
            pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(0, 7, _safe(stripped[3:]))
            pdf.ln(1)
        elif stripped.startswith("# "):
            pdf.set_font("Helvetica", "B", 13)
            pdf.multi_cell(0, 8, _safe(stripped[2:]))
            pdf.ln(1)
        # Horizontal rule
        elif re.match(r"^---+$", stripped):
            pdf.set_draw_color(180, 180, 180)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.set_draw_color(0, 0, 0)
            pdf.ln(3)
        # Bullet
        elif re.match(r"^[\*\-]\s", stripped):
            content = _safe(_strip_bold(re.sub(r"^[\*\-]\s+", "", stripped)))
            pdf.set_font("Helvetica", "", 10)
            pdf.set_x(pdf.l_margin + 5)
            pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin - 5, 5, "- " + content)
        # Numbered list
        elif re.match(r"^\d+\.\s", stripped):
            content = _safe(_strip_bold(re.sub(r"^\d+\.\s+", "", stripped)))
            num = re.match(r"^(\d+)\.", stripped).group(1)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_x(pdf.l_margin + 5)
            pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin - 5, 5, f"{num}. {content}")
        # Empty line
        elif stripped == "":
            pdf.ln(3)
        # Regular text
        else:
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 5, _safe(_strip_bold(stripped)))

        i += 1


def generate_recommendations_pdf(criteria, rankings, original_message):
    pdf = FPDF()
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "Supplier Recommendations", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M')}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    # Project Brief
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Project Brief", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _safe(original_message))
    pdf.ln(6)

    # Criteria
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Screening Criteria", new_x="LMARGIN", new_y="NEXT")
    render_markdown(pdf, criteria)
    pdf.ln(4)

    # Rankings
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Supplier Screening Results", new_x="LMARGIN", new_y="NEXT")
    render_markdown(pdf, rankings)

    filename = os.path.join(OUTPUT_FOLDER, "Supplier_Recommendations.pdf")
    pdf.output(filename)
    return filename


# =========================
# PIPELINE
# =========================

def process_message(message_text, channel_id, thread_ts, client):
    try:
        criteria = create_supplier_criteria(message_text)
        logging.info("Criteria created")

        supplier_docs = get_supplier_docs()

        if not supplier_docs:
            client.chat_postMessage(
                channel=channel_id, thread_ts=thread_ts,
                text="No supplier profiles found in the Google Drive folder."
            )
            return

        rankings = rank_suppliers(criteria, supplier_docs)
        logging.info("Rankings complete")

        pdf_path = generate_recommendations_pdf(criteria, rankings, message_text)

        client.files_upload_v2(
            channel=channel_id,
            thread_ts=thread_ts,
            file=pdf_path,
            filename="Supplier_Recommendations.pdf",
            initial_comment="Here are your supplier recommendations! 📋"
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
    return "OK", 200


if __name__ == "__main__":
    flask_app.run(port=int(os.getenv("PORT", 3000)))