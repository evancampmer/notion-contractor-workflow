import os
import io
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

# Slack
slack_app = App(
    token=os.getenv("SLACK_BOT_TOKEN"),
    signing_secret=os.getenv("SLACK_SIGNING_SECRET")
)

# OpenRouter
openrouter = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY_ALINA")
)

MODEL = "anthropic/claude-sonnet-4-5"

# Google Drive
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

CRITERIA_SYSTEM_PROMPT = """Role:
You are a business analyst in charge of converting consulting project descriptions into requirements.

Context:
You are helping a consulting and staffing firm find the best profiles within their supplier database to fill the need for a specific project. You will be receiving information from a business user with project description info (could be one line or an interview transcript). You will need to sort through this to identify 1) what is required in a consultant who will support the project (specific credentials, experiences, or qualities), 2) a specific rubric that will allow another agent to identify the right people based on the criteria you create.
Also all components of the query whether it's a short sentence or a long interview transcript are integral parts to the criteria process. If it's said explicitly that the criteria is there or if it's implied with the certain wording it uses.

## Your Task
Analyze the project description and extract comprehensive evaluation criteria in THREE categories:
### 1. Core Technical Requirements
Skills, tools, certifications explicitly mentioned or directly implied.
Assign importance (1-5) based on how critical each is.
### 2. Contextual Fit Requirements
Extract these by analyzing:
- Company type/stage: What does "early-stage VC" tell you about work environment?
- Industry domain: Does industry experience matter?
- Implicit work style: What does this role actually need day-to-day?
Create scorable criteria like:
- "Startup or VC experience"
- "Entrepreneurial/generalist mindset"
- "Comfort with ambiguity and rapid change"
Assign importance (1-5) - these should be AS IMPORTANT as technical skills.
### 3. Adjacent/Complementary Skills
What would make someone even better, based on the context?
## Output Format
For EACH criterion across all three groups, output:
- Criterion name
- Importance (1-5)
- Brief rationale
Make contextual criteria just as concrete and scorable as technical ones.

Output:
You will create summary output with a 1-3 sentence summary of the project and a table with the information above."""

SCREENING_SYSTEM_PROMPT = """Role:
You are a supplier screening agent. You work for a consulting and staffing firm, helping select the right suppliers to support a specific team and project need.

Input:
You will receive summary output with a 1-3 sentence summary of the project and a table with 1) criterion (could be a credential, experience, quality, or something else), 2) importance of that criterion on a scale of 1 [low] - 5 [high], and 2) explanation & rationale for why it's needed in the project.

Process to follow:

* You will look through every consultant profile provided to you.
* You will screen each profile against the criteria you received in input. This is going to be you going through each profile and searching for the key words in the input query that match exactly to words or experiences and abilities that the profile contains. Not just assumptions of experience but out right text that verifies the comparison between the query and the files.
* You will assess the fit of each person to these criteria, and give them a fit score between 1 and 100 based on how well they fit each criterion and the importance of that criterion.

## Your Task
Review every supplier profile provided and score each against ALL criteria you received (technical + contextual + adjacent).

## Scoring Process
For each supplier:
1. Calculate score for EACH individual criterion (0-5 scale)
2. Multiply by importance weight
3. Sum weighted scores
4. Normalize to 100-point scale
Show breakdown:
- Technical subtotal
- Contextual subtotal
- Overall total

## Output Format
The table should have these columns:
1) Name
2) Total Score (out of 100)
3) Technical subtotal (out of 100)
4) Contextual subtotal (out of 100)
5) Each Criteria Score
6) Strengths Summary
7) Weakness Summary
8) Overall Summary on fit for position

Present candidates showing HOW they scored across dimensions, not just final number."""

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
# AI CALLS
# =========================

def create_supplier_criteria(message_text):
    """Step 1: Generate rubric from client brief."""
    logging.info("Creating supplier criteria...")

    response = openrouter.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": CRITERIA_SYSTEM_PROMPT},
            {"role": "user", "content": (
                "Please follow your system instructions to create supplier criteria "
                "based on the project description or interview transcript you just received.\n\n"
                f"{message_text}"
            )}
        ]
    )

    return response.choices[0].message.content


def rank_suppliers(criteria, supplier_docs):
    """Step 2: Score each supplier against the criteria."""
    logging.info(f"Ranking {len(supplier_docs)} suppliers...")

    supplier_text = "\n\n---\n\n".join(
        f"SUPPLIER: {doc['name']}\n\n{doc['text']}"
        for doc in supplier_docs
    )

    user_message = (
        f"Please look through the supplier profiles below and find the best suppliers "
        f"to meet this project need based on the criteria you just received.\n\n"
        f"CRITERIA:\n{criteria}\n\n"
        f"SUPPLIER PROFILES:\n{supplier_text}"
    )

    response = openrouter.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SCREENING_SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
    )

    return response.choices[0].message.content


# =========================
# DOCX GENERATION
# =========================

def generate_recommendations_docx(criteria, rankings, original_message):
    """Generate the Supplier Recommendations DOCX."""
    doc = Document()

    doc.add_heading("Supplier Recommendations", 0)
    doc.add_paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M')}")

    doc.add_heading("Project Brief", 1)
    doc.add_paragraph(original_message)

    doc.add_heading("Supplier Criteria", 1)
    doc.add_paragraph(criteria)

    doc.add_heading("Ranked Supplier Results", 1)
    doc.add_paragraph(rankings)

    filename = os.path.join(OUTPUT_FOLDER, "Supplier_Recommendations.docx")
    doc.save(filename)

    return filename


# =========================
# PIPELINE
# =========================

def process_message(message_text, channel_id, thread_ts, client):
    """Full pipeline — runs in background thread."""
    try:
        # Step 1: Create criteria
        criteria = create_supplier_criteria(message_text)
        logging.info("Criteria created")

        # Step 2: Get supplier docs from Drive
        supplier_docs = get_supplier_docs()

        if not supplier_docs:
            client.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_ts,
                text="No supplier profiles found in the Google Drive folder."
            )
            return

        # Step 3: Rank suppliers
        rankings = rank_suppliers(criteria, supplier_docs)
        logging.info("Rankings complete")

        # Step 4: Generate DOCX
        docx_path = generate_recommendations_docx(criteria, rankings, message_text)

        # Step 5: Upload to Slack thread
        client.files_upload_v2(
            channel=channel_id,
            thread_ts=thread_ts,
            file=docx_path,
            filename="Supplier_Recommendations.docx",
            initial_comment="Here are your supplier recommendations! 📋"
        )

        logging.info("Supplier recommendations sent to Slack")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text=f"Something went wrong processing your request: {str(e)}"
        )


# =========================
# SLACK EVENT HANDLER
# =========================

@slack_app.event("message")
def handle_message(event, client):
    # Skip bot messages and subtypes (joins, leaves, etc.)
    if event.get("bot_id") or event.get("subtype"):
        return

    channel_id = event.get("channel")
    if channel_id != SLACK_CHANNEL_ID:
        return

    message_text = event.get("text", "").strip()
    thread_ts = event.get("ts")

    if not message_text:
        return

    logging.info(f"Received message: {message_text[:100]}...")

    # Acknowledge immediately so user knows it's working
    client.chat_postMessage(
        channel=channel_id,
        thread_ts=thread_ts,
        text="Got it! Analyzing your request and scanning the supplier database... this may take a minute. 🔍"
    )

    # Process in background thread so Slack doesn't time out
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
    logging.info(f"Incoming request headers: {dict(request.headers)}")
    logging.info(f"Incoming request body: {request.get_data(as_text=True)[:500]}")
    return handler.handle(request)


@flask_app.route("/health", methods=["GET"])
def health():
    return "OK", 200


if __name__ == "__main__":
    flask_app.run(port=int(os.getenv("PORT", 3000)))