"""
Physics Book Chatbot -- FastAPI backend
Serves the chat API + the static PWA frontend from one app, for simple single-service deployment.
"""

import os
import re
import json
import time
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import chromadb
from fastembed import TextEmbedding
from google import genai

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

APP_DIR = os.path.dirname(os.path.abspath(__file__))

print("Loading embedding model (ONNX, lightweight)...")
embed_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

def embed_text(text: str):
    return list(embed_model.embed([text]))[0].tolist()

print("Connecting to vector store...")
chroma_client = chromadb.PersistentClient(path=os.path.join(APP_DIR, "chroma_db"))
collection = chroma_client.get_collection("physics_book")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
gemini = genai.Client(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-3.5-flash-lite"
MAX_DISTANCE = 1.0  # empirically calibrated -- see Day 2 testing notes

TRANSLATE_PROMPT = """You will be given a student's physics question, which may be written in
English, Urdu (native script), Sindhi (native script), or Roman Urdu (Urdu written in English/Latin letters).

Detect the language, then respond with ONLY a JSON object (no markdown, no explanation) in this exact form:
{"language": "<English|Urdu|Sindhi|Roman Urdu>", "english_translation": "<the question translated to English>"}
"""

ANSWER_SYSTEM_PROMPT = """You are a helpful physics tutor for a Matric-level (Class 10, Sindh Textbook Board) student.
Answer the student's question using ONLY the textbook excerpts provided below.

Before answering, check: do the excerpts actually define, explain, or directly discuss the specific concept being asked about -- not just share a word or two in common with it?
If the excerpts do not substantively cover the exact concept asked about, say clearly (in the student's own language) that this topic is not covered in the textbook material provided.

When you do have real supporting content, keep answers clear, use the same terms and explanation style as the textbook, and include relevant formulas if present in the excerpts.

IMPORTANT: Respond in the SAME language the student asked in (given to you below as "Respond in: <language>").
If the language is "Roman Urdu", write your answer in Roman Urdu too (Urdu words spelled out in English letters), not in Urdu script or English.
"""

NOT_COVERED_TEXT = "Based on the textbook excerpts provided, this topic is not covered in the textbook material."

# ---------------------------------------------------------------------------
# Simple in-memory rate limiting (fine for single-instance free-tier hosting)
# ---------------------------------------------------------------------------

RATE_LIMIT_PER_MINUTE = 6          # per IP
RATE_LIMIT_PER_DAY = 80            # per IP
_minute_hits = defaultdict(deque)
_day_hits = defaultdict(deque)

def check_rate_limit(ip: str):
    now = time.time()

    dq = _minute_hits[ip]
    while dq and now - dq[0] > 60:
        dq.popleft()
    if len(dq) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Too many questions too fast -- please wait a minute and try again.")

    dq_day = _day_hits[ip]
    while dq_day and now - dq_day[0] > 86400:
        dq_day.popleft()
    if len(dq_day) >= RATE_LIMIT_PER_DAY:
        raise HTTPException(status_code=429, detail="Daily question limit reached -- please try again tomorrow.")

    dq.append(now)
    dq_day.append(now)

# ---------------------------------------------------------------------------
# Core pipeline (same logic validated in Day 2 testing)
# ---------------------------------------------------------------------------

def translate_to_english(question: str):
    response = gemini.models.generate_content(
        model=MODEL_NAME,
        config=genai.types.GenerateContentConfig(
            system_instruction=TRANSLATE_PROMPT, max_output_tokens=200
        ),
        contents=question,
    )
    raw = re.sub(r'^```json\s*|\s*```$', '', response.text.strip())
    try:
        data = json.loads(raw)
        return data["language"], data["english_translation"]
    except Exception:
        return "English", question

def retrieve(english_question: str, n_results: int = 3, max_distance: float = MAX_DISTANCE):
    q_embedding = embed_text(english_question)
    results = collection.query(
        query_embeddings=[q_embedding], n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    return [(d, m, dist) for d, m, dist in zip(docs, metas, dists) if dist <= max_distance]

def generate_answer(original_question: str, language: str, retrieved):
    if not retrieved:
        response = gemini.models.generate_content(
            model=MODEL_NAME,
            config=genai.types.GenerateContentConfig(
                system_instruction=f"Tell the student, in {language}, that this topic isn't covered in the textbook. Be brief and kind.",
                max_output_tokens=100,
            ),
            contents=original_question,
        )
        return response.text, []

    context = "\n\n---\n\n".join(
        f"[Unit {m['unit_id']}: {m['unit_title']}]\n{text}" for text, m, _ in retrieved
    )
    user_message = f"""Textbook excerpts:
{context}

Respond in: {language}

Student question (original): {original_question}"""

    response = gemini.models.generate_content(
        model=MODEL_NAME,
        config=genai.types.GenerateContentConfig(
            system_instruction=ANSWER_SYSTEM_PROMPT, max_output_tokens=700
        ),
        contents=user_message,
    )
    sources = [{"unit_id": m["unit_id"], "unit_title": m["unit_title"]} for _, m, _ in retrieved]
    # de-duplicate while preserving order
    seen = set()
    unique_sources = []
    for s in sources:
        key = s["unit_id"]
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)
    return response.text, unique_sources

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Physics Book Chatbot")

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    language: str
    sources: list

@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    check_rate_limit(client_ip)

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if len(question) > 500:
        raise HTTPException(status_code=400, detail="Question is too long.")

    try:
        language, english_q = translate_to_english(question)
        retrieved = retrieve(english_q)
        answer, sources = generate_answer(question, language, retrieved)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong generating an answer: {e}")

    return AskResponse(answer=answer, language=language, sources=sources)

@app.get("/api/health")
def health():
    return {"status": "ok"}

# Serve the PWA frontend
app.mount("/", StaticFiles(directory=os.path.join(APP_DIR, "static"), html=True), name="static")
