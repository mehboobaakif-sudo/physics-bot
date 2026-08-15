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

TRANSLATE_PROMPT = """You will be given a student's latest message, and possibly some recent
conversation history before it. It may be written in English, Urdu (native script),
Sindhi (native script), or Roman Urdu (Urdu written in English/Latin letters).

First, classify the latest message as one of:
- "greeting": a hello/greeting/thanks/bye or pure small talk with no physics content (e.g. "hi", "salam", "thanks", "ok bye")
- "question": an actual physics question or a follow-up on a physics topic (e.g. "what is ohm's law", "I don't understand it", "explain simpler")

If the latest message is a vague follow-up on its own (like "I don't understand", "explain simpler",
"what does that mean", "why"), use the conversation history to figure out what physics topic it
actually refers to, and produce an english_translation that captures the real intent -- e.g. if the
student previously asked about the resistance formula and now says "I don't understand it", the
english_translation should be something like "explain the resistance formula R = V/I in a simpler way",
NOT a literal translation of "I don't understand it" alone.

Detect the language of the LATEST message, then respond with ONLY a JSON object (no markdown, no
explanation) in this exact form:
{"language": "<English|Urdu|Sindhi|Roman Urdu>", "message_type": "<greeting|question>", "english_translation": "<the real intent, translated/expanded to English -- for greetings, just translate the greeting itself>"}
"""

GREETING_PROMPT = """You are a friendly physics tutor chatbot for a Class 10 (Sindh Textbook Board) student.
The student just sent a greeting or small talk, not a physics question. Reply warmly and briefly (1-2 sentences)
in the SAME language they used (told to you as "Respond in: <language>"), and if it's a first greeting,
mention you can help with Units 10-20 of their physics textbook (waves, sound, light, electricity, electronics,
nuclear physics). No LaTeX or markdown. If it's a thanks/goodbye, just respond warmly and briefly without
re-introducing yourself.
"""

ANSWER_SYSTEM_PROMPT = """You are a helpful physics tutor for a Matric-level (Class 10, Sindh Textbook Board) student,
chatting with the student turn by turn -- use any conversation history given to you so follow-ups like
"I don't understand" or "explain simpler" are answered about the right topic, in a genuinely simpler way,
rather than being treated as a brand new unrelated question.

Answer using ONLY the textbook excerpts provided below.

Before answering, check: do the excerpts actually define, explain, or directly discuss the specific concept being asked about -- not just share a word or two in common with it?
If the excerpts do not substantively cover the exact concept asked about, say clearly (in the student's own language) that this topic is not covered in the textbook material provided.

When you do have real supporting content, keep answers clear, use the same terms and explanation style as the textbook, and include relevant formulas if present in the excerpts.

FORMATTING: Never use LaTeX or markdown math notation (no $, $$, \\frac, \\times, etc). Write formulas
in plain text instead, e.g. "R = V / I" or "F = (k * q1 * q2) / r^2". Keep the rest of the answer as
plain, simple text too -- no markdown headers or bullet asterisks, since this is a plain chat bubble.

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

def format_history(history):
    if not history:
        return ""
    lines = []
    for turn in history[-6:]:  # last 3 exchanges max
        role = "Student" if turn.get("role") == "user" else "Tutor"
        lines.append(f"{role}: {turn.get('content', '')}")
    return "\n".join(lines)

def translate_to_english(question: str, history=None):
    history_text = format_history(history)
    contents = question if not history_text else f"Recent conversation:\n{history_text}\n\nLatest message: {question}"
    response = gemini.models.generate_content(
        model=MODEL_NAME,
        config=genai.types.GenerateContentConfig(
            system_instruction=TRANSLATE_PROMPT, max_output_tokens=250
        ),
        contents=contents,
    )
    raw = re.sub(r'^```json\s*|\s*```$', '', response.text.strip())
    try:
        data = json.loads(raw)
        return data["language"], data.get("message_type", "question"), data["english_translation"]
    except Exception:
        return "English", "question", question

def generate_greeting_reply(original_message: str, language: str, history=None):
    history_text = format_history(history)
    contents = f"Respond in: {language}\n\nStudent's message: {original_message}"
    if history_text:
        contents = f"Recent conversation:\n{history_text}\n\n{contents}"
    response = gemini.models.generate_content(
        model=MODEL_NAME,
        config=genai.types.GenerateContentConfig(
            system_instruction=GREETING_PROMPT, max_output_tokens=150
        ),
        contents=contents,
    )
    return response.text

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

def generate_answer(original_question: str, language: str, retrieved, history=None):
    history_text = format_history(history)
    history_block = f"\nRecent conversation:\n{history_text}\n" if history_text else ""

    if not retrieved:
        response = gemini.models.generate_content(
            model=MODEL_NAME,
            config=genai.types.GenerateContentConfig(
                system_instruction=f"Tell the student, in {language}, that this topic isn't covered in the textbook. Be brief and kind. No LaTeX or markdown.",
                max_output_tokens=100,
            ),
            contents=f"{history_block}\nStudent's latest message: {original_question}",
        )
        return response.text, []

    context = "\n\n---\n\n".join(
        f"[Unit {m['unit_id']}: {m['unit_title']}]\n{text}" for text, m, _ in retrieved
    )
    user_message = f"""Textbook excerpts:
{context}
{history_block}
Respond in: {language}

Student's latest message (original): {original_question}"""

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
    history: list = []   # list of {"role": "user"|"bot", "content": str}, most recent last

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

    history = req.history[-6:] if req.history else []

    try:
        language, message_type, english_q = translate_to_english(question, history)

        if message_type == "greeting":
            answer = generate_greeting_reply(question, language, history)
            return AskResponse(answer=answer, language=language, sources=[])

        retrieved = retrieve(english_q)
        answer, sources = generate_answer(question, language, retrieved, history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong generating an answer: {e}")

    return AskResponse(answer=answer, language=language, sources=sources)

@app.get("/api/health")
def health():
    return {"status": "ok"}

# Serve the PWA frontend
app.mount("/", StaticFiles(directory=os.path.join(APP_DIR, "static"), html=True), name="static")
