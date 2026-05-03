from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from groq import Groq
from ddgs import DDGS

import datetime
import os
import glob
import json

import firebase_admin
from firebase_admin import credentials, messaging


app = FastAPI(
    title="AI Agent API — Groq Edition",
    description="Production API powered by Groq + Firebase Admin SDK",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

GROQ_MODEL = "llama-3.1-8b-instant"
FCM_CHANNEL_ID = "daily_briefing"
TOKEN_FILE = "fcm_tokens.txt"

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ---------------------------------------------------------------------
# Firebase
# ---------------------------------------------------------------------

def init_firebase():
    """
    Initializes Firebase Admin SDK using Railway env variable:
    FIREBASE_SERVICE_ACCOUNT_JSON
    """
    if firebase_admin._apps:
        return

    service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")

    if not service_account_json:
        raise RuntimeError("FIREBASE_SERVICE_ACCOUNT_JSON missing in Railway Variables")

    cred_dict = json.loads(service_account_json)
    cred = credentials.Certificate(cred_dict)

    firebase_admin.initialize_app(cred)


async def send_push_notification(token: str, title: str, body: str, briefing: str):
    """
    Sends push notification using Firebase Admin SDK.
    Uses Android channel_id=daily_briefing, so Android app must create same channel.
    """
    init_firebase()

    token = token.strip()
    if not token:
        return {
            "success": False,
            "reason": "FCM token empty"
        }

    try:
        message = messaging.Message(
            token=token,
            notification=messaging.Notification(
                title=title,
                body=body
            ),
            data={
                "title": title,
                "body": body,
                "briefing": briefing[:3000]
            },
            android=messaging.AndroidConfig(
                priority="high",
                notification=messaging.AndroidNotification(
                    channel_id=FCM_CHANNEL_ID,
                    sound="default"
                )
            )
        )

        message_id = messaging.send(message)

        return {
            "success": True,
            "message_id": message_id
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def is_invalid_fcm_token_error(error: str) -> bool:
    error = error or ""

    return (
        "Requested entity was not found" in error
        or "registration-token-not-registered" in error
        or "The registration token is not a valid FCM registration token" in error
        or "SenderId mismatch" in error
        or "not a valid FCM registration token" in error
    )


# ---------------------------------------------------------------------
# Token storage
# ---------------------------------------------------------------------

def get_tokens() -> list[str]:
    try:
        with open(TOKEN_FILE, "r") as f:
            return [
                line.strip()
                for line in f.readlines()
                if line.strip()
            ]
    except FileNotFoundError:
        return []
    except Exception:
        return []


def save_token(token: str):
    token = token.strip()
    if not token:
        return

    tokens = set(get_tokens())
    tokens.add(token)

    with open(TOKEN_FILE, "w") as f:
        for t in sorted(tokens):
            f.write(t + "\n")


def remove_token(token: str):
    token = token.strip()
    tokens = set(get_tokens())
    tokens.discard(token)

    with open(TOKEN_FILE, "w") as f:
        for t in sorted(tokens):
            f.write(t + "\n")


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

class ResearchRequest(BaseModel):
    topic: str
    session_id: Optional[str] = "default"


class ResearchResponse(BaseModel):
    topic: str
    report: str
    model: str
    timestamp: str


class ChatRequest(BaseModel):
    message: str
    system_prompt: Optional[str] = "You are a helpful AI assistant."


class ChatResponse(BaseModel):
    message: str
    reply: str
    model: str
    timestamp: str


class BriefingResponse(BaseModel):
    briefing: str
    model: str
    timestamp: str
    date: str


class FollowUpsResponse(BaseModel):
    questions: list[str]
    timestamp: str


class TokenRequest(BaseModel):
    token: str
    user_id: Optional[str] = "default"


# ---------------------------------------------------------------------
# Groq + Search
# ---------------------------------------------------------------------

def groq_chat(messages: list, max_tokens: int = 2048) -> str:
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content


def search_web(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            if not results:
                return "No results found"

            return "\n\n".join([
                f"TITLE: {r.get('title', '')}\n"
                f"URL: {r.get('href', '')}\n"
                f"SUMMARY: {r.get('body', '')[:300]}"
                for r in results
            ])

    except Exception as e:
        return f"Search failed: {e}"


def run_research_pipeline(topic: str) -> str:
    web_results = search_web(topic)
    tech_results = search_web(f"{topic} tutorial learn resources")

    report = groq_chat([
        {
            "role": "system",
            "content": (
                "You are a technical writer creating a daily briefing for an Android developer "
                "transitioning to AI Engineering. Always include real URLs from the research data provided."
            )
        },
        {
            "role": "user",
            "content": f"""Write a detailed daily briefing on: {topic}

Research data:
{web_results}

Technical/Learning data:
{tech_results}

Write in this EXACT format. Use real URLs from the research data above:

# Daily Briefing — {datetime.datetime.now().strftime('%B %d, %Y')}

## 📱 Android News
### [First article title from research]
Summary: 2-3 sentences about what this means for Android developers.
Read more: [real URL from research data]

### [Second article title from research]
Summary: 2-3 sentences of insight.
Read more: [real URL from research data]

## 🧠 AI Engineering
### [First AI article title from research]
Summary: 2-3 sentences relevant to LangGraph, agents, or LLMs.
Read more: [real URL from research data]

### [Second AI article title from research]
Summary: 2-3 sentences of insight.
Read more: [real URL from research data]

## 💼 Jobs & Salary
### Android + AI Engineer roles in India
Summary: Current salary ranges, companies hiring, skills in demand based on research.
Read more: [real URL from research data]

## 📚 Learn Today
### Recommended resource
Summary: One specific tutorial or resource to learn today based on the research.
Read more: [real URL from research data]

IMPORTANT: Only use URLs that appear in the research data above. Do not invent URLs."""
        }
    ], max_tokens=3000)

    filename = f"groq_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(filename, "w") as f:
        f.write(report)

    return report


def run_briefing_pipeline() -> str:
    today = datetime.datetime.now().strftime('%B %Y')

    android_results = search_web(f"Android development news {today}")
    ai_results = search_web(f"AI engineering LLM agents news {today}")
    jobs_results = search_web(f"Android AI engineer jobs India salary {today}")
    learn_results = search_web("LangGraph CrewAI FastAPI tutorial 2026")

    all_results = f"""
=== ANDROID NEWS ===
{android_results}

=== AI ENGINEERING NEWS ===
{ai_results}

=== JOBS & SALARY ===
{jobs_results}

=== LEARNING RESOURCES ===
{learn_results}
"""

    report = groq_chat([
        {
            "role": "system",
            "content": (
                "You are a technical writer creating a daily briefing for Anil Kumar, "
                "an Android developer transitioning to AI Engineering. "
                "Always include real URLs from the research data."
            )
        },
        {
            "role": "user",
            "content": f"""Write a personalized daily briefing using this research data:

{all_results}

Write in this EXACT format using real URLs from the data above:

# Daily Briefing — {datetime.datetime.now().strftime('%A, %B %d, %Y')}

## 📱 Android News
### [Title from Android research]
Summary: What this means for an Android developer like Anil.
Read more: [real URL from Android research]

### [Second title from Android research]
Summary: Key insight for Android development.
Read more: [real URL from Android research]

## 🧠 AI Engineering
### [Title from AI research]
Summary: How this relates to LangGraph, agents, RAG, or LLMs Anil has built.
Read more: [real URL from AI research]

### [Second title from AI research]
Summary: Key insight for AI engineering.
Read more: [real URL from AI research]

## 💼 Jobs & Salary in India
### Android + AI Engineer opportunities
Summary: Specific salary ranges and companies hiring for Android+AI hybrid roles in India.
Read more: [real URL from jobs research]

## 📚 Learn Today
### [Specific tutorial or resource title]
Summary: One concrete thing Anil should learn or practice today based on his LangGraph/CrewAI background.
Read more: [real URL from learning research]

CRITICAL: Only use URLs that actually appear in the research data. Never invent URLs."""
        }
    ], max_tokens=3000)

    return report


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "AI Agent API",
        "file": "api_groq.py",
        "version": "2.1.0",
        "routes": [
            "/health",
            "/briefing",
            "/register-token",
            "/test-push",
            "/tokens",
            "/chat",
            "/assistant",
            "/research",
            "/followups",
            "/reports"
        ]
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": GROQ_MODEL,
        "backend": "groq",
        "firebase_initialized": bool(firebase_admin._apps),
        "tokens_count": len(get_tokens()),
        "timestamp": datetime.datetime.now().isoformat()
    }


@app.get("/routes")
async def list_routes():
    return {
        "routes": [
            {
                "path": route.path,
                "methods": sorted(list(route.methods))
            }
            for route in app.routes
        ]
    }


@app.post("/register-token")
async def register_token(request: TokenRequest):
    save_token(request.token)

    return {
        "status": "registered",
        "token_start": request.token[:20],
        "tokens_count": len(get_tokens())
    }


@app.get("/tokens")
async def list_registered_tokens():
    tokens = get_tokens()

    return {
        "tokens_count": len(tokens),
        "tokens": [
            {
                "token_start": token[:20],
                "token_end": token[-10:] if len(token) > 10 else token
            }
            for token in tokens
        ]
    }


@app.post("/test-push")
async def test_push():
    tokens = get_tokens()

    if not tokens:
        return {
            "status": "failed",
            "reason": "No FCM tokens found. Call /register-token first."
        }

    results = []
    removed_tokens = []

    for token in tokens:
        result = await send_push_notification(
            token=token,
            title="Railway Test",
            body="Notification from Railway backend",
            briefing="Test briefing from Railway"
        )

        error = str(result.get("error", ""))
        should_remove = is_invalid_fcm_token_error(error)

        if should_remove:
            remove_token(token)
            removed_tokens.append(token[:20])

        results.append({
            "token_start": token[:20],
            "removed": should_remove,
            "result": result
        })

    return {
        "status": "done",
        "tokens_count_before_send": len(tokens),
        "tokens_count_after_cleanup": len(get_tokens()),
        "removed_tokens_count": len(removed_tokens),
        "removed_tokens": removed_tokens,
        "results": results
    }


@app.get("/briefing", response_model=BriefingResponse)
async def get_briefing():
    briefing = run_briefing_pipeline()

    tokens = get_tokens()
    removed_tokens = []

    for token in tokens:
        result = await send_push_notification(
            token=token,
            title="Daily AI Briefing",
            body="Your AI + Android briefing is ready",
            briefing=briefing
        )

        error = str(result.get("error", ""))
        should_remove = is_invalid_fcm_token_error(error)

        if should_remove:
            remove_token(token)
            removed_tokens.append(token[:20])

    return BriefingResponse(
        briefing=briefing,
        model=GROQ_MODEL,
        timestamp=datetime.datetime.now().isoformat(),
        date=datetime.datetime.now().strftime("%A, %B %d, %Y")
    )


@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest):
    report = run_research_pipeline(request.topic)

    return ResearchResponse(
        topic=request.topic,
        report=report,
        model=GROQ_MODEL,
        timestamp=datetime.datetime.now().isoformat()
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    reply = groq_chat([
        {"role": "system", "content": request.system_prompt},
        {"role": "user", "content": request.message}
    ])

    return ChatResponse(
        message=request.message,
        reply=reply,
        model=GROQ_MODEL,
        timestamp=datetime.datetime.now().isoformat()
    )


@app.post("/assistant", response_model=ChatResponse)
async def learning_assistant(request: ChatRequest):
    system = """You are Anil Kumar's personal AI learning assistant and career advisor.
Anil is a 7-year Android developer with expertise in Kotlin, MVVM, payment gateways, NFC, and OpenCV.
He recently completed an 8-week AI Engineering roadmap and built:
LLM APIs, RAG pipelines, AI agents, LangGraph, CrewAI, FastAPI, Streamlit, deployed on Railway.
His goal is to transition to an AI Engineer role in India.
Give specific, personalized advice always referencing his actual background.
Be concise — this is a mobile app, keep responses under 200 words."""

    reply = groq_chat([
        {"role": "system", "content": system},
        {"role": "user", "content": request.message}
    ])

    return ChatResponse(
        message=request.message,
        reply=reply,
        model=GROQ_MODEL,
        timestamp=datetime.datetime.now().isoformat()
    )


@app.post("/followups", response_model=FollowUpsResponse)
async def get_follow_ups(request: ChatRequest):
    reply = groq_chat([
        {
            "role": "system",
            "content": """Generate exactly 3 short follow-up questions based on the conversation.
Return ONLY a JSON array of 3 strings. No explanation, no markdown, just the JSON array.
Example: ["Question 1?", "Question 2?", "Question 3?"]
Questions should be specific, actionable and relevant to the context."""
        },
        {
            "role": "user",
            "content": f"Conversation:\n{request.message}\n\nGenerate 3 follow-up questions."
        }
    ], max_tokens=200)

    import re

    questions = []

    try:
        match = re.search(r'\[.*?\]', reply, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            questions = [q for q in parsed if isinstance(q, str)][:3]
    except Exception:
        questions = []

    if not questions:
        questions = [
            "Can you explain this in more detail?",
            "What should I do first?",
            "How does this apply to my background?"
        ]

    return FollowUpsResponse(
        questions=questions,
        timestamp=datetime.datetime.now().isoformat()
    )


@app.get("/reports")
async def list_reports():
    files = sorted(glob.glob("*.md"), reverse=True)

    return {
        "reports": [
            {
                "filename": f,
                "size": os.path.getsize(f)
            }
            for f in files
        ],
        "total": len(files)
    }