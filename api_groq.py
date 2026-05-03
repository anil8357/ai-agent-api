from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from groq import Groq
from ddgs import DDGS
import datetime
import os
import glob
import httpx
import json
import re


app = FastAPI(
    title="AI Agent API — Groq Edition",
    description="Production API powered by Groq — fast, free, no local GPU needed",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


GROQ_MODEL = "llama-3.1-8b-instant"


def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set in Railway environment variables")
    return Groq(api_key=api_key)


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


@app.get("/")
async def root():
    return {
        "status": "running",
        "message": "AI Agent API is live",
        "backend": "groq",
        "model": GROQ_MODEL,
        "routes": {
            "health": "/health",
            "docs": "/docs",
            "chat": "/chat",
            "assistant": "/assistant",
            "research": "/research",
            "briefing": "/briefing",
            "reports": "/reports",
            "followups": "/followups",
            "register_token": "/register-token"
        },
        "timestamp": datetime.datetime.now().isoformat()
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": GROQ_MODEL,
        "backend": "groq",
        "timestamp": datetime.datetime.now().isoformat()
    }


def groq_chat(messages: list, max_tokens: int = 2048) -> str:
    client = get_groq_client()

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=max_tokens
    )

    return response.choices[0].message.content or ""


def search_web(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

            if not results:
                return "No results found"

            formatted_results = []

            for r in results:
                title = r.get("title", "")
                href = r.get("href", "")
                body = r.get("body", "")

                formatted_results.append(
                    f"TITLE: {title}\nURL: {href}\nSUMMARY: {body[:300]}"
                )

            return "\n\n".join(formatted_results)

    except Exception as e:
        return f"Search failed: {e}"


def run_research_pipeline(topic: str) -> str:
    web_results = search_web(topic)
    tech_results = search_web(f"{topic} tutorial learn resources")

    report = groq_chat(
        [
            {
                "role": "system",
                "content": (
                    "You are a technical writer creating a daily briefing for an Android "
                    "developer transitioning to AI Engineering. Always include real URLs "
                    "from the research data provided."
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
        ],
        max_tokens=3000
    )

    filename = f"groq_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(report)

    return report


def run_briefing_pipeline() -> str:
    today = datetime.datetime.now().strftime("%B %Y")

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

    report = groq_chat(
        [
            {
                "role": "system",
                "content": (
                    "You are a technical writer creating a daily briefing for Anil Kumar, "
                    "an Android developer transitioning to AI Engineering. Always include "
                    "real URLs from the research data."
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
        ],
        max_tokens=3000
    )

    return report


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
    reply = groq_chat(
        [
            {
                "role": "system",
                "content": request.system_prompt
            },
            {
                "role": "user",
                "content": request.message
            }
        ]
    )

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

    reply = groq_chat(
        [
            {
                "role": "system",
                "content": system
            },
            {
                "role": "user",
                "content": request.message
            }
        ]
    )

    return ChatResponse(
        message=request.message,
        reply=reply,
        model=GROQ_MODEL,
        timestamp=datetime.datetime.now().isoformat()
    )


@app.get("/briefing", response_model=BriefingResponse)
async def get_briefing():
    briefing = run_briefing_pipeline()
    date = datetime.datetime.now().strftime("%A, %B %d, %Y")

    tokens = get_tokens()

    for token in tokens:
        await send_push_notification(
            token=token,
            title=f"🤖 Daily Briefing — {date}",
            body="Your AI + Android briefing is ready. Tap to read.",
            briefing=briefing[:500]
        )

    return BriefingResponse(
        briefing=briefing,
        model=GROQ_MODEL,
        timestamp=datetime.datetime.now().isoformat(),
        date=date
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


@app.post("/followups", response_model=FollowUpsResponse)
async def get_follow_ups(request: ChatRequest):
    reply = groq_chat(
        [
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
        ],
        max_tokens=200
    )

    questions = []

    try:
        match = re.search(r"\[.*?\]", reply, re.DOTALL)

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


async def send_push_notification(token: str, title: str, body: str, briefing: str):
    fcm_key = os.getenv("FCM_SERVER_KEY")

    if not fcm_key or not token:
        return

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            await client.post(
                "https://fcm.googleapis.com/fcm/send",
                headers={
                    "Authorization": f"key={fcm_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "to": token,
                    "notification": {
                        "title": title,
                        "body": body[:100]
                    },
                    "data": {
                        "briefing": briefing[:500]
                    }
                }
            )

    except Exception as e:
        print(f"FCM error: {e}")


def save_token(token: str):
    token = token.strip()

    if not token:
        return

    existing_tokens = get_tokens()

    if token in existing_tokens:
        return

    with open("fcm_tokens.txt", "a", encoding="utf-8") as f:
        f.write(token + "\n")


def get_tokens() -> list[str]:
    try:
        with open("fcm_tokens.txt", "r", encoding="utf-8") as f:
            tokens = f.read().splitlines()
            return list(set(token.strip() for token in tokens if token.strip()))

    except FileNotFoundError:
        return []

    except Exception:
        return []


@app.post("/register-token")
async def register_token(request: TokenRequest):
    save_token(request.token)

    return {
        "status": "registered",
        "user_id": request.user_id,
        "timestamp": datetime.datetime.now().isoformat()
    }