from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from groq import Groq
from ddgs import DDGS
import datetime
import os
import glob

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

# Groq client — reads GROQ_API_KEY from environment
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_MODEL = "llama-3.1-8b-instant"  # free, very fast

# ── Request/Response models ────────────────────────────────
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

# ── Helper functions ───────────────────────────────────────
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
            results = list(ddgs.text(query, max_results=4))
            if not results:
                return "No results found"
            return "\n\n".join([
                f"{r['title']}: {r['body'][:300]}"
                for r in results
            ])
    except Exception as e:
        return f"Search failed: {e}"

# ── Research pipeline ──────────────────────────────────────
def run_research_pipeline(topic: str) -> str:
    # Step 1: Search
    web_results = search_web(topic)
    tech_results = search_web(f"{topic} technical 2026")

    # Step 2: Write report
    report = groq_chat([
        {
            "role": "system",
            "content": "You are a technical writer. Write comprehensive markdown reports."
        },
        {
            "role": "user",
            "content": f"""Write a markdown report on: {topic}

Research data:
{web_results}

Technical data:
{tech_results}

Include: # Title, ## Overview, ## Key Findings, ## Technical Details, ## Conclusion
*Generated: {datetime.datetime.now().strftime('%B %d, %Y')}*"""
        }
    ])

    # Step 3: Review
    review = groq_chat([
        {
            "role": "system",
            "content": "You are a strict editor. Reply APPROVED or NEEDS REVISION: [issues]"
        },
        {
            "role": "user",
            "content": f"Review this report:\n{report}"
        }
    ])

    # Step 4: Revise if needed
    if "APPROVED" not in review.upper():
        report = groq_chat([
            {
                "role": "system",
                "content": "You are a technical writer. Improve the report based on feedback."
            },
            {
                "role": "user",
                "content": f"Improve this report:\n{report}\n\nFeedback:\n{review}"
            }
        ])

    # Save to file
    filename = f"groq_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(filename, "w") as f:
        f.write(report)

    return report

# ── Endpoints ──────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": GROQ_MODEL,
        "backend": "groq",
        "timestamp": datetime.datetime.now().isoformat()
    }

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

@app.get("/reports")
async def list_reports():
    files = sorted(glob.glob("*.md"), reverse=True)
    return {
        "reports": [{"filename": f, "size": os.path.getsize(f)} for f in files],
        "total": len(files)
    }