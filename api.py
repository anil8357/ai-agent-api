from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import datetime
import os
import glob

from ddgs import DDGS
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
import operator

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")


# ── App setup ─────────────────────────────────────────────
app = FastAPI(
    title="AI Agent API",
    description="LangGraph agents accessible via REST — call from Android",
    version="1.0.0"
)

# Allow Android app to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = OllamaLLM(
    model="llama3.1:8b",
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
)

# ── Request/Response models ────────────────────────────────
# Like data classes in Kotlin
class ResearchRequest(BaseModel):
    topic: str
    session_id: Optional[str] = "default"
    max_iterations: Optional[int] = 2

class ResearchResponse(BaseModel):
    topic: str
    report: str
    iterations: int
    filename: str
    timestamp: str

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    message: str
    reply: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_available: list

# ── LangGraph State ────────────────────────────────────────
class ResearchState(TypedDict):
    topic: str
    search_results: Annotated[list, operator.add]
    draft: str
    feedback: str
    final_report: str
    iteration: int

# ── Agent functions ────────────────────────────────────────
def web_searcher(state: ResearchState) -> ResearchState:
    results = []
    with DDGS() as ddgs:
        try:
            hits = list(ddgs.text(state["topic"], max_results=3))
            for r in hits:
                results.append(f"[WEB] {r['title']}: {r['body'][:250]}")
        except Exception as e:
            results.append(f"Search failed: {e}")
    return {**state, "search_results": results}

def tech_researcher(state: ResearchState) -> ResearchState:
    results = []
    with DDGS() as ddgs:
        try:
            hits = list(ddgs.text(f"{state['topic']} technical", max_results=3))
            for r in hits:
                results.append(f"[TECH] {r['title']}: {r['body'][:250]}")
        except Exception as e:
            results.append(f"Search failed: {e}")
    return {**state, "search_results": results}

def writer(state: ResearchState) -> ResearchState:
    all_results = "\n".join(state["search_results"])
    feedback_section = f"\nAddress: {state['feedback']}" if state.get("feedback") else ""
    prompt = f"""Write a comprehensive markdown report on: {state['topic']}
Sources: {all_results}
{feedback_section}
Format: # Title, ## Overview, ## Key Findings, ## Technical Details, ## Conclusion"""
    draft = llm.invoke(prompt)
    return {**state, "draft": draft}

def reviewer(state: ResearchState) -> ResearchState:
    prompt = f"""Review this report. Reply ONLY with:
APPROVED - if comprehensive
REVISION NEEDED: [issues] - if not
{state['draft']}"""
    feedback = llm.invoke(prompt)
    return {
        **state,
        "feedback": feedback,
        "iteration": state.get("iteration", 0) + 1
    }

def finalizer(state: ResearchState) -> ResearchState:
    final = state["draft"]
    final += f"\n\n---\n*Generated: {datetime.datetime.now().strftime('%B %d, %Y')} | Iterations: {state.get('iteration', 1)}*"
    filename = f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(filename, "w") as f:
        f.write(final)
    return {**state, "final_report": final}

def reviewer_router(state: ResearchState) -> str:
    if state.get("iteration", 0) >= 2:
        return "finalize"
    if "APPROVED" in state.get("feedback", "").upper():
        return "finalize"
    return "revise"

# ── Build graph once at startup ────────────────────────────
def build_graph():
    workflow = StateGraph(ResearchState)
    workflow.add_node("web_searcher",    web_searcher)
    workflow.add_node("tech_researcher", tech_researcher)
    workflow.add_node("writer",          writer)
    workflow.add_node("reviewer",        reviewer)
    workflow.add_node("finalizer",       finalizer)
    workflow.set_entry_point("web_searcher")
    workflow.add_edge("web_searcher",    "tech_researcher")
    workflow.add_edge("tech_researcher", "writer")
    workflow.add_edge("writer",          "reviewer")
    workflow.add_conditional_edges(
        "reviewer", reviewer_router,
        {"revise": "writer", "finalize": "finalizer"}
    )
    workflow.add_edge("finalizer", END)
    return workflow.compile(checkpointer=MemorySaver())

research_graph = build_graph()

# ── Endpoints ──────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if API is running — call this first from Android"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.datetime.now().isoformat(),
        models_available=["llama3.1:8b", "llama3.2", "llava", "nomic-embed-text"]
    )

@app.post("/research", response_model=ResearchResponse)
async def run_research(request: ResearchRequest):
    """Run the research agent on a topic — main endpoint"""
    config = {"configurable": {"thread_id": request.session_id}}
    initial_state = {
        "topic": request.topic,
        "search_results": [],
        "draft": "",
        "feedback": "",
        "final_report": "",
        "iteration": 0
    }
    result = research_graph.invoke(initial_state, config=config)

    # Find the saved filename
    files = sorted(glob.glob("report_*.md"), reverse=True)
    filename = files[0] if files else "report.md"

    return ResearchResponse(
        topic=request.topic,
        report=result["final_report"],
        iterations=result["iteration"],
        filename=filename,
        timestamp=datetime.datetime.now().isoformat()
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Simple chat endpoint — stateless LLM call"""
    response = llm.invoke(request.message)
    return ChatResponse(
        message=request.message,
        reply=response,
        timestamp=datetime.datetime.now().isoformat()
    )

@app.get("/reports")
async def list_reports():
    """List all saved report files"""
    files = sorted(glob.glob("report_*.md"), reverse=True)
    reports = []
    for f in files:
        size = os.path.getsize(f)
        modified = datetime.datetime.fromtimestamp(os.path.getmtime(f))
        reports.append({
            "filename": f,
            "size_bytes": size,
            "created": modified.isoformat()
        })
    return {"reports": reports, "total": len(reports)}

@app.get("/reports/{filename}")
async def get_report(filename: str):
    """Get content of a specific report"""
    if not filename.endswith(".md") or "/" in filename:
        return {"error": "Invalid filename"}
    try:
        with open(filename, "r") as f:
            content = f.read()
        return {"filename": filename, "content": content}
    except FileNotFoundError:
        return {"error": "Report not found"}