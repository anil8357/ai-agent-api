import streamlit as st
from ddgs import DDGS
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
import operator
import datetime

llm = OllamaLLM(model="llama3.1:8b", temperature=0)

# ── State ──────────────────────────────────────────────────
class ResearchState(TypedDict):
    topic: str
    search_results: Annotated[list, operator.add]
    draft: str
    feedback: str
    final_report: str
    iteration: int
    status: str

# ── Agents ─────────────────────────────────────────────────
def web_searcher(state: ResearchState) -> ResearchState:
    st.session_state.logs.append("🔍 Web Searcher: searching...")
    results = []
    with DDGS() as ddgs:
        try:
            hits = list(ddgs.text(state["topic"], max_results=3))
            for r in hits:
                results.append(f"[WEB] {r['title']}: {r['body'][:250]}")
        except Exception as e:
            results.append(f"Search failed: {e}")
    st.session_state.logs.append(f"   Found {len(results)} results")
    return {**state, "search_results": results, "status": "searching"}

def tech_researcher(state: ResearchState) -> ResearchState:
    st.session_state.logs.append("🔬 Tech Researcher: deep diving...")
    results = []
    with DDGS() as ddgs:
        try:
            hits = list(ddgs.text(f"{state['topic']} technical", max_results=3))
            for r in hits:
                results.append(f"[TECH] {r['title']}: {r['body'][:250]}")
        except Exception as e:
            results.append(f"Search failed: {e}")
    st.session_state.logs.append(f"   Found {len(results)} technical results")
    return {**state, "search_results": results, "status": "researching"}

def writer(state: ResearchState) -> ResearchState:
    st.session_state.logs.append("✍️  Writer: drafting report...")
    all_results = "\n".join(state["search_results"])
    feedback_section = f"\nAddress this feedback:\n{state['feedback']}" if state.get("feedback") else ""

    prompt = f"""Write a comprehensive markdown report on: {state['topic']}

Sources:
{all_results}
{feedback_section}

Format:
# {state['topic']}
## Overview
## Key Findings
## Technical Details
## Practical Applications  
## Conclusion
*Generated: {datetime.datetime.now().strftime('%B %d, %Y')}*"""

    draft = llm.invoke(prompt)
    st.session_state.logs.append(f"   Draft: {len(draft)} chars")
    return {**state, "draft": draft, "status": "writing"}

def reviewer(state: ResearchState) -> ResearchState:
    st.session_state.logs.append("🔎 Reviewer: reviewing...")
    prompt = f"""Review this report. Reply ONLY with:
APPROVED - if good
REVISION NEEDED: [issues] - if not

{state['draft']}"""

    feedback = llm.invoke(prompt)
    approved = "APPROVED" in feedback.upper()
    st.session_state.logs.append(f"   {'✅ Approved' if approved else '❌ Needs revision'}")
    return {
        **state,
        "feedback": feedback,
        "iteration": state.get("iteration", 0) + 1,
        "status": "approved" if approved else "needs_revision"
    }

def finalizer(state: ResearchState) -> ResearchState:
    st.session_state.logs.append("💾 Finalizer: saving...")
    final = state["draft"]
    final += f"\n\n---\n*Iterations: {state.get('iteration', 1)}*"
    filename = f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(filename, "w") as f:
        f.write(final)
    st.session_state.logs.append(f"   Saved: {filename}")
    return {**state, "final_report": final, "status": "complete"}

def reviewer_router(state: ResearchState) -> str:
    if state.get("iteration", 0) >= 2:
        return "finalize"
    if "APPROVED" in state.get("feedback", "").upper():
        return "finalize"
    return "revise"

# ── Build Graph ────────────────────────────────────────────
@st.cache_resource
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
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# ── Streamlit UI ───────────────────────────────────────────
st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 AI Research Agent")
st.caption("Powered by LangGraph + Llama 3.1 + DuckDuckGo — 100% local")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    model = st.selectbox("Model", ["llama3.1:8b", "llama3.2"])
    max_iter = st.slider("Max iterations", 1, 4, 2)
    st.divider()
    st.header("📋 Session")
    session_id = st.text_input("Session ID", value="default")
    st.divider()
    st.header("💡 Example Topics")
    examples = [
        "Android AI integration 2026",
        "Kotlin Multiplatform vs Flutter",
        "On-device ML with ML Kit",
        "LangGraph for mobile developers",
        "RAG pipeline best practices",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.topic_input = ex
            st.rerun()

# Main area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Research Topic")
    if "topic_input" not in st.session_state:
        st.session_state.topic_input = ""

    topic = st.text_input(
        "Enter topic",
        value=st.session_state.topic_input,
        placeholder="e.g. Android AI integration trends 2026"
    )
    run_btn = st.button("🚀 Start Research", type="primary", use_container_width=True)

    # Agent log
    st.subheader("🤖 Agent Activity")
    log_container = st.container(height=300)

with col2:
    st.subheader("📄 Generated Report")
    report_container = st.empty()

# Run agent
if run_btn and st.session_state.topic_input:
    topic = st.session_state.topic_input
    if "logs" not in st.session_state:
        st.session_state.logs = []
    st.session_state.logs = [f"Starting research: {topic}"]

    app = build_graph()
    config = {"configurable": {"thread_id": session_id}}

    initial_state = {
        "topic": topic,
        "search_results": [],
        "draft": "",
        "feedback": "",
        "final_report": "",
        "iteration": 0,
        "status": "starting"
    }

    with st.spinner("Agents working..."):
        try:
            result = app.invoke(initial_state, config=config)

            # Show logs
            with log_container:
                for log in st.session_state.logs:
                    st.text(log)

            # Show report
            with report_container.container():
                st.markdown(result["final_report"])
                st.download_button(
                    "⬇️ Download Report",
                    data=result["final_report"],
                    file_name=f"report_{topic[:30].replace(' ','_')}.md",
                    mime="text/markdown"
                )
            st.success(f"✅ Done in {result['iteration']} iteration(s)")

        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state.logs.append(f"❌ Error: {e}")