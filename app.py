"""Streamlit Web UI for Supervisor Agent.

Run with:
    streamlit run app.py
"""

import streamlit as st
from agent import build_agent, AgentState

st.set_page_config(
    page_title="Supervisor Agent Hub",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for clean, professional look
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #6c757d;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .step-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 0.5rem;
        margin-bottom: 0.4rem;
        background-color: #e9ecef;
    }
    .article-box {
        background-color: #f8f9fa;
        border-left: 4px solid #198754;
        padding: 1.2rem;
        border-radius: 4px;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🤖 Team Settings")
    st.markdown("### System Architecture")
    st.info(
        "**Hub-and-Spoke Architecture:**\n"
        "- 👑 **Supervisor** (Decision Maker)\n"
        "- 🔍 **Researcher** (Wikipedia Tool)\n"
        "- ✍️ **Writer** (~80-word article)\n"
        "- 📝 **Proofreader** (Grammar & Polish)"
    )

    st.markdown("---")
    st.markdown("### Model & Stack")
    st.markdown("- **LLM**: Ollama `llama3.2` (temp=0)")
    st.markdown("- **Orchestration**: LangGraph StateGraph")
    st.markdown("- **API**: Wikipedia REST API")

    st.markdown("---")
    st.caption("Ensure Ollama is running locally: `ollama run llama3.2`")


# ---------------------------------------------------------------------------
# App Initialization
# ---------------------------------------------------------------------------
@st.cache_resource
def get_agent():
    return build_agent()


try:
    agent_app = get_agent()
except Exception as e:
    st.error(f"Failed to initialize agent graph: {e}")
    agent_app = None

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">Supervisor Agent Team</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Watch a supervisor LLM orchestrate researcher, writer, and proofreader agents in real time.</div>',
    unsafe_allow_html=True,
)

# Quick sample topic buttons
col_quick1, col_quick2, col_quick3, col_quick4 = st.columns(4)
sample_topic = None
if col_quick1.button("🌌 Black holes"):
    sample_topic = "Black holes"
if col_quick2.button("☀️ The Sun"):
    sample_topic = "Sun"
if col_quick3.button("🧠 Artificial intelligence"):
    sample_topic = "Artificial intelligence"
if col_quick4.button("🔭 James Webb Telescope"):
    sample_topic = "James Webb Space Telescope"

# User topic input
topic = st.text_input(
    "Enter a topic for the agent team:",
    value=sample_topic if sample_topic else "",
    placeholder="e.g., Quantum computing, Aurora borealis, Solar energy...",
)

run_button = st.button("🚀 Run Agent Team", type="primary", use_container_width=True)

if run_button:
    if not topic.strip():
        st.warning("Please enter a topic to begin.")
    elif agent_app is None:
        st.error("Agent could not be loaded. Please check your Ollama installation.")
    else:
        st.markdown("---")
        st.subheader("⚡ Live Multi-Agent Workflow")

        status_container = st.container()
        notes_expander = st.expander("🔍 Research Notes (Wikipedia)", expanded=False)
        draft_expander = st.expander("✍️ Draft Article (Writer)", expanded=False)
        result_container = st.container()

        state: AgentState = {
            "task": topic.strip(),
            "notes": "",
            "article": "",
            "next": "",
            "steps": 0,
        }

        progress_bar = st.progress(0, text="Supervisor analyzing task...")
        step_logs = []

        try:
            current_notes = ""
            current_draft = ""
            current_final = ""

            for step_output in agent_app.stream(state):
                for node_name, node_state in step_output.items():
                    if node_name == "supervisor":
                        decision = node_state.get("next", "")
                        step_num = node_state.get("steps", 0)
                        step_logs.append(f"**Step {step_num}**: `[supervisor] -> {decision}`")

                        if decision == "researcher":
                            progress_bar.progress(25, text="Supervisor assigned task to **Researcher**...")
                        elif decision == "writer":
                            progress_bar.progress(50, text="Supervisor assigned task to **Writer**...")
                        elif decision == "proofreader":
                            progress_bar.progress(75, text="Supervisor assigned task to **Proofreader**...")
                        elif decision == "FINISH":
                            progress_bar.progress(100, text="Supervisor confirmed task is **FINISH**ed! 🎉")

                    elif node_name == "researcher":
                        current_notes = node_state.get("notes", "")
                        with notes_expander:
                            st.markdown(current_notes)

                    elif node_name == "writer":
                        current_draft = node_state.get("article", "")
                        with draft_expander:
                            st.markdown(current_draft)

                    elif node_name == "proofreader":
                        current_final = node_state.get("article", "")

            # Display supervisor decisions
            with status_container:
                st.markdown("#### Supervisor Decisions")
                cols = st.columns(len(step_logs))
                for idx, log in enumerate(step_logs):
                    with cols[idx]:
                        st.info(log)

            # Display final article
            with result_container:
                st.markdown("### 📄 Final Polished Article")
                if current_final:
                    st.markdown(
                        f"""
                        <div class="article-box">
                        {current_final.replace(chr(10), '<br>')}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning("No article was produced.")

        except Exception as ex:
            st.error(f"Execution error: {ex}")
            st.info("Check that Ollama is active (`ollama list`) and accessible.")
