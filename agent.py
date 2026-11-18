"""Supervisor-with-workers agent CLI in Python using LangChain and LangGraph."""

import sys
import warnings
from typing import TypedDict
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

from tools import wiki_summary

# Suppress deprecation warnings for cleaner CLI experience
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SUPERVISOR_PROMPT = """You are a supervisor managing a team of workers:
  researcher -> finds facts (Wikipedia)
  writer     -> turns the facts into a short article
  proofreader-> fixes grammar and polishes the article
Look at the progress so far and decide who works NEXT.
Typical order: researcher, then writer, then proofreader, then FINISH.
Answer with EXACTLY ONE word: researcher, writer, proofreader, or FINISH."""

RESEARCHER_PROMPT = "You are a researcher. Use wiki_summary to get facts about the topic. Report the key facts as bullet points."
WRITER_PROMPT = "You are a writer. Using the research notes, write a clear 80-word article. Return only the article."
PROOFREADER_PROMPT = "You are a proofreader. Fix grammar and polish this article without changing its meaning. Return only the final article."

# ---------------------------------------------------------------------------
# State & Configuration
# ---------------------------------------------------------------------------
MAX_STEPS = 6


class AgentState(TypedDict):
    task: str
    notes: str
    article: str
    next: str
    steps: int


# LLM instance (Ollama llama3.2, temperature=0)
llm = ChatOllama(model="llama3.2", temperature=0)

# Prebuilt react agent for researcher
research_agent = create_react_agent(llm, [wiki_summary], prompt=RESEARCHER_PROMPT)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def supervisor_node(state: AgentState) -> dict:
    steps = state.get("steps", 0) + 1

    # Guard rail: hard stop against infinite loops
    if steps >= MAX_STEPS:
        decision = "FINISH"
        print(f"  [supervisor] -> {decision}")
        return {"next": decision, "steps": steps}

    notes = state.get("notes", "").strip()
    article = state.get("article", "").strip()
    task = state.get("task", "").strip()

    prompt = (
        f"{SUPERVISOR_PROMPT}\n\n"
        f"Topic: {task}\n"
        f"Notes:\n{notes if notes else 'None'}\n\n"
        f"Article:\n{article if article else 'None'}\n\n"
        f"Decision:"
    )

    try:
        response = llm.invoke(prompt)
        raw_text = response.content.strip() if response and response.content else ""
        first_word = raw_text.split()[0].strip(".,;:!? ") if raw_text else ""
    except Exception:
        first_word = ""

    # GUARD RAILS (small models slip):
    # no notes -> researcher;
    # no article -> writer;
    # article not proofread -> proofreader;
    # proofread -> FINISH;
    # steps >= MAX_STEPS(6) -> FINISH
    if not notes:
        decision = "researcher"
    elif not article:
        decision = "writer"
    elif "(proofread ✔)" not in article:
        decision = "proofreader"
    else:
        decision = "FINISH"

    print(f"  [supervisor] -> {decision}")
    return {"next": decision, "steps": steps}


def researcher_node(state: AgentState) -> dict:
    task = state.get("task", "")
    message_content = f"Research the following topic and extract key facts: {task}"
    result = research_agent.invoke({"messages": [("user", message_content)]})
    messages = result.get("messages", [])
    if messages:
        content = messages[-1].content
        if isinstance(content, list):
            notes = "\n".join(str(item) for item in content)
        else:
            notes = str(content)
    else:
        notes = "No facts found."
    return {"notes": notes}


def writer_node(state: AgentState) -> dict:
    task = state.get("task", "")
    notes = state.get("notes", "")
    prompt = (
        f"{WRITER_PROMPT}\n\n"
        f"Topic: {task}\n\n"
        f"Research Notes:\n{notes}\n\n"
        f"Article:"
    )
    response = llm.invoke(prompt)
    article = response.content.strip() if response and response.content else ""
    return {"article": article}


def proofreader_node(state: AgentState) -> dict:
    task = state.get("task", "")
    article = state.get("article", "")
    prompt = (
        f"{PROOFREADER_PROMPT}\n\n"
        f"Topic: {task}\n\n"
        f"Draft Article:\n{article}\n\n"
        f"Final Proofread Article:"
    )
    response = llm.invoke(prompt)
    polished = response.content.strip() if response and response.content else article
    final_article = f"{polished}\n\n(proofread ✔)"
    return {"article": final_article}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def route_supervisor(state: AgentState) -> str:
    next_step = state.get("next", "FINISH")
    if next_step in ("researcher", "writer", "proofreader"):
        return next_step
    return END


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------
def build_agent():
    graph = StateGraph(AgentState)

    # Hub node & worker nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("proofreader", proofreader_node)

    # Start at supervisor
    graph.add_edge(START, "supervisor")

    # Supervisor conditional edges to workers or END
    graph.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "researcher": "researcher",
            "writer": "writer",
            "proofreader": "proofreader",
            END: END,
        },
    )

    # Workers report back to supervisor (hub-and-spoke)
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer", "supervisor")
    graph.add_edge("proofreader", "supervisor")

    return graph.compile()


# ---------------------------------------------------------------------------
# Execution & CLI
# ---------------------------------------------------------------------------
def ask(topic: str, app=None) -> str:
    if app is None:
        app = build_agent()

    initial_state: AgentState = {
        "task": topic,
        "notes": "",
        "article": "",
        "next": "",
        "steps": 0,
    }

    result = app.invoke(initial_state)
    return result.get("article", "")


def main():
    print("=" * 60)
    print("Supervisor Agent CLI (LangChain + LangGraph + Ollama)")
    print("Type 'quit', 'exit', or 'q' to stop.")
    print("=" * 60)

    app = build_agent()

    while True:
        try:
            topic = input("\nTopic: ").strip()
            if not topic:
                continue
            if topic.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            article = ask(topic, app=app)
            print(f"\nArticle:\n{article}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    main()
