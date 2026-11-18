# Supervisor Agent

A hub-and-spoke multi-agent system built using **LangChain**, **LangGraph**, and local **Ollama (`llama3.2`)**.

---

## Architecture

```
                    +--------------------+
                    |     START          |
                    +---------+----------+
                              |
                              v
                    +---------+----------+
      +------------>|    supervisor      |<------------+
      |             +----+----+----+-----+             |
      |                  |    |    |                   |
      |       researcher |    |    | proofreader       |
      |                  |    |    |                   |
      |                  v    |    v                   |
+-----+--------+   +-----+--+ |  +-+------------+      |
|  researcher  |   | writer | |  | proofreader  |      |
| (ReAct Agent)|   +----+---+ |  +-------+------+      |
| [wiki_summary]        |     |          |             |
+--------------+        |     | writer   |             |
      |                 +-----+          |             |
      +----------------------------------+             |
                              |                        |
                              | FINISH                 |
                              v                        |
                           +--+--+                     |
                           | END |                     |
                           +-----+                     |
```

### Components

1. **Supervisor**: Inspects the shared state and determines the next worker (`researcher`, `writer`, `proofreader`, or `FINISH`).
   - Guard rails guarantee logical flow and prevent model slips.
   - Enforces `MAX_STEPS = 6` hard stop against infinite loops.
2. **Researcher**: LangGraph ReAct agent powered by `wiki_summary` (Wikipedia REST API) to gather facts.
3. **Writer**: Turns notes into a concise ~80-word article draft.
4. **Proofreader**: Polishes the draft and marks it with `(proofread ✔)`.

---

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) running locally with `llama3.2`:
  ```bash
  ollama run llama3.2
  ```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Ways to Run

### 1. Interactive Web Dashboard (Streamlit - Recommended)
Runs a full interactive web app with live progress bars, supervisor decision timelines, and expandable notes:
```bash
streamlit run app.py
```
Open your browser at: `http://localhost:8501`

### 2. Web Server & API (FastAPI)
Runs a lightweight web server with a built-in clean HTML/Tailwind frontend and a `POST /generate` REST endpoint:
```bash
python server.py
```
Open your browser at: `http://localhost:8000`

### 3. Terminal CLI
Runs directly in your terminal:
```bash
python agent.py
```
Type any topic, or type `quit` / `q` to exit.
