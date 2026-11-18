"""FastAPI web server with built-in responsive web interface.

Run with:
    python server.py
    or
    uvicorn server:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from agent import build_agent, AgentState

app = FastAPI(title="Supervisor Agent API")
agent_app = build_agent()


class TopicRequest(BaseModel):
    topic: str


class GenerateResponse(BaseModel):
    topic: str
    article: str
    notes: str
    steps: int


HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Supervisor Multi-Agent Hub</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-50 min-h-screen text-slate-800 flex flex-col items-center p-6">
  <div class="max-w-3xl w-full bg-white shadow-xl rounded-2xl p-8 border border-slate-200 mt-6">
    <div class="flex items-center space-x-3 mb-2">
      <span class="text-3xl">🤖</span>
      <h1 class="text-2xl font-bold text-slate-900">Supervisor Agent Hub</h1>
    </div>
    <p class="text-slate-500 mb-6">Orchestrating Researcher, Writer, and Proofreader agents powered by Ollama llama3.2.</p>

    <!-- Input Section -->
    <div class="flex gap-3 mb-4">
      <input id="topicInput" type="text" placeholder="Enter a topic (e.g. Black holes, Mars, Artificial Intelligence)..." 
             class="flex-1 px-4 py-2.5 rounded-lg border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500">
      <button id="runBtn" onclick="runAgent()" 
              class="bg-blue-600 hover:bg-blue-700 text-white font-medium px-6 py-2.5 rounded-lg transition duration-150">
        Run Agent
      </button>
    </div>

    <!-- Quick pills -->
    <div class="flex gap-2 mb-6 flex-wrap">
      <button onclick="setTopic('Black holes')" class="text-xs bg-slate-100 hover:bg-slate-200 px-3 py-1 rounded-full text-slate-700">🌌 Black holes</button>
      <button onclick="setTopic('Sun')" class="text-xs bg-slate-100 hover:bg-slate-200 px-3 py-1 rounded-full text-slate-700">☀️ Sun</button>
      <button onclick="setTopic('Artificial intelligence')" class="text-xs bg-slate-100 hover:bg-slate-200 px-3 py-1 rounded-full text-slate-700">🧠 AI</button>
      <button onclick="setTopic('James Webb Space Telescope')" class="text-xs bg-slate-100 hover:bg-slate-200 px-3 py-1 rounded-full text-slate-700">🔭 JWST</button>
    </div>

    <!-- Status loading -->
    <div id="loader" class="hidden mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg text-blue-700 flex items-center space-x-3">
      <svg class="animate-spin h-5 w-5 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
      </svg>
      <span>Supervisor is coordinating workers: <strong>Researcher → Writer → Proofreader</strong>...</span>
    </div>

    <!-- Output Section -->
    <div id="results" class="hidden space-y-4">
      <div>
        <h3 class="font-semibold text-slate-900 mb-2">Final Article</h3>
        <div id="articleBox" class="p-4 bg-slate-50 border-l-4 border-green-500 rounded whitespace-pre-line text-slate-800 leading-relaxed font-sans"></div>
      </div>

      <details class="group bg-slate-50 rounded-lg p-3 border border-slate-200">
        <summary class="font-medium cursor-pointer text-slate-700">View Researcher Notes</summary>
        <div id="notesBox" class="mt-2 text-sm text-slate-600 whitespace-pre-line border-t border-slate-200 pt-2"></div>
      </details>
    </div>
  </div>

  <script>
    function setTopic(val) {
      document.getElementById('topicInput').value = val;
    }

    async function runAgent() {
      const topic = document.getElementById('topicInput').value.trim();
      if (!topic) return alert('Please enter a topic');

      const loader = document.getElementById('loader');
      const results = document.getElementById('results');
      const runBtn = document.getElementById('runBtn');

      loader.classList.remove('hidden');
      results.classList.add('hidden');
      runBtn.disabled = true;
      runBtn.classList.add('opacity-50');

      try {
        const response = await fetch('/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ topic })
        });
        const data = await response.json();

        document.getElementById('articleBox').textContent = data.article;
        document.getElementById('notesBox').textContent = data.notes;
        results.classList.remove('hidden');
      } catch (err) {
        alert('Error: ' + err.message);
      } finally {
        loader.classList.add('hidden');
        runBtn.disabled = false;
        runBtn.classList.remove('opacity-50');
      }
    }
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


@app.post("/generate", response_model=GenerateResponse)
def generate(req: TopicRequest):
    topic = req.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Topic cannot be empty")

    initial_state: AgentState = {
        "task": topic,
        "notes": "",
        "article": "",
        "next": "",
        "steps": 0,
    }

    result = agent_app.invoke(initial_state)
    return GenerateResponse(
        topic=topic,
        article=result.get("article", ""),
        notes=result.get("notes", ""),
        steps=result.get("steps", 0),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
