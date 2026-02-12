import os
import json
import time

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from rag_core import RAGAssistant

load_dotenv()

app = FastAPI(title="SME AI Adoption Assistant (RAG demo)")

# Optional access key for the demo (set DEMO_KEY in Render env vars)
DEMO_KEY = os.getenv("DEMO_KEY")

# Lazy init so the app can start even if OPENAI_API_KEY is not set yet
rag = None

def get_rag():
    global rag
    if rag is None:
        # Store index in /tmp for cloud environments
        rag = RAGAssistant(faq_path="faq.txt", index_path="/tmp/embeddings.json")
    return rag


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=600)
    top_k: int = 5


@app.get("/", response_class=HTMLResponse)
def home():
    # If you use DEMO_KEY, put it here so the UI calls /ask?k=...
    # For production, don't hardcode the key into the UI; use a better auth method.
    demo_key = os.getenv("DEMO_KEY", "")

    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>SME AI Adoption Assistant</title>
</head>
<body style="font-family: sans-serif; max-width: 900px; margin: 40px auto;">
  <h2>SME AI Adoption Assistant (RAG demo)</h2>
  <p>Ask a question about AI adoption & automation for small businesses.</p>

  <textarea id="q" rows="4" style="width: 100%;" placeholder="Type your question..."></textarea><br><br>
  <button onclick="ask()">Ask</button>

  <pre id="out" style="white-space: pre-wrap; background: #f6f6f6; padding: 12px; margin-top: 16px;"></pre>

<script>
async function ask() {{
  const out = document.getElementById('out');
  out.textContent = "Loading...";

  const question = document.getElementById('q').value;

  try {{
    const url = "/ask{('?k=' + demo_key) if demo_key else ''}";
    const res = await fetch(url, {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{question, top_k: 5}})
    }});

    const text = await res.text();

    if (!res.ok) {{
      out.textContent = `ERROR ${{res.status}}:\\n` + text;
      return;
    }}

    const data = JSON.parse(text);

    out.textContent =
      "ANSWER:\\n" + data.answer +
      "\\n\\nSOURCES: " + data.sources.join(", ") +
      "\\n\\nTOP MATCHES:\\n" + data.top_matches.map(m =>
        `- [${{m.id}}] score=${{m.score}} | ${{m.question}}`
      ).join("\\n");

  }} catch (e) {{
    out.textContent = "REQUEST FAILED:\\n" + e;
  }}
}}
</script>

</body>
</html>
"""


@app.post("/ask")
def ask(req: AskRequest, k: str = Query(default="")):
    # Optional demo protection
    if DEMO_KEY and k != DEMO_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        result = get_rag().answer(req.question, top_k=req.top_k)

        # Optional audit log (safe, no keys stored)
        log_line = {
            "ts": int(time.time()),
            "question": req.question,
            "sources": result.get("sources", [])
        }
        with open("/tmp/logs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_line, ensure_ascii=False) + "\n")

        return result

    except Exception as e:
        # Return a readable error during development; for production, remove details
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reindex")
def reindex(k: str = Query(default="")):
    if DEMO_KEY and k != DEMO_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return get_rag().reindex()


@app.get("/health")
def health():
    return {"status": "ok"}
