import json, time
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from rag_core import RAGAssistant

load_dotenv()

app = FastAPI(title="SME AI Adoption Assistant (RAG)")
rag = None

def get_rag():
    global rag
    if rag is None:
        rag = RAGAssistant(faq_path="faq.txt", index_path="embeddings.json")
    return rag


class AskRequest(BaseModel):
    question: str
    top_k: int = 5

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!doctype html>
<html>
<head><meta charset="utf-8"><title>SME AI Adoption Assistant</title></head>
<body style="font-family: sans-serif; max-width: 900px; margin: 40px auto;">
  <h2>SME AI Adoption Assistant (RAG demo)</h2>
  <p>Ask a question about AI adoption & automation for small businesses.</p>
  <textarea id="q" rows="4" style="width: 100%;" placeholder="Type your question..."></textarea><br><br>
  <button onclick="ask()">Ask</button>
  <pre id="out" style="white-space: pre-wrap; background: #f6f6f6; padding: 12px; margin-top: 16px;"></pre>

<script>
async function ask() {
  const question = document.getElementById('q').value;
  const res = await fetch('/ask', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({question, top_k: 5})
  });
  const data = await res.json();
  document.getElementById('out').textContent =
    "ANSWER:\\n" + data.answer +
    "\\n\\nSOURCES: " + data.sources.join(", ") +
    "\\n\\nTOP MATCHES:\\n" + data.top_matches.map(m =>
      `- [${m.id}] score=${m.score} | ${m.question}`
    ).join("\\n");
}
</script>
</body>
</html>
"""

@app.post("/ask")
def ask(req: AskRequest):
  return get_rag().answer(req.question, top_k=req.top_k)


@app.get("/health")
def health():
    return {"status": "ok"}

