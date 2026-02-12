import os
import json
import time
import traceback

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
    question: str = Field(min_length=1, max_length=800)
    top_k: int = 5


@app.get("/", response_class=HTMLResponse)
def home():
    demo_key = os.getenv("DEMO_KEY", "")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SME AI Adoption Assistant (RAG demo)</title>
  <style>
    :root {{
      --bg: #0b1220;
      --card: rgba(255,255,255,0.06);
      --text: rgba(255,255,255,0.92);
      --muted: rgba(255,255,255,0.66);
      --line: rgba(255,255,255,0.12);
      --accent: #7c3aed;
      --accent2: #22c55e;
      --danger: #ef4444;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
      background: radial-gradient(1200px 600px at 20% 10%, rgba(124,58,237,0.25), transparent 55%),
                  radial-gradient(900px 500px at 80% 30%, rgba(34,197,94,0.18), transparent 55%),
                  var(--bg);
      color: var(--text);
    }}
    .wrap {{
      max-width: 980px;
      margin: 48px auto;
      padding: 0 18px;
    }}
    .top {{
      display: flex;
      gap: 14px;
      align-items: flex-start;
      justify-content: space-between;
      margin-bottom: 16px;
    }}
    .title {{
      font-size: 34px;
      letter-spacing: -0.02em;
      margin: 0 0 8px 0;
    }}
    .subtitle {{
      margin: 0;
      color: var(--muted);
      line-height: 1.4;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(255,255,255,0.04);
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
      height: fit-content;
    }}
    .dot {{
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: var(--accent2);
      box-shadow: 0 0 18px rgba(34,197,94,0.55);
    }}
    .card {{
      border: 1px solid var(--line);
      background: linear-gradient(180deg, var(--card), rgba(255,255,255,0.03));
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 18px 40px rgba(0,0,0,0.35);
    }}
    .row {{
      display: grid;
      grid-template-columns: 1fr 220px;
      gap: 14px;
      align-items: start;
    }}
    textarea {{
      width: 100%;
      min-height: 120px;
      resize: vertical;
      border-radius: 14px;
      border: 1px solid var(--line);
      padding: 12px 12px;
      background: rgba(0,0,0,0.18);
      color: var(--text);
      outline: none;
      font-size: 15px;
      line-height: 1.4;
    }}
    textarea:focus {{
      border-color: rgba(124,58,237,0.65);
      box-shadow: 0 0 0 4px rgba(124,58,237,0.16);
    }}
    .btns {{
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    button {{
      appearance: none;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.06);
      color: var(--text);
      border-radius: 14px;
      padding: 11px 12px;
      font-weight: 600;
      cursor: pointer;
      transition: 140ms ease;
    }}
    button:hover {{
      transform: translateY(-1px);
      border-color: rgba(124,58,237,0.55);
      background: rgba(124,58,237,0.10);
    }}
    .primary {{
      background: linear-gradient(135deg, rgba(124,58,237,0.85), rgba(124,58,237,0.45));
      border-color: rgba(124,58,237,0.85);
    }}
    .primary:hover {{
      background: linear-gradient(135deg, rgba(124,58,237,0.95), rgba(124,58,237,0.55));
    }}
    .ghost {{
      background: rgba(255,255,255,0.04);
    }}
    .examples {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 14px;
      padding-top: 14px;
      border-top: 1px solid var(--line);
    }}
    .chip {{
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.04);
      color: var(--muted);
      padding: 8px 10px;
      border-radius: 999px;
      font-size: 13px;
      cursor: pointer;
      transition: 140ms ease;
      user-select: none;
    }}
    .chip:hover {{
      border-color: rgba(34,197,94,0.55);
      color: var(--text);
      background: rgba(34,197,94,0.10);
    }}
    .out {{
      margin-top: 14px;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: rgba(0,0,0,0.20);
      padding: 14px;
      min-height: 120px;
    }}
    .status {{
      color: var(--muted);
      font-size: 13px;
      margin: 10px 0 0 0;
    }}
    .section-title {{
      margin: 0 0 6px 0;
      font-size: 13px;
      letter-spacing: .08em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .answer {{
      white-space: pre-wrap;
      line-height: 1.55;
      font-size: 15px;
    }}
    .meta {{
      margin-top: 12px;
      display: grid;
      gap: 10px;
    }}
    .box {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      background: rgba(255,255,255,0.04);
      color: var(--muted);
      font-size: 13px;
      white-space: pre-wrap;
    }}
    .error {{
      border-color: rgba(239,68,68,0.55);
      background: rgba(239,68,68,0.10);
      color: rgba(255,255,255,0.9);
    }}
    .footer {{
      margin-top: 14px;
      color: rgba(255,255,255,0.5);
      font-size: 12px;
      line-height: 1.4;
    }}
    code {{
      background: rgba(255,255,255,0.07);
      padding: 2px 6px;
      border-radius: 8px;
      border: 1px solid var(--line);
    }}
    @media (max-width: 860px) {{
      .row {{ grid-template-columns: 1fr; }}
      .btns {{ flex-direction: row; flex-wrap: wrap; }}
      button {{ flex: 1; min-width: 160px; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="top">
      <div>
        <h1 class="title">SME AI Adoption Assistant <span style="opacity:.75">(RAG demo)</span></h1>
        <p class="subtitle">Ask a question about AI adoption & automation for small businesses. Answers are grounded in a small FAQ knowledge base and return cited sources.</p>
      </div>
      <div class="badge"><span class="dot"></span> Live demo</div>
    </div>

    <div class="card">
      <div class="row">
        <div>
          <textarea id="q" placeholder="Type your question... e.g. 'What is the safest first AI use case for a small company?'"></textarea>
          <div class="examples">
            <div class="chip" onclick="setQ('What’s the safest way to start using AI in a small business?')">Safe start</div>
            <div class="chip" onclick="setQ('How do we choose the first AI use case for automation?')">First use case</div>
            <div class="chip" onclick="setQ('What are typical risks of AI adoption and how do we reduce them?')">Risks</div>
            <div class="chip" onclick="setQ('How do we measure ROI for AI automation in a practical way?')">ROI</div>
            <div class="chip" onclick="setQ('Do we need a vector database to start a small RAG assistant?')">Vector DB?</div>
          </div>
          <p class="status" id="status">Tip: try a question above, then review <code>Sources</code> and <code>Top matches</code> below.</p>
        </div>

        <div class="btns">
          <button class="primary" onclick="ask()">Ask</button>
          <button class="ghost" onclick="clearAll()">Clear</button>
          <button class="ghost" onclick="reindex()">Reindex FAQ</button>
          <button class="ghost" onclick="openDocs()">API docs</button>
        </div>
      </div>

      <div class="out" id="outBox">
        <div class="section-title">Result</div>
        <div id="answer" class="answer"></div>

        <div class="meta" id="meta" style="display:none;">
          <div>
            <div class="section-title">Sources</div>
            <div id="sources" class="box"></div>
          </div>
          <div>
            <div class="section-title">Top matches</div>
            <div id="matches" class="box"></div>
          </div>
        </div>

        <div class="footer">
          This is a demo assistant. Don’t paste confidential data. If the answer is not covered by the FAQ, it should say so.
        </div>
      </div>
    </div>
  </div>

<script>
const DEMO_KEY = "{demo_key}";

function setQ(text) {{
  document.getElementById('q').value = text;
}}

function clearAll() {{
  document.getElementById('q').value = "";
  document.getElementById('answer').textContent = "";
  document.getElementById('sources').textContent = "";
  document.getElementById('matches').textContent = "";
  document.getElementById('meta').style.display = "none";
  document.getElementById('status').textContent = "Cleared.";
  document.getElementById('outBox').classList.remove('error');
}}

function openDocs() {{
  window.open('/docs', '_blank');
}}

async function reindex() {{
  const status = document.getElementById('status');
  status.textContent = "Reindexing...";
  try {{
    const url = DEMO_KEY ? `/reindex?k=${{encodeURIComponent(DEMO_KEY)}}` : '/reindex';
    const res = await fetch(url, {{ method: 'POST' }});
    const text = await res.text();
    if (!res.ok) {{
      status.textContent = `Reindex failed (${{res.status}}).`;
      document.getElementById('answer').textContent = text;
      document.getElementById('outBox').classList.add('error');
      return;
    }}
    document.getElementById('outBox').classList.remove('error');
    status.textContent = "Reindexed OK.";
  }} catch (e) {{
    status.textContent = "Reindex request failed.";
    document.getElementById('answer').textContent = String(e);
    document.getElementById('outBox').classList.add('error');
  }}
}}

async function ask() {{
  const q = document.getElementById('q').value.trim();
  const status = document.getElementById('status');
  const answerEl = document.getElementById('answer');
  const metaEl = document.getElementById('meta');
  const outBox = document.getElementById('outBox');

  answerEl.textContent = "";
  metaEl.style.display = "none";
  outBox.classList.remove('error');

  if (!q) {{
    status.textContent = "Type a question first.";
    return;
  }}

  status.textContent = "Thinking...";
  answerEl.textContent = "Loading...";

  try {{
    const url = DEMO_KEY ? `/ask?k=${{encodeURIComponent(DEMO_KEY)}}` : '/ask';
    const res = await fetch(url, {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ question: q, top_k: 5 }})
    }});

    const text = await res.text();

    if (!res.ok) {{
      status.textContent = `Error (${{res.status}}).`;
      answerEl.textContent = text;
      outBox.classList.add('error');
      return;
    }}

    const data = JSON.parse(text);

    answerEl.textContent = data.answer || "(empty answer)";

    const sources = (data.sources || []).join(", ");
    document.getElementById('sources').textContent = sources || "(none)";

    const matches = (data.top_matches || []).map(m =>
      `• [${{m.id}}] score=${{m.score}}\\n  Q: ${{m.question}}`
    ).join("\\n\\n");
    document.getElementById('matches').textContent = matches || "(none)";

    metaEl.style.display = "grid";
    status.textContent = "Done.";
  }} catch (e) {{
    status.textContent = "Request failed.";
    answerEl.textContent = String(e);
    outBox.classList.add('error');
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
        print("ASK ERROR:\n" + "".join(traceback.format_exception(type(e), e, e.__traceback__)))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reindex")
def reindex(k: str = Query(default="")):
    if DEMO_KEY and k != DEMO_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return get_rag().reindex()


@app.get("/health")
def health():
    return {"status": "ok"}
