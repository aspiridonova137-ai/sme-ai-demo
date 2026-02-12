import os
import json
import numpy as np
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def read_text_with_fallback(path: str) -> str:
    encodings = ("utf-8", "utf-8-sig", "cp1252", "cp1250", "cp1251")
    last = None
    for enc in encodings:
        try:
            with open(path, encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError as e:
            last = e
    raise RuntimeError(f"Cannot decode {path}. Last error: {last}")

def parse_faq_txt(path: str):
    """
    Format: blocks separated by blank line:
    Q: ...
    A: ...
    """
    text = read_text_with_fallback(path).strip()
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]

    rows = []
    idx = 1
    for b in blocks:
        q, a = "", ""
        for line in [l.strip() for l in b.splitlines() if l.strip()]:
            low = line.lower()
            if low.startswith("q:"):
                q = line[2:].strip()
            elif low.startswith("a:"):
                a = (a + " " + line[2:].strip()).strip()
            else:
                if a:
                    a += " " + line

        if q and a:
            rows.append({
                "id": str(idx),
                "question": q,
                "answer": a,
                "text": f"Q: {q}\nA: {a}"
            })
            idx += 1

    if not rows:
        raise ValueError("FAQ parsed as empty. Check faq.txt format (Q:/A: blocks separated by blank lines).")
    return rows

class RAGAssistant:
    def __init__(self, faq_path: str = "faq.txt", index_path: str = "embeddings.json"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=api_key)
        self.faq_path = faq_path
        self.index_path = index_path

    def _embed(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=EMBED_MODEL, input=texts)
        return [d.embedding for d in resp.data]

    def build_or_load_index(self):
        if os.path.exists(self.index_path):
            with open(self.index_path, encoding="utf-8") as f:
                return json.load(f)

        rows = parse_faq_txt(self.faq_path)
        vectors = self._embed([r["text"] for r in rows])

        data = []
        for r, v in zip(rows, vectors):
            data.append({
                "id": r["id"],
                "question": r["question"],
                "answer": r["answer"],
                "embedding": v
            })

        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        return data

    def retrieve(self, query: str, top_k: int = 5):
        index = self.build_or_load_index()
        q_vec = np.array(self._embed([query])[0], dtype=np.float32)

        scored = []
        for item in index:
            v = np.array(item["embedding"], dtype=np.float32)
            scored.append((cosine(q_vec, v), item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def answer(self, query: str, top_k: int = 5):
        top = self.retrieve(query, top_k=top_k)

        context_blocks = []
        sources = []
        top_matches = []
        for score, item in top:
            sources.append(item["id"])
            top_matches.append({
                "id": item["id"],
                "score": round(float(score), 4),
                "question": item["question"],
                "answer": item["answer"]
            })
            context_blocks.append(f"[FAQ {item['id']}] Q: {item['question']}\nA: {item['answer']}")

        context = "\n\n".join(context_blocks)

        system_msg = (
            "You are an AI Adoption & Automation assistant for SMEs. "
            "Answer ONLY using the provided FAQ context. "
            "If the context is insufficient, say: "
            "'I don't have that information in the FAQ yet.' "
            "Be concise, practical, and business-oriented."
        )
        user_msg = (
            f"User question: {query}\n\n"
            f"FAQ context:\n{context}\n\n"
            "Return the final answer in plain text."
        )

        completion = self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )

        final_answer = completion.choices[0].message.content.strip()

        return {
            "answer": final_answer,
            "sources": [f"FAQ {i}" for i in sources],
            "top_matches": top_matches
        }
