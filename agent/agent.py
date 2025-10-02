"""
Run: python agent.py
Then type your question at the prompt.
If OPENAI_API_KEY is set, the agent will use OpenAI (gpt-3.5-turbo). Otherwise it will synthesize from retrieved text.
Uses sentence-transformers/all-MiniLM-L6-v2 for embeddings.
"""

import os
import math
import numpy as np
from typing import List, Tuple

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    print("Missing dependency: sentence-transformers. Install with: pip install sentence-transformers")
    raise e

USE_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))
if USE_OPENAI:
    try:
        import openai
        openai.api_key = os.environ["OPENAI_API_KEY"]
    except Exception as e:
        print("OpenAI not available. Falling back to local synthesis.")
        USE_OPENAI = False

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def load_docs(folder: str = "docs") -> dict:
    docs = {}
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        if os.path.isfile(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                docs[fname] = f.read()
    return docs


def chunk_text(text: str, max_chars: int = 800) -> List[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    sentences = text.replace("\n", " ").split(". ")
    chunks, cur = [], ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        piece = (s + ("" if s.endswith(".") else "."))
        if len(cur) + len(piece) <= max_chars:
            cur += (" " if cur else "") + piece
        else:
            if cur:
                chunks.append(cur.strip())
            cur = piece
    if cur:
        chunks.append(cur.strip())
    return chunks


class RAGAgent:
    def __init__(self, docs: dict):
        self.model = SentenceTransformer(MODEL_NAME)
        self.docs = docs
        self.chunks = []
        self.embeddings = None
        self._build_index()

    def _build_index(self):
        idx = 0
        for fname, text in self.docs.items():
            for c in chunk_text(text):
                self.chunks.append({"id": idx, "text": c, "source": fname})
                idx += 1
        texts = [c["text"] for c in self.chunks]
        if texts:
            embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10)
            self.embeddings = embs
        else:
            self.embeddings = np.zeros((0, EMBEDDING_DIM))

    def _cosine_search(self, query: str, top_k: int = 3) -> List[Tuple[float, dict]]:
        q_emb = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-10)
        sims = (self.embeddings @ q_emb.T).squeeze(axis=1)
        idxs = np.argsort(-sims)[:top_k]
        return [(float(sims[i]), self.chunks[i]) for i in idxs if not math.isnan(float(sims[i]))]

    def answer(self, question: str, top_k: int = 3) -> str:
        """
        This function will search top_k related files and generate the CONTEXT based on the user question.
        It will use the fixed format of out put if the openai raise any exception.
        """
        retrieved = self._cosine_search(question, top_k=top_k)
        if not retrieved:
            return "No relevant information found in documents."
        sources = []
        concat_parts = []
        for score, chunk in retrieved:
            concat_parts.append(f"Source: {chunk['source']}\n{chunk['text']}")
            sources.append(chunk['source'])
        sources = list(dict.fromkeys(sources))
        context = "\n\n---\n\n".join(concat_parts)
        if USE_OPENAI:
            prompt = (
                "You are a helpful assistant. Use only the provided CONTEXT to answer the question. "
                "If the answer is not contained in the CONTEXT, say you don't know. "
                "At the end, list the source file names like: SOURCES: [file1, file2]\n\n"
                f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nAnswer concisely."
            )
            try:
                resp = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=512
                )
                return resp["choices"][0]["message"]["content"].strip()
            except Exception:
                pass
        answer = "Based on the documents provided:\n\n"
        for i, (score, chunk) in enumerate(retrieved, 1):
            answer += f"{i}. ({chunk['source']}) {chunk['text']}\n\n"
        answer += "SOURCES: " + ", ".join(sources)
        return answer


def main():
    docs = load_docs("docs")
    if not docs:
        print("No documents found in ./docs folder.")
        return
    agent = RAGAgent(docs)
    print("Mini RAG agent ready. Type your question (Ctrl+C to exit).")
    try:
        while True:
            q = input("\nQuestion: ").strip()
            if not q:
                print("Please type a question.")
                continue
            if q.lower() in ['exit','quit']:
                print('Successfully Exited.')
                break
            resp = agent.answer(q, top_k=3)
            print("\nAnswer:\n")
            print(resp)
            print('Ask your next question or enter Exit')
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    main()
