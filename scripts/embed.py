"""Embeds every page of each filing (as documents) and every question (as queries) with one model, on the same page
text as Page Search read (data/pages), the first MAX_CHARS characters of each page. Writes vectors/<model>/.

    python scripts/embed.py voyage    OPENROUTER_API_KEY: voyage-4-large through OpenRouter, input_type document/query
    python scripts/embed.py gemini    GCP_PROJECT and GCP_TOKEN (gcloud auth print-access-token): gemini-embedding-2
                                      on Vertex AI, task types RETRIEVAL_DOCUMENT and RETRIEVAL_QUERY
    python scripts/embed.py openai    OPENAI_API_KEY: text-embedding-3-large
"""
import concurrent.futures as cf
import json
import os
import sys
import time

import httpx
import numpy as np

from common import MAX_CHARS, VECTORS, pages, questions


def post(url, headers, body):
    for attempt in range(8):
        try:
            r = httpx.post(url, headers=headers, json=body, timeout=120)
            if r.status_code == 200:
                return r.json()
            if r.status_code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
        except httpx.TransportError:
            pass
        time.sleep(min(30, 2 ** attempt))
    raise RuntimeError("no answer after 8 attempts")


def batched(texts, batch_chars, call):
    """Batches of texts under batch_chars, sent four at a time; the vectors in order."""
    batches, cur, size = [], [], 0
    for t in texts:
        if cur and size + len(t) > batch_chars:
            batches.append(cur)
            cur, size = [], 0
        cur.append(t)
        size += len(t)
    batches += [cur] if cur else []
    with cf.ThreadPoolExecutor(4) as pool:
        return [v for vs in pool.map(call, batches) for v in vs]


def voyage(texts, kind):
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
    return batched(texts, 200_000, lambda b: [d["embedding"] for d in sorted(post(
        "https://openrouter.ai/api/v1/embeddings", headers,
        {"model": "voyageai/voyage-4-large", "input": b, "input_type": kind})["data"], key=lambda d: d["index"])])


def openai(texts, kind):
    headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
    return batched(texts, 400_000, lambda b: [d["embedding"] for d in sorted(post(
        "https://api.openai.com/v1/embeddings", headers,
        {"model": "text-embedding-3-large", "input": b})["data"], key=lambda d: d["index"])])


def gemini(texts, kind):
    url = (f"https://aiplatform.googleapis.com/v1/projects/{os.environ['GCP_PROJECT']}/locations/global/"
           "publishers/google/models/gemini-embedding-2:embedContent")
    headers = {"Authorization": f"Bearer {os.environ['GCP_TOKEN']}"}
    task = "RETRIEVAL_DOCUMENT" if kind == "document" else "RETRIEVAL_QUERY"
    with cf.ThreadPoolExecutor(16) as pool:  # one text a request
        return list(pool.map(lambda t: post(url, headers, {
            "content": {"parts": [{"text": t}]}, "taskType": task, "autoTruncate": True})["embedding"]["values"], texts))


def unit(v):
    v = np.array(v, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def main(model):
    embed = {"voyage": voyage, "gemini": gemini, "openai": openai}[model]
    out = VECTORS / model
    out.mkdir(parents=True, exist_ok=True)
    qs = questions()
    for filing in sorted({f for q in qs for f in q["files"]}):
        path = out / f"{filing}.npy"
        if not path.exists():
            np.save(path, unit(embed([t[:MAX_CHARS].strip() or "(empty page)" for t in pages(filing)], "document")))
            print(f"{model}: {filing}", flush=True)
    path = out / "questions.json"
    if not path.exists():
        vectors = unit(embed([q["question"] for q in qs], "query"))
        path.write_text(json.dumps({q["id"]: v.tolist() for q, v in zip(qs, vectors)}))


if __name__ == "__main__":
    main(sys.argv[1])
