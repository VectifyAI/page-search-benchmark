"""Times a new 160-page 10-K and one question, five times each, the same way for four searches: Page Search API
(upload, then search; a fresh copy of the file each time, so it is processed, not found again), and three
embeddings through OpenRouter (every page embedded, 8 batches of 20 pages at once, then the question).

Run it from a machine in Google Cloud us-central1, the API's region, so sending the file takes milliseconds:

    OPENROUTER_API_KEY=... PAGEINDEX_SEARCH_API_KEY=pis_live_... python timing.py 3M_2018_10K.pdf
"""
import concurrent.futures as cf
import json
import os
import statistics
import sys
import time
import urllib.request
import uuid

import pymupdf
from pageindex_search import PageSearch

PDF = sys.argv[1] if len(sys.argv) > 1 else "3M_2018_10K.pdf"
Q = "What was capital expenditure in 2018?"
RUNS = 5
texts = [p.get_text()[:12000].strip() or "(empty page)" for p in pymupdf.open(PDF)]


def embed(model, batch, kind):
    body = {"model": model, "input": batch}
    if model.startswith("voyage"):
        body["input_type"] = kind
    for attempt in range(4):
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/embeddings", data=json.dumps(body).encode(), headers={
                "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1)


def embedding(model):
    t = time.time()
    with cf.ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda b: embed(model, b, "document"), [texts[i:i + 20] for i in range(0, len(texts), 20)]))
    embed(model, [Q], "query")
    return time.time() - t


client = PageSearch(timeout=300)


def ours():
    doc = pymupdf.open(PDF)
    doc.set_metadata({"title": "timing " + uuid.uuid4().hex})  # a new file: processed, not found again
    data = doc.tobytes()
    t = time.time()
    d = client.upload(data, filename=PDF)
    assert not d.duplicate
    client.search(d, [Q])
    took = time.time() - t
    client.delete(d)
    return took


embedding("voyageai/voyage-4-large")  # warm the connection once
for name, run in [("Page Search API", ours), ("Voyage 4 Large", lambda: embedding("voyageai/voyage-4-large")),
                  ("Gemini Embedding 2", lambda: embedding("google/gemini-embedding-2")),
                  ("OpenAI text-embedding-3-large", lambda: embedding("openai/text-embedding-3-large"))]:
    times = [run() for _ in range(RUNS)]
    print(f"{name:30s} median {statistics.median(times):.2f}s   " + ", ".join(f"{x:.2f}" for x in times), flush=True)
