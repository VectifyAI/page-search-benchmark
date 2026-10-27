"""Page Search API's relevance for every page, for each question: uploads the question's filing and searches it
with top_k as large as the filing, so every page comes back with its relevance. Writes results/page_search/<id>.json.

    PAGEINDEX_SEARCH_API_KEY=pis_live_... BENCHMARKS=../pageindex-benchmarks python scripts/page_search.py [ID ...]

The filings are pageindex-benchmarks' financial-qa/corpus/pdfs (its corpus/fetch.py downloads them).
"""
import json
import os
import sys
from pathlib import Path

from pageindex_search import PageSearch

from common import RESULTS, ROOT, questions

PDFS = Path(os.environ.get("BENCHMARKS", ROOT.parent / "pageindex-benchmarks")) / "financial-qa" / "corpus" / "pdfs"


def main():
    client = PageSearch(timeout=300)
    RESULTS.mkdir(parents=True, exist_ok=True)
    docs = {}
    only = set(sys.argv[1:])
    for q in questions():
        (filing,) = q["files"]
        if only and q["id"] not in only:
            continue
        if not (PDFS / filing).exists():
            print(f"{q['id']}: {filing} is not in {PDFS}; skipped", flush=True)
            continue
        if filing not in docs:
            docs[filing] = client.upload(PDFS / filing)  # the same file again is the same document
        doc = docs[filing]
        result = client.search(doc, [q["question"]], top_k=doc.pages)
        relevance = [0.0] * doc.pages
        for p in result.results[0].pages:
            relevance[p.page - 1] = p.relevance
        (RESULTS / f"{q['id']}.json").write_text(json.dumps({"question": q["id"], "file": filing, "relevance": relevance}))
        print(f"{q['id']}: {filing}, {doc.pages} pages, {result.seconds} s", flush=True)


if __name__ == "__main__":
    main()
