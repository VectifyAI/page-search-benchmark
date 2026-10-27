"""The comparison: for each question, the share of its evidence pages among the first k pages each method returns,
for k = 5, 10 and 20; and, for Page Search by default (no top_k), how many pages it lists and how many of the
filing's pages it leaves out. Embeddings rank pages by cosine similarity to the question, ties by page number.

    python scripts/compare.py        (Page Search from results/; each embedding whose vectors/ are there)
"""
import json
import statistics

import numpy as np

from common import RESULTS, VECTORS, evidence, questions

KS = (5, 10, 20)
NAMES = {"page_search": "Page Search API", "voyage": "Voyage 4 Large", "gemini": "Gemini Embedding 2",
         "openai": "OpenAI text-embedding-3-large"}


def ranked(scores):
    """Page numbers, from 1, most relevant first."""
    return [p + 1 for p in sorted(range(len(scores)), key=lambda p: (-scores[p], p))]


def default_listing(scores):
    """The pages the API lists without top_k: relevance 0.3 or more, or 0.2 or more when fewer than 3 are."""
    strong = [p for p in ranked(scores) if scores[p - 1] >= 0.3]
    return strong if len(strong) >= 3 else [p for p in ranked(scores) if scores[p - 1] >= 0.2]


def main():
    # Some BLAS builds (Apple's Accelerate) warn on these products without cause; the scores are finite.
    np.seterr(all="ignore")
    qs = questions()
    scores = {"page_search": {q["id"]: json.loads((RESULTS / f"{q['id']}.json").read_text())["relevance"] for q in qs}}
    for model in ("voyage", "gemini", "openai"):
        if (VECTORS / model / "questions.json").exists():
            qv = json.loads((VECTORS / model / "questions.json").read_text())
            scores[model] = {q["id"]: list(np.load(VECTORS / model / f"{q['files'][0]}.npy").astype(np.float64) @ np.array(qv[q["id"]]))
                             for q in qs}

    print(f"{len(qs)} questions ({sum(q['type'] == 'SET' for q in qs)} SET, {sum(q['type'] == 'WHY' for q in qs)} WHY), "
          f"{len({q['files'][0] for q in qs})} filings\n")
    print("Evidence pages found among the first k returned (all questions' evidence pages together)")
    print(f"{'':32s}" + "".join(f"{'top ' + str(k):>10s}" for k in KS) + f"{'all evidence, top 10':>24s}")
    for method, by_q in scores.items():
        found = {k: 0 for k in KS}
        total = complete = 0
        for q in qs:
            order, ev = ranked(by_q[q["id"]]), evidence(q)
            total += len(ev)
            for k in KS:
                found[k] += len(ev & set(order[:k]))
            complete += ev <= set(order[:10])
        print(f"{NAMES[method]:32s}" + "".join(f"{100 * found[k] / total:9.1f}%" for k in KS) + f"{100 * complete / len(qs):23.1f}%")

    listed = [default_listing(scores["page_search"][q["id"]]) for q in qs]
    sizes = [len(scores["page_search"][q["id"]]) for q in qs]
    found = sum(len(evidence(q) & set(l)) for q, l in zip(qs, listed)) / sum(len(evidence(q)) for q in qs)
    print(f"\nPage Search by default: {statistics.mean(map(len, listed)):.1f} pages a question on average "
          f"(median {statistics.median(map(len, listed))}) of {statistics.mean(sizes):.0f}; "
          f"{100 * found:.1f}% of evidence pages found; "
          f"{100 * statistics.mean(1 - len(l) / n for l, n in zip(listed, sizes)):.1f}% of pages left out")


if __name__ == "__main__":
    main()
