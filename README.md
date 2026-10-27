# Page Search benchmark

How well [Page Search API](https://search.pageindex.ai) finds the pages that answer a question in a long financial
filing, and how fast, against three embedding models: Voyage 4 Large, Gemini Embedding 2 and OpenAI
text-embedding-3-large. The numbers on search.pageindex.ai come from here.

## Results

**Evidence pages found** among the first *k* pages each method returns, over all questions' evidence pages:

| | Top 5 | Top 10 | Top 20 |
| --- | ---: | ---: | ---: |
| **Page Search API** | **93.2%** | **97.5%** | **99.3%** |
| Voyage 4 Large | 65.5% | 83.8% | 90.3% |
| Gemini Embedding 2 | 54.0% | 70.1% | 86.3% |
| OpenAI text-embedding-3-large | 47.5% | 68.0% | 79.9% |

Questions with **every** evidence page in the first 10: Page Search 93.8%, Voyage 65.0%, Gemini 45.0%, OpenAI 38.8%.

**By default** (no `top_k`), Page Search lists the pages likely to help: 4.7 pages a question on average (median 4)
out of a filing's 186. They hold 95.0% of the evidence pages, and
leave out 97.1% of the filing's pages, and so of the tokens you would send your LLM.

**Time** to search a new 160-page 10-K (3M, 2018) with one question, processing or embedding the pages included and
sending the file not; the median of five runs from a VM in Google Cloud us-central1:

| | Seconds |
| --- | ---: |
| **Page Search API** (upload, then search) | **1.1** |
| OpenAI text-embedding-3-large | 2.4 |
| Voyage 4 Large | 2.8 |
| Gemini Embedding 2 | 3.2 |

## What is measured

**Questions and filings.** The financial-qa set of [VectifyAI/pageindex-benchmarks](https://github.com/VectifyAI/pageindex-benchmarks):
questions about the 10-K filings of eight companies, each with the pages that hold its answer (its evidence). We use
its **80 single-filing questions** (52 SET, 28 WHY) over 39 filings. Its other **20 questions (XCO) compare several
companies** and need several filings at once; a Page Search search reads one document, so they are left out. The
filings themselves are in that repository (its `corpus/fetch.py` downloads them).

**Each question searches its whole filing**: every page of the 10-K, 186 on average.

**Page Search API** reads every page against the question and gives each page a relevance from 0 to 1;
`results/page_search/` has every page's relevance for each question. With `top_k` it returns the most relevant
pages; the comparison ranks all of them.

**The embeddings** embed every page of the filing as a document and the question as a query, and rank the pages by
cosine similarity. They read the same page text Page Search made of each filing (`data/pages/`, one string a page),
the first 12,000 characters of each page.

- Voyage 4 Large: `voyage-4-large` through OpenRouter, `input_type` document and query.
- Gemini Embedding 2: `gemini-embedding-2` on Vertex AI, task types `RETRIEVAL_DOCUMENT` and `RETRIEVAL_QUERY`.
- OpenAI: `text-embedding-3-large`.

**Time** covers what each needs to answer a question about a document it has not seen: Page Search processes the PDF
and searches it; an embedding embeds every page, then the question. Sending the PDF over the network is left out by
measuring from the API's region. The embeddings are called through OpenRouter with 8 batches of 20 pages at once.

## Reproduce

```bash
git clone https://github.com/VectifyAI/pageindex-benchmarks ../pageindex-benchmarks
python ../pageindex-benchmarks/financial-qa/corpus/fetch.py      # the filings, for Page Search
pip install -r requirements.txt

PAGEINDEX_SEARCH_API_KEY=pis_live_... python scripts/page_search.py   # results/page_search/
OPENROUTER_API_KEY=... python scripts/embed.py voyage                  # vectors/voyage/
GCP_PROJECT=... GCP_TOKEN=$(gcloud auth print-access-token) python scripts/embed.py gemini
OPENAI_API_KEY=... python scripts/embed.py openai
python scripts/compare.py                                              # the tables above
python scripts/timing.py path/to/3M_2018_10K.pdf                       # from a VM in us-central1
```

`compare.py` runs on the results already here, and on each embedding whose vectors you have made.

| | |
| --- | --- |
| `data/pages/` | Each filing's page text, gzipped JSON, one string a page from page 1. |
| `results/page_search/` | Page Search API's relevance for every page, for each question. |
| `scripts/` | `page_search.py`, `embed.py`, `compare.py`, `timing.py`; `common.py` is what they share. |
