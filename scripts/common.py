"""What the scripts share: the questions, each filing's page text, and where results and vectors go."""
import gzip
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGES = ROOT / "data" / "pages"
RESULTS = ROOT / "results" / "page_search"
VECTORS = ROOT / "vectors"  # made by embed.py; not kept in the repository
# Each embedding sees at most this many characters of a page, as the comparison did.
MAX_CHARS = 12000


def questions():
    """financial-qa's single-filing questions (SET and WHY), from a checkout of VectifyAI/pageindex-benchmarks.

    The 20 cross-company (XCO) questions need several filings at once; a Page Search search reads one document, so
    they are left out.
    """
    path = Path(os.environ.get("BENCHMARKS", ROOT.parent / "pageindex-benchmarks")) / "financial-qa" / "questions" / "questions.json"
    return [q for q in json.loads(path.read_text()) if q["type"] != "XCO"]


def pages(filing):
    """A filing's page text, one string a page from page 1: the text the Page Search API made of it."""
    with gzip.open(PAGES / f"{Path(filing).stem}.json.gz", "rt") as f:
        return json.load(f)


def evidence(q):
    """The pages (numbered from 1) that hold the question's answer."""
    return {e["page"] for e in q["evidence"]}
