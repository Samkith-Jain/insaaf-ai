"""
Query reformulation.
Detects incomplete or under-specified user queries (missing dispute type,
missing legal-domain framing, sentence fragments) and rewrites them into a
clearer, well-formed query before they enter the retrieval pipeline —
per Objective 5 in the project spec.

Rule-based by design: reformulation here means structural/completeness
fixes (fragment -> full query, add missing domain anchor), not semantic
paraphrasing — an LLM-based reformulation step is the natural upgrade path
once the agentic pipeline (stage 4+) exists to consume it.
"""
import re

DISPUTE_TYPE_KEYWORDS = {
    "possession", "refund", "delay", "flat", "apartment", "builder",
    "insurance", "claim", "medical", "negligence", "product", "defective",
    "warranty", "unfair", "trade", "service", "deficiency",
}

DOMAIN_ANCHOR = "consumer dispute"

QUESTION_STARTERS = ("what", "why", "how", "when", "who", "can", "is", "does", "did")


def is_fragment(query: str) -> bool:
    """A handful of bare keywords with no verb/structure, e.g. 'flat delay refund'."""
    words = query.strip().split()
    if len(words) <= 4 and not any(w.lower() in QUESTION_STARTERS for w in words):
        return True
    return False


def has_domain_anchor(query: str) -> bool:
    lower = query.lower()
    return DOMAIN_ANCHOR in lower or "ncdrc" in lower or "consumer protection act" in lower


def missing_dispute_type(query: str) -> bool:
    lower = query.lower()
    return not any(kw in lower for kw in DISPUTE_TYPE_KEYWORDS)


def reformulate(query: str) -> tuple[str, list[str]]:
    notes = []
    result = query.strip()

    if not result:
        return result, ["empty query — nothing to reformulate"]

    if is_fragment(result):
        result = f"Find precedent cases and applicable statutes regarding {result}"
        notes.append("expanded sentence fragment into a full retrieval query")

    if not has_domain_anchor(result):
        result = f"{result} under the Consumer Protection Act, 2019 (NCDRC)"
        notes.append("added missing legal-domain anchor (Consumer Protection Act / NCDRC)")

    if missing_dispute_type(result):
        notes.append("warning: no recognizable dispute-type keyword found — query may be too vague for precise retrieval")

    # normalize whitespace / capitalization
    result = re.sub(r"\s+", " ", result).strip()
    if result and not result[0].isupper():
        result = result[0].upper() + result[1:]
    if result and result[-1] not in ".?":
        result += "."

    return result, notes
