"""
Vocabulary used for spell-correction matching.
This sandbox has no network access to install a spellchecker package
(pyspellchecker/enchant/nltk) or download a system dictionary
(/usr/share/dict/words is empty here), so correction is matched against a
curated in-repo vocabulary instead: common English function words + the
consumer-dispute legal domain terms this system actually needs to get right
(statute names, forum names, case-type vocabulary). This is deliberately
narrow — it's built to catch typos in the terms that matter for retrieval,
not to be a general English spellchecker.
"""

COMMON_ENGLISH = {
    "the", "a", "an", "of", "for", "and", "or", "is", "was", "were", "in", "on",
    "at", "to", "by", "with", "from", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "under", "over",
    "again", "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "can", "will", "just", "should", "now", "my", "i", "me", "he",
    "she", "it", "we", "they", "them", "his", "her", "its", "our", "their",
    "this", "that", "these", "those", "am", "are", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "would",
    "could", "seeking", "filed", "against", "regarding", "concerning", "due",
    "because", "since", "delayed", "wants", "want", "need", "needs", "please",
    "claim", "case", "money", "paid", "amount", "full", "partial", "months",
    "years", "years", "months", "days",
}

LEGAL_DOMAIN_TERMS = {
    # forums / bodies
    "ncdrc", "commission", "tribunal", "forum", "authority", "court",
    # parties / roles
    "complainant", "respondent", "opposite", "party", "petitioner",
    "appellant", "builder", "developer", "insurer", "insurance", "bank",
    "consumer", "manufacturer", "seller", "dealer",
    # acts / statutes
    "act", "section", "consumer", "protection", "1986", "2019", "rera",
    "provision", "statute", "clause",
    # dispute-type vocabulary
    "deficiency", "service", "negligence", "medical", "possession", "refund",
    "compensation", "delay", "delayed", "defective", "product", "warranty",
    "unfair", "trade", "practice", "misleading", "advertisement", "damages",
    "interest", "litigation", "cost", "costs", "penalty", "execution",
    "jurisdiction", "pecuniary", "limitation", "appeal", "revision",
    "order", "judgment", "decree", "settlement", "agreement", "flat",
    "apartment", "shop", "booking", "allotment", "occupancy", "certificate",
    "construction", "premium", "policy", "claim", "hospital", "surgery",
    "treatment", "diagnosis", "accident", "vehicle", "loan", "emi",
}

VOCAB = COMMON_ENGLISH | LEGAL_DOMAIN_TERMS
