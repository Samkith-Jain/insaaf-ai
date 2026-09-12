"""
Curated, authenticated statute provisions for CiteVerify to check citations
against. This stands in for the "curated database of authenticated NCDRC
judgments and Consumer Protection Act, 2019 provisions" described in the
project spec — scoped here to the Act text itself (the judgment side of that
curated DB is structuring/db.py's `judgments` table, already populated from
real ingested judgments).

Sourced from the text of the Consumer Protection Act, 2019 (key consumer-
dispute-relevant sections) plus the superseded Consumer Protection Act, 1986
(most ingested pre-2019 judgments cite this Act, so a real verifier has to
recognize both).
"""

CPA_2019_SECTIONS = {
    "2": "Definitions",
    "17": "Complaint to District Commission",
    "21": "Reference to mediation",
    "34": "Jurisdiction of District Commission",
    "35": "Manner in which complaint shall be made",
    "38": "Procedure on admission of complaint",
    "39": "Findings of District Commission",
    "41": "Appeal against order of District Commission",
    "47": "Jurisdiction of State Commission",
    "51": "Appeal against order of State Commission",
    "58": "Jurisdiction of National Commission",
    "67": "Appeal against order of National Commission",
    "71": "Enforcement of orders of District Commission, State Commission or National Commission",
    "72": "Penalty for non-compliance of order",
    "85": "Liability of manufacturer for defective products",
    "86": "Liability of product service provider",
    "88": "Liability for unfair trade practices",
}

CPA_1986_SECTIONS = {
    "2": "Definitions",
    "11": "Jurisdiction of the District Forum",
    "12": "Manner in which complaint shall be made",
    "13": "Procedure on admission of complaint",
    "14": "Findings of the District Forum",
    "15": "Appeal against order of District Forum",
    "17": "Jurisdiction of the State Commission",
    "19": "Appeal against order of State Commission",
    "21": "Jurisdiction of the National Commission",
    "23": "Appeal against order of National Commission",
    "24A": "Limitation period",
    "27": "Penalties",
}

ACT_REGISTRY = {
    "consumer protection act, 2019": CPA_2019_SECTIONS,
    "consumer protection act 2019": CPA_2019_SECTIONS,
    "consumer protection act, 1986": CPA_1986_SECTIONS,
    "consumer protection act 1986": CPA_1986_SECTIONS,
}
