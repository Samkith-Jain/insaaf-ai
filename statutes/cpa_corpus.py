"""
Provision corpus for the Statute Retrieval Agent.

Scope
-----
The consumer-dispute provisions of the Consumer Protection Act, 2019 and the
superseded Consumer Protection Act, 1986, with, for each provision:
  - a plain-language SUMMARY (see the honesty note below),
  - the ELEMENTS a complainant must establish to bring a case under it,
    each with match patterns, so the agent can report *which* element a fact
    satisfies rather than asserting a bare section number,
  - the concept hooks the Fact Extraction Agent emits (`CPA_HOOKS`),
  - which side the provision helps (`favours`), and
  - the corresponding provision under the other Act (`equivalent`), so the
    agent can apply the Act that was actually in force.

HONESTY NOTE - read before relying on this file
-----------------------------------------------
`summary` is a plain-language restatement written for retrieval and
explanation. It is NOT the statutory text: every entry carries
`verbatim=False`. Before this system is used on a live matter, these entries
must be replaced with the text of the Act as published in the Gazette of
India. Two specific things to check against the gazette:

  1. Chapter VI (product liability) section NUMBERING. This repo's
     `verification/curated_provisions.py` maps s.85 to manufacturer liability;
     the numbering used elsewhere maps s.84 to manufacturer liability and s.85
     to the product service provider. Rather than pick a side, Chapter VI is
     entered here as a single grouped provision with
     `numbering_verified=False`, so the agent can surface the chapter without
     asserting a section number it cannot authenticate.
  2. PECUNIARY LIMITS (see `PECUNIARY_LIMITS`). These were revised by
     subordinate rules after the Act commenced and can be revised again;
     each band carries the date it took effect.

Anything in here flagged `verbatim=False` or `numbering_verified=False`
should be treated by downstream agents as a retrieval aid, not as an
authenticated citation - CiteVerify remains the gate.
"""
from __future__ import annotations

from datetime import date

CPA_2019 = "Consumer Protection Act, 2019"
CPA_1986 = "Consumer Protection Act, 1986"

# The 2019 Act's consumer-dispute provisions were brought into force on
# 20 July 2020. A cause of action that accrued and was filed before that date
# is governed by the 1986 Act, which is why most pre-2020 judgments in the
# corpus cite 1986 provisions.
CPA_2019_COMMENCEMENT = date(2020, 7, 20)

# Pecuniary jurisdiction. `from_date` is when the band took effect.
# The 2021 bands were set by subordinate rules, not by the Act itself, and are
# the most likely entry in this file to be out of date - verify before use.
PECUNIARY_LIMITS = [
    {
        "regime": "CPA 1986",
        "from_date": date(1993, 6, 18),
        "act": CPA_1986,
        "bands": [("District Forum", 0, 2_000_000),
                  ("State Commission", 2_000_000, 10_000_000),
                  ("National Commission", 10_000_000, None)],
        "basis": "value of goods or services and compensation claimed",
        "verified": False,
    },
    {
        "regime": "CPA 2019 (as enacted)",
        "from_date": CPA_2019_COMMENCEMENT,
        "act": CPA_2019,
        "bands": [("District Commission", 0, 10_000_000),
                  ("State Commission", 10_000_000, 100_000_000),
                  ("National Commission", 100_000_000, None)],
        "basis": "value of goods or services paid as consideration",
        "verified": False,
    },
    {
        "regime": "CPA 2019 (as revised by the 2021 jurisdiction rules)",
        "from_date": date(2021, 12, 30),
        "act": CPA_2019,
        "bands": [("District Commission", 0, 5_000_000),
                  ("State Commission", 5_000_000, 20_000_000),
                  ("National Commission", 20_000_000, None)],
        "basis": "value of goods or services paid as consideration",
        "verified": False,
    },
]

LIMITATION_YEARS = 2  # s.69 (2019) / s.24A (1986)


# A provision's ROLE in a consumer complaint. A complaint is founded on a
# cause of action; definitions support it; remedies follow from it; and
# jurisdiction, procedure and limitation are the frame around it. Without this,
# ranking treats a limitation defence as interchangeable with the cause of
# action it defeats, and the top provision for a builder-delay case comes back
# as s.69 rather than s.2(11).
ROLE_PRIOR = {
    "cause_of_action": 1.00,   # the wrong complained of
    "definition": 0.80,        # establishes an element of that wrong
    "remedy": 0.60,            # what the Commission may order
    "limitation": 0.45,        # the OP's usual threshold defence
    "jurisdiction": 0.40,      # which forum, not whether there is a case
    "procedure": 0.25,         # how the complaint is run
}


def _p(section, act, title, summary, elements, concepts, role, favours="complainant",
       equivalent=None, keywords="", numbering_verified=True, notes=""):
    if role not in ROLE_PRIOR:
        raise ValueError(f"unknown role {role!r} for section {section}")
    return {
        "section": section,
        "act": act,
        "title": title,
        "role": role,
        "summary": summary,
        "elements": elements,           # list of (element_name, match_pattern)
        "concepts": concepts,           # CPA_HOOKS concept names from Agent 1
        "favours": favours,             # complainant | opposite_party | neutral
        "equivalent": equivalent,       # (section, act) under the other Act
        "keywords": keywords,           # extra retrieval surface
        "verbatim": False,              # never the gazette text - see module docstring
        "numbering_verified": numbering_verified,
        "notes": notes,
    }


# --------------------------------------------------------------------------
# Consumer Protection Act, 2019
# --------------------------------------------------------------------------
PROVISIONS_2019 = [
    _p("2(7)", CPA_2019, "Consumer",
       "A person who buys goods or hires or avails services for consideration, "
       "including under deferred payment; excludes a person who obtains goods for resale "
       "or services for a commercial purpose.",
       [("goods bought or service availed",
         r"\b(purchas\w+|bought|booked|hired|availed|subscrib\w+|allot\w+|took (a )?(loan|policy))\b"),
        ("for consideration",
         r"\b(paid|payment|consideration|price|premium|instalment|booking amount|deposit\w*)\b|Rs\.?\s*[\d,]")],
       ["deficiency_in_service", "defect_in_goods", "unfair_trade_practice"], "definition",
       equivalent=("2(1)(d)", CPA_1986),
       keywords="consumer standing locus buyer purchaser beneficiary commercial purpose",
       notes="Standing is the first thing an OP attacks; check the commercial-purpose exclusion."),

    _p("2(6)", CPA_2019, "Complainant",
       "The persons who may file a complaint, including a consumer, a recognised "
       "consumer association, one or more consumers with the same interest, and the legal "
       "heir or representative of a deceased consumer.",
       [("person competent to complain",
         r"\b(complainant|consumer association|legal heir|legal representative|joint complaint)\b")], [],
       "definition", favours="neutral", equivalent=("2(1)(b)", CPA_1986),
       keywords="complainant standing who may file association legal heir"),

    _p("2(11)", CPA_2019, "Deficiency",
       "Any fault, imperfection, shortcoming or inadequacy in the quality, nature or manner "
       "of performance of a service that is required to be maintained by law or undertaken "
       "under a contract; includes negligent acts or omissions causing loss, and deliberate "
       "withholding of information.",
       [("a service was hired or availed",
         r"\b(service|booked|allot\w+|agreement|policy|loan|treatment|hired|availed)\b"),
        ("standard owed by law or contract",
         r"\b(agreement|contract|undertak\w+|promised|committed|policy terms|was to be)\b"),
        ("shortfall in performance",
         r"\b(delay\w*|not (complete|completed|delivered|offered|handed|refunded)|failed to|failure to|"
         r"deficien\w+|negligen\w+|non[- ]performance|did not (deliver|offer|respond|refund|repair|replace))\b")],
       ["deficiency_in_service"], "cause_of_action", equivalent=("2(1)(g)", CPA_1986),
       keywords="deficiency in service shortcoming inadequacy negligence delay failure to perform"),

    _p("2(10)", CPA_2019, "Defect",
       "Any fault, imperfection or shortcoming in the quality, quantity, potency, purity or "
       "standard of goods required to be maintained by law or under a contract, or as claimed "
       "by the trader.",
       [("goods supplied",
         r"\b(goods|product|vehicle|appliance|refrigerator|mobile|unit|item|delivered)\b"),
        ("standard required or claimed",
         r"\b(warranty|guarantee|specification|standard|as (promised|described|advertised)|brochure)\b"),
        ("fault in the goods",
         r"\b(defect(ive|s)?|malfunction\w*|not working|faulty|substandard|damaged)\b")], ["defect_in_goods"],
       "cause_of_action", equivalent=("2(1)(f)", CPA_1986),
       keywords="defect goods quality quantity potency purity standard faulty product"),

    _p("2(42)", CPA_2019, "Service",
       "Service of any description made available to potential users, including banking, "
       "financing, insurance, transport, housing construction, entertainment and boarding; "
       "excludes services rendered free of charge or under a contract of personal service.",
       [("service of a described kind",
         r"\b(banking|financ\w+|insurance|transport|housing construction|boarding|lodging|"
         r"telecom|electricity|medical|education|builder|developer)\b")], ["deficiency_in_service"],
       "definition", favours="neutral", equivalent=("2(1)(o)", CPA_1986),
       keywords="service housing construction banking insurance definition free of charge",
       notes="Housing construction is expressly a service - the usual answer to a builder's "
             "argument that a flat sale is a pure sale of immovable property."),

    _p("2(47)", CPA_2019, "Unfair trade practice",
       "A trade practice adopting a deceptive method to promote a sale, including false "
       "representation about standard or quality, misleading advertising, non-issue of a bill, "
       "and refusal to withdraw or take back defective goods within the stipulated period.",
       [("a trade practice in promoting a sale",
         r"\b(advertis\w+|brochure|represent\w+|market\w+|offer\w*|scheme|promot\w+|sold|sale)\b"),
        ("deception or false representation",
         r"\b(misrepresent\w+|false|misleading|deceptive|concealed|suppress\w+|"
         r"not as (promised|described|advertised)|overcharg\w+|refus(ed|al) to (refund|replace|take back))\b")], ["unfair_trade_practice"],
       "cause_of_action", equivalent=("2(1)(r)", CPA_1986),
       keywords="unfair trade practice misleading advertisement false representation deceptive"),

    _p("2(9)", CPA_2019, "Consumer rights",
       "The rights of consumers: protection against hazardous goods and services, information "
       "about quality and price, access to a variety of goods at competitive prices, a hearing "
       "and due consideration, redress against unfair or restrictive trade practices, and "
       "consumer awareness.",
       [("right asserted", r"\b(right|entitled|redress|informed|hearing)\b")], [],
       "definition", favours="neutral", equivalent=("6", CPA_1986),
       keywords="consumer rights redress information awareness"),

    _p("34", CPA_2019, "Jurisdiction of the District Commission",
       "The District Commission entertains complaints where the value of the goods or services "
       "paid as consideration does not exceed the prescribed limit, and where a party resides "
       "or carries on business or the cause of action arises wholly or in part.",
       [("value within the district band", r"Rs\.?\s*[\d,]|\bconsideration\b|\bvalue\b"),
        ("territorial nexus", r"\b(resides|carries on business|branch office|cause of action)\b")],
       ["pecuniary_jurisdiction_issue"], "jurisdiction", favours="neutral", equivalent=("11", CPA_1986),
       keywords="pecuniary jurisdiction district commission territorial cause of action"),

    _p("47", CPA_2019, "Jurisdiction of the State Commission",
       "The State Commission entertains complaints within its prescribed pecuniary band, "
       "appeals against District Commission orders, and transfer applications.",
       [("value within the state band", r"Rs\.?\s*[\d,]|\bconsideration\b|\bvalue\b"),
        ("appellate or original jurisdiction invoked", r"\b(appeal|state commission|original)\b")],
       ["pecuniary_jurisdiction_issue"], "jurisdiction", favours="neutral", equivalent=("17", CPA_1986),
       keywords="pecuniary jurisdiction state commission appeal transfer"),

    _p("58", CPA_2019, "Jurisdiction of the National Commission",
       "The National Commission entertains complaints above the prescribed pecuniary limit, "
       "appeals against State Commission orders, and revision petitions.",
       [("value above the state band", r"Rs\.?\s*[\d,]|\bconsideration\b|\bvalue\b"),
        ("national forum invoked", r"\b(national commission|revision petition|appeal)\b")],
       ["pecuniary_jurisdiction_issue"], "jurisdiction", favours="neutral", equivalent=("21", CPA_1986),
       keywords="pecuniary jurisdiction national commission ncdrc revision appeal"),

    _p("35", CPA_2019, "Manner in which complaint shall be made",
       "How a complaint is to be instituted before the District Commission, by whom, and in "
       "what form, including electronic filing.",
       [("complaint instituted", r"\b(filed|instituted|preferred|complaint no|consumer case no)\b")], [],
       "procedure", favours="neutral", equivalent=("12", CPA_1986),
       keywords="manner of complaint filing procedure electronic filing"),

    _p("38", CPA_2019, "Procedure on admission of complaint",
       "The procedure the District Commission follows after admitting a complaint, including "
       "notice to the opposite party, the time to file a reply, testing of goods, and "
       "ex parte proceedings on default.",
       [("procedural step taken", r"\b(notice|written version|written statement|reply|ex[- ]?parte|adjourn)\b")],
       [], "procedure", favours="neutral", equivalent=("13", CPA_1986),
       keywords="procedure admission notice written version reply ex parte testing"),

    _p("39", CPA_2019, "Findings of the District Commission (reliefs)",
       "The reliefs a Commission may order once a complaint is proved: removal of the defect, "
       "replacement of the goods, refund of the price, compensation for loss or injury, "
       "discontinuance of the unfair trade practice, punitive damages, and costs.",
       [("relief sought", r"\b(refund|replace\w*|compensation|damages|interest|cost of litigation|"
                          r"remove the defect|discontinue)\b"),
        ("loss or injury pleaded", r"\b(loss|injury|harass\w+|mental agony|suffered)\b")], ["deficiency_in_service", "defect_in_goods", "unfair_trade_practice"],
       "remedy",
       equivalent=("14", CPA_1986),
       keywords="reliefs refund replacement compensation punitive damages costs findings"),

    _p("69", CPA_2019, "Limitation period",
       "A complaint must be filed within two years from the date on which the cause of action "
       "arose; a complaint filed later may still be entertained if the complainant satisfies "
       "the Commission that there was sufficient cause for the delay, with reasons recorded.",
       [("date of cause of action identifiable", r"\b(cause of action|dated|on \d{1,2}[./-]\d{1,2}[./-]\d{4})\b"),
        ("filing outside two years alleged", r"\b(limitation|time[- ]barred|barred by limitation|delay in filing|"
                                             r"condonation|sufficient cause)\b")],
       ["limitation_issue"], "limitation", favours="opposite_party", equivalent=("24A", CPA_1986),
       keywords="limitation two years cause of action condonation delay time barred",
       notes="Where possession is never delivered, tribunals have treated the cause of action as "
             "recurring - check the precedents before treating a delayed filing as barred."),

    _p("71", CPA_2019, "Enforcement of orders",
       "Orders of the District, State and National Commissions are enforceable as a decree of a "
       "civil court.",
       [("enforcement sought", r"\b(execution|enforce\w+|decree|non[- ]compliance)\b")],
       [], "remedy", favours="complainant", equivalent=("25", CPA_1986),
       keywords="enforcement execution decree civil court"),

    _p("72", CPA_2019, "Penalty for non-compliance of order",
       "Punishment for failing to comply with an order of a Commission, by imprisonment, fine, "
       "or both.",
       [("non-compliance alleged", r"\b(non[- ]compliance|failed to comply|did not comply|contempt)\b")],
       [], "remedy", favours="complainant", equivalent=("27", CPA_1986),
       keywords="penalty non-compliance punishment imprisonment fine"),

    _p("Chapter VI (ss. 82-87)", CPA_2019, "Product liability",
       "The product liability chapter: when a product manufacturer, product service provider or "
       "product seller is liable for harm caused by a defective product, and the exceptions to "
       "that liability.",
       [("harm from a product", r"\b(harm|injur\w+|damage|defect(ive)?|manufactur\w+|product seller|"
                                r"product service provider)\b")], ["defect_in_goods"],
       "cause_of_action", equivalent=None,
       keywords="product liability manufacturer seller service provider defective product harm",
       numbering_verified=False,
       notes="Entered as a chapter, not a section: the individual section numbers in this chapter "
             "must be confirmed against the gazette before any of them is cited (see module docstring)."),
]

# --------------------------------------------------------------------------
# Consumer Protection Act, 1986 (for pre-commencement causes of action)
# --------------------------------------------------------------------------
PROVISIONS_1986 = [
    _p("2(1)(d)", CPA_1986, "Consumer",
       "A person who buys goods or hires or avails services for consideration; excludes goods "
       "obtained for resale or a commercial purpose.",
       [("goods bought or service availed",
         r"\b(purchas\w+|bought|booked|hired|availed|allot\w+)\b"),
        ("for consideration", r"\b(paid|payment|consideration|price|premium|instalment)\b|Rs\.?\s*[\d,]")],
       ["deficiency_in_service", "defect_in_goods"], "definition", equivalent=("2(7)", CPA_2019),
       keywords="consumer definition 1986 buyer hirer consideration"),

    _p("2(1)(g)", CPA_1986, "Deficiency",
       "Any fault, imperfection, shortcoming or inadequacy in the quality, nature and manner of "
       "performance required to be maintained by law or undertaken under a contract.",
       [("a service was hired or availed", r"\b(service|booked|allot\w+|agreement|policy|loan|hired)\b"),
        ("shortfall in performance",
         r"\b(delay\w*|not (complete|delivered|offered)|failed to|deficien\w+|negligen\w+)\b")], ["deficiency_in_service"],
       "cause_of_action", equivalent=("2(11)", CPA_2019),
       keywords="deficiency 1986 service shortcoming inadequacy"),

    _p("2(1)(f)", CPA_1986, "Defect",
       "Any fault, imperfection or shortcoming in the quality, quantity, potency, purity or "
       "standard of goods required to be maintained under law or contract.",
       [("goods supplied", r"\b(goods|product|vehicle|appliance|unit|item)\b"),
        ("fault in the goods", r"\b(defect(ive|s)?|malfunction\w*|faulty|substandard)\b")], ["defect_in_goods"],
       "cause_of_action", equivalent=("2(10)", CPA_2019),
       keywords="defect 1986 goods quality standard"),

    _p("2(1)(r)", CPA_1986, "Unfair trade practice",
       "A trade practice adopting an unfair method or deceptive practice to promote a sale, "
       "including false representation and misleading advertising.",
       [("practice promoting a sale", r"\b(advertis\w+|brochure|represent\w+|promot\w+|sale)\b"),
        ("deception", r"\b(misrepresent\w+|false|misleading|deceptive|overcharg\w+)\b")], ["unfair_trade_practice"],
       "cause_of_action", equivalent=("2(47)", CPA_2019),
       keywords="unfair trade practice 1986 misleading false representation"),

    _p("11", CPA_1986, "Jurisdiction of the District Forum",
       "The District Forum's pecuniary and territorial jurisdiction.",
       [("value within the district band", r"Rs\.?\s*[\d,]|\bvalue\b"),
        ("territorial nexus", r"\b(resides|carries on business|cause of action)\b")],
       ["pecuniary_jurisdiction_issue"], "jurisdiction", favours="neutral", equivalent=("34", CPA_2019),
       keywords="district forum jurisdiction 1986 pecuniary territorial"),

    _p("17", CPA_1986, "Jurisdiction of the State Commission",
       "The State Commission's original, appellate and revisional jurisdiction.",
       [("value within the state band", r"Rs\.?\s*[\d,]|\bvalue\b"),
        ("state forum invoked", r"\b(state commission|appeal)\b")],
       ["pecuniary_jurisdiction_issue"], "jurisdiction", favours="neutral", equivalent=("47", CPA_2019),
       keywords="state commission jurisdiction 1986"),

    _p("21", CPA_1986, "Jurisdiction of the National Commission",
       "The National Commission's original, appellate and revisional jurisdiction.",
       [("value above the state band", r"Rs\.?\s*[\d,]|\bvalue\b"),
        ("national forum invoked", r"\b(national commission|revision petition|appeal)\b")],
       ["pecuniary_jurisdiction_issue"], "jurisdiction", favours="neutral", equivalent=("58", CPA_2019),
       keywords="national commission ncdrc jurisdiction 1986"),

    _p("14", CPA_1986, "Findings of the District Forum (reliefs)",
       "The reliefs available once a complaint is proved: removal of the defect, replacement, "
       "refund of the price, compensation, discontinuance of the unfair trade practice, and costs.",
       [("relief sought", r"\b(refund|replace\w*|compensation|damages|interest|costs?)\b")], ["deficiency_in_service", "defect_in_goods", "unfair_trade_practice"],
       "remedy",
       equivalent=("39", CPA_2019),
       keywords="reliefs 1986 refund replacement compensation costs"),

    _p("24A", CPA_1986, "Limitation period",
       "A complaint must be filed within two years of the cause of action, subject to condonation "
       "of delay for sufficient cause recorded in writing.",
       [("date of cause of action identifiable", r"\b(cause of action|dated)\b"),
        ("filing outside two years alleged",
         r"\b(limitation|time[- ]barred|barred by limitation|condonation)\b")],
       ["limitation_issue"], "limitation", favours="opposite_party", equivalent=("69", CPA_2019),
       keywords="limitation 1986 two years 24A condonation time barred"),
]

ALL_PROVISIONS = PROVISIONS_2019 + PROVISIONS_1986

PROVISIONS_BY_ACT = {
    CPA_2019: PROVISIONS_2019,
    CPA_1986: PROVISIONS_1986,
}


# Some `equivalent` pointers name a real provision of the other Act that is
# outside the modelled subset here (the 1986 entries cover the provisions that
# actually appear in the ingested judgments, not the whole Act). Those pointers
# are still legally correct and useful to display, so they are kept - and
# listed here so nothing dangles silently. `get_provision` returns None for
# them, which callers must handle.
UNMODELLED_EQUIVALENTS = {
    ("2(1)(b)", CPA_1986),   # complainant
    ("2(1)(o)", CPA_1986),   # service
    ("6", CPA_1986),         # consumer rights
    ("12", CPA_1986),        # manner of complaint
    ("13", CPA_1986),        # procedure on admission
    ("25", CPA_1986),        # enforcement of orders
    ("27", CPA_1986),        # penalties
}


def equivalent_is_modelled(provision: dict) -> bool:
    """False when `equivalent` names a real provision outside this corpus's subset."""
    eq = provision.get("equivalent")
    return bool(eq) and tuple(eq) not in UNMODELLED_EQUIVALENTS


def get_provision(section: str, act: str) -> dict | None:
    for p in ALL_PROVISIONS:
        if p["section"].lower() == section.lower() and p["act"].lower() == act.lower():
            return p
    return None


def provision_text(p: dict) -> str:
    """The retrieval surface for a provision: title + summary + keywords + element names."""
    elements = " ".join(name for name, _ in p["elements"])
    return f"{p['title']}. {p['summary']} {p['keywords']} {elements}"


def act_in_force(reference: date | None) -> tuple[str, str]:
    """
    Which Act governs, and why.

    The 2019 Act's consumer-dispute provisions commenced on 20 July 2020.
    A cause of action accruing before that is governed by the 1986 Act - which
    is why the agent must not cite s.2(11) at a 2016 dispute.
    """
    if reference is None:
        return CPA_2019, ("No reference date found in the fact sheet; defaulting to the "
                          "Consumer Protection Act, 2019 as the Act currently in force.")
    if reference < CPA_2019_COMMENCEMENT:
        return CPA_1986, (f"Reference date {reference.isoformat()} precedes the commencement of the "
                          f"Consumer Protection Act, 2019 ({CPA_2019_COMMENCEMENT.isoformat()}); "
                          "the 1986 Act governs.")
    return CPA_2019, (f"Reference date {reference.isoformat()} falls on or after the commencement of "
                      f"the Consumer Protection Act, 2019 ({CPA_2019_COMMENCEMENT.isoformat()}).")


def pecuniary_regime(reference: date | None) -> dict:
    """The pecuniary-jurisdiction bands in force on a given date (latest if unknown)."""
    if reference is None:
        return PECUNIARY_LIMITS[-1]
    applicable = [r for r in PECUNIARY_LIMITS if r["from_date"] <= reference]
    return applicable[-1] if applicable else PECUNIARY_LIMITS[0]


def forum_for_value(value_inr: int, reference: date | None) -> tuple[str, dict]:
    """Which forum has pecuniary jurisdiction over a claim of this value."""
    regime = pecuniary_regime(reference)
    for forum, lo, hi in regime["bands"]:
        if value_inr > lo and (hi is None or value_inr <= hi):
            return forum, regime
    return regime["bands"][0][0], regime
