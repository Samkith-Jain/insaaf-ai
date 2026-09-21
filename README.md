# Insaaf AI

Agentic RAG-powered legal judgment prediction system for Indian consumer dispute adjudication (NCDRC + Consumer Protection Act, 2019).

## Pipeline
1. Prompt Correction — grammar/spell + query reformulation
2. Data Pipeline — OCR (pytesseract), cleaning (pdfplumber)
3. Structuring — segmentation, NER, LLM field extraction
4. Structured DB — SQLite + Qdrant embeddings
5. Hybrid Retrieval — dense (Sentence Transformers) + BM25 + re-ranking
6. Multi-Agent Reasoning — Fact Extraction, Statute Retrieval, Precedent Retrieval, Argument Analysis, Prediction, Verification, Explanation
7. CiteVerify (SPMA) — rule-based citation verification
8. Output — IRAC explanation + FastAPI + React

## Status
Zeroth review: design + literature review complete.
Implemented so far: **stages 0-4** (of 8 in the architecture diagram) —
prompt correction, extraction, cleaning, structuring/citation NER, SQLite
legal DB, hybrid (BM25 + dense) retrieval, CiteVerify (SPMA) citation
verification, IRAC explanation generation, and a backend API + frontend UI.
Verified end-to-end against 4 real judgment files (3 NCDRC, 1 State
Commission).

**All 5 reasoning agents implemented:**
- **Fact Extraction Agent** (`agents/fact_extraction.py`)
- **Statute Retrieval Agent** (`agents/statute_retrieval.py`)
- **Precedent Retrieval Agent** (`agents/precedent_retrieval.py`)
- **Argument Analysis Agent** (`agents/argument_analysis.py`)
- **Prediction Agent** (`agents/prediction.py`)

They chain on one shared state dict: Agent 2 writes `state["statutes"]`, which
Agent 3 reads; Agent 4 consumes all three bundles and writes
`state["argument_map"]`; Agent 5 consumes all four and writes
`state["prediction"]`. Run the whole chain with
`scripts/run_prediction_agent.py`.

### Note on offline substitutions
This was built in a network-isolated dev environment, so a few components
use dependency-free / offline stand-ins with the same interface as the
production target, swap-in-ready once network access is available:
- **Dense retrieval**: TF-IDF + Truncated SVD (scikit-learn) in place of
  Sentence Transformers + Qdrant — same cosine-similarity ranking interface.
- **NER**: regex-based statute/precedent citation extraction in place of
  spaCy — legal citations follow tight, predictable patterns, so this is a
  reasonable baseline ahead of swapping in a trained NER model. Party-name
  boundaries are imperfect (long/nested citations sometimes over- or
  under-capture) — a known limitation of regex NER, not hidden.
- **BM25**: implemented from scratch in place of the `rank_bm25` package.
- **Backend**: stdlib `http.server` in place of FastAPI — same route/JSON
  contract, mechanical port later.
- **Frontend**: plain HTML/JS in place of React — no build step needed,
  same component boundaries.
- **IRAC generation**: rule-based extractive NLG (keyword + position
  heuristics to locate Issue/Rule/Application/Conclusion sentences) in
  place of LLM-based generation — a real, inspectable extraction, not a
  placeholder, but terser than LLM-generated prose would be.
- **CiteVerify precedent matching**: difflib `SequenceMatcher` textual
  similarity in place of Sentence-Transformer semantic similarity.

## Setup & run
```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Drop judgment PDFs/HTML/txt into data/raw/ or data/ncdrc_judgments/, then:
python scripts/run_pipeline.py --query "refund for delayed possession of flat by builder"
```
This ingests every file in `data/raw/` and `data/ncdrc_judgments/`, extracts
and structures each judgment into `data_pipeline/legal.db`, builds a hybrid
BM25 + dense index, and prints ranked retrieval results for the query.

### Running the web app (backend + frontend)
```bash
# 1. seed the database (run the pipeline once, or reuse existing legal.db)
python scripts/run_pipeline.py

# 2. start the API server
python -m api.server
# serves on http://localhost:8000 — routes:
#   GET  /api/judgments
#   GET  /api/judgments/{id}
#   POST /api/query   { "query": "...", "top_k": 5 }
#   POST /api/facts       { "judgment_id": 1 }  (Agent 1)
#   POST /api/statutes    { "judgment_id": 1, "with_precedents": true }  (Agent 2)
#   POST /api/arguments   { "judgment_id": 1 }  (Agents 1-4)
#   POST /api/precedents  { "judgment_id": 1, "top_k": 5 }  (Agent 3)

# 3. open the frontend — no build step needed
#    just open frontend/index.html directly in a browser,
#    or serve it: python -m http.server 5500 --directory frontend
```
Backend is stdlib `http.server` (not FastAPI — no network access to
pip-install it in this sandbox) and frontend is plain HTML/JS (not React —
no network access to npm-install it). Same route contract / component
boundaries either way, so porting to FastAPI + React is a mechanical swap
once network access is available, not a redesign.

## Fact Extraction Agent (`agents/`)
Turns a case file (+ optional query) into an auditable fact sheet.
```bash
python scripts/run_fact_agent.py data/ncdrc_judgments/gunjan_aggarwal_257_2019.txt
python scripts/run_fact_agent.py case.txt --json          # machine-readable
python -m unittest tests.test_fact_extraction -v          # 19 tests
# API: POST /api/facts  {"judgment_id": 1}  or  {"case_text": "...", "query": "..."}
```
- **Argument mining**: each sentence -> transaction / payment / deficiency
  (facts), claim, defence, procedural, legal_reasoning (kept out of facts).
- **Entities**: parties, dates (normalised ISO), rupee amounts (lakh/crore
  aware), durations, percentages -> timeline.
- **Derived facts** (computed, flagged `derived=True`, inputs listed):
  promised delivery deadline and delay days.
- **Outcome-leakage control**: the operative order is stripped before
  extraction (`strip_outcome=True`), so Prediction never sees the answer.
- **Transparency**: every fact has paragraph no. + character offsets and a
  verified `grounded` flag; nothing is generated.
- **Hand-off**: writes `fact_sheet`, `retrieval_query`, `legal_hooks` into the
  shared pipeline state for the Statute Retrieval Agent.
- **Rules vs LLM**: offline rule-based by default (same trade-off as NER/IRAC
  above). `FactExtractionAgent(llm=AnthropicBackend())` lets an LLM re-label
  sentences (needs `ANTHROPIC_API_KEY`); any failure falls back to rules.
  The LLM backend is covered by stub tests only, not yet run against the live API.
- Known limits: keyword rules misjudge mixed sentences (one sentence gets one
  primary label) and category detection is coarse (e.g. a refrigerator
  complaint scores "ecommerce_retail" over "product_defect").

## Statute Retrieval Agent (`agents/statute_retrieval.py`)
Agent 2. Identifies the statutory basis of the pending case and checks the two
statutory tests that turn on computable facts.
```bash
python scripts/run_statute_agent.py data/ncdrc_judgments/gunjan_aggarwal_257_2019.txt
python scripts/run_statute_agent.py --judgment-id 1 --with-precedents   # Agents 1 -> 2 -> 3
python scripts/run_statute_agent.py case.txt --both-acts --json
python -m unittest tests.test_statute_retrieval -v                      # 49 tests
# API: POST /api/statutes  {"judgment_id": 1}  or  {"case_text": "...", "with_precedents": true}
```
- **Provision corpus** (`statutes/cpa_corpus.py`): the consumer-dispute
  provisions of the CPA 2019 and the superseded CPA 1986, each with the
  elements a complainant must establish, the concept hooks Agent 1 emits, the
  side it favours, and its equivalent under the other Act.
- **Which Act applies** is settled before anything is retrieved. The 2019
  Act's consumer-dispute provisions commenced on 20 July 2020, so a 2016
  cause of action is governed by the 1986 Act — citing s.2(11) at it is simply
  wrong. The agent picks the Act from the case's reference date, says why, and
  can show the other Act's equivalents flagged (`--both-acts`).
- **Element matching** is the substance of the agent. For every provision it
  reports which statutory element is satisfied *by which fact IDs* and which
  elements are unsatisfied. A bare section number is an assertion; "element 2
  satisfied by F3 and F7, element 3 unsatisfied" is an argument Agent 4 can
  use and a reader can check against Agent 1's character offsets.
- **Four retrieval channels**, each recorded as provenance: concept hooks,
  element matching, BM25 over the provision corpus, and corpus evidence (how
  often ingested judgments actually cite the provision).
- **Role-aware ranking**: a cause of action outranks a definition, a remedy, a
  jurisdiction provision or a limitation defence (`ROLE_PRIOR`). Without this a
  builder-delay case returns s.69 as its leading provision, because
  "limitation" is lexically prominent in a judgment that *rejects* that
  defence. Hook matching is weighted by Agent 1's cue counts rather than being
  binary, for the same reason.
- **Pecuniary jurisdiction** computed from the amount in issue against the
  bands in force on the relevant date (the bands changed after the Act
  commenced, so they are versioned with effect dates).
- **Limitation** (s.69 / s.24A) computed from the cause-of-action date — the
  date of the *breach*, not of the purchase. Where the facts show a continuing
  wrong (possession never delivered), it reports `indeterminate` with that
  reason rather than `barred`, because tribunals treat such causes of action as
  recurring. Missing dates give `indeterminate`, never a guess.
- **Both sides**: provisions the complainant relies on and those the opposite
  party relies on are separated for Agent 4.
- **Verification-first**: every provision goes through CiteVerify. This agent
  also fixed a real gap — the curated provisions list held whole-section titles
  only, so s.2(11) and s.2(47) (the refs Agent 1 emits) could not be
  authenticated and verified as UNVERIFIED. The corpus sub-clauses are now
  registered into that list (merge-only; hand-curated titles win on conflict),
  keeping CiteVerify the single verification gate.
- **Rules vs LLM**: offline by default; `StatuteRetrievalAgent(llm=...)` blends
  an LLM applicability score 50/50 into the rule score. The LLM re-scores only
  provisions already in the curated corpus, so it cannot invent a section —
  tested (`test_llm_cannot_introduce_a_section`).
- **Known limits, stated plainly**: the corpus holds *plain-language summaries,
  not gazette text* — every entry is flagged `verbatim=False` and the flag
  propagates to every match and to the bundle. Chapter VI (product liability)
  is entered as a chapter, not a section, because this repo's two sources
  disagree on its numbering; it carries `numbering_verified=False` and is not
  citable as-is. The pecuniary bands are likewise unverified and the agent
  warns whenever it relies on them. Before any live use, replace the summaries
  and the bands with the Gazette of India text.

## Precedent Retrieval Agent (`agents/precedent_retrieval.py`)
Agent 3. Turns a fact sheet into a ranked, verified set of governing prior
decisions, plus the outcome prior the Prediction Agent conditions on.
```bash
# seed the DB first: python scripts/run_pipeline.py
python scripts/run_precedent_agent.py data/ncdrc_judgments/gunjan_aggarwal_257_2019.txt
python scripts/run_precedent_agent.py --judgment-id 1 --top-k 3     # case already in the DB
python scripts/run_precedent_agent.py case.txt --json
python -m unittest tests.test_precedent_retrieval -v                # 36 tests
# API: POST /api/precedents  {"judgment_id": 1}  or  {"case_text": "...", "query": "..."}
```
- **Query construction**: builds a precedent query from the fact sheet
  (category + CPA hooks + fact-pattern signals + relief sought), not from the
  user's wording — a precedent is relevant for its issue, not its phrasing.
- **Retrieval**: existing hybrid BM25 + dense retriever over the structured
  legal DB, with an automatic BM25-only fallback on corpora too small to fit
  a dense index (reported in `retrieval_mode`, not hidden).
- **Legal re-ranking**: six transparent signals — retrieval score, dispute
  category, statutory/hook overlap, fact-pattern overlap, forum authority
  (Supreme Court > High Court > NCDRC > State > District), and recency
  (8-year half-life). Weights live in `SCORING_WEIGHTS`; every component is
  reported per match in `score_components` + `why_relevant`, so no ranking is
  unexplained.
- **Evidence per match**: candidate *ratio decidendi* sentences (extractive,
  offset-grounded, re-sliced and checked — never generated), the operative
  outcome (allowed / partly allowed / dismissed / disposed with directions),
  the relief actually granted (amounts, interest rate, relief types), and
  which side the authority helps.
- **Distinguishing factors**: where the precedent's facts diverge — the
  points opposing counsel would use to distinguish it. Argument Analysis
  needs these as much as the similarities.
- **Outcome-leakage control**: the case under consideration is excluded from
  its own results, detected three ways (judgment_id, case number, or ≥0.85
  text overlap). Without this, a decided judgment sitting in the corpus
  retrieves itself and hands Prediction the answer — the same concern the
  Fact Extraction Agent addresses by stripping the operative order.
- **Verification-first (Objective 4)**: every precedent surfaced, *and* every
  authority cited inside it (citation-graph expansion via the
  `precedent_citations` table), is run through CiteVerify. Unverifiable
  citations are shown flagged UNVERIFIED with a reason and
  `admissible_as_authority=False` — never silently dropped, never silently
  passed.
- **Prior for Agent 5**: outcome distribution, complainant success rate, and
  an authority-weighted success rate (three District Forum orders should not
  outweigh one NCDRC decision). Only verified matches count toward either.
- **Rules vs LLM**: offline rule-based by default.
  `PrecedentRetrievalAgent(llm=AnthropicBackend())` blends an LLM relevance
  score 50/50 into the rule score. The LLM only re-scores candidates already
  retrieved from the authenticated corpus, so it can re-order but cannot
  introduce a case — the hallucination surface is zero by construction, and
  it is tested (`test_llm_cannot_introduce_a_case`). Any failure falls back
  to the rule ranking and says so in `warnings`.
- Known limits: with only 4 ingested judgments the outcome distribution is
  indicative, not a statistical prior (the agent emits this warning itself).
  Holding extraction is cue-based, so a holding stated without any of the
  usual markers ("we are of the view", "amounts to", "held that") is missed,
  and a long quotation of another court's reasoning can be picked up as the
  deciding court's own. Test fixtures in `tests/test_precedent_retrieval.py`
  are clearly-labelled synthetic judgments, not real decisions.

## Argument Analysis Agent (`agents/argument_analysis.py`)
Agent 4. Turns the facts, statutes and precedents into a structured argument
map: what each side contends on each issue, what supports it, what rebuts it,
and where the issue stands.
```bash
python scripts/run_argument_agent.py data/ncdrc_judgments/gunjan_aggarwal_257_2019.txt
python scripts/run_argument_agent.py --judgment-id 1 --json
python scripts/run_argument_agent.py case.txt --no-precedents       # Agents 1 -> 2 -> 4
python scripts/run_argument_agent.py case.txt --count-unverified    # ablation
python -m unittest tests.test_argument_analysis -v                  # 48 tests
# API: POST /api/arguments  {"judgment_id": 1}  or  {"case_text": "...", "query": "..."}
```
- **Issues, not one question.** A complaint raises threshold issues (consumer
  status, jurisdiction, limitation), merits issues (deficiency, defect, unfair
  trade practice) and a relief issue. Issues are framed only when something in
  *this* case raises them — a provision Agent 2 returned, a hook Agent 1
  emitted, or a pleaded sentence — so the map is of the dispute, not of the Act.
- **Contentions stay traceable.** Every contention keeps its fact ID,
  paragraph and offsets, so it can be read back in the source. Claims and
  defences are pleaded contentions; the complainant's *factual averments* are
  the case on the merits ("the seller failed to replace it" is the deficiency
  case; "I want a refund" is only the relief).
- **Voice attribution.** The tribunal is not a party. Sentences in the court's
  own voice are excluded from the map and listed separately — on a decided
  judgment the court's threshold findings *are* part of the outcome being
  predicted, so counting them would leak. Reported speech overrides this: "the
  objection taken by the OP is that this Commission lacks jurisdiction" is the
  OP's plea, not a finding. Procedural sentences that cannot be attributed to
  a party are left out rather than handed to the opposite party by default.
- **Rebuttal and unrebutted averments.** Rebuttal is tracked at issue level
  (sentence-to-sentence pairing on legal prose is more noise than signal), and
  contentions the other side never answered are reported — an uncontroverted
  averment is a real evidentiary point.
- **Four strength components per side**: factual grounding, statutory support
  (element coverage from Agent 2), precedential support (authority-weighted),
  and whether contentions were rebutted. Weights in `STRENGTH_WEIGHTS`, every
  component reported.
- **Precedents attach by outcome.** A dismissed complaint on comparable facts
  is authority for the opposite party. Threshold issues take only precedents
  that actually hold on that threshold, so a builder-delay authority is not
  treated as authority on limitation.
- **Threshold risk is kept separate from merits.** A strong merits case must
  not mask a fatal limitation problem, so `merits_balance` and
  `threshold_risk` are reported independently and a threshold issue leaning
  against the complainant is flagged as a `dispositive_risk`.
- **One-sided pleading records are flagged**, not presented as settled: a side
  can hold on-point admissible authority and still score zero because it
  pleaded nothing on that issue — usually an artefact of a decided judgment,
  where the answering argument sits in the tribunal's voice and is excluded.
- **Verification-first**: unverified provisions and inadmissible precedents are
  attached and visible with a reason, but excluded from the strength scores
  (`--count-unverified` flips this, for ablation only).
- **Evidential gaps** — unsatisfied statutory elements, sides with contentions
  but no support — are collected for the Explanation Agent to concede.
- Known limits: issue assignment is cue-based, so a contention phrased without
  the usual markers is missed and one phrased across two issues may attach to
  both. Voice attribution assumes the narrative in a judgment is the
  complainant's case, which is right for that input but would need the `party`
  field set explicitly for a two-sided pleading bundle.

## Prediction Agent (`agents/prediction.py`)
Agent 5. Predicts the disposition with a calibrated confidence, and shows its
working — the score decomposes exactly into named, auditable terms.
```bash
python scripts/run_prediction_agent.py data/ncdrc_judgments/gunjan_aggarwal_257_2019.txt
python scripts/run_prediction_agent.py --judgment-id 1 --json
python scripts/run_prediction_agent.py case.txt --no-precedents     # Agents 1 -> 2 -> 4 -> 5
python scripts/run_prediction_agent.py case.txt --no-prior          # ablate the precedential prior
python scripts/run_prediction_agent.py case.txt --abstain 0.0       # force a call on every case
python scripts/run_prediction_agent.py --backtest --judgment-id 1   # vs. the real disposition
python scripts/run_prediction_agent.py --backtest-all               # aggregate accuracy + Brier
python -m unittest tests.test_prediction -v                         # 62 tests
# API: POST /api/predict  {"judgment_id": 1}  or  {"case_text": "...", "query": "..."}
```
- **A transparent Bayesian update, not a black box.** The prior is the
  authority-weighted rate at which *verified* comparable decisions went for the
  complainant (Agent 3). This case's own evidence then moves it in log-odds:
  merits balance, statutory element coverage, unrebutted contentions, and a
  penalty for evidential gaps. Every term is reported in `drivers` with its
  log-odds contribution, and `posterior = logit(prior) + sum(drivers)` holds
  exactly — there's a test asserting it, so the confidence score can be audited
  term by term instead of taken on trust.
- **Small-corpus shrinkage.** Two concordant precedents give a raw success rate
  of 1.0; shrunk toward a neutral prior by a pseudo-count, that becomes 0.75 —
  what two cases actually support. This is the main guard against
  overconfidence in the regime this system really operates in.
- **Threshold issues are gated, not averaged.** A tribunal that holds a
  complaint time-barred never reaches deficiency, so a fatal threshold issue
  short-circuits the merits rather than being blended into an overall score —
  otherwise the model confidently predicts a win in a case being thrown out at
  the door. `decided_at` records which stage disposed of the case, and
  threshold risk is the *worst* threshold issue, not the mean, because they are
  alternative fatal grounds. Agent 2's computed findings (`prima_facie_barred`,
  a pecuniary forum mismatch) set a floor on that risk, since they come from
  dates and amounts rather than inferred from prose.
- **`allowed` vs `partly_allowed` is predicted, not collapsed.** Consumer
  commissions routinely grant the principal with interest while trimming the
  compensation claimed, so always predicting "allowed" for a complainant win
  would be wrong on most cases it got directionally right. The split is
  informed by the corpus distribution and by the strength of the relief issue.
- **Confidence is capped by the record behind it.** Directional certainty is
  cheap — two sentences and one precedent can produce a lopsided score — so
  confidence is scaled by an `evidence_quality` term (verified precedents,
  grounded contentions, statutory grounding, input completeness) and capped
  below certainty. No configuration can make this agent report 100%.
- **It abstains.** Below `abstain_threshold` the agent returns `indeterminate`
  and states what is missing, rather than guessing. `--abstain 0.0` forces a
  call on every case for a coverage/accuracy trade-off curve.
- **Counterfactuals.** "Dismissal, 71% confident" tells a user nothing they can
  act on; "clear the limitation objection and this becomes a 94% allowance"
  tells them what the case turns on. Each scenario is recomputed through the
  same model and the same disposition split, so these are real model outputs
  under changed inputs, not commentary.
- **Relief forecast, grounded twice.** A head is forecast only if the
  complainant claimed it *and* a verified precedent granted it; heads claimed
  but unprecedented are surfaced separately rather than dropped. Interest rates
  are the median of named comparable decisions. Quantum is explicitly the
  Commission's, not the model's.
- **Verification-first**: only CiteVerify-passed precedents enter the prior;
  the rest are listed in `excluded_from_prior` with a reason
  (`--count-unverified` flips this, for ablation only).
- **Leakage**: the agent reads only the four upstream bundles, never
  `state["case_text"]`, so it cannot see the order it is predicting. If Agent 1
  reports the outcome was *not* stripped, the prediction is flagged
  `leakage_suspect` — a backtest on such a case measures nothing.
- **Rules vs LLM**: offline by default; `PredictionAgent(llm=...)` admits an
  LLM's merits view only as a *bounded* log-odds shift reported as its own
  driver. It can shade a borderline case but never overturn the evidence — a
  maximally biased stub LLM cannot convert a case with a live dispositive
  ground into a predicted win; at most it raises uncertainty into abstention.
- **Backtesting** (`score_against_actual`) reports direction accuracy, exact
  disposition accuracy and Brier score separately, using the same outcome
  vocabulary Agent 3 reads off decided judgments. Direction is the headline;
  Brier penalises being right but overconfident, which is exactly what should
  happen to a legal prediction system.
- Known limits: the weights are reasoned defaults, not fitted — with a real
  labelled split they should be fit and the reported prior replaced by the
  corpus base rate. The shipped corpus is far too small for the backtest
  numbers to mean anything, and the harness says so on every run.

## Structure
- `prompt_correction/` — input correction module
- `data_pipeline/` — OCR, cleaning, raw/processed judgment storage
- `structuring/` — segmentation, NER, field extraction
- `retrieval/` — dense + sparse retrieval, re-ranking
- `agents/` — multi-agent reasoning pipeline (Agents 1-5 implemented)
- `statutes/` — CPA 2019 / 1986 provision corpus with statutory elements
- `verification/` — CiteVerify / SPMA
- `explanation/` — IRAC generation
- `api/` — FastAPI backend
- `frontend/` — React frontend
- `data/` — source judgments, CPA 2019 text
- `tests/`, `scripts/`, `docs/`
