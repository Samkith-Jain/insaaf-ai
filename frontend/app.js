const API_BASE = "http://localhost:8000/api";

const el = (id) => document.getElementById(id);

async function loadJudgments() {
  const listEl = el("judgments-list");
  try {
    const res = await fetch(`${API_BASE}/judgments`);
    const data = await res.json();
    if (!data.judgments || data.judgments.length === 0) {
      listEl.innerHTML = "<p>No judgments ingested yet. Run scripts/run_pipeline.py first.</p>";
      return;
    }
    listEl.innerHTML = data.judgments.map(j => `
      <div class="judgment-row" data-id="${j.judgment_id}">
        <span><strong>${j.case_number || "Unnumbered"}</strong> — ${j.forum || "Unknown forum"}</span>
        <span>${j.date || ""}</span>
      </div>
    `).join("");
    listEl.querySelectorAll(".judgment-row").forEach(row => {
      row.addEventListener("click", () => loadDetail(row.dataset.id));
    });
  } catch (e) {
    listEl.innerHTML = `<p style="color:red">Could not reach API at ${API_BASE}. Is api/server.py running?</p>`;
  }
}

async function loadDetail(id) {
  const res = await fetch(`${API_BASE}/judgments/${id}`);
  const d = await res.json();
  const panel = el("detail-panel");
  const content = el("detail-content");

  const statutes = (d.statute_citations || [])
    .map(s => `<span class="citation-tag">Sec. ${s.section} of ${s.act}</span>`).join("") || "<em>none detected</em>";
  const precedents = (d.precedent_citations || [])
    .map(p => `<span class="citation-tag">${p.case_name}${p.year ? " (" + p.year + ")" : ""}</span>`).join("") || "<em>none detected</em>";

  content.innerHTML = `
    <div class="row"><span class="label">Case No.</span><span>${d.case_number || "-"}</span></div>
    <div class="row"><span class="label">Forum</span><span>${d.forum || "-"}</span></div>
    <div class="row"><span class="label">Date</span><span>${d.date || "-"}</span></div>
    <div class="row"><span class="label">Bench</span><span>${d.bench || "-"}</span></div>
    <div class="row"><span class="label">Statutes</span><span>${statutes}</span></div>
    <div class="row"><span class="label">Precedents</span><span>${precedents}</span></div>
    <p class="label" style="margin-top:0.75rem">Full text</p>
    <pre>${(d.full_text || "").slice(0, 4000)}</pre>
  `;
  panel.classList.remove("hidden");
  panel.scrollIntoView({ behavior: "smooth" });
}

async function runQuery() {
  const query = el("query-input").value.trim();
  if (!query) return;
  const btn = el("submit-btn");
  btn.disabled = true;
  btn.textContent = "Running...";

  try {
    const res = await fetch(`${API_BASE}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: 5 }),
    });
    const data = await res.json();
    if (data.error) {
      alert(data.error);
      return;
    }
    renderCorrection(data);
    renderResults(data.results);
  } catch (e) {
    alert("Could not reach API. Is api/server.py running on port 8000?");
  } finally {
    btn.disabled = false;
    btn.textContent = "Run pipeline";
  }
}

function renderCorrection(data) {
  el("correction-panel").classList.remove("hidden");
  el("orig-query").textContent = data.original_query;
  el("corrected-query").textContent = data.corrected_query;

  const fixesEl = el("spelling-fixes");
  fixesEl.innerHTML = (data.spelling_fixes || []).length
    ? "Spelling fixes: " + data.spelling_fixes.map(f => `<span class="fix-chip">${f.original} → ${f.corrected}</span>`).join("")
    : "";

  const notesEl = el("reformulation-notes");
  notesEl.innerHTML = (data.reformulation_notes || []).length
    ? data.reformulation_notes.map(n => `<div>• ${n}</div>`).join("")
    : "";
}

function renderResults(results) {
  const panel = el("results-panel");
  const list = el("results-list");
  panel.classList.remove("hidden");
  if (!results || results.length === 0) {
    list.innerHTML = "<p>No results.</p>";
    return;
  }
  list.innerHTML = results.map(r => `
    <div class="result-card" data-id="${r.judgment_id}">
      <span class="case-no">${r.case_number || "Unnumbered"}</span>
      <span class="forum-badge">${r.forum || "?"}</span>
      <div class="scores">hybrid=${r.hybrid_score} · bm25=${r.bm25_score} · dense=${r.dense_score}</div>
      <div class="snippet">${r.snippet}</div>
    </div>
  `).join("");
  list.querySelectorAll(".result-card").forEach(card => {
    card.addEventListener("click", () => loadDetail(card.dataset.id));
  });
}

el("submit-btn").addEventListener("click", runQuery);
el("query-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && e.ctrlKey) runQuery();
});

loadJudgments();
