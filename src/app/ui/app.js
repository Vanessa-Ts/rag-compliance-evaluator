// ── Utilities ────────────────────────────────────────────────────────────────

function showEl(id) { document.getElementById(id).classList.remove("hidden"); }
function hideEl(id) { document.getElementById(id).classList.add("hidden"); }

function showToast(msg, ms = 1500) {
  const toast = document.getElementById("toast");
  toast.textContent = msg;
  toast.classList.remove("hidden");
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => toast.classList.add("hidden"), ms);
}

function animateValue(el, target, suffix, duration = 600) {
  const start = performance.now();
  function step(now) {
    const pct = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - pct, 3);
    const val = target * eased;
    el.textContent = (suffix === " ms" ? val.toFixed(0) : val.toFixed(1)) + suffix;
    if (pct < 1) requestAnimationFrame(step);
    else el.textContent = (suffix === " ms" ? target.toFixed(0) : target.toFixed(1)) + suffix;
  }
  requestAnimationFrame(step);
}

// ── Skeleton ─────────────────────────────────────────────────────────────────

function showSkeleton() {
  hideEl("answer-box");
  hideEl("ask-error");
  showEl("ask-skeleton");
}

function hideSkeleton() {
  hideEl("ask-skeleton");
}

// ── Sample chips ─────────────────────────────────────────────────────────────

function fillSample(text) {
  const ta = document.getElementById("question");
  ta.value = text;
  ta.focus();
}

// ── Copy answer ──────────────────────────────────────────────────────────────

function copyAnswer() {
  const text = document.getElementById("answer-text").innerText;
  navigator.clipboard.writeText(text).then(() => showToast("Copied to clipboard"));
}

// ── Citations ─────────────────────────────────────────────────────────────────

const JURISDICTION_COLORS = {
  EU: "#60a5fa", DE: "#4ade80", FR: "#a78bfa",
  ES: "#fb923c", NL: "#34d399",
};

function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function renderCitations(citations) {
  const container = document.getElementById("citations");
  container.innerHTML = "";
  if (!citations || !citations.length) return;

  const list = document.createElement("div");
  list.className = "citation-list";

  citations.forEach((c) => {
    const card = document.createElement("div");
    card.className = "citation-card";
    card.style.borderLeftColor = JURISDICTION_COLORS[c.jurisdiction] || "var(--border)";

    const scorePct = Math.round(Math.max(0, Math.min(1, c.score)) * 100);
    card.innerHTML = `
      <div class="citation-card-header">
        <a href="${escHtml(c.source_url)}" target="_blank" rel="noopener">${escHtml(c.title)}</a>
        <span class="badge">${escHtml(c.jurisdiction)}</span>
      </div>
      <div class="score-bar-wrap">
        <div class="score-bar-track"><div class="score-bar-fill" style="width:${scorePct}%"></div></div>
        <span class="score-label">score ${c.score.toFixed(3)}</span>
      </div>
      <div class="snippet">${escHtml(c.snippet)}</div>`;
    list.appendChild(card);
  });

  container.appendChild(list);
}

// ── SSE stream parser ─────────────────────────────────────────────────────────

async function* readSSE(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop();
    for (const part of parts) {
      const lines = part.trim().split("\n");
      let eventType = "message", data = "";
      for (const line of lines) {
        if (line.startsWith("event: ")) eventType = line.slice(7).trim();
        if (line.startsWith("data: "))  data      = line.slice(6);
      }
      if (data) yield { eventType, data: JSON.parse(data) };
    }
  }
}

// ── Ask ───────────────────────────────────────────────────────────────────────

async function askQuestion() {
  const question = document.getElementById("question").value.trim();
  if (!question) return;

  const jurisdiction = document.getElementById("jurisdiction").value || null;
  const btn = document.getElementById("ask-btn");
  const answerEl = document.getElementById("answer-text");
  const box = document.getElementById("answer-box");

  btn.disabled = true;
  btn.textContent = "Asking…";
  showSkeleton();
  hideEl("ask-error");

  let accumulated = "";
  let firstToken = true;

  try {
    const resp = await fetch("/query/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, k: 4, jurisdiction }),
    });
    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || resp.statusText);
    }

    for await (const { eventType, data } of readSSE(resp)) {
      if (eventType === "token") {
        if (firstToken) {
          hideSkeleton();
          answerEl.textContent = "";
          box.classList.remove("hidden", "fadein");
          void box.offsetWidth;
          box.classList.add("fadein");
          firstToken = false;
        }
        accumulated += data.token;
        answerEl.textContent = accumulated;

      } else if (eventType === "done") {
        answerEl.innerHTML = (typeof marked !== "undefined")
          ? marked.parse(data.answer)
          : escHtml(data.answer).replace(/\n/g, "<br>");
        document.getElementById("answer-meta").textContent =
          `Provider: ${data.provider} · Model: ${data.model} · ${data.latency_ms.toFixed(0)} ms`;
        renderCitations(data.citations);

      } else if (eventType === "error") {
        throw new Error(data.detail || "Stream error");
      }
    }
  } catch (e) {
    document.getElementById("ask-error").textContent = e.message;
    showEl("ask-error");
  } finally {
    hideSkeleton();
    btn.disabled = false;
    btn.textContent = "Ask";
  }
}

document.getElementById("question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); askQuestion(); }
});

// ── Eval progress ─────────────────────────────────────────────────────────────

let _evalStart = 0;

function setProgress(done, total) {
  const pct = total > 0 ? Math.round((done / total) * 100) : 0;
  document.getElementById("eval-bar").style.width = pct + "%";

  let etaStr = "";
  if (done > 0 && done < total) {
    const elapsed = (Date.now() - _evalStart) / 1000;
    const remaining = Math.round((elapsed / done) * (total - done));
    etaStr = ` · ~${remaining}s left`;
  }
  document.getElementById("eval-progress-text").textContent =
    `Running… ${done} / ${total}  (${pct}%)${etaStr}`;
}

// ── Metric coloring ───────────────────────────────────────────────────────────

function colorMetricCard(cardId, value, goodThresh, warnThresh, invert = false) {
  const card = document.getElementById(cardId);
  card.classList.remove("metric-good", "metric-warn", "metric-bad");
  let cls;
  if (!invert) {
    cls = value >= goodThresh ? "metric-good" : value >= warnThresh ? "metric-warn" : "metric-bad";
  } else {
    cls = value <= goodThresh ? "metric-good" : value <= warnThresh ? "metric-warn" : "metric-bad";
  }
  card.classList.add(cls);
}

// ── Sort table ────────────────────────────────────────────────────────────────

const _sortState = { col: -1, dir: 1 };

function sortTable(colIndex) {
  const tbody = document.querySelector("#eval-detail tbody");
  if (!tbody) return;

  _sortState.dir = (_sortState.col === colIndex) ? _sortState.dir * -1 : 1;
  _sortState.col = colIndex;

  document.querySelectorAll("#eval-detail th.sortable").forEach((th, i) => {
    th.classList.remove("sort-asc", "sort-desc");
    const icon = th.querySelector(".sort-icon");
    if (i === colIndex) {
      th.classList.add(_sortState.dir === 1 ? "sort-asc" : "sort-desc");
      if (icon) icon.textContent = _sortState.dir === 1 ? "▲" : "▼";
    } else {
      if (icon) icon.textContent = "⇅";
    }
  });

  const rows = Array.from(tbody.querySelectorAll("tr.data-row"));
  rows.sort((a, b) => {
    const av = a.cells[colIndex]?.dataset.sort ?? a.cells[colIndex]?.textContent ?? "";
    const bv = b.cells[colIndex]?.dataset.sort ?? b.cells[colIndex]?.textContent ?? "";
    const an = parseFloat(av), bn = parseFloat(bv);
    if (!isNaN(an) && !isNaN(bn)) return (an - bn) * _sortState.dir;
    return av.localeCompare(bv) * _sortState.dir;
  });

  rows.forEach(row => {
    tbody.appendChild(row);
    const next = row.nextSibling;
    if (next && next.classList && next.classList.contains("expanded-row")) tbody.appendChild(next);
  });
}

function toggleRow(i) {
  const row = document.getElementById(`expanded-${i}`);
  const btn = document.getElementById(`expand-${i}`);
  const open = row.classList.toggle("open");
  btn.classList.toggle("open", open);
  btn.setAttribute("aria-expanded", String(open));
}

// ── Render eval results ───────────────────────────────────────────────────────

function renderEvalResults(summary, config, timestamp, perQuestion, animate = true) {
  const s = summary;

  colorMetricCard("mc-precision", s.retrieval_precision_at_k,        0.8,  0.5);
  colorMetricCard("mc-hitrate",   s.hit_rate_at_k,                   0.8,  0.5);
  colorMetricCard("mc-faithful",  s.mean_faithfulness,               0.75, 0.5);
  colorMetricCard("mc-relevance", s.mean_context_relevance ?? 0,     0.75, 0.5);
  colorMetricCard("mc-latency",   s.p95_latency_ms,                  1000, 2500, true);

  const set = (id, val, suffix) => {
    const el = document.getElementById(id);
    if (animate) animateValue(el, val, suffix);
    else el.textContent = (suffix === " ms" ? val.toFixed(0) : val.toFixed(1)) + suffix;
  };

  set("mv-precision", s.retrieval_precision_at_k * 100,    "%");
  set("mv-hitrate",   s.hit_rate_at_k * 100,               "%");
  set("mv-faithful",  s.mean_faithfulness * 100,           "%");
  set("mv-relevance", (s.mean_context_relevance ?? 0) * 100, "%");
  set("mv-latency",   s.p95_latency_ms,                    " ms");

  document.getElementById("eval-meta").textContent =
    `${s.n} questions · Provider: ${config.provider} · Model: ${config.model} · k=${config.k} · ${timestamp}`;

  const detail = document.getElementById("eval-detail");
  if (perQuestion && perQuestion.length) {
    const rows = perQuestion.map((q, i) => {
      const q_short = q.question.length > 52 ? q.question.slice(0, 52) + "…" : q.question;
      const docIds = (q.retrieved_doc_ids || []).join(", ") || "—";
      const ctxRel = (q.context_relevance_score ?? 0) * 100;
      const reasoning = q.faithfulness_reasoning ? escHtml(q.faithfulness_reasoning) : "";
      return `
        <tr class="data-row">
          <td><button class="expand-btn" onclick="toggleRow(${i})" id="expand-${i}" aria-expanded="false" aria-label="Expand">›</button></td>
          <td title="${escHtml(q.question)}" data-sort="${escHtml(q.question)}">${escHtml(q_short)}</td>
          <td>${escHtml(q.jurisdiction || "—")}</td>
          <td data-sort="${q.precision_at_k}">${(q.precision_at_k * 100).toFixed(0)}%</td>
          <td class="${q.hit ? 'pass' : 'fail'}" data-sort="${q.hit ? 1 : 0}">${q.hit ? "✓" : "✗"}</td>
          <td class="${q.faithful ? 'pass' : 'fail'}" data-sort="${q.faithful ? 1 : 0}">${q.faithful ? "✓" : "✗"}</td>
          <td data-sort="${ctxRel}">${ctxRel.toFixed(0)}%</td>
          <td data-sort="${q.latency_ms}">${q.latency_ms.toFixed(0)}</td>
        </tr>
        <tr class="expanded-row" id="expanded-${i}">
          <td colspan="8">Retrieved: ${escHtml(docIds)}${reasoning ? `<br><em>Faithfulness: ${reasoning}</em>` : ""}</td>
        </tr>`;
    }).join("");

    detail.innerHTML = `<div class="result-table-wrap">
      <table class="result-table">
        <thead><tr>
          <th style="width:2rem"></th>
          <th class="sortable" scope="col" onclick="sortTable(1)">Question<span class="sort-icon">⇅</span></th>
          <th class="sortable" scope="col" onclick="sortTable(2)">Jur<span class="sort-icon">⇅</span></th>
          <th class="sortable" scope="col" onclick="sortTable(3)">Prec@k<span class="sort-icon">⇅</span></th>
          <th class="sortable" scope="col" onclick="sortTable(4)">Hit<span class="sort-icon">⇅</span></th>
          <th class="sortable" scope="col" onclick="sortTable(5)">Faithful<span class="sort-icon">⇅</span></th>
          <th class="sortable" scope="col" onclick="sortTable(6)">Ctx rel<span class="sort-icon">⇅</span></th>
          <th class="sortable" scope="col" onclick="sortTable(7)">Lat ms<span class="sort-icon">⇅</span></th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
  } else {
    detail.innerHTML = "";
  }

  showEl("eval-box");
}

// ── Run evaluation ────────────────────────────────────────────────────────────

async function runEval() {
  const btn = document.getElementById("eval-btn");

  _evalStart = Date.now();
  btn.disabled = true;
  btn.textContent = "Running…";
  hideEl("eval-box");
  hideEl("eval-error");
  showEl("eval-progress");
  setProgress(0, 0);

  const perQuestion = [];

  try {
    const resp = await fetch("/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || resp.statusText);
    }
    const { job_id } = await resp.json();

    await new Promise((resolve, reject) => {
      const source = new EventSource(`/evaluate/stream/${job_id}`);
      source.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        if (msg.event === "progress") {
          perQuestion.push(msg.result);
          setProgress(msg.n_done, msg.n_total);
        } else if (msg.event === "done") {
          source.close();
          hideEl("eval-progress");
          renderEvalResults(msg.summary, msg.config, msg.timestamp, perQuestion, true);
          resolve();
        }
      };
      source.onerror = () => {
        source.close();
        reject(new Error("Stream connection lost. The evaluation may still be running — try View last result."));
      };
    });
  } catch (e) {
    hideEl("eval-progress");
    document.getElementById("eval-error").textContent = e.message;
    showEl("eval-error");
  } finally {
    btn.disabled = false;
    btn.textContent = "▶ Run evaluation";
  }
}

// ── Load last eval ────────────────────────────────────────────────────────────

async function loadLastEval() {
  const btn = document.getElementById("last-btn");

  btn.disabled = true;
  btn.textContent = "Loading…";
  hideEl("eval-box");
  hideEl("eval-error");

  try {
    const resp = await fetch("/evaluate/last");
    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || resp.statusText);
    }
    const data = await resp.json();
    renderEvalResults(data.summary, data.config, data.timestamp, data.per_question, false);
  } catch (e) {
    document.getElementById("eval-error").textContent = e.message;
    showEl("eval-error");
  } finally {
    btn.disabled = false;
    btn.textContent = "↺ View last result";
  }
}
