function fillSample(text) {
  const input = document.getElementById("question");
  input.value = text;
  input.focus();
}

async function askQuestion() {
  const question = document.getElementById("question").value.trim();
  if (!question) return;

  const jurisdiction = document.getElementById("jurisdiction").value || null;
  const btn = document.getElementById("ask-btn");
  const answerBox = document.getElementById("answer-box");
  const errorBox = document.getElementById("ask-error");

  btn.disabled = true;
  btn.textContent = "Asking…";
  answerBox.classList.add("hidden");
  errorBox.classList.add("hidden");

  try {
    const resp = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, k: 4, jurisdiction }),
    });
    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || resp.statusText);
    }
    const data = await resp.json();

    document.getElementById("answer-text").textContent = data.answer;
    document.getElementById("answer-meta").textContent =
      `Provider: ${data.provider} · Model: ${data.model} · ${data.latency_ms.toFixed(0)} ms`;

    const citeEl = document.getElementById("citations");
    citeEl.innerHTML = "";
    if (data.citations && data.citations.length) {
      const ul = document.createElement("ul");
      ul.className = "citations";
      data.citations.forEach((c) => {
        const li = document.createElement("li");
        li.innerHTML = `<a href="${c.source_url}" target="_blank">${c.title}</a>
          <span class="badge">${c.jurisdiction}</span>
          <span class="score">score ${c.score.toFixed(3)}</span>
          <div class="snippet">${c.snippet}</div>`;
        ul.appendChild(li);
      });
      citeEl.appendChild(ul);
    }

    answerBox.classList.remove("hidden");
  } catch (e) {
    errorBox.textContent = e.message;
    errorBox.classList.remove("hidden");
  } finally {
    btn.disabled = false;
    btn.textContent = "Ask";
  }
}

async function runEval() {
  const btn = document.getElementById("eval-btn");
  const evalBox = document.getElementById("eval-box");
  const errorBox = document.getElementById("eval-error");

  btn.disabled = true;
  btn.textContent = "Running…";
  evalBox.classList.add("hidden");
  errorBox.classList.add("hidden");

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
    const data = await resp.json();
    const s = data.summary;

    document.getElementById("eval-summary").innerHTML = `
      <table class="metrics-table">
        <tr><th>Questions</th><td>${s.n}</td></tr>
        <tr><th>Retrieval precision@k</th><td>${(s.retrieval_precision_at_k * 100).toFixed(1)}%</td></tr>
        <tr><th>Hit rate@k</th><td>${(s.hit_rate_at_k * 100).toFixed(1)}%</td></tr>
        <tr><th>Mean faithfulness</th><td>${(s.mean_faithfulness * 100).toFixed(1)}%</td></tr>
        <tr><th>Mean latency</th><td>${s.mean_latency_ms.toFixed(0)} ms</td></tr>
        <tr><th>p95 latency</th><td>${s.p95_latency_ms.toFixed(0)} ms</td></tr>
      </table>
      <div class="meta">Provider: ${data.config.provider} · Model: ${data.config.model} · k=${data.config.k} · ${data.timestamp}</div>`;

    const detail = document.getElementById("eval-detail");
    detail.innerHTML = "<pre>" + JSON.stringify(data.per_question, null, 2) + "</pre>";

    evalBox.classList.remove("hidden");
  } catch (e) {
    errorBox.textContent = e.message;
    errorBox.classList.remove("hidden");
  } finally {
    btn.disabled = false;
    btn.textContent = "Run evaluation";
  }
}

document.getElementById("question").addEventListener("keydown", (e) => {
  if (e.key === "Enter") askQuestion();
});
