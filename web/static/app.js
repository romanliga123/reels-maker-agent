let sessionId = null;
let ws = null;

const fileInput = document.getElementById("file-input");
const uploadBtn = document.getElementById("upload-btn");
const progressSection = document.getElementById("progress-section");
const progressFeed = document.getElementById("progress-feed");
const candidatesSection = document.getElementById("candidates-section");
const candidatesList = document.getElementById("candidates-list");
const manualStart = document.getElementById("manual-start");
const manualEnd = document.getElementById("manual-end");
const manualAddBtn = document.getElementById("manual-add-btn");
const renderBtn = document.getElementById("render-btn");
const resultsSection = document.getElementById("results-section");
const resultsList = document.getElementById("results-list");
const downloadAllBtn = document.getElementById("download-all-btn");

fileInput.addEventListener("change", () => {
  uploadBtn.disabled = !fileInput.files.length;
});

async function ensureSession() {
  if (sessionId) return sessionId;
  const resp = await fetch("/api/session", { method: "POST" });
  const data = await resp.json();
  sessionId = data.session_id;
  connectWs();
  return sessionId;
}

function connectWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/${sessionId}`);
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    appendProgress(msg.text, msg.type);
    if (msg.type === "ready") {
      loadCandidates();
    }
    if (msg.type === "render_done") {
      loadResults();
    }
  };
}

function appendProgress(text, kind) {
  progressSection.classList.remove("hidden");
  const line = document.createElement("div");
  if (kind === "error" || kind === "ready") line.className = kind;
  line.textContent = text;
  progressFeed.appendChild(line);
  progressFeed.scrollTop = progressFeed.scrollHeight;
}

uploadBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) return;
  uploadBtn.disabled = true;
  await ensureSession();

  const formData = new FormData();
  formData.append("file", file);

  appendProgress(`⬆️ Загружаю ${file.name}…`, "progress");
  try {
    const resp = await fetch(`/api/${sessionId}/upload`, { method: "POST", body: formData });
    if (!resp.ok) {
      const err = await resp.json();
      appendProgress(`❌ ${err.detail || "Ошибка загрузки"}`, "error");
      uploadBtn.disabled = false;
      return;
    }
    appendProgress("✅ Файл загружен, начинаю анализ…", "progress");
  } catch (e) {
    appendProgress(`❌ ${e}`, "error");
    uploadBtn.disabled = false;
  }
});

function fmtTime(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

async function loadCandidates() {
  const resp = await fetch(`/api/${sessionId}/candidates`);
  const data = await resp.json();
  renderCandidates(data.candidates || []);
}

function renderCandidates(candidates) {
  candidatesSection.classList.remove("hidden");
  candidatesList.innerHTML = "";
  if (!candidates.length) {
    candidatesList.innerHTML = '<p class="hint">Кандидатов не найдено. Можно добавить клип вручную ниже.</p>';
    return;
  }
  for (const c of candidates) {
    candidatesList.appendChild(renderCandidateCard(c));
  }
}

function renderCandidateCard(c) {
  const card = document.createElement("div");
  card.className = "candidate-card" + (c.approved ? " approved" : "");
  card.dataset.id = c.id;

  card.innerHTML = `
    <div class="reason">${c.reason} <span class="score">(${c.score.toFixed(1)})</span></div>
    <div class="excerpt">${c.transcript_excerpt || ""}</div>
    <div class="row">
      <label>✅ <input type="checkbox" class="approve-cb" ${c.approved ? "checked" : ""}></label>
      <label>Начало <input type="number" class="start-input" value="${c.start.toFixed(1)}" step="0.5"></label>
      <label>Конец <input type="number" class="end-input" value="${c.end.toFixed(1)}" step="0.5"></label>
      <label><input type="radio" name="sub-${c.id}" class="sub-dynamic" ${c.subtitle_style === "dynamic" ? "checked" : ""}> Караоке</label>
      <label><input type="radio" name="sub-${c.id}" class="sub-static" ${c.subtitle_style === "static" ? "checked" : ""}> Статичные</label>
      <span class="hint">${fmtTime(c.start)}–${fmtTime(c.end)}</span>
    </div>
  `;

  const patch = (body) => fetch(`/api/${sessionId}/candidates/${c.id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  card.querySelector(".approve-cb").addEventListener("change", (e) => {
    card.classList.toggle("approved", e.target.checked);
    patch({ approved: e.target.checked });
  });
  card.querySelector(".start-input").addEventListener("change", (e) => {
    patch({ start: parseFloat(e.target.value) });
  });
  card.querySelector(".end-input").addEventListener("change", (e) => {
    patch({ end: parseFloat(e.target.value) });
  });
  card.querySelector(".sub-dynamic").addEventListener("change", () => patch({ subtitle_style: "dynamic" }));
  card.querySelector(".sub-static").addEventListener("change", () => patch({ subtitle_style: "static" }));

  return card;
}

manualAddBtn.addEventListener("click", async () => {
  const start = parseFloat(manualStart.value);
  const end = parseFloat(manualEnd.value);
  if (isNaN(start) || isNaN(end) || end <= start) {
    appendProgress("❌ Укажите корректные начало и конец", "error");
    return;
  }
  await ensureSession();
  const resp = await fetch(`/api/${sessionId}/candidates/manual`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ start, end }),
  });
  if (!resp.ok) {
    const err = await resp.json();
    appendProgress(`❌ ${err.detail || "Ошибка добавления клипа"}`, "error");
    return;
  }
  manualStart.value = "";
  manualEnd.value = "";
  loadCandidates();
});

renderBtn.addEventListener("click", async () => {
  renderBtn.disabled = true;
  const resp = await fetch(`/api/${sessionId}/render`, { method: "POST" });
  if (!resp.ok) {
    const err = await resp.json();
    appendProgress(`❌ ${err.detail || "Ошибка запуска рендера"}`, "error");
    renderBtn.disabled = false;
    return;
  }
  appendProgress("🎬 Рендер запущен…", "progress");
});

async function loadResults() {
  const resp = await fetch(`/api/${sessionId}/render/status`);
  const data = await resp.json();
  renderResults(data.results || {});
  renderBtn.disabled = false;
}

function renderResults(results) {
  resultsSection.classList.remove("hidden");
  resultsList.innerHTML = "";
  const entries = Object.entries(results);
  if (!entries.length) {
    resultsList.innerHTML = '<p class="hint">Пока нет готовых клипов.</p>';
    return;
  }
  for (const [clipId, r] of entries) {
    const row = document.createElement("div");
    row.className = "result-item";
    if (r.error) {
      row.innerHTML = `<span>❌ ${clipId}: ${r.error}</span>`;
    } else {
      row.innerHTML = `<span>🎬 ${clipId} (${r.duration.toFixed(1)}с)</span>
        <a href="/api/${sessionId}/download/${clipId}" download>Скачать</a>`;
    }
    resultsList.appendChild(row);
  }
}

downloadAllBtn.addEventListener("click", () => {
  window.location.href = `/api/${sessionId}/download_all`;
});
