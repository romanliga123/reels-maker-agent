let sessionId = null;
let ws = null;
let s3Enabled = false;
let configLoaded = (async () => {
  try {
    const resp = await fetch("/api/config");
    const data = await resp.json();
    s3Enabled = !!data.s3_enabled;
  } catch (e) {
    s3Enabled = false;
  }
})();

const fileInput = document.getElementById("file-input");
const uploadBtn = document.getElementById("upload-btn");
const progressSection = document.getElementById("progress-section");
const progressFeed = document.getElementById("progress-feed");
const candidatesSection = document.getElementById("candidates-section");
const candidatesList = document.getElementById("candidates-list");
const manualStart = document.getElementById("manual-start");
const manualEnd = document.getElementById("manual-end");
const manualAddBtn = document.getElementById("manual-add-btn");
const selectAllBtn = document.getElementById("select-all-btn");
const renderBtn = document.getElementById("render-btn");
const cancelRenderBtn = document.getElementById("cancel-render-btn");
const resultsSection = document.getElementById("results-section");
const resultsList = document.getElementById("results-list");
const downloadAllBtn = document.getElementById("download-all-btn");

const previewModal = document.getElementById("preview-modal");
const previewVideo = document.getElementById("preview-video");
const previewRange = document.getElementById("preview-range");
const previewMarkStartBtn = document.getElementById("preview-mark-start");
const previewMarkEndBtn = document.getElementById("preview-mark-end");
const previewSaveBtn = document.getElementById("preview-save-btn");
const previewCloseBtn = document.getElementById("preview-close-btn");
let previewCandidateId = null;
let previewStart = 0;
let previewEnd = 0;

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
    if (msg.type === "ping") return;
    if (msg.type === "stage_progress") {
      updateStageProgressLine(msg.text);
      return;
    }
    clearStageProgressLine();
    appendProgress(msg.text, msg.type);
    if (msg.type === "ready") {
      loadCandidates();
    }
    if (msg.type === "render_done") {
      loadResults();
    }
  };
  // Долгая загрузка большого файла может надолго оставить соединение без трафика —
  // некоторые прокси (в т.ч. у Render) рвут такие idle-сокеты. Переподключаемся
  // автоматически, чтобы не терять последующий прогресс анализа/рендера.
  ws.onclose = () => {
    if (sessionId) setTimeout(connectWs, 1000);
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

function updateStageProgressLine(text) {
  progressSection.classList.remove("hidden");
  let line = document.getElementById("stage-progress-line");
  if (!line) {
    line = document.createElement("div");
    line.id = "stage-progress-line";
    progressFeed.appendChild(line);
  }
  line.textContent = text;
  progressFeed.scrollTop = progressFeed.scrollHeight;
}

function clearStageProgressLine() {
  document.getElementById("stage-progress-line")?.remove();
}

uploadBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) return;
  uploadBtn.disabled = true;
  await ensureSession();
  await configLoaded;

  try {
    if (s3Enabled) {
      await uploadViaS3(file);
    } else {
      await uploadDirect(file);
    }
  } catch (e) {
    uploadBtn.disabled = false;
  }
});

async function uploadDirect(file) {
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
}

async function uploadViaS3(file) {
  appendProgress("⬆️ Получаю ссылку для загрузки…", "progress");
  let data;
  try {
    const urlResp = await fetch(`/api/${sessionId}/upload-url`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: file.name, size: file.size }),
    });
    if (!urlResp.ok) {
      const err = await urlResp.json();
      appendProgress(`❌ ${err.detail || "Ошибка получения ссылки на загрузку"}`, "error");
      uploadBtn.disabled = false;
      return;
    }
    data = await urlResp.json();
  } catch (e) {
    appendProgress(`❌ ${e}`, "error");
    uploadBtn.disabled = false;
    return;
  }

  appendProgress(`⬆️ Загружаю ${file.name} напрямую в хранилище…`, "progress");
  let completeBody = {};
  try {
    if (data.mode === "multipart") {
      completeBody = { parts: await uploadMultipart(file, data) };
    } else {
      await putFileWithProgress(data.upload_url, file, data.content_type);
    }
  } catch (e) {
    appendProgress(`❌ Загрузка не удалась: ${e.message || e}`, "error");
    uploadBtn.disabled = false;
    return;
  }

  appendProgress("✅ Файл загружен, начинаю анализ…", "progress");
  try {
    const completeResp = await fetch(`/api/${sessionId}/upload-complete`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(completeBody),
    });
    if (!completeResp.ok) {
      const err = await completeResp.json();
      appendProgress(`❌ ${err.detail || "Ошибка завершения загрузки"}`, "error");
      uploadBtn.disabled = false;
    }
  } catch (e) {
    appendProgress(`❌ ${e}`, "error");
    uploadBtn.disabled = false;
  }
}

async function uploadMultipart(file, { part_size, parts }) {
  const total = file.size;
  let uploadedBefore = 0;
  const completed = [];
  for (const part of parts) {
    const start = (part.part_number - 1) * part_size;
    const end = Math.min(start + part_size, total);
    const blob = file.slice(start, end);
    const etag = await putPartWithProgress(part.url, blob, (loaded) => {
      updateUploadProgressLine(Math.round(((uploadedBefore + loaded) / total) * 100));
    });
    uploadedBefore += end - start;
    completed.push({ part_number: part.part_number, etag });
    updateUploadProgressLine(Math.round((uploadedBefore / total) * 100));
  }
  return completed;
}

function putPartWithProgress(url, blob, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", url);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(e.loaded);
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(xhr.getResponseHeader("ETag"));
      else reject(new Error(`HTTP ${xhr.status}: ${xhr.responseText.slice(0, 300)}`));
    };
    xhr.onerror = () => reject(new Error("ошибка сети"));
    xhr.send(blob);
  });
}

function putFileWithProgress(url, file, contentType) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", url);
    // Должен совпасть с Content-Type, под который подписан presigned URL — иначе
    // S3 отвергнет запрос (подпись включает этот заголовок, если он был передан
    // при генерации ссылки). Без правильного типа браузер потом не проигрывает
    // видео по прямой ссылке, а скачивает его как application/octet-stream.
    if (contentType) xhr.setRequestHeader("Content-Type", contentType);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        updateUploadProgressLine(Math.round((e.loaded / e.total) * 100));
      }
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve();
      else reject(new Error(`HTTP ${xhr.status}: ${xhr.responseText.slice(0, 300)}`));
    };
    xhr.onerror = () => reject(new Error("ошибка сети"));
    xhr.send(file);
  });
}

function updateUploadProgressLine(pct) {
  progressSection.classList.remove("hidden");
  let line = document.getElementById("upload-progress-line");
  if (!line) {
    line = document.createElement("div");
    line.id = "upload-progress-line";
    progressFeed.appendChild(line);
  }
  line.textContent = `⬆️ Загрузка: ${pct}%`;
}

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
      <span class="hint" data-role="time-range">${fmtTime(c.start)}–${fmtTime(c.end)}</span>
      <button type="button" class="preview-btn secondary">▶ Просмотр</button>
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
  card.querySelector(".preview-btn").addEventListener("click", () => openPreview(c));

  return card;
}

function openPreview(candidate) {
  previewCandidateId = candidate.id;
  previewStart = candidate.start;
  previewEnd = candidate.end;
  if (previewVideo.dataset.sessionId !== sessionId) {
    previewVideo.src = `/api/${sessionId}/preview`;
    previewVideo.dataset.sessionId = sessionId;
  }
  previewVideo.onloadedmetadata = () => {
    previewVideo.currentTime = Math.min(candidate.start, previewVideo.duration || candidate.start);
  };
  updatePreviewRangeDisplay();
  previewModal.classList.remove("hidden");
  if (previewVideo.readyState >= 1) previewVideo.currentTime = candidate.start;
}

function updatePreviewRangeDisplay() {
  previewRange.textContent = `Начало: ${fmtTime(previewStart)} — Конец: ${fmtTime(previewEnd)}`;
}

function closePreview() {
  previewModal.classList.add("hidden");
  previewVideo.pause();
}

previewModal.querySelectorAll("[data-seek]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const delta = parseFloat(btn.dataset.seek);
    const duration = previewVideo.duration || Infinity;
    previewVideo.currentTime = Math.max(0, Math.min(previewVideo.currentTime + delta, duration));
  });
});

previewMarkStartBtn.addEventListener("click", () => {
  previewStart = previewVideo.currentTime;
  updatePreviewRangeDisplay();
});

previewMarkEndBtn.addEventListener("click", () => {
  previewEnd = previewVideo.currentTime;
  updatePreviewRangeDisplay();
});

previewSaveBtn.addEventListener("click", async () => {
  if (previewEnd <= previewStart) {
    appendProgress("❌ Конец должен быть позже начала", "error");
    return;
  }
  await fetch(`/api/${sessionId}/candidates/${previewCandidateId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ start: previewStart, end: previewEnd }),
  });
  const card = candidatesList.querySelector(`[data-id="${previewCandidateId}"]`);
  if (card) {
    card.querySelector(".start-input").value = previewStart.toFixed(1);
    card.querySelector(".end-input").value = previewEnd.toFixed(1);
    card.querySelector('[data-role="time-range"]').textContent = `${fmtTime(previewStart)}–${fmtTime(previewEnd)}`;
  }
  closePreview();
});

previewCloseBtn.addEventListener("click", closePreview);

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

selectAllBtn.addEventListener("click", async () => {
  const cards = candidatesList.querySelectorAll(".candidate-card");
  selectAllBtn.disabled = true;
  await Promise.all(Array.from(cards).map(async (card) => {
    const cb = card.querySelector(".approve-cb");
    if (cb.checked) return;
    cb.checked = true;
    card.classList.add("approved");
    await fetch(`/api/${sessionId}/candidates/${card.dataset.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ approved: true }),
    });
  }));
  selectAllBtn.disabled = false;
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
  cancelRenderBtn.classList.remove("hidden");
  cancelRenderBtn.disabled = false;
});

cancelRenderBtn.addEventListener("click", async () => {
  cancelRenderBtn.disabled = true;
  await fetch(`/api/${sessionId}/render/cancel`, { method: "POST" });
});

async function loadResults() {
  const resp = await fetch(`/api/${sessionId}/render/status`);
  const data = await resp.json();
  renderResults(data.results || {});
  renderBtn.disabled = false;
  cancelRenderBtn.classList.add("hidden");
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
