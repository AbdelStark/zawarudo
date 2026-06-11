const MODEL_ID = "lewm-pusht";
const ACTION_DIM = 10;
const IMAGE_SIZE = 224;
const HISTORY = 3;

const state = {
  metadata: null,
  backend: "mock",
  liveTimer: null,
  metricsTimer: null,
  log: [],
  pixelCache: new Map(),
};

const els = {};

function $(id) {
  return document.getElementById(id);
}

function initElements() {
  [
    "ready-state",
    "backend-state",
    "model-state",
    "revision-state",
    "refresh-button",
    "metric-total",
    "metric-rate",
    "metric-inflight",
    "metric-error-rate",
    "metric-p95",
    "metric-compute",
    "metric-queue",
    "metric-errors",
    "payload-mode",
    "operation-select",
    "candidates-input",
    "horizon-input",
    "iterations-input",
    "seed-input",
    "batch-input",
    "send-button",
    "batch-button",
    "mix-button",
    "live-button",
    "stop-button",
    "payload-editor",
    "chart-rate",
    "chart-latency",
    "chart-errors",
    "operation-table",
    "metrics-age",
    "response-log",
    "clear-log-button",
  ].forEach((id) => {
    els[id] = $(id);
  });
}

function numberValue(id, fallback) {
  const value = Number(els[id].value);
  return Number.isFinite(value) ? value : fallback;
}

function formatSeconds(value) {
  if (!Number.isFinite(value)) return "n/a";
  if (value < 0.001) return `${(value * 1000000).toFixed(0)} us`;
  if (value < 1) return `${(value * 1000).toFixed(1)} ms`;
  return `${value.toFixed(2)} s`;
}

function formatRate(value) {
  if (!Number.isFinite(value)) return "0/s";
  if (value < 1) return `${value.toFixed(2)}/s`;
  return `${value.toFixed(1)}/s`;
}

function formatCount(value) {
  if (!Number.isFinite(value)) return "0";
  return new Intl.NumberFormat().format(Math.round(value));
}

function formatPerMinute(value) {
  if (!Number.isFinite(value)) return "0/min";
  if (value < 1) return `${value.toFixed(2)}/min`;
  return `${value.toFixed(1)}/min`;
}

function formatWorkload(row) {
  const parts = [];
  if (Number.isFinite(row.candidates)) parts.push(`S${Math.round(row.candidates)}`);
  if (Number.isFinite(row.horizon)) parts.push(`T${Math.round(row.horizon)}`);
  return parts.length ? parts.join(" / ") : "n/a";
}

function requestId(operation, seed) {
  return `ui-${operation}-${Date.now()}-${seed}`;
}

function rng(seed) {
  let stateValue = seed >>> 0;
  return () => {
    stateValue = (stateValue * 1664525 + 1013904223) >>> 0;
    return stateValue / 4294967296;
  };
}

function randomBase64Bytes(byteLength, seed) {
  const key = `${byteLength}:${seed}`;
  if (state.pixelCache.has(key)) {
    return state.pixelCache.get(key);
  }
  const next = rng(seed);
  const bytes = new Uint8Array(byteLength);
  for (let index = 0; index < byteLength; index += 1) {
    bytes[index] = Math.floor(next() * 256);
  }
  let binary = "";
  const chunk = 8192;
  for (let offset = 0; offset < bytes.length; offset += chunk) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunk));
  }
  const encoded = btoa(binary);
  state.pixelCache.set(key, encoded);
  return encoded;
}

function tensorRef(shape, layout, options = {}) {
  const encoding = options.encoding || "uri";
  const ref = {
    kind: "tensor",
    encoding,
    dtype: options.dtype || "float32",
    shape,
    layout,
  };
  if (encoding === "inline") {
    ref.data = options.data;
  } else if (encoding === "base64") {
    ref.data_b64 = options.data_b64;
  } else {
    ref.uri = options.uri || "memory://dashboard/tensor.npy";
  }
  return ref;
}

function pixelNode(kind, seed) {
  const frames = kind === "goal" ? 1 : HISTORY;
  const shape = [1, frames, 3, IMAGE_SIZE, IMAGE_SIZE];
  const encoding = state.backend === "lewm" ? "base64" : "uri";
  const tensor = encoding === "base64"
    ? tensorRef(shape, kind === "goal" ? "B,G,C,224,224" : "B,H,C,224,224", {
      encoding,
      dtype: "uint8",
      data_b64: randomBase64Bytes(shape.reduce((a, b) => a * b, 1), seed),
    })
    : tensorRef(shape, kind === "goal" ? "B,G,C,224,224" : "B,H,C,224,224", {
      encoding,
      dtype: "uint8",
      uri: `memory://dashboard/${kind}.npy`,
    });
  return { modality: "rgb", tensor };
}

function actionData(candidates, horizon, seed) {
  const next = rng(seed);
  const batch = [];
  const candidateRows = [];
  for (let candidate = 0; candidate < candidates; candidate += 1) {
    const horizonRows = [];
    for (let step = 0; step < horizon; step += 1) {
      const action = [];
      for (let dim = 0; dim < ACTION_DIM; dim += 1) {
        action.push(Number((next() * 2 - 1).toFixed(5)));
      }
      horizonRows.push(action);
    }
    candidateRows.push(horizonRows);
  }
  batch.push(candidateRows);
  return batch;
}

function actionCandidates(candidates, horizon, seed) {
  return {
    space: "continuous",
    tensor: tensorRef([1, candidates, horizon, ACTION_DIM], "B,S,T,A", {
      encoding: "inline",
      dtype: "float32",
      data: actionData(candidates, horizon, seed),
    }),
    bounds: { low: Array(ACTION_DIM).fill(-1), high: Array(ACTION_DIM).fill(1) },
  };
}

function buildPayload(operation, overrides = {}) {
  const candidates = Number(overrides.candidates || numberValue("candidates-input", 16));
  const horizon = Number(overrides.horizon || numberValue("horizon-input", 8));
  const iterations = Number(overrides.iterations || numberValue("iterations-input", 5));
  const seed = Number(overrides.seed || numberValue("seed-input", 11));
  const base = {
    wmcp_version: "0.1",
    request_id: requestId(operation, seed),
    operation,
    model: MODEL_ID,
    inputs: {},
    parameters: { seed },
    return_options: {
      include_candidate_costs: true,
      include_best_index: true,
      include_diagnostics: true,
    },
  };

  if (operation === "encode") {
    base.inputs.observation_history = pixelNode("history", seed);
    base.parameters.history_size = HISTORY;
  }
  if (operation === "rollout") {
    base.inputs.observation_history = pixelNode("history", seed);
    base.inputs.action_candidates = actionCandidates(candidates, horizon, seed);
    base.parameters.history_size = HISTORY;
    base.parameters.horizon = horizon;
  }
  if (operation === "score") {
    base.inputs.observation_history = pixelNode("history", seed);
    base.inputs.goal = pixelNode("goal", seed + 1);
    base.inputs.action_candidates = actionCandidates(candidates, horizon, seed);
    base.parameters.history_size = HISTORY;
    base.parameters.horizon = horizon;
  }
  if (operation === "plan") {
    base.inputs.observation_history = pixelNode("history", seed);
    base.inputs.goal = pixelNode("goal", seed + 1);
    base.parameters = {
      planner: "cem",
      horizon,
      iterations,
      candidates,
      elite_fraction: 0.1,
      seed,
      action_bounds: { low: Array(ACTION_DIM).fill(-1), high: Array(ACTION_DIM).fill(1) },
    };
  }
  return base;
}

function renderPayload() {
  const operation = els["operation-select"].value;
  els["payload-editor"].value = JSON.stringify(buildPayload(operation), null, 2);
  els["payload-mode"].textContent = state.backend === "lewm" ? "base64 pixels" : "uri pixels";
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  let body = {};
  try {
    body = text ? JSON.parse(text) : {};
  } catch (error) {
    body = { raw: text };
  }
  if (!response.ok) {
    const detail = body.detail || body.error || body;
    const message = detail.message || response.statusText;
    const code = detail.code || "HTTP_ERROR";
    const err = new Error(`${response.status} ${code}: ${message}`);
    err.body = body;
    err.status = response.status;
    throw err;
  }
  return body;
}

async function refreshStatus() {
  try {
    const ready = await fetchJson("/api/readyz");
    const metadata = await fetchJson(`/api/wmcp/v1/models/${MODEL_ID}`);
    state.metadata = metadata;
    state.backend = metadata.runtime?.backend || ready.backend || "mock";
    els["ready-state"].textContent = ready.status || "ready";
    els["backend-state"].textContent = state.backend;
    els["model-state"].textContent = metadata.model_id || MODEL_ID;
    els["revision-state"].textContent = metadata.model_revision || "unknown";
    renderPayload();
  } catch (error) {
    els["ready-state"].textContent = "unreachable";
    els["backend-state"].textContent = "unknown";
    addLog("status", false, 0, error.body || { message: error.message });
  }
}

async function promQuery(query) {
  const url = `/prometheus/api/v1/query?query=${encodeURIComponent(query)}`;
  const body = await fetchJson(url);
  return body.data?.result || [];
}

async function promRange(query, seconds = 900, step = 15) {
  const end = Math.floor(Date.now() / 1000);
  const start = end - seconds;
  const params = new URLSearchParams({
    query,
    start: String(start),
    end: String(end),
    step: String(step),
  });
  const body = await fetchJson(`/prometheus/api/v1/query_range?${params.toString()}`);
  return body.data?.result || [];
}

function firstValue(vector) {
  const value = vector?.[0]?.value?.[1];
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function vectorMap(vector, label, valueName) {
  const out = new Map();
  vector.forEach((item) => {
    const key = item.metric?.[label] || "all";
    const value = Number(item.value?.[1]);
    out.set(key, { ...(out.get(key) || {}), [valueName]: Number.isFinite(value) ? value : 0 });
  });
  return out;
}

function mergeMaps(...maps) {
  const merged = new Map();
  maps.forEach((map) => {
    map.forEach((value, key) => {
      merged.set(key, { ...(merged.get(key) || {}), ...value });
    });
  });
  return merged;
}

function drawSparkline(svg, matrix, color = "#24746f") {
  const series = matrix?.[0]?.values || [];
  const values = series.map((point) => Number(point[1])).filter((value) => Number.isFinite(value));
  svg.replaceChildren();
  if (values.length < 2) {
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", "18");
    text.setAttribute("y", "48");
    text.setAttribute("fill", "#6a665d");
    text.textContent = "waiting for samples";
    svg.appendChild(text);
    return;
  }
  const width = 320;
  const height = 84;
  const pad = 10;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(max - min, 0.000001);
  const points = values.map((value, index) => {
    const x = pad + (index / (values.length - 1)) * (width - pad * 2);
    const y = height - pad - ((value - min) / span) * (height - pad * 2);
    return [x, y];
  });
  const line = points.map((point, index) => `${index === 0 ? "M" : "L"}${point[0].toFixed(2)},${point[1].toFixed(2)}`).join(" ");
  const area = `${line} L${points[points.length - 1][0].toFixed(2)},${height - pad} L${points[0][0].toFixed(2)},${height - pad} Z`;
  const areaPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
  areaPath.setAttribute("class", "spark-area");
  areaPath.setAttribute("d", area);
  const linePath = document.createElementNS("http://www.w3.org/2000/svg", "path");
  linePath.setAttribute("class", "spark-line");
  linePath.setAttribute("d", line);
  linePath.setAttribute("stroke", color);
  svg.append(areaPath, linePath);
}

function renderOperationTable(rows) {
  const container = els["operation-table"];
  container.replaceChildren();
  const header = document.createElement("div");
  header.className = "op-row";
  header.innerHTML = "<span>Operation</span><span>Rate</span><span>Error</span><span>p95</span><span>compute</span><span>queue</span><span>workload</span>";
  container.appendChild(header);
  ["score", "plan", "rollout", "encode"].forEach((operation) => {
    const row = rows.get(operation) || {};
    const element = document.createElement("div");
    element.className = "op-row";
    element.innerHTML = `
      <strong>${operation}</strong>
      <span>${formatPerMinute((row.rate || 0) * 60)}</span>
      <span>${formatPerMinute((row.errorRate || 0) * 60)}</span>
      <span>${formatSeconds(row.latency)}</span>
      <span>${formatSeconds(row.compute)}</span>
      <span>${formatSeconds(row.queue)}</span>
      <span>${formatWorkload(row)}</span>
    `;
    container.appendChild(element);
  });
}

async function refreshMetrics() {
  try {
    const [
      total,
      rate,
      inflight,
      errorRate,
      errors,
      p95,
      compute,
      queue,
      rateByOperation,
      errorsByOperation,
      p95ByOperation,
      computeByOperation,
      queueByOperation,
      candidatesByOperation,
      horizonByOperation,
      rateSeries,
      latencySeries,
      errorSeries,
    ] = await Promise.all([
      promQuery("sum(increase(wmcp_requests_total[15m]))"),
      promQuery("sum(rate(wmcp_requests_total[2m]))"),
      promQuery("sum(wmcp_inflight_requests) OR on() vector(0)"),
      promQuery("sum(rate(wmcp_request_errors_total[5m])) OR on() vector(0)"),
      promQuery("sum(increase(wmcp_input_validation_errors_total[15m]))"),
      promQuery("histogram_quantile(0.95, sum(rate(wmcp_request_latency_seconds_bucket[5m])) by (le))"),
      promQuery("histogram_quantile(0.95, sum(rate(wmcp_model_compute_seconds_bucket[5m])) by (le))"),
      promQuery("histogram_quantile(0.95, sum(rate(wmcp_queue_wait_seconds_bucket[5m])) by (le))"),
      promQuery("sum(rate(wmcp_requests_total[2m])) by (operation)"),
      promQuery("sum(rate(wmcp_request_errors_total[5m])) by (operation)"),
      promQuery("histogram_quantile(0.95, sum(rate(wmcp_request_latency_seconds_bucket[5m])) by (le, operation))"),
      promQuery("histogram_quantile(0.95, sum(rate(wmcp_model_compute_seconds_bucket[5m])) by (le, operation))"),
      promQuery("histogram_quantile(0.95, sum(rate(wmcp_queue_wait_seconds_bucket[5m])) by (le, operation))"),
      promQuery("histogram_quantile(0.95, sum(rate(wmcp_candidate_count_bucket[5m])) by (le, operation))"),
      promQuery("histogram_quantile(0.95, sum(rate(wmcp_rollout_horizon_bucket[5m])) by (le, operation))"),
      promRange("sum(rate(wmcp_requests_total[1m]))"),
      promRange("histogram_quantile(0.95, sum(rate(wmcp_request_latency_seconds_bucket[5m])) by (le))"),
      promRange("sum(rate(wmcp_request_errors_total[1m])) OR on() vector(0)"),
    ]);
    els["metric-total"].textContent = formatCount(firstValue(total));
    els["metric-rate"].textContent = formatRate(firstValue(rate));
    els["metric-inflight"].textContent = formatCount(firstValue(inflight));
    els["metric-error-rate"].textContent = formatRate(firstValue(errorRate));
    els["metric-p95"].textContent = formatSeconds(firstValue(p95));
    els["metric-compute"].textContent = formatSeconds(firstValue(compute));
    els["metric-queue"].textContent = formatSeconds(firstValue(queue));
    els["metric-errors"].textContent = formatCount(firstValue(errors));
    els["metrics-age"].textContent = new Date().toLocaleTimeString();
    drawSparkline(els["chart-rate"], rateSeries, "#24746f");
    drawSparkline(els["chart-latency"], latencySeries, "#b65f3b");
    drawSparkline(els["chart-errors"], errorSeries, "#b3384b");
    renderOperationTable(mergeMaps(
      vectorMap(rateByOperation, "operation", "rate"),
      vectorMap(errorsByOperation, "operation", "errorRate"),
      vectorMap(p95ByOperation, "operation", "latency"),
      vectorMap(computeByOperation, "operation", "compute"),
      vectorMap(queueByOperation, "operation", "queue"),
      vectorMap(candidatesByOperation, "operation", "candidates"),
      vectorMap(horizonByOperation, "operation", "horizon"),
    ));
  } catch (error) {
    els["metrics-age"].textContent = "metrics unavailable";
  }
}

async function postOperation(operation, payload) {
  const started = performance.now();
  const response = await fetchJson(`/api/wmcp/v1/models/${MODEL_ID}:${operation}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ ...payload, operation, model: MODEL_ID }),
  });
  const elapsed = performance.now() - started;
  addLog(operation, true, elapsed, response);
  return response;
}

function addLog(operation, ok, elapsed, body) {
  state.log.unshift({ operation, ok, elapsed, body, at: new Date() });
  state.log = state.log.slice(0, 40);
  renderLog();
}

function renderLog() {
  const container = els["response-log"];
  container.replaceChildren();
  if (state.log.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-log";
    empty.textContent = "No calls yet.";
    container.appendChild(empty);
    return;
  }
  state.log.forEach((entry) => {
    const detail = document.createElement("details");
    detail.className = "log-entry";
    const statusClass = entry.ok ? "ok" : "error";
    const statusText = entry.ok ? "ok" : "error";
    detail.innerHTML = `
      <summary>
        <span class="${statusClass}">${statusText}</span>
        <span>${entry.operation}</span>
        <span>${formatSeconds(entry.elapsed / 1000)} · ${entry.at.toLocaleTimeString()}</span>
      </summary>
      <pre>${escapeHtml(JSON.stringify(entry.body, null, 2))}</pre>
    `;
    container.appendChild(detail);
  });
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[char]);
}

async function sendFromEditor() {
  const operation = els["operation-select"].value;
  try {
    const payload = JSON.parse(els["payload-editor"].value);
    await postOperation(operation, payload);
    await refreshMetrics();
  } catch (error) {
    addLog(operation, false, 0, error.body || { message: error.message });
  }
}

async function runBatch() {
  const count = Math.max(1, Math.min(50, numberValue("batch-input", 6)));
  const operation = els["operation-select"].value;
  const baseSeed = numberValue("seed-input", 11);
  for (let index = 0; index < count; index += 1) {
    const payload = buildPayload(operation, { seed: baseSeed + index });
    try {
      await postOperation(operation, payload);
    } catch (error) {
      addLog(operation, false, 0, error.body || { message: error.message });
    }
  }
  await refreshMetrics();
}

function mixPayloads(seedOffset = 0) {
  const safeLeWM = state.backend === "lewm";
  const scoreCandidates = safeLeWM ? 6 : 32;
  const rolloutCandidates = safeLeWM ? 4 : 16;
  const planCandidates = safeLeWM ? 10 : 64;
  const horizon = safeLeWM ? 4 : 8;
  return [
    ["score", buildPayload("score", { candidates: scoreCandidates, horizon, seed: 101 + seedOffset })],
    ["plan", buildPayload("plan", { candidates: planCandidates, horizon, iterations: safeLeWM ? 2 : 5, seed: 102 + seedOffset })],
    ["encode", buildPayload("encode", { seed: 103 + seedOffset })],
    ["rollout", buildPayload("rollout", { candidates: rolloutCandidates, horizon, seed: 104 + seedOffset })],
    ["score", { wmcp_version: "0.1", inputs: {} }],
  ];
}

async function runPregeneratedMix(seedOffset = 0) {
  for (const [operation, payload] of mixPayloads(seedOffset)) {
    try {
      await postOperation(operation, payload);
    } catch (error) {
      addLog(operation, false, 0, error.body || { message: error.message });
    }
  }
  await refreshMetrics();
}

function startStimulator() {
  if (state.liveTimer) return;
  let tick = 0;
  els["live-button"].disabled = true;
  els["stop-button"].disabled = false;
  state.liveTimer = window.setInterval(() => {
    tick += 1;
    runPregeneratedMix(tick * 10);
  }, state.backend === "lewm" ? 9000 : 5000);
  runPregeneratedMix(0);
}

function stopStimulator() {
  if (state.liveTimer) {
    window.clearInterval(state.liveTimer);
    state.liveTimer = null;
  }
  els["live-button"].disabled = false;
  els["stop-button"].disabled = true;
}

function bindEvents() {
  ["operation-select", "candidates-input", "horizon-input", "iterations-input", "seed-input"].forEach((id) => {
    els[id].addEventListener("change", renderPayload);
    els[id].addEventListener("input", renderPayload);
  });
  els["refresh-button"].addEventListener("click", async () => {
    await refreshStatus();
    await refreshMetrics();
  });
  els["send-button"].addEventListener("click", sendFromEditor);
  els["batch-button"].addEventListener("click", runBatch);
  els["mix-button"].addEventListener("click", () => runPregeneratedMix(0));
  els["live-button"].addEventListener("click", startStimulator);
  els["stop-button"].addEventListener("click", stopStimulator);
  els["clear-log-button"].addEventListener("click", () => {
    state.log = [];
    renderLog();
  });
}

async function boot() {
  initElements();
  bindEvents();
  renderLog();
  renderPayload();
  await refreshStatus();
  await refreshMetrics();
  state.metricsTimer = window.setInterval(refreshMetrics, 5000);
}

boot();
