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
  logSeq: 0,
  pixelCache: new Map(),
  cancel: false,
  serviceReachable: null,
  metricsAvailable: null,
  activeRunController: null,
  lastServiceAnnouncement: "",
  lastMetricsAnnouncement: "",
};

const els = {};

const DEFAULT_BUTTON_LABELS = {
  "refresh-button": "Refresh",
  "send-button": "Send",
  "batch-button": "Run batch",
  "mix-button": "Run mix",
  "live-button": "Start stimulator",
  "stop-button": "Stop",
};

function $(id) {
  return document.getElementById(id);
}

function cssVar(name, fallback) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

function initElements() {
  [
    "ready-state",
    "status-band",
    "readiness-item",
    "status-hint",
    "service-announcer",
    "backend-state",
    "model-state",
    "revision-state",
    "refresh-button",
    "metrics-grid",
    "metrics-announcer",
    "metric-total",
    "metric-rate",
    "metric-inflight",
    "metric-error-rate",
    "metric-p95",
    "metric-compute",
    "metric-queue",
    "metric-errors",
    "payload-mode",
    "request-panel",
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

function setButtonState(id, { disabled = false, busy = false, label = DEFAULT_BUTTON_LABELS[id] } = {}) {
  const button = els[id];
  button.disabled = disabled;
  button.classList.toggle("is-busy", busy);
  button.setAttribute("aria-busy", busy ? "true" : "false");
  button.textContent = label;
}

function announce(id, message, key) {
  if (!message || state[key] === message) return;
  state[key] = message;
  els[id].textContent = message;
}

function isAbortError(error) {
  return error?.name === "AbortError";
}

function beginRun(kind) {
  if (state.activeRunController) return null;
  state.cancel = false;
  state.activeRunController = new AbortController();
  els["request-panel"].setAttribute("aria-busy", "true");
  setButtonState("send-button", { disabled: true });
  setButtonState("batch-button", { disabled: true });
  setButtonState("mix-button", { disabled: true });
  setButtonState("live-button", { disabled: true, label: kind === "stimulator" ? "Stimulator on" : DEFAULT_BUTTON_LABELS["live-button"] });
  setButtonState("stop-button", { disabled: false });
  return state.activeRunController;
}

function endRun(controller) {
  if (controller && state.activeRunController !== controller) return;
  const stimulatorOn = Boolean(state.liveTimer);
  state.activeRunController = null;
  els["request-panel"].removeAttribute("aria-busy");
  setButtonState("send-button");
  setButtonState("batch-button", { disabled: stimulatorOn });
  setButtonState("mix-button", { disabled: stimulatorOn });
  setButtonState("live-button", { disabled: stimulatorOn, label: stimulatorOn ? "Stimulator on" : DEFAULT_BUTTON_LABELS["live-button"] });
  setButtonState("stop-button", { disabled: !stimulatorOn });
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
    const previousBackend = state.backend;
    state.metadata = metadata;
    state.backend = metadata.runtime?.backend || ready.backend || "mock";
    els["ready-state"].textContent = ready.status || "ready";
    els["backend-state"].textContent = state.backend;
    els["model-state"].textContent = metadata.model_id || MODEL_ID;
    els["revision-state"].textContent = metadata.model_revision || "unknown";
    setServiceState(false);
    state.serviceReachable = true;
    // Only regenerate the editor payload when the backend actually changes, so the periodic
    // self-heal below never clobbers a payload the user is editing. The format (uri vs base64
    // pixels) depends on the backend, so a stale `mock` must correct itself to `lewm`.
    if (state.backend !== previousBackend) {
      renderPayload();
    }
  } catch (error) {
    els["ready-state"].textContent = "unreachable";
    els["backend-state"].textContent = "unknown";
    setServiceState(true, error);
    // Log only on the transition to unreachable, so the 5s poll doesn't spam the call log.
    if (state.serviceReachable !== false) {
      addLog("status", false, 0, error.body || { message: error.message });
    }
    state.serviceReachable = false;
  }
}

function setServiceState(down, error) {
  const mockBackend = !down && state.backend === "mock";
  els["status-band"].classList.toggle("is-down", down);
  els["status-band"].classList.toggle("is-mock", mockBackend);
  els["readiness-item"].classList.toggle("is-down", down);
  els["backend-state"].parentElement.classList.toggle("is-mock", mockBackend);
  const hint = els["status-hint"];
  hint.classList.remove("is-error", "is-warning");
  if (down) {
    const reason = error?.message ? ` (${error.message})` : "";
    hint.textContent =
      `Service unreachable${reason}. Start the backend (\`make demo\`) or check WMCP_BACKEND, then Refresh.`;
    hint.classList.add("visible", "is-error");
    announce("service-announcer", "WMCP service unreachable. Start the backend or check WMCP_BACKEND.", "lastServiceAnnouncement");
  } else if (mockBackend) {
    const revision = els["revision-state"].textContent || "unknown";
    const revisionLabel = revision === "mock" ? " / revision=mock" : ` / revision=${revision}`;
    hint.textContent =
      `backend=mock${revisionLabel} means synthetic demo responses. Use make demo-lewm for real checkpoint inference.`;
    hint.classList.add("visible", "is-warning");
    announce("service-announcer", "WMCP service ready with mock backend. Responses are synthetic.", "lastServiceAnnouncement");
  } else {
    hint.textContent = "";
    hint.classList.remove("visible");
    announce("service-announcer", `WMCP service ready with backend ${state.backend}.`, "lastServiceAnnouncement");
  }
}

function setMetricsState(available) {
  const changed = state.metricsAvailable !== available;
  state.metricsAvailable = available;
  els["metrics-grid"].setAttribute("aria-busy", "false");
  els["metrics-grid"].classList.toggle("is-stale", !available);
  els["metrics-age"].classList.toggle("is-error", !available);
  els["metrics-age"].classList.toggle("is-warning", false);
  if (changed) {
    announce(
      "metrics-announcer",
      available ? "Prometheus metrics available." : "Prometheus metrics unavailable.",
      "lastMetricsAnnouncement",
    );
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

function drawSparkline(svg, matrix, color = cssVar("--teal", "#176f72")) {
  const series = matrix?.[0]?.values || [];
  const values = series.map((point) => Number(point[1])).filter((value) => Number.isFinite(value));
  svg.replaceChildren();
  if (values.length < 2) {
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", "18");
    text.setAttribute("y", "48");
    text.setAttribute("fill", cssVar("--muted", "#687068"));
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
  els["metrics-grid"].setAttribute("aria-busy", "true");
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
    els["metrics-age"].classList.remove("is-error", "is-warning");
    drawSparkline(els["chart-rate"], rateSeries, cssVar("--teal", "#176f72"));
    drawSparkline(els["chart-latency"], latencySeries, cssVar("--clay", "#b45b42"));
    drawSparkline(els["chart-errors"], errorSeries, cssVar("--bad", "#b3384b"));
    renderOperationTable(mergeMaps(
      vectorMap(rateByOperation, "operation", "rate"),
      vectorMap(errorsByOperation, "operation", "errorRate"),
      vectorMap(p95ByOperation, "operation", "latency"),
      vectorMap(computeByOperation, "operation", "compute"),
      vectorMap(queueByOperation, "operation", "queue"),
      vectorMap(candidatesByOperation, "operation", "candidates"),
      vectorMap(horizonByOperation, "operation", "horizon"),
    ));
    setMetricsState(true);
  } catch (error) {
    els["metrics-age"].textContent = "metrics unavailable";
    setMetricsState(false);
  }
}

async function postOperation(operation, payload, options = {}) {
  const requestBody = { ...payload, operation, model: MODEL_ID };
  const started = performance.now();
  const response = await fetchJson(`/api/wmcp/v1/models/${MODEL_ID}:${operation}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(requestBody),
    signal: options.signal,
  });
  const elapsed = performance.now() - started;
  addLog(operation, true, elapsed, response, requestBody);
  return response;
}

function addLog(operation, ok, elapsed, body, request = null) {
  const entry = { id: (state.logSeq += 1), operation, ok, elapsed, body, request, at: new Date() };
  state.log.unshift(entry);
  state.log = state.log.slice(0, 40);

  // Prepend a single node (instead of rebuilding) so an open/expanded entry stays open as new
  // calls stream in.
  const container = els["response-log"];
  const empty = container.querySelector(".empty-log");
  if (empty) empty.remove();
  container.prepend(createLogEntry(entry));
  while (container.children.length > 40) {
    container.lastElementChild.remove();
  }
}

function renderLog() {
  const container = els["response-log"];
  container.replaceChildren();
  if (state.log.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-log";
    empty.textContent = "No WMCP calls recorded in this browser session.";
    container.appendChild(empty);
    return;
  }
  state.log.forEach((entry) => container.appendChild(createLogEntry(entry)));
}

function createLogEntry(entry) {
  const details = document.createElement("details");
  details.className = "log-entry";
  details.dataset.entryId = String(entry.id);
  const statusClass = entry.ok ? "ok" : "error";
  const hasRequest = entry.request != null;
  details.innerHTML = `
    <summary>
      <span class="${statusClass}">${entry.ok ? "ok" : "error"}</span>
      <span>${entry.operation}</span>
      <span class="log-meta">${formatSeconds(entry.elapsed / 1000)} · ${entry.at.toLocaleTimeString()}</span>
    </summary>
    <div class="log-detail">
      <div class="log-toolbar">
        <button type="button" class="log-tab is-active" data-tab="response">Response</button>
        ${hasRequest ? '<button type="button" class="log-tab" data-tab="request">Request</button>' : ""}
        <span class="log-spacer"></span>
        <button type="button" class="copy-btn" title="Copy the visible JSON to the clipboard">Copy</button>
      </div>
      <pre class="json-view" data-tab="response"></pre>
    </div>
  `;
  details.querySelector(".json-view").innerHTML = highlightJson(previewJson(entry.body));
  return details;
}

// Delegated handler: tab switch (Response/Request) and copy-to-clipboard for any log entry.
function onLogClick(event) {
  const details = event.target.closest(".log-entry");
  if (!details) return;
  const entry = state.log.find((item) => String(item.id) === details.dataset.entryId);
  if (!entry) return;

  const tabButton = event.target.closest(".log-tab");
  if (tabButton) {
    const tab = tabButton.dataset.tab;
    details.querySelectorAll(".log-tab").forEach((button) => {
      button.classList.toggle("is-active", button === tabButton);
    });
    const view = details.querySelector(".json-view");
    view.dataset.tab = tab;
    view.innerHTML = highlightJson(previewJson(tab === "request" ? entry.request : entry.body));
    return;
  }

  const copyButton = event.target.closest(".copy-btn");
  if (copyButton) {
    const tab = details.querySelector(".json-view").dataset.tab || "response";
    const data = tab === "request" ? entry.request : entry.body;
    copyToClipboard(JSON.stringify(data, null, 2), copyButton);
  }
}

async function copyToClipboard(text, button) {
  const original = button.textContent;
  let copied = false;
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      copied = true;
    }
  } catch (error) {
    copied = false;
  }
  if (!copied) {
    // Fallback for non-secure contexts / older browsers.
    const area = document.createElement("textarea");
    area.value = text;
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.focus();
    area.select();
    try {
      copied = document.execCommand("copy");
    } catch (error) {
      copied = false;
    }
    area.remove();
  }
  button.textContent = copied ? "Copied" : "Copy failed";
  button.classList.toggle("is-copied", copied);
  window.setTimeout(() => {
    button.textContent = original;
    button.classList.remove("is-copied");
  }, 1300);
}

// The preview abbreviates giant payloads (base64 pixels, large inline action arrays) so the view
// stays readable and light; Copy always uses the full, untruncated value.
function previewJson(data) {
  if (data === undefined || data === null) return "(none)";
  return JSON.stringify(abbreviate(data), null, 2);
}

function abbreviate(value) {
  if (typeof value === "string") {
    return value.length > 140 ? `${value.slice(0, 96)}… [${value.length} chars]` : value;
  }
  if (Array.isArray(value)) {
    const limit = 16;
    const head = value.slice(0, limit).map(abbreviate);
    if (value.length > limit) head.push(`… [${value.length - limit} more of ${value.length}]`);
    return head;
  }
  if (value && typeof value === "object") {
    const out = {};
    for (const [key, val] of Object.entries(value)) out[key] = abbreviate(val);
    return out;
  }
  return value;
}

// Lightweight JSON syntax highlighter: tokenize the raw string, escape each piece, wrap in spans.
function highlightJson(jsonString) {
  const tokenRe = /("(?:\\u[a-fA-F0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(?:true|false)\b|\bnull\b|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/g;
  let html = "";
  let last = 0;
  let match;
  while ((match = tokenRe.exec(jsonString)) !== null) {
    html += escapeHtml(jsonString.slice(last, match.index));
    const token = match[0];
    let cls = "tok-num";
    if (token.startsWith('"')) {
      cls = /:\s*$/.test(token) ? "tok-key" : "tok-str";
    } else if (token === "true" || token === "false") {
      cls = "tok-bool";
    } else if (token === "null") {
      cls = "tok-null";
    }
    html += `<span class="${cls}">${escapeHtml(token)}</span>`;
    last = match.index + token.length;
  }
  html += escapeHtml(jsonString.slice(last));
  return html;
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
  const button = els["send-button"];
  if (button.disabled) return;
  setButtonState("send-button", { disabled: true, busy: true, label: "Sending..." });
  els["request-panel"].setAttribute("aria-busy", "true");
  let payload = null;
  try {
    payload = JSON.parse(els["payload-editor"].value);
    await postOperation(operation, payload);
    await refreshMetrics();
  } catch (error) {
    addLog(operation, false, 0, error.body || { message: error.message }, payload);
  } finally {
    setButtonState("send-button");
    if (!state.activeRunController) {
      els["request-panel"].removeAttribute("aria-busy");
    }
  }
}

async function runBatch() {
  const controller = beginRun("batch");
  if (!controller) return;
  const count = Math.max(1, Math.min(50, numberValue("batch-input", 6)));
  const operation = els["operation-select"].value;
  const baseSeed = numberValue("seed-input", 11);
  let completed = 0;
  try {
    for (let index = 0; index < count; index += 1) {
      if (state.cancel) break;
      setButtonState("batch-button", { disabled: true, busy: true, label: `Batch ${index + 1}/${count}` });
      const payload = buildPayload(operation, { seed: baseSeed + index });
      try {
        await postOperation(operation, payload, { signal: controller.signal });
        completed += 1;
      } catch (error) {
        if (isAbortError(error)) {
          addLog(operation, false, 0, {
            code: "OPERATOR_STOP",
            message: `Batch stopped by operator after ${completed} of ${count} calls.`,
          }, payload);
          break;
        }
        addLog(operation, false, 0, error.body || { message: error.message }, payload);
      }
    }
  } finally {
    endRun(controller);
  }
  await refreshMetrics();
}

function mixPayloads(seedOffset = 0) {
  const safeLeWM = state.backend === "lewm";
  const scoreCandidates = safeLeWM ? 6 : 32;
  const rolloutCandidates = safeLeWM ? 4 : 16;
  const planCandidates = safeLeWM ? 10 : 64;
  const horizon = safeLeWM ? 4 : 8;
  // One valid request per operation — a clean "all green" demo. Validation-error metrics are fed by
  // the traffic-generator and stress-tester (which include intentional invalids by design), so the
  // dashboard mix does not need to ship a guaranteed failure.
  return [
    ["score", buildPayload("score", { candidates: scoreCandidates, horizon, seed: 101 + seedOffset })],
    ["plan", buildPayload("plan", { candidates: planCandidates, horizon, iterations: safeLeWM ? 2 : 5, seed: 102 + seedOffset })],
    ["encode", buildPayload("encode", { seed: 103 + seedOffset })],
    ["rollout", buildPayload("rollout", { candidates: rolloutCandidates, horizon, seed: 104 + seedOffset })],
  ];
}

async function runPregeneratedMix(seedOffset = 0, options = {}) {
  const kind = options.stimulator ? "stimulator" : "mix";
  const controller = beginRun(kind);
  if (!controller) return;
  const payloads = mixPayloads(seedOffset);
  let completed = 0;
  try {
    for (const [index, [operation, payload]] of payloads.entries()) {
      if (state.cancel) break;
      if (kind === "mix") {
        setButtonState("mix-button", { disabled: true, busy: true, label: `Mix ${index + 1}/${payloads.length}` });
      } else {
        setButtonState("live-button", { disabled: true, busy: true, label: `Stimulating ${index + 1}/${payloads.length}` });
      }
      try {
        await postOperation(operation, payload, { signal: controller.signal });
        completed += 1;
      } catch (error) {
        if (isAbortError(error)) {
          addLog(operation, false, 0, {
            code: "OPERATOR_STOP",
            message: `Traffic run stopped by operator after ${completed} of ${payloads.length} calls.`,
          }, payload);
          break;
        }
        addLog(operation, false, 0, error.body || { message: error.message }, payload);
      }
    }
  } finally {
    endRun(controller);
  }
  await refreshMetrics();
}

function startStimulator() {
  if (state.liveTimer) return;
  state.cancel = false;
  let tick = 0;
  setButtonState("batch-button", { disabled: true });
  setButtonState("mix-button", { disabled: true });
  setButtonState("live-button", { disabled: true, label: "Stimulator on" });
  setButtonState("stop-button", { disabled: false });
  state.liveTimer = window.setInterval(() => {
    tick += 1;
    runPregeneratedMix(tick * 10, { stimulator: true });
  }, state.backend === "lewm" ? 9000 : 5000);
  runPregeneratedMix(0, { stimulator: true });
}

// Unified stop: cancels the stimulator interval and breaks any in-flight
// batch / mix loop on its next iteration.
function requestStop() {
  state.cancel = true;
  if (state.activeRunController) {
    state.activeRunController.abort();
  }
  if (state.liveTimer) {
    window.clearInterval(state.liveTimer);
    state.liveTimer = null;
  }
  if (!state.activeRunController) {
    endRun();
  } else {
    setButtonState("stop-button", { disabled: true, label: "Stopping..." });
  }
}

function bindEvents() {
  ["operation-select", "candidates-input", "horizon-input", "iterations-input", "seed-input"].forEach((id) => {
    els[id].addEventListener("change", renderPayload);
    els[id].addEventListener("input", renderPayload);
  });
  els["refresh-button"].addEventListener("click", async () => {
    setButtonState("refresh-button", { disabled: true, busy: true, label: "Refreshing..." });
    try {
      await refreshStatus();
      await refreshMetrics();
    } finally {
      setButtonState("refresh-button");
    }
  });
  els["send-button"].addEventListener("click", sendFromEditor);
  els["batch-button"].addEventListener("click", runBatch);
  els["mix-button"].addEventListener("click", () => runPregeneratedMix(0));
  els["live-button"].addEventListener("click", startStimulator);
  els["stop-button"].addEventListener("click", requestStop);
  els["payload-editor"].addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      sendFromEditor();
    }
  });
  els["response-log"].addEventListener("click", onLogClick);
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
  // Re-detect status (backend, readiness) alongside metrics so a `state.backend` that was stuck
  // at the `mock` default — e.g. the page loaded while the backend was briefly unreachable — heals
  // itself within one interval instead of requiring a manual Refresh.
  state.metricsTimer = window.setInterval(() => {
    refreshMetrics();
    refreshStatus();
  }, 5000);
}

boot();
