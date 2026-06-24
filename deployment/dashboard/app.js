const MODEL_ID = "lewm-pusht";
const ACTION_DIM = 10;
const IMAGE_SIZE = 224;
const HISTORY = 3;

const OP_COLORS = {
  score: "--teal",
  plan: "--olive",
  rollout: "--clay",
  encode: "--amber",
};

const state = {
  metadata: null,
  backend: "mock",
  liveTimer: null,
  metricsTimer: null,
  ageTimer: null,
  log: [],
  logSeq: 0,
  pixelCache: new Map(),
  cancel: false,
  serviceReachable: null,
  metricsAvailable: null,
  chartsReady: false,
  lastMetricsAt: null,
  activeRunController: null,
  lastServiceAnnouncement: "",
  lastMetricsAnnouncement: "",
};

const els = {};
const lineCharts = {};

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

function colorFor(token, fallback) {
  return cssVar(token, fallback);
}

function mix(token, withColor, amount, fallback) {
  // color-mix resolved lazily by the browser is fine inline; build a string the CSS engine reads.
  return `color-mix(in srgb, ${cssVar(token, fallback)}, ${withColor} ${amount}%)`;
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
    "spark-rate",
    "spark-p95",
    "spark-errors",
    "spark-compute",
    "livepulse",
    "now-rate",
    "now-latency",
    "now-errors",
    "foot-rate",
    "foot-errors",
    "legend-latency",
    "chart-composition",
    "composition-legend",
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

function axisRate(value) {
  return value < 1 ? value.toFixed(2) : value.toFixed(1);
}

function axisSeconds(value) {
  return `${value.toFixed(2)}s`;
}

function formatWorkload(row) {
  const parts = [];
  if (Number.isFinite(row.candidates)) parts.push(`S${Math.round(row.candidates)}`);
  if (Number.isFinite(row.horizon)) parts.push(`T${Math.round(row.horizon)}`);
  return parts.length ? parts.join(" / ") : "n/a";
}

function formatClock(date) {
  return date.toLocaleTimeString([], { hour12: false });
}

function formatAgo(date) {
  if (!date) return "waiting";
  const secs = Math.max(0, Math.round((Date.now() - date.getTime()) / 1000));
  if (secs < 2) return "just now";
  if (secs < 60) return `${secs}s ago`;
  const mins = Math.floor(secs / 60);
  return `${mins}m ${secs % 60}s ago`;
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
    setStateText(els["ready-state"], ready.status || "ready");
    setStateText(els["backend-state"], state.backend);
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
    setStateText(els["ready-state"], "unreachable");
    setStateText(els["backend-state"], "unknown");
    setServiceState(true, error);
    // Log only on the transition to unreachable, so the 5s poll doesn't spam the call log.
    if (state.serviceReachable !== false) {
      addLog("status", false, 0, error.body || { message: error.message });
    }
    state.serviceReachable = false;
  }
}

function setLivepulse(stateName) {
  els["livepulse"].dataset.state = stateName;
}

// Render a status value as a state dot + text. Built via DOM (not innerHTML) because the text can
// come from the service response; this keeps server-supplied strings out of the HTML parser.
function setStateText(el, text) {
  el.replaceChildren();
  const dot = document.createElement("span");
  dot.className = "state-dot";
  el.append(dot, document.createTextNode(text));
}

function setServiceState(down, error) {
  const mockBackend = !down && state.backend === "mock";
  els["status-band"].classList.toggle("is-down", down);
  els["status-band"].classList.toggle("is-mock", mockBackend);
  els["readiness-item"].classList.toggle("is-down", down);
  els["backend-state"].parentElement.classList.toggle("is-mock", mockBackend);
  if (down) setLivepulse("down");
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
  if (available) {
    setLivepulse("live");
  } else if (state.serviceReachable === false) {
    setLivepulse("down");
  } else {
    setLivepulse("stale");
  }
  if (changed) {
    announce(
      "metrics-announcer",
      available ? "Prometheus metrics available." : "Prometheus metrics unavailable.",
      "lastMetricsAnnouncement",
    );
  }
}

function tickAge() {
  const el = els["metrics-age"];
  if (state.serviceReachable === false) {
    el.textContent = "offline";
  } else if (state.metricsAvailable === false) {
    el.textContent = "no metrics";
  } else {
    el.textContent = formatAgo(state.lastMetricsAt);
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

// ----------------------------------------------------------------- charting

const SVGNS = "http://www.w3.org/2000/svg";

function svgEl(tag, attrs = {}) {
  const node = document.createElementNS(SVGNS, tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null) continue;
    node.setAttribute(key, String(value));
  }
  return node;
}

function extractSeries(matrix) {
  const series = matrix?.[0]?.values || [];
  return series
    .map((point) => [Number(point[0]), Number(point[1])])
    .filter(([t, v]) => Number.isFinite(t) && Number.isFinite(v));
}

// Monotone cubic interpolation (Fritsch–Carlson) over pixel-space points; returns an SVG path
// that follows the data without the overshoot a naive cubic spline produces.
function monotonePath(points) {
  const n = points.length;
  if (n === 0) return "";
  if (n === 1) return `M${points[0][0]},${points[0][1]}`;
  if (n === 2) return `M${points[0][0]},${points[0][1]} L${points[1][0]},${points[1][1]}`;

  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const dx = [];
  const slope = [];
  for (let i = 0; i < n - 1; i += 1) {
    const h = xs[i + 1] - xs[i] || 1e-6;
    dx.push(h);
    slope.push((ys[i + 1] - ys[i]) / h);
  }
  const m = new Array(n);
  m[0] = slope[0];
  m[n - 1] = slope[n - 2];
  for (let i = 1; i < n - 1; i += 1) {
    if (slope[i - 1] * slope[i] <= 0) {
      m[i] = 0;
    } else {
      m[i] = (slope[i - 1] + slope[i]) / 2;
    }
  }
  for (let i = 0; i < n - 1; i += 1) {
    if (slope[i] === 0) {
      m[i] = 0;
      m[i + 1] = 0;
    } else {
      const a = m[i] / slope[i];
      const b = m[i + 1] / slope[i];
      const s = a * a + b * b;
      if (s > 9) {
        const t = 3 / Math.sqrt(s);
        m[i] = t * a * slope[i];
        m[i + 1] = t * b * slope[i];
      }
    }
  }
  let d = `M${xs[0].toFixed(2)},${ys[0].toFixed(2)}`;
  for (let i = 0; i < n - 1; i += 1) {
    const c1x = xs[i] + dx[i] / 3;
    const c1y = ys[i] + (m[i] * dx[i]) / 3;
    const c2x = xs[i + 1] - dx[i] / 3;
    const c2y = ys[i + 1] - (m[i + 1] * dx[i]) / 3;
    d += ` C${c1x.toFixed(2)},${c1y.toFixed(2)} ${c2x.toFixed(2)},${c2y.toFixed(2)} ${xs[i + 1].toFixed(2)},${ys[i + 1].toFixed(2)}`;
  }
  return d;
}

function niceTicks(min, max) {
  if (!(max > min)) return [min];
  const mid = (min + max) / 2;
  return [max, mid, min];
}

function registerChart(id, config) {
  const svg = els[id];
  const canvas = svg.closest(".chart-canvas");
  const entry = {
    svg,
    canvas,
    tip: canvas.querySelector(".chart-tip"),
    accent: config.accent,
    accentToken: config.accentToken,
    format: config.format,
    axisFormat: config.axisFormat || config.format,
    seriesDefs: config.series,
    data: null,
    geom: null,
  };
  svg.closest(".chart-card").dataset.accent = config.accentToken.replace("--", "");
  lineCharts[id] = entry;
  if (canvas) {
    canvas.addEventListener("pointermove", (event) => onChartHover(entry, event));
    canvas.addEventListener("pointerleave", () => hideChartHover(entry));
  }
  if (typeof ResizeObserver !== "undefined") {
    const ro = new ResizeObserver(() => renderLineChart(entry));
    ro.observe(svg);
  }
  return entry;
}

function setLineChart(id, seriesData) {
  const entry = lineCharts[id];
  if (!entry) return;
  entry.data = seriesData;
  renderLineChart(entry);
}

function chartEmpty(svg, W, H, message) {
  svg.replaceChildren();
  const text = svgEl("text", { x: W / 2, y: H / 2 + 4, "text-anchor": "middle", class: "c-empty" });
  text.textContent = message;
  svg.appendChild(text);
}

function renderLineChart(entry) {
  const { svg } = entry;
  const rect = svg.getBoundingClientRect();
  const W = Math.max(Math.round(rect.width) || 600, 160);
  const H = Math.max(Math.round(rect.height) || 180, 90);
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("preserveAspectRatio", "none");

  const data = entry.data;
  const series = (data || []).map((s, index) => ({ ...entry.seriesDefs[index], points: s }));
  const primary = series.find((s) => s.primary) || series[0];
  if (!primary || primary.points.length < 2) {
    entry.geom = null;
    chartEmpty(svg, W, H, "waiting for samples");
    return;
  }

  const pad = { l: 44, r: 12, t: 14, b: 17 };
  const allValues = [];
  let tMin = Infinity;
  let tMax = -Infinity;
  series.forEach((s) => s.points.forEach(([t, v]) => {
    allValues.push(v);
    if (t < tMin) tMin = t;
    if (t > tMax) tMax = t;
  }));
  let yMin = Math.min(...allValues);
  let yMax = Math.max(...allValues);
  const span = yMax - yMin || Math.abs(yMax) || 1;
  yMax += span * 0.18;
  yMin = Math.max(0, yMin - span * 0.18);
  if (yMax === yMin) yMax = yMin + 1;
  const tSpan = tMax - tMin || 1;

  const plotW = W - pad.l - pad.r;
  const plotH = H - pad.t - pad.b;
  const xScale = (t) => pad.l + ((t - tMin) / tSpan) * plotW;
  const yScale = (v) => pad.t + (1 - (v - yMin) / (yMax - yMin)) * plotH;

  const frag = document.createDocumentFragment();

  // gridlines + y-axis labels
  niceTicks(yMin, yMax).forEach((tick, index) => {
    const y = yScale(tick);
    frag.appendChild(svgEl("line", {
      x1: pad.l, x2: W - pad.r, y1: y, y2: y,
      class: index === 2 ? "c-grid-base" : "c-grid",
    }));
    const label = svgEl("text", { x: pad.l - 8, y: y + 3, "text-anchor": "end", class: "c-axis-label" });
    label.textContent = entry.axisFormat(tick);
    frag.appendChild(label);
  });

  // x-axis time hints
  const spanMin = Math.round(tSpan / 60);
  const leftTick = svgEl("text", { x: pad.l, y: H - 5, "text-anchor": "start", class: "c-axis-label" });
  leftTick.textContent = spanMin >= 1 ? `-${spanMin}m` : "";
  const rightTick = svgEl("text", { x: W - pad.r, y: H - 5, "text-anchor": "end", class: "c-axis-label" });
  rightTick.textContent = "now";
  frag.append(leftTick, rightTick);

  // area for primary series
  const primaryPts = primary.points.map(([t, v]) => [xScale(t), yScale(v)]);
  const baseY = yScale(yMin);
  const linePath = monotonePath(primaryPts);
  const areaPath = `${linePath} L${primaryPts[primaryPts.length - 1][0].toFixed(2)},${baseY.toFixed(2)} L${primaryPts[0][0].toFixed(2)},${baseY.toFixed(2)} Z`;
  const accent = cssVar(entry.accentToken, entry.accent);
  frag.appendChild(svgEl("path", {
    class: "c-area", d: areaPath,
    fill: `color-mix(in srgb, ${accent}, transparent 84%)`,
  }));

  // secondary lines (drawn under the primary)
  series.forEach((s) => {
    if (s.primary) return;
    const pts = s.points.map(([t, v]) => [xScale(t), yScale(v)]);
    frag.appendChild(svgEl("path", {
      class: `c-line ${s.cls || "c-line--ghost"}`,
      d: monotonePath(pts),
      stroke: cssVar(s.colorToken, s.color),
    }));
  });

  // primary line
  frag.appendChild(svgEl("path", { class: "c-line", d: linePath, stroke: accent }));

  // live leading-edge dot + halo
  const last = primaryPts[primaryPts.length - 1];
  frag.appendChild(svgEl("circle", { class: "c-dot-halo live-halo", cx: last[0], cy: last[1], r: 4, fill: accent }));
  frag.appendChild(svgEl("circle", { class: "c-dot", cx: last[0], cy: last[1], r: 3.4, fill: accent }));

  const overlay = svgEl("g", { class: "c-overlay" });

  svg.replaceChildren(frag, overlay);

  // geometry for hover
  const samples = primary.points.map(([t], i) => ({
    x: primaryPts[i][0],
    y: primaryPts[i][1],
    t,
    vals: series.map((s) => ({
      label: s.label,
      color: cssVar(s.colorToken || entry.accentToken, s.color || entry.accent),
      v: s.points[i] ? s.points[i][1] : undefined,
      y: s.points[i] ? yScale(s.points[i][1]) : undefined,
    })),
  }));
  entry.geom = { samples, pad, W, H, overlay, multi: series.length > 1 };
}

function onChartHover(entry, event) {
  const geom = entry.geom;
  if (!geom || !geom.samples.length) return;
  const rect = entry.svg.getBoundingClientRect();
  const scaleX = geom.W / rect.width;
  const px = (event.clientX - rect.left) * scaleX;
  let best = geom.samples[0];
  let bestDist = Infinity;
  for (const sample of geom.samples) {
    const dist = Math.abs(sample.x - px);
    if (dist < bestDist) { bestDist = dist; best = sample; }
  }
  const o = geom.overlay;
  o.replaceChildren();
  o.appendChild(svgEl("line", { class: "c-crosshair", x1: best.x, x2: best.x, y1: geom.pad.t, y2: geom.H - geom.pad.b }));
  best.vals.forEach((val) => {
    if (val.y === undefined) return;
    o.appendChild(svgEl("circle", { class: "c-marker", cx: best.x, cy: val.y, r: 3, fill: val.color }));
  });

  const tip = entry.tip;
  const time = new Date(best.t * 1000);
  const rows = best.vals
    .filter((val) => val.v !== undefined)
    .map((val) => geom.multi
      ? `<div class="tip-row"><span class="tip-swatch" style="background:${val.color}"></span>${val.label} <b>${entry.format(val.v)}</b></div>`
      : `<div class="tip-row"><b>${entry.format(val.v)}</b></div>`)
    .join("");
  tip.innerHTML = `<div class="tip-time">${formatClock(time)}</div>${rows}`;
  tip.hidden = false;
  const leftPx = best.x / scaleX;
  const topPx = (geom.multi ? geom.pad.t : best.y) / (geom.H / rect.height);
  tip.style.left = `${Math.max(6, Math.min(rect.width - 6, leftPx))}px`;
  tip.style.top = `${topPx}px`;
}

function hideChartHover(entry) {
  if (entry.geom?.overlay) entry.geom.overlay.replaceChildren();
  if (entry.tip) entry.tip.hidden = true;
}

function drawMicroSpark(svg, values, accentToken) {
  if (!svg) return;
  const rect = svg.getBoundingClientRect();
  const W = Math.max(Math.round(rect.width) || 120, 40);
  const H = Math.max(Math.round(rect.height) || 26, 16);
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("preserveAspectRatio", "none");
  if (values.length < 2) {
    svg.replaceChildren();
    return;
  }
  const pad = 3;
  const ys = values.map((p) => p[1]);
  let min = Math.min(...ys);
  let max = Math.max(...ys);
  const span = max - min || Math.abs(max) || 1;
  min -= span * 0.12;
  max += span * 0.12;
  const pts = values.map(([, v], i) => [
    pad + (i / (values.length - 1)) * (W - pad * 2),
    H - pad - ((v - min) / (max - min)) * (H - pad * 2),
  ]);
  const line = monotonePath(pts);
  const color = mix(accentToken, "white", 32, "#9fc");
  const area = `${line} L${pts[pts.length - 1][0].toFixed(2)},${H - pad} L${pts[0][0].toFixed(2)},${H - pad} Z`;
  const frag = document.createDocumentFragment();
  frag.appendChild(svgEl("path", { d: area, fill: `color-mix(in srgb, ${cssVar(accentToken, "#9fc")}, transparent 80%)`, stroke: "none" }));
  frag.appendChild(svgEl("path", { d: line, fill: "none", stroke: color, "stroke-width": 1.8, "stroke-linecap": "round", "stroke-linejoin": "round" }));
  frag.appendChild(svgEl("circle", { cx: pts[pts.length - 1][0], cy: pts[pts.length - 1][1], r: 2, fill: color }));
  svg.replaceChildren(frag);
}

const SPARK_TOKENS = {
  "spark-rate": "--teal",
  "spark-p95": "--clay",
  "spark-errors": "--bad",
  "spark-compute": "--moss",
};
const sparkData = {};

function setMicroSpark(id, values) {
  sparkData[id] = values;
  drawMicroSpark(els[id], values, SPARK_TOKENS[id]);
}

// Repaint micro-sparklines on container resize (they use preserveAspectRatio="none", so a width
// change between 5s polls would otherwise stretch the cached path until the next poll).
function initSparks() {
  if (typeof ResizeObserver === "undefined") return;
  Object.keys(SPARK_TOKENS).forEach((id) => {
    const svg = els[id];
    if (!svg) return;
    new ResizeObserver(() => drawMicroSpark(svg, sparkData[id] || [], SPARK_TOKENS[id])).observe(svg);
  });
}

function seriesStats(values) {
  const v = values.map((p) => p[1]).filter(Number.isFinite);
  if (!v.length) return null;
  return {
    min: Math.min(...v),
    max: Math.max(...v),
    avg: v.reduce((a, b) => a + b, 0) / v.length,
  };
}

function renderFoot(el, stats, format) {
  if (!el) return;
  if (!stats) { el.replaceChildren(); return; }
  el.innerHTML = `
    <span class="stat">min <b>${format(stats.min)}</b></span>
    <span class="stat">avg <b>${format(stats.avg)}</b></span>
    <span class="stat">peak <b>${format(stats.max)}</b></span>`;
}

function drawDonut(rows) {
  const svg = els["chart-composition"];
  const legend = els["composition-legend"];
  const entries = ["score", "plan", "rollout", "encode"]
    .map((op) => ({ op, rate: Math.max(0, rows.get(op)?.rate || 0) }))
    .filter((item) => item.rate > 0)
    .sort((a, b) => b.rate - a.rate);
  const total = entries.reduce((sum, item) => sum + item.rate, 0);

  svg.replaceChildren();
  const cx = 60;
  const cy = 60;
  const r = 44;
  const C = 2 * Math.PI * r;
  svg.appendChild(svgEl("circle", { class: "donut-track", cx, cy, r }));

  if (total > 0) {
    const group = svgEl("g", { transform: `rotate(-90 ${cx} ${cy})` });
    const gap = entries.length > 1 ? 2 : 0;
    let acc = 0;
    entries.forEach((item) => {
      const frac = item.rate / total;
      const drawLen = Math.max(frac * C - gap, 0.5);
      const seg = svgEl("circle", {
        class: "donut-seg",
        cx, cy, r,
        stroke: cssVar(OP_COLORS[item.op], "#176f72"),
        "stroke-dasharray": `${drawLen.toFixed(2)} ${(C - drawLen).toFixed(2)}`,
        "stroke-dashoffset": `${(-acc).toFixed(2)}`,
      });
      group.appendChild(seg);
      acc += frac * C;
    });
    svg.appendChild(group);
  }

  const totalText = svgEl("text", { class: "donut-total", x: cx, y: cy + 1 });
  totalText.textContent = total > 0 ? total.toFixed(total < 10 ? 1 : 0) : "0.0";
  const totalLabel = svgEl("text", { class: "donut-total-label", x: cx, y: cy + 13 });
  totalLabel.textContent = "req/s";
  svg.append(totalText, totalLabel);

  legend.replaceChildren();
  if (total <= 0) {
    const empty = document.createElement("div");
    empty.className = "donut-empty";
    empty.textContent = "No live traffic. Run a mix or start the stimulator.";
    legend.appendChild(empty);
    return;
  }
  entries.forEach((item) => {
    const row = document.createElement("div");
    row.className = "donut-row";
    const pct = Math.round((item.rate / total) * 100);
    row.innerHTML = `
      <span class="swatch" style="background:${cssVar(OP_COLORS[item.op], "#176f72")}"></span>
      <span class="op">${item.op}</span>
      <span class="pct">${formatRate(item.rate)} · ${pct}%</span>`;
    legend.appendChild(row);
  });
}

function renderOperationTable(rows) {
  const container = els["operation-table"];
  container.replaceChildren();
  const header = document.createElement("div");
  header.className = "op-row op-head";
  header.innerHTML =
    "<span>Operation</span><span>Rate</span><span>Error</span><span>p95</span><span>compute</span><span>queue</span><span>workload</span>";
  container.appendChild(header);

  const ops = ["score", "plan", "rollout", "encode"];
  const maxRate = Math.max(0.0001, ...ops.map((op) => rows.get(op)?.rate || 0));

  ops.forEach((operation) => {
    const row = rows.get(operation) || {};
    const rate = row.rate || 0;
    const errorRate = row.errorRate || 0;
    const barPct = Math.max(2, Math.round((rate / maxRate) * 100));
    const element = document.createElement("div");
    element.className = `op-row op-data${rate > 0 ? "" : " is-quiet"}`;
    element.innerHTML = `
      <span class="op-name">${operation}<span class="op-bar"><i style="width:${barPct}%;background:${cssVar(OP_COLORS[operation], "#176f72")}"></i></span></span>
      <span class="num">${formatPerMinute(rate * 60)}</span>
      <span class="${errorRate > 0 ? "error" : "muted"}">${formatPerMinute(errorRate * 60)}</span>
      <span class="num">${formatSeconds(row.latency)}</span>
      <span class="num">${formatSeconds(row.compute)}</span>
      <span class="num">${formatSeconds(row.queue)}</span>
      <span class="muted">${formatWorkload(row)}</span>
    `;
    container.appendChild(element);
  });
}

function setChartsLoading(loading) {
  document.querySelectorAll(".chart-canvas").forEach((canvas) => {
    canvas.classList.toggle("is-loading", loading);
  });
}

async function refreshMetrics() {
  if (!state.chartsReady) setChartsLoading(true);
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
      latencyP50Series,
      latencySeries,
      latencyP99Series,
      computeSeries,
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
      promQuery("sum(rate(wmcp_requests_total[5m])) by (operation)"),
      promQuery("sum(rate(wmcp_request_errors_total[5m])) by (operation)"),
      promQuery("histogram_quantile(0.95, sum(rate(wmcp_request_latency_seconds_bucket[5m])) by (le, operation))"),
      promQuery("histogram_quantile(0.95, sum(rate(wmcp_model_compute_seconds_bucket[5m])) by (le, operation))"),
      promQuery("histogram_quantile(0.95, sum(rate(wmcp_queue_wait_seconds_bucket[5m])) by (le, operation))"),
      promQuery("histogram_quantile(0.95, sum(rate(wmcp_candidate_count_bucket[5m])) by (le, operation))"),
      promQuery("histogram_quantile(0.95, sum(rate(wmcp_rollout_horizon_bucket[5m])) by (le, operation))"),
      promRange("sum(rate(wmcp_requests_total[1m]))"),
      promRange("histogram_quantile(0.50, sum(rate(wmcp_request_latency_seconds_bucket[5m])) by (le))"),
      promRange("histogram_quantile(0.95, sum(rate(wmcp_request_latency_seconds_bucket[5m])) by (le))"),
      promRange("histogram_quantile(0.99, sum(rate(wmcp_request_latency_seconds_bucket[5m])) by (le))"),
      promRange("histogram_quantile(0.95, sum(rate(wmcp_model_compute_seconds_bucket[5m])) by (le))"),
      promRange("sum(rate(wmcp_request_errors_total[1m])) OR on() vector(0)"),
    ]);

    const rateNow = firstValue(rate);
    const p95Now = firstValue(p95);
    const errorNow = firstValue(errorRate);
    const computeNow = firstValue(compute);

    els["metric-total"].textContent = formatCount(firstValue(total));
    els["metric-rate"].textContent = formatRate(rateNow);
    els["metric-inflight"].textContent = formatCount(firstValue(inflight));
    els["metric-error-rate"].textContent = formatRate(errorNow);
    els["metric-p95"].textContent = formatSeconds(p95Now);
    els["metric-compute"].textContent = formatSeconds(computeNow);
    els["metric-queue"].textContent = formatSeconds(firstValue(queue));
    els["metric-errors"].textContent = formatCount(firstValue(errors));

    els["now-rate"].textContent = formatRate(rateNow);
    els["now-latency"].textContent = formatSeconds(p95Now);
    els["now-errors"].textContent = formatRate(errorNow);

    const rateValues = extractSeries(rateSeries);
    const p50Values = extractSeries(latencyP50Series);
    const p95Values = extractSeries(latencySeries);
    const p99Values = extractSeries(latencyP99Series);
    const computeValues = extractSeries(computeSeries);
    const errorValues = extractSeries(errorSeries);

    setLineChart("chart-rate", [rateValues]);
    setLineChart("chart-latency", [p50Values, p95Values, p99Values]);
    setLineChart("chart-errors", [errorValues]);

    renderFoot(els["foot-rate"], seriesStats(rateValues), formatRate);
    renderFoot(els["foot-errors"], seriesStats(errorValues), formatRate);
    renderLatencyLegend();

    setMicroSpark("spark-rate", rateValues);
    setMicroSpark("spark-p95", p95Values);
    setMicroSpark("spark-errors", errorValues);
    setMicroSpark("spark-compute", computeValues);

    const opRows = mergeMaps(
      vectorMap(rateByOperation, "operation", "rate"),
      vectorMap(errorsByOperation, "operation", "errorRate"),
      vectorMap(p95ByOperation, "operation", "latency"),
      vectorMap(computeByOperation, "operation", "compute"),
      vectorMap(queueByOperation, "operation", "queue"),
      vectorMap(candidatesByOperation, "operation", "candidates"),
      vectorMap(horizonByOperation, "operation", "horizon"),
    );
    renderOperationTable(opRows);
    drawDonut(opRows);

    state.lastMetricsAt = new Date();
    state.chartsReady = true;
    setChartsLoading(false);
    tickAge();
    setMetricsState(true);
  } catch (error) {
    setChartsLoading(false);
    setMetricsState(false);
    tickAge();
  }
}

function renderLatencyLegend() {
  const el = els["legend-latency"];
  if (!el) return;
  el.innerHTML = `
    <span class="legend-item"><span class="legend-swatch" style="background:${cssVar("--moss", "#7b9652")}"></span>p50</span>
    <span class="legend-item"><span class="legend-swatch" style="background:${cssVar("--clay", "#b45b42")}"></span>p95</span>
    <span class="legend-item" style="color:${cssVar("--muted", "#687068")}"><span class="legend-swatch is-dashed"></span>p99</span>`;
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
    empty.textContent = "No WMCP calls recorded in this browser session yet — send one from the workbench.";
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
      <span class="log-status ${statusClass}">${entry.ok ? "ok" : "error"}</span>
      <span class="log-op">${entry.operation}</span>
      <span class="log-meta">${formatSeconds(entry.elapsed / 1000)} · ${formatClock(entry.at)}</span>
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

function initCharts() {
  registerChart("chart-rate", {
    accentToken: "--teal",
    accent: "#176f72",
    format: formatRate,
    axisFormat: axisRate,
    series: [{ primary: true, label: "rate", colorToken: "--teal" }],
  });
  registerChart("chart-latency", {
    accentToken: "--clay",
    accent: "#b45b42",
    format: formatSeconds,
    axisFormat: axisSeconds,
    series: [
      { label: "p50", colorToken: "--moss", cls: "c-line--ghost" },
      { primary: true, label: "p95", colorToken: "--clay" },
      { label: "p99", colorToken: "--ink", cls: "c-line--faint" },
    ],
  });
  registerChart("chart-errors", {
    accentToken: "--bad",
    accent: "#b3384b",
    format: formatRate,
    axisFormat: axisRate,
    series: [{ primary: true, label: "errors", colorToken: "--bad" }],
  });
}

async function boot() {
  initElements();
  initCharts();
  initSparks();
  bindEvents();
  renderLog();
  renderPayload();
  setChartsLoading(true);
  await refreshStatus();
  await refreshMetrics();
  // Re-detect status (backend, readiness) alongside metrics so a `state.backend` that was stuck
  // at the `mock` default — e.g. the page loaded while the backend was briefly unreachable — heals
  // itself within one interval instead of requiring a manual Refresh.
  state.metricsTimer = window.setInterval(() => {
    refreshMetrics();
    refreshStatus();
  }, 5000);
  state.ageTimer = window.setInterval(tickAge, 1000);
}

boot();
