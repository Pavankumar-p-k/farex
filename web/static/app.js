const fpsNode = document.getElementById("fps");
const knownCountNode = document.getElementById("known-count");
const powerLevelNode = document.getElementById("power-level");
const activeCameraChipNode = document.getElementById("active-camera-chip");
const clockNode = document.getElementById("clock");

const healthPowerNode = document.getElementById("health-power");
const healthKnownNode = document.getElementById("health-known");
const healthTracksNode = document.getElementById("health-tracks");
const healthUnknownNode = document.getElementById("health-unknown");

const recognizedListNode = document.getElementById("recognized-list");
const pendingCardNode = document.getElementById("pending-card");
const pendingEmptyNode = document.getElementById("pending-empty");
const pendingPreviewNode = document.getElementById("pending-preview");
const pendingInfoNode = document.getElementById("pending-info");
const saveFormNode = document.getElementById("save-form");
const dismissBtnNode = document.getElementById("dismiss-btn");
const errorBoxNode = document.getElementById("error-box");
const eventLogNode = document.getElementById("event-log");

const facePointsNode = document.getElementById("face-points");
const trackedCountNode = document.getElementById("tracked-count");
const unknownCountNode = document.getElementById("unknown-count");
const cameraStateNode = document.getElementById("camera-state");
const skeletonStateNode = document.getElementById("skeleton-state");
const cameraStreamNode = document.getElementById("camera-stream");
const skeletonStreamNode = document.getElementById("skeleton-stream");
const holoCanvasNode = document.getElementById("holo-overlay");
const radarCanvasNode = document.getElementById("radar-canvas");

const cameraSelectNode = document.getElementById("camera-select");
const applyCameraBtnNode = document.getElementById("apply-camera-btn");
const refreshCamerasBtnNode = document.getElementById("refresh-cameras-btn");
const cameraAutoSelectNode = document.getElementById("camera-auto-select");
const cameraSourceInfoNode = document.getElementById("camera-source-info");
const mobileLinksNode = document.getElementById("mobile-links");
let targetFpsDisplay = 20;

let pendingToken = null;
let lastErrorMessage = "";
let hadPending = false;
let isSwitchingCamera = false;
let stateFetchFailureCount = 0;
const dashboardBootAt = Date.now();

const events = [];
const telemetry = {
  fps: 0,
  knownCount: 0,
  trackedCount: 0,
  unknownCount: 0,
  facePoints: 0,
  power: 0
};

function updateClock() {
  const now = new Date();
  clockNode.textContent = now.toLocaleTimeString([], { hour12: false });
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function formatEventTime(ts) {
  return new Date(ts).toLocaleTimeString([], { hour12: false });
}

function pushEvent(message) {
  events.unshift({ message, ts: Date.now() });
  if (events.length > 30) {
    events.length = 30;
  }
  renderEvents();
}

function renderEvents() {
  if (events.length === 0) {
    eventLogNode.innerHTML = '<p class="muted">Waiting for recognition events.</p>';
    return;
  }

  eventLogNode.innerHTML = events
    .map((item) => {
      return `<div class="event-row"><span>${escapeHtml(item.message)}</span><span>${formatEventTime(item.ts)}</span></div>`;
    })
    .join("");
}

function renderRecognized(items) {
  if (!items || items.length === 0) {
    recognizedListNode.innerHTML = '<p class="muted">No faces recognized yet.</p>';
    return;
  }

  recognizedListNode.innerHTML = items
    .map((item) => {
      const score = Number(item.score || 0);
      const pct = Math.round(clamp(score, 0, 1) * 100);
      const seenMs = Number(item.last_seen_ms || 0);
      return `
        <div class="recognized-row">
          <div class="recognized-head">
            <span>${escapeHtml(item.name || "Unknown")}</span>
            <span>${pct}%</span>
          </div>
          <div class="recognized-meta">Last seen ${seenMs} ms ago</div>
          <div class="confidence"><span style="width:${pct}%;"></span></div>
        </div>
      `;
    })
    .join("");
}

function renderPending(pending) {
  if (!pending) {
    pendingToken = null;
    pendingCardNode.classList.add("hidden");
    pendingEmptyNode.classList.remove("hidden");
    return;
  }

  pendingToken = pending.token;
  pendingPreviewNode.src = `data:image/jpeg;base64,${pending.preview_b64}`;
  pendingInfoNode.textContent = `Samples captured: ${pending.sample_count}`;
  pendingCardNode.classList.remove("hidden");
  pendingEmptyNode.classList.add("hidden");
}

function renderError(error) {
  if (!error) {
    errorBoxNode.classList.add("hidden");
    errorBoxNode.textContent = "";
    return;
  }
  errorBoxNode.classList.remove("hidden");
  errorBoxNode.textContent = error;
}

function updatePowerBadge(value) {
  powerLevelNode.textContent = `Engine: ${value}%`;
  healthPowerNode.textContent = `${value}%`;
  if (value >= 82) {
    powerLevelNode.style.color = "#a7f8e6";
  } else if (value >= 55) {
    powerLevelNode.style.color = "#ffd18f";
  } else {
    powerLevelNode.style.color = "#ff9191";
  }
}

function renderSystemState(state) {
  const recognized = state.recognized || [];
  const pendingUnknown = state.pending_unknown;
  const cameraLive = state.camera_live !== false;
  const hasError = Boolean(state.error);
  const unknownSignals = Number.isFinite(Number(state.unknown_count))
    ? Number(state.unknown_count)
    : (pendingUnknown ? 1 : 0);
  const tracked = Number.isFinite(Number(state.tracked_count))
    ? Number(state.tracked_count)
    : (recognized.length + unknownSignals);
  const facePoints = Number(state.face_points || 0);
  const fps = Number(state.fps || 0);

  // Power reflects stream health/stability, not whether a face is currently visible.
  const fpsWeight = clamp(fps / targetFpsDisplay, 0, 1);
  let power = 0;
  if (!hasError && cameraLive) {
    // Keep health meter around 70-80 for normal live operation.
    power = Math.round(72 + fpsWeight * 8);
    if (!state.pose_locked) {
      power = Math.max(68, power - 2);
    }
  } else if (cameraLive) {
    power = Math.round(45 + fpsWeight * 20);
  } else {
    power = Math.round(20 + fpsWeight * 15);
  }

  telemetry.fps = fps;
  telemetry.knownCount = Number(state.known_count || 0);
  telemetry.trackedCount = tracked;
  telemetry.unknownCount = unknownSignals;
  telemetry.facePoints = facePoints;
  telemetry.power = power;

  facePointsNode.textContent = facePoints || "--";
  trackedCountNode.textContent = tracked;
  unknownCountNode.textContent = unknownSignals;

  healthKnownNode.textContent = telemetry.knownCount;
  healthTracksNode.textContent = tracked;
  healthUnknownNode.textContent = unknownSignals;

  const backendLabel = state.camera_backend ? `/${state.camera_backend}` : "";
  activeCameraChipNode.textContent = `Source: ${state.camera_index ?? "--"}${backendLabel}`;
  updatePowerBadge(power);
}

function updateStreamStatus(error, poseLocked, cameraLive) {
  if (error || cameraLive === false) {
    cameraStateNode.textContent = "DEGRADED";
    cameraStateNode.classList.remove("live");
    skeletonStateNode.textContent = "DEGRADED";
    skeletonStateNode.classList.remove("live");
    return;
  }

  cameraStateNode.textContent = "LIVE";
  cameraStateNode.classList.add("live");
  skeletonStateNode.textContent = poseLocked ? "LOCKED" : "SCANNING";
  skeletonStateNode.classList.add("live");
}

function renderCameraInventory(payload) {
  const cameras = payload.cameras || [];
  if (!cameras.length) {
    cameraSelectNode.innerHTML = "";
    cameraAutoSelectNode.checked = false;
    activeCameraChipNode.textContent = "Source: --";
    cameraSourceInfoNode.textContent = "No camera sources found.";
    return;
  }

  cameraSelectNode.innerHTML = cameras
    .map((item) => {
      const selected = Number(item.index) === Number(payload.active_camera_index) ? "selected" : "";
      const fpsLabel = item.fps ? `${item.fps.toFixed(1)} fps` : "-- fps";
      const backend = item.backend ? `, ${item.backend}` : "";
      return `<option value="${item.index}" ${selected}>Camera ${item.index} (${item.width}x${item.height}, ${fpsLabel}${backend})</option>`;
    })
    .join("");

  cameraAutoSelectNode.checked = Boolean(payload.auto_select);
  const active = cameras.find((item) => Number(item.index) === Number(payload.active_camera_index));
  const activeBackend = active && active.backend ? ` via ${active.backend}` : "";
  cameraSourceInfoNode.textContent = `Active source: ${payload.active_camera_index}${activeBackend} (${cameras.length} detected)`;
  const chipBackend = active && active.backend ? `/${active.backend}` : "";
  activeCameraChipNode.textContent = `Source: ${payload.active_camera_index}${chipBackend}`;
}

async function fetchCameraInventory(forceRefresh = false) {
  try {
    const url = forceRefresh ? "/api/cameras?refresh=true" : "/api/cameras";
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      throw new Error("Failed to scan camera sources.");
    }
    const payload = await response.json();
    renderCameraInventory(payload);
  } catch (error) {
    cameraSourceInfoNode.textContent = error.message;
  }
}

async function applyCameraSelection() {
  if (!cameraSelectNode.value || isSwitchingCamera) {
    return;
  }
  isSwitchingCamera = true;
  try {
    const cameraIndex = Number(cameraSelectNode.value);
    const response = await fetch("/api/camera/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        camera_index: cameraIndex,
        pin_manual: true
      })
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Failed to switch camera.");
    }
    renderCameraInventory(payload);
    pushEvent(`Camera source switched to ${cameraIndex}`);
  } catch (error) {
    renderError(error.message);
  } finally {
    isSwitchingCamera = false;
  }
}

async function setAutoCameraMode(enabled) {
  try {
    const response = await fetch("/api/camera/auto-select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled: Boolean(enabled) })
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Failed to set auto camera mode.");
    }
    renderCameraInventory(payload);
    pushEvent(enabled ? "Auto camera select enabled" : "Auto camera select disabled");
  } catch (error) {
    renderError(error.message);
    cameraAutoSelectNode.checked = !enabled;
  }
}

async function fetchMobileLinks() {
  try {
    const response = await fetch("/api/mobile-access", { cache: "no-store" });
    if (!response.ok) {
      throw new Error("Failed to load mobile access links.");
    }
    const payload = await response.json();
    const urls = payload.urls || [];
    if (!urls.length) {
      mobileLinksNode.innerHTML = '<div class="link-item">No reachable network address found.</div>';
      return;
    }
    mobileLinksNode.innerHTML = urls
      .map((url) => `<div class="link-item"><a href="${escapeHtml(url)}" target="_blank" rel="noreferrer">${escapeHtml(url)}</a></div>`)
      .join("");
  } catch (error) {
    mobileLinksNode.innerHTML = `<div class="link-item">${escapeHtml(error.message)}</div>`;
  }
}

async function fetchState() {
  try {
    const response = await fetch("/api/state", { cache: "no-store" });
    if (!response.ok) {
      throw new Error("Failed to load state.");
    }

    const state = await response.json();
    stateFetchFailureCount = 0;
    const nextTarget = Number(state.target_fps || targetFpsDisplay || 20);
    targetFpsDisplay = Number.isFinite(nextTarget) && nextTarget > 0 ? nextTarget : 20;
    fpsNode.textContent = `FPS: ${state.fps ?? "--"} / ${targetFpsDisplay}`;
    knownCountNode.textContent = `Known: ${state.known_count ?? "--"}`;
    renderRecognized(state.recognized);
    renderPending(state.pending_unknown);
    renderError(state.error);
    renderSystemState(state);
    updateStreamStatus(state.error, state.pose_locked, state.camera_live);

    if (state.announcements && state.announcements.length) {
      state.announcements.forEach((item) => pushEvent(item));
    }

    if (state.pending_unknown && !hadPending) {
      pushEvent("Unknown signal captured");
    }
    hadPending = Boolean(state.pending_unknown);

    if (state.error && state.error !== lastErrorMessage) {
      pushEvent(`System alert: ${state.error}`);
    }
    lastErrorMessage = state.error || "";
  } catch (error) {
    stateFetchFailureCount += 1;
    renderError(error.message);
    updateStreamStatus(error.message, false, false);
    const bootAgeMs = Date.now() - dashboardBootAt;
    const startupGrace = bootAgeMs < 45000;
    if (!startupGrace && stateFetchFailureCount >= 3 && error.message !== lastErrorMessage) {
      pushEvent(`Network alert: ${error.message}`);
      lastErrorMessage = error.message;
    }
  }
}

async function registerPending(event) {
  event.preventDefault();
  const employeeId = document.getElementById("employee-id").value.trim();
  const employeeName = document.getElementById("employee-name").value.trim();

  if (!employeeId || !employeeName || !pendingToken) {
    renderError("Pending face data not ready yet.");
    return;
  }

  try {
    const response = await fetch("/api/register-pending", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        employee_id: employeeId,
        name: employeeName
      })
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Failed to save user.");
    }

    saveFormNode.reset();
    renderError("");
    pushEvent(`New face registered: ${employeeName}`);
    await fetchState();
  } catch (error) {
    renderError(error.message);
  }
}

async function clearPending() {
  try {
    await fetch("/api/clear-pending", { method: "POST" });
    pushEvent("Pending unknown dismissed");
    await fetchState();
  } catch (error) {
    renderError(error.message);
  }
}

function fitCanvasToNode(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return { ctx: null, width: 0, height: 0 };
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width: rect.width, height: rect.height };
}

function drawHoloOverlay(ctx, width, height, t) {
  ctx.clearRect(0, 0, width, height);
  const cx = width * 0.5;
  const cy = height * 0.54;
  const maxRadius = Math.min(width, height) * 0.38;
  const sweep = (t * 0.0013) % (Math.PI * 2);

  ctx.strokeStyle = "rgba(150, 218, 255, 0.2)";
  ctx.lineWidth = 1;
  for (let i = 1; i <= 4; i += 1) {
    ctx.beginPath();
    ctx.arc(cx, cy, (maxRadius * i) / 4, 0, Math.PI * 2);
    ctx.stroke();
  }

  const gradient = ctx.createRadialGradient(cx, cy, maxRadius * 0.1, cx, cy, maxRadius);
  gradient.addColorStop(0, "rgba(120, 232, 255, 0.24)");
  gradient.addColorStop(1, "rgba(120, 232, 255, 0)");
  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(cx, cy, maxRadius, sweep - 0.34, sweep + 0.34);
  ctx.lineTo(cx, cy);
  ctx.closePath();
  ctx.fill();
}

function drawRadar(ctx, width, height, t) {
  ctx.clearRect(0, 0, width, height);
  const cx = width * 0.5;
  const cy = height * 0.5;
  const r = Math.min(width, height) * 0.44;
  const sweep = (t * 0.0018) % (Math.PI * 2);

  ctx.fillStyle = "rgba(2, 18, 30, 0.94)";
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "rgba(126, 214, 255, 0.18)";
  ctx.lineWidth = 1;
  for (let i = 1; i <= 4; i += 1) {
    ctx.beginPath();
    ctx.arc(cx, cy, (r * i) / 4, 0, Math.PI * 2);
    ctx.stroke();
  }

  const g = ctx.createRadialGradient(cx, cy, r * 0.08, cx, cy, r);
  g.addColorStop(0, "rgba(122, 236, 188, 0.26)");
  g.addColorStop(1, "rgba(122, 236, 188, 0)");
  ctx.fillStyle = g;
  ctx.beginPath();
  ctx.arc(cx, cy, r, sweep - 0.26, sweep + 0.26);
  ctx.lineTo(cx, cy);
  ctx.closePath();
  ctx.fill();

  const dots = Math.max(2, telemetry.trackedCount + telemetry.unknownCount);
  for (let i = 0; i < dots; i += 1) {
    const angle = (Math.PI * 2 * i) / dots + t * 0.00042;
    const radius = r * (0.32 + (i % 4) * 0.15);
    const x = cx + Math.cos(angle) * radius;
    const y = cy + Math.sin(angle) * radius;
    ctx.beginPath();
    ctx.arc(x, y, 2.2, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(157, 255, 215, 0.95)";
    ctx.fill();
  }
}

function animateHud(time) {
  const holo = fitCanvasToNode(holoCanvasNode);
  if (holo.ctx) {
    drawHoloOverlay(holo.ctx, holo.width, holo.height, time);
  }

  const radar = fitCanvasToNode(radarCanvasNode);
  if (radar.ctx) {
    drawRadar(radar.ctx, radar.width, radar.height, time);
  }

  requestAnimationFrame(animateHud);
}

saveFormNode.addEventListener("submit", registerPending);
dismissBtnNode.addEventListener("click", clearPending);
applyCameraBtnNode.addEventListener("click", applyCameraSelection);
refreshCamerasBtnNode.addEventListener("click", () => fetchCameraInventory(true));
cameraAutoSelectNode.addEventListener("change", () => {
  setAutoCameraMode(cameraAutoSelectNode.checked);
});

updateClock();
setInterval(updateClock, 1000);
fetchState();
setInterval(fetchState, 1000);
fetchCameraInventory(true);
setInterval(() => fetchCameraInventory(false), 8000);
fetchMobileLinks();
setInterval(fetchMobileLinks, 20000);
requestAnimationFrame(animateHud);

cameraStreamNode.addEventListener("error", () => {
  pushEvent("Camera stream dropped. Retrying...");
  cameraStateNode.textContent = "DEGRADED";
  cameraStateNode.classList.remove("live");
});

skeletonStreamNode.addEventListener("error", () => {
  pushEvent("Skeleton stream dropped. Retrying...");
  skeletonStateNode.textContent = "DEGRADED";
  skeletonStateNode.classList.remove("live");
});
