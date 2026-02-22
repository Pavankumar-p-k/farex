const inputNode = document.getElementById("server-url");
const saveBtnNode = document.getElementById("save-btn");
const openBtnNode = document.getElementById("open-btn");
const statusNode = document.getElementById("status");

const STORAGE_KEY = "face_attendance_dashboard_url";

function inferDefaultDashboardUrl() {
  const protocol = window.location.protocol || "http:";
  const host = window.location.hostname || "127.0.0.1";
  return `${protocol}//${host}:8000`;
}

function normalizeUrl(raw) {
  const value = String(raw || "").trim();
  if (!value) {
    return "";
  }
  if (value.startsWith("http://") || value.startsWith("https://")) {
    return value;
  }
  return `http://${value}`;
}

function setStatus(message) {
  statusNode.textContent = message;
}

function loadSavedUrl() {
  const saved = localStorage.getItem(STORAGE_KEY) || "";
  if (saved) {
    inputNode.value = saved;
    setStatus(`Saved URL: ${saved}`);
  } else {
    const inferred = inferDefaultDashboardUrl();
    inputNode.value = inferred;
    setStatus(`Default URL: ${inferred}`);
  }
}

function saveUrl() {
  const url = normalizeUrl(inputNode.value);
  if (!url) {
    setStatus("Enter a valid URL.");
    return;
  }
  localStorage.setItem(STORAGE_KEY, url);
  inputNode.value = url;
  setStatus("URL saved.");
}

function openDashboard() {
  const url = normalizeUrl(inputNode.value || localStorage.getItem(STORAGE_KEY));
  if (!url) {
    setStatus("Enter a valid URL first.");
    return;
  }
  window.location.href = url;
}

saveBtnNode.addEventListener("click", saveUrl);
openBtnNode.addEventListener("click", openDashboard);
loadSavedUrl();
