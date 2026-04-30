// ===================================================================
//  NeuroAds - Frontend JS
// ===================================================================

const STATE = {
  jobId:     null,
  pollTimer: null,
  charts:    {},
  lastResults: null,
  fileName:    "",
};

const STATUS_LABELS = {
  queued:            "INITIALIZING NEURAL PORT",
  loading_model:     "LOADING TRIBE V2 KERNEL...",
  extracting_events: "EXTRACTING MULTIMODAL FEATURES...",
  predicting:        "RUNNING CORTICAL PREDICTION...",
  analysing:         "EXECUTING MARKETING AUDIT...",
  done:              "ANALYSIS COMPLETE",
  error:             "SYSTEM FAULT",
};

const STEP_ORDER = ["queued","loading_model","extracting_events","predicting","analysing","done"];
const UPLOADED_FILES = [];

// -- DOM refs ---------------------------------------------------
const $    = id => document.getElementById(id);
const form = $("upload-form");

// -- Char counter for textarea ----------------------------------
const textInput = $("text-input");
const charCount = $("char-count");
if (textInput && charCount) {
  textInput.addEventListener("input", () => {
    const len = textInput.value.length;
    charCount.textContent = len + " character" + (len !== 1 ? "s" : "");
    charCount.style.color = len > 0 ? "rgba(0,255,163,0.8)" : "var(--text-muted)";
  });
}

// -- Script Toggle ----------------------------------------------
const toggleBtn = $("toggle-script-btn");
const scriptBox = $("script-collapsible");
if (toggleBtn && scriptBox) {
  toggleBtn.addEventListener("click", () => {
    const isOpen = scriptBox.classList.toggle("open");
    toggleBtn.classList.toggle("active", isOpen);
  });
}

// -- Unified Upload Handler -------------------------------------
const unifiedInput = $("unified-input");
const unifiedZone  = $("unified-zone");
const fileList     = $("file-list-container");

if (unifiedInput && unifiedZone) {
  unifiedInput.addEventListener("change", (e) => {
    handleFiles(e.target.files);
  });

  unifiedZone.addEventListener("dragover",  e => { e.preventDefault(); unifiedZone.classList.add("drag-over"); });
  unifiedZone.addEventListener("dragleave", () => unifiedZone.classList.remove("drag-over"));
  unifiedZone.addEventListener("drop",      e => {
    e.preventDefault();
    unifiedZone.classList.remove("drag-over");
    handleFiles(e.dataTransfer.files);
  });
}

function handleFiles(files) {
  Array.from(files).forEach(file => {
    if (UPLOADED_FILES.some(f => f.name === file.name && f.size === file.size)) return;
    UPLOADED_FILES.push(file);
    renderFileList();
  });
}

function renderFileList() {
  const defaultContent = $("upload-content-default");
  const previewContainer = $("upload-preview");
  
  if (!defaultContent || !previewContainer) return;

  if (UPLOADED_FILES.length === 0) {
    defaultContent.style.display = "block";
    previewContainer.style.display = "none";
    previewContainer.innerHTML = "";
    return;
  }

  defaultContent.style.display = "none";
  previewContainer.style.display = "flex";
  previewContainer.innerHTML = "";

  UPLOADED_FILES.forEach((file, idx) => {
    const item = document.createElement("div");
    item.className = "preview-item fade-up";
    
    let icon = "📄";
    const type = file.type.toLowerCase();
    
    if (type.startsWith("image/")) {
       const url = URL.createObjectURL(file);
       item.innerHTML = `
         <div class="preview-media"><img src="${url}" /></div>
         <div class="preview-meta">
           <span class="preview-name">${file.name}</span>
           <span class="preview-remove" onclick="removeFile(event, ${idx})">Remove</span>
         </div>
       `;
    } else if (type.startsWith("video/")) {
       const url = URL.createObjectURL(file);
       item.innerHTML = `
         <div class="preview-media">
           <video src="${url}" controls onclick="event.stopPropagation()" style="max-width:350px; max-height:240px; border-radius:8px;"></video>
         </div>
         <div class="preview-meta">
           <span class="preview-name">${file.name}</span>
           <span class="preview-remove" onclick="removeFile(event, ${idx})">Remove</span>
         </div>
       `;
    } else if (type.startsWith("audio/")) {
       const url = URL.createObjectURL(file);
       item.innerHTML = `
         <div class="preview-media">
           <audio src="${url}" controls onclick="event.stopPropagation()" style="width:350px;"></audio>
         </div>
         <div class="preview-meta">
           <span class="preview-name">${file.name}</span>
           <span class="preview-remove" onclick="removeFile(event, ${idx})">Remove</span>
         </div>
       `;
    } else {
       if (type.includes("video")) icon = "🎬";
       else if (type.includes("audio")) icon = "🎧";
       
       item.innerHTML = `
         <div class="preview-icon-box">${icon}</div>
         <div class="preview-meta">
           <span class="preview-name">${file.name}</span>
           <span class="preview-remove" onclick="removeFile(event, ${idx})">Remove</span>
         </div>
       `;
    }
    previewContainer.appendChild(item);
  });
}

window.removeFile = function(event, idx) {
  event.stopPropagation(); // Prevent triggering the file input click
  UPLOADED_FILES.splice(idx, 1);
  renderFileList();
};

// -- Form submit -----------------------------------------------
form.addEventListener("submit", async e => {
  e.preventDefault();
  clearError();
  resetResults();

  const fd = new FormData();

  let hasVideo = false, hasAudio = false, hasImage = false;
  
  UPLOADED_FILES.forEach((file, idx) => {
    if (idx === 0) STATE.fileName = file.name;
    const type = file.type.toLowerCase();
    if (type.includes("video") && !hasVideo) {
      fd.append("video", file);
      hasVideo = true;
    } else if (type.includes("audio") && !hasAudio) {
      fd.append("audio", file);
      hasAudio = true;
    } else if (type.includes("image") && !hasImage) {
      fd.append("image", file);
      hasImage = true;
    } else if (file.name.endsWith(".txt")) {
      fd.append("text", file);
    }
  });

  const textContent = textInput.value.trim();
  const hasText = textContent.length > 0;
  if (hasText) {
    const blob = new Blob([textContent], { type: "text/plain" });
    fd.append("text", blob, "script.txt");
  }

  const hfToken = $("hf-token").value.trim();
  if (hfToken) fd.append("hf_token", hfToken);

  if (!hasVideo && !hasAudio && !hasText && !hasImage && !UPLOADED_FILES.length) {
    showError("Please upload at least one file or enter your ad script text.");
    return;
  }

  setSubmitting(true);
  showProgress();

  try {
    const res  = await fetch("/api/analyse", { method: "POST", body: fd });
    const data = await res.json();

    if (data.error) { showError(data.error); setSubmitting(false); hideProgress(); return; }

    STATE.jobId = data.job_id;
    startPolling();
  } catch (err) {
    showError("Network error: " + err.message);
    setSubmitting(false);
    hideProgress();
  }
});

// -- Polling ---------------------------------------------------
function startPolling() {
  if (STATE.pollTimer) clearInterval(STATE.pollTimer);
  STATE.pollTimer = setInterval(async () => {
    try {
      const res  = await fetch(`/api/status/${STATE.jobId}`);
      const data = await res.json();
      updateProgress(data);
      if (data.status === "done")  { clearInterval(STATE.pollTimer); renderResults(data.results); }
      if (data.status === "error") { clearInterval(STATE.pollTimer); showError(data.error || "Unknown error"); hideProgress(); setSubmitting(false); }
    } catch {}
  }, 1500);
}

function updateProgress(data) {
  const pct   = data.progress || 0;
  const label = STATUS_LABELS[data.status] || data.status;
  $("progress-pct").textContent      = pct + "%";
  $("progress-status").textContent   = label;
  $("progress-fill").style.width     = pct + "%";

  STEP_ORDER.forEach(s => {
    const chip = document.querySelector(`[data-step="${s}"]`);
    if (!chip) return;
    const idx      = STEP_ORDER.indexOf(s);
    const curIdx   = STEP_ORDER.indexOf(data.status);
    chip.className = "step-chip";
    if (idx < curIdx)  chip.classList.add("done");
    if (idx === curIdx) chip.classList.add("active");
  });
}

// -- UI helpers ------------------------------------------------
function showProgress()  { $("progress-section").style.display = "block"; }
function hideProgress()  { $("progress-section").style.display = "none";  }
function setSubmitting(v){ $("submit-btn").disabled = v; }
function showError(msg)  { const b = $("error-box"); b.textContent = "Warning: " + msg; b.style.display = "block"; }
function clearError()    { $("error-box").style.display = "none"; }
function resetResults()  {
  $("results-section").style.display = "none";
  Object.values(STATE.charts).forEach(c => c.destroy());
  STATE.charts = {};
  STATE.lastResults = null;
}

// -- Results renderer ------------------------------------------
function renderResults(r) {
  setSubmitting(false);
  hideProgress();
  STATE.lastResults = r;
  const sec = $("results-section");
  sec.style.display = "block";
  sec.classList.add("fade-up");
  renderKPIs(r.kpis);
  renderCharts(r);
  renderHooks(r.top_hooks);
  renderInsights(r.kpis);
  renderTable(r.events_preview);
}

function renderKPIs(k) {
  $("kpi-avg").textContent      = k.avg_attention.toFixed(1);
  $("kpi-peak").textContent     = k.peak_attention.toFixed(1);
  $("kpi-hook").textContent     = k.hook_score.toFixed(1);
  $("kpi-timesteps").textContent= k.n_timesteps;
  $("kpi-vertices").textContent = (k.n_vertices / 1000).toFixed(0) + "k";
  const badge = $("engagement-badge");
  badge.textContent = k.engagement;
  badge.className   = "engagement-badge";
  if (k.avg_attention >= 75)      badge.classList.add("eng-high");
  else if (k.avg_attention >= 50) badge.classList.add("eng-mod");
  else if (k.avg_attention >= 25) badge.classList.add("eng-low");
  else                            badge.classList.add("eng-very");
}

const CHART_DEFAULTS = {
  responsive: true,
  maintainAspectRatio: false,
  interaction: { mode: "index", intersect: false },
  plugins: {
    legend: { labels: { color: "#9A9AB0", font: { family: "Space Grotesk", size: 11, weight: "600" }, boxWidth: 10, padding: 20 } },
    tooltip: { 
      backgroundColor: "rgba(10, 12, 30, 0.9)", 
      borderColor: "rgba(167, 139, 255, 0.3)", 
      borderWidth: 1, 
      titleColor: "#A78BFF", 
      bodyColor: "#F0F0FF",
      padding: 12,
      cornerRadius: 8,
      titleFont: { family: "Space Grotesk", weight: "bold" }
    },
  },
  scales: {
    x: { 
      ticks: { color: "#5A5A75", font: { size: 10, family: "Space Grotesk" } }, 
      grid: { color: "rgba(255,255,255,0.03)", drawBorder: false } 
    },
    y: { 
      ticks: { color: "#5A5A75", font: { size: 10, family: "Space Grotesk" } }, 
      grid: { color: "rgba(255,255,255,0.03)", drawBorder: false } 
    },
  },
};

function makeLabels(len) { return Array.from({ length: len }, (_, i) => `${(i * 2).toFixed(0)}s`); }

function renderCharts(r) {
  const ts  = r.timeseries;
  const roi = r.roi;
  const labels = makeLabels(ts.attention_scores.length);
  STATE.charts.attention = new Chart($("chart-attention"), {
    type: "line",
    data: {
      labels,
      datasets: [
        { 
          label: "Attention Envelope", 
          data: ts.attention_scores, 
          borderColor: "#A78BFF", 
          backgroundColor: "rgba(167, 139, 255, 0.1)", 
          borderWidth: 3, 
          fill: true, 
          tension: 0.4, 
          pointRadius: 0, 
          pointHoverRadius: 6,
          pointHoverBackgroundColor: "#A78BFF"
        }, 
        { 
          label: "Retention Pacing", 
          data: ts.retention_curve, 
          borderColor: "#00F0FF", 
          borderWidth: 2, 
          borderDash: [5, 5], 
          fill: false, 
          tension: 0.4, 
          pointRadius: 0 
        }
      ],
    },
    options: { ...CHART_DEFAULTS },
  });
  const roiLabels = makeLabels(roi.visual.length);
  STATE.charts.roi = new Chart($("chart-roi"), {
    type: "line",
    data: {
      labels: roiLabels,
      datasets: [
        { label: "Visual",    data: roi.visual,    borderColor: "#FF3CAD", backgroundColor: "rgba(255,60,172,0.05)", borderWidth: 2.5, fill: true, tension: 0.4, pointRadius: 0 },
        { label: "Auditory",  data: roi.auditory,  borderColor: "#00F0FF", backgroundColor: "rgba(0,240,255,0.05)",  borderWidth: 2.5, fill: true, tension: 0.4, pointRadius: 0 },
        { label: "Language",  data: roi.language,  borderColor: "#A78BFF", backgroundColor: "rgba(167, 139, 255, 0.05)", borderWidth: 2.5, fill: true, tension: 0.4, pointRadius: 0 },
        { label: "Attention", data: roi.attention, borderColor: "#00FFA3", backgroundColor: "rgba(0,255,163,0.05)",  borderWidth: 2.5, fill: true, tension: 0.4, pointRadius: 0 },
      ],
    },
    options: { ...CHART_DEFAULTS },
  });
  const segs = r.segments;
  STATE.charts.segments = new Chart($("chart-segments"), {
    type: "bar",
    data: {
      labels: segs.map(s => `${s.start.toFixed(0)}s`),
      datasets: [{ 
        label: "Attention Density", 
        data: segs.map(s => s.attention), 
        backgroundColor: segs.map(s => 
          s.attention >= 75 ? "rgba(0, 255, 163, 0.6)" : 
          s.attention >= 50 ? "rgba(167, 139, 255, 0.5)" : 
          s.attention >= 25 ? "rgba(255, 184, 0, 0.5)" : 
          "rgba(255, 60, 172, 0.4)"
        ), 
        borderRadius: 6 
      }],
    },
    options: { ...CHART_DEFAULTS, scales: { ...CHART_DEFAULTS.scales, y: { ...CHART_DEFAULTS.scales.y, min: 0, max: 100 } } },
  });
  const avg = arr => arr.reduce((a,b)=>a+b,0)/arr.length;
  STATE.charts.radar = new Chart($("chart-radar"), {
    type: "radar",
    data: {
      labels: ["Visual Cortex", "Auditory Cortex", "Semantic Focus", "Pre-Frontal Core", "Overall Neural Pulse"],
      datasets: [{ 
        label: "Activation Topology", 
        data: [
          parseFloat((avg(roi.visual)*100).toFixed(1)), 
          parseFloat((avg(roi.auditory)*100).toFixed(1)), 
          parseFloat((avg(roi.language)*100).toFixed(1)), 
          parseFloat((avg(roi.attention)*100).toFixed(1)), 
          r.kpis.avg_attention
        ], 
        backgroundColor: "rgba(167, 139, 255, 0.2)", 
        borderColor: "#A78BFF", 
        pointBackgroundColor: "#00F0FF", 
        borderWidth: 3, 
        pointRadius: 5 
      }],
    },
    options: { 
      responsive: true, 
      maintainAspectRatio: false,
      plugins: CHART_DEFAULTS.plugins, 
      scales: { 
        r: { 
          angleLines: { color: "rgba(255,255,255,0.05)" },
          ticks: { color: "#5A5A75", font:{size:9}, backdropColor:"transparent" }, 
          grid: { color: "rgba(255,255,255,0.08)" }, 
          pointLabels: { color: "#9A9AB0", font:{size:10, family: "Space Grotesk", weight: "bold"} }, 
          min: 0, max: 100 
        } 
      } 
    },
  });
}

function renderHooks(hooks) {
  const list = $("hooks-list");
  list.innerHTML = "";
  if (!hooks || !hooks.length) { list.innerHTML = '<p style="color:var(--text-muted);font-size:0.85rem;">No segment data available.</p>'; return; }
  hooks.forEach((h, i) => {
    const pct = h.attention.toFixed(0);
    list.innerHTML += `
      <div class="hook-item fade-up" style="animation-delay:${i*0.1}s">
        <div class="hook-rank" style="background:linear-gradient(135deg, var(--accent-primary), var(--accent-tertiary))">${i+1}</div>
        <div class="hook-info">
          <div class="hook-text" title="${h.text||''}">${h.text||'Neural signal detected'}</div>
          <div class="hook-time">${h.start.toFixed(1)}s - ${h.end.toFixed(1)}s</div>
        </div>
        <div class="hook-score-bar">
          <div class="hook-bar-bg"><div class="hook-bar-fill" style="width:${pct}%; background:linear-gradient(90deg, var(--accent-primary), var(--accent-secondary))"></div></div>
          <span class="hook-score-num">${pct}</span>
        </div>
      </div>`;
  });
}

function renderInsights(k) {
  const wrap = $("insights-wrap");
  const insights = [
    { icon:"⚡", title:"Neural Impulse", text: k.hook_score >= 60 ? `Exceptional opening sequence detected (${k.hook_score.toFixed(0)}/100).` : `Initial cortical response is sub-optimal (${k.hook_score.toFixed(0)}/100).` },
    { icon:"🌀", title:"Deep Focus", text: `Maximum neural synchronization detected at ~${k.peak_timestep*2}s.` },
    { icon:"💎", title:"Value Mapping", text: `System-wide mean activation: ${k.avg_attention.toFixed(1)}/100.` },
    { icon:"🔭", title:"Terminal Verdict", text: k.avg_attention >= 75 ? "Target engagement achieved." : "Further signal refinement recommended." }
  ];
  wrap.innerHTML = insights.map(i => `<div class="insight-card fade-up"><span class="insight-icon">${i.icon}</span><div class="insight-title">${i.title}</div><div class="insight-text">${i.text}</div></div>`).join("");
}

function renderTable(events) {
  const tbody = $("events-tbody");
  tbody.innerHTML = "";
  if (!events || !events.length) { tbody.innerHTML = '<tr><td colspan="5">No data.</td></tr>'; return; }
  events.forEach(ev => {
    tbody.innerHTML += `<tr><td><span class="td-type">${ev.type}</span></td><td>${ev.start.toFixed(2)}s</td><td>${ev.duration.toFixed(2)}s</td><td>${ev.text||"—"}</td><td style="color:var(--text-muted)">${ev.context||"—"}</td></tr>`;
  });
}

// -- Professional PDF Export -----------------------------------
async function getLogoBase64() {
  return new Promise(resolve => {
    try {
      const img = new Image();
      img.crossOrigin = "Anonymous";
      img.onload = () => {
        const canvas = document.createElement("canvas");
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0);
        resolve({ data: canvas.toDataURL("image/png"), w: img.width, h: img.height });
      };
      img.onerror = () => {
        console.warn("PDF Logo failed to load.");
        resolve(null);
      };
      img.src = "/static/logo/obbserv.ai-2.png";
    } catch(e) { resolve(null); }
  });
}

async function downloadPDF() {
  const r = STATE.lastResults;
  if (!r) { alert("Please run an analysis first to generate results."); return; }

  // Resilient lib detection
  let chosenJsPDF = null;
  if (window.jspdf && window.jspdf.jsPDF) {
    chosenJsPDF = window.jspdf.jsPDF;
  } else if (window.jsPDF) {
    chosenJsPDF = window.jsPDF;
  }

  if (!chosenJsPDF) {
    alert("Error: PDF Generation library (jsPDF) is not loaded correctly. Please check your internet connection and refresh.");
    return;
  }

  try {
    const doc = new chosenJsPDF({ orientation: "p", unit: "mm", format: "a4" });
    const W = 210, H = 297, k = r.kpis;
    const logo = await getLogoBase64();

    // Branding Colors
    const COLORS = {
      dark: [5, 6, 15],      // Cyber-Deep
      purple: [167, 139, 255],
      cyan: [0, 240, 255],
      text: [248, 249, 250],
      muted: [154, 154, 176]
    };

    // Helpers
    const fillBG = (c) => { doc.setFillColor(...c); doc.rect(0,0,W,H,"F"); };
    const getLogoDims = (targetW) => {
      if (!logo) return { w:0, h:0 };
      const ratio = logo.h / logo.w;
      return { w: targetW, h: targetW * ratio };
    };

    // Safe Text Helper - Enforced String and Polish
    const txt = (text, x, y, size, color=COLORS.text, align="left", style="normal") => {
      if (text === undefined || text === null) text = "—";
      doc.setFontSize(size); 
      doc.setTextColor(...color); 
      doc.setFont("helvetica", style);
      
      // Force string and strip non-ASCII (emojis/special chars) for PDF stability
      let content = Array.isArray(text) ? text.map(String) : String(text);
      if (typeof content === "string") {
        content = content.replace(/[^\x00-\x7F]/g, "").trim();
      } else if (Array.isArray(content)) {
        content = content.map(s => s.replace(/[^\x00-\x7F]/g, "").trim());
      }
      
      doc.text(content, x, y, { align: align });
    };

    let pdfPage = 1;
    const addFooter = () => {
      doc.setDrawColor(...COLORS.purple); doc.setLineWidth(0.1); doc.line(14, 282, 196, 282);
      txt("NeuroAds AI Analysis • Proprietary Marketing Intelligence", 14, 287, 6, COLORS.muted);
      txt(`Internal Report • Page ${pdfPage}`, 196, 287, 6, COLORS.muted, "right");
      pdfPage++;
    };


    // ===================== PAGE 1: COVER =====================
    fillBG(COLORS.dark);
    doc.setDrawColor(...COLORS.purple); doc.setLineWidth(2); doc.line(0, 0, W, 0); // Bold Top accent
    
    const coverLogo = getLogoDims(65);
    if (logo) doc.addImage(logo.data, "PNG", (W-coverLogo.w)/2, 65, coverLogo.w, coverLogo.h);

    txt("BRAIN RESPONSE", W/2, 135, 36, COLORS.text, "center", "bold");
    txt("PREDICTION REPORT", W/2, 149, 26, COLORS.purple, "center", "bold");
    
    if (STATE.fileName) {
      txt(STATE.fileName.toUpperCase(), W/2, 158, 11, COLORS.muted, "center", "bold");
    }

    doc.setDrawColor(...COLORS.purple); doc.setLineWidth(0.4); doc.line(W/2-25, 162, W/2+25, 162);

    txt("TECHNOLOGY POWERED BY", W/2, 220, 9, COLORS.muted, "center");
    txt("TRIBE v2 Neural Core • Meta FAIR", W/2, 228, 12, COLORS.text, "center", "bold");

    txt(`REPORT ID: ${Date.now()}`, W/2, 260, 8, COLORS.muted, "center");
    txt(`GENERATED ON: ${new Date().toLocaleDateString()} at ${new Date().toLocaleTimeString()}`, W/2, 266, 8, COLORS.muted, "center");

    addFooter();

    doc.addPage(); fillBG(COLORS.dark);
    const tinyLogo = getLogoDims(22);
    if (logo) doc.addImage(logo.data, "PNG", 14, 10, tinyLogo.w, tinyLogo.h);
    txt("Executive Intelligence Summary", 196, 18, 14, COLORS.text, "right", "bold");
    
    const engColor = k.avg_attention >= 75 ? [0, 255, 163] : k.avg_attention >= 50 ? [0, 229, 255] : [255, 60, 172];
    doc.setDrawColor(...engColor); doc.setLineWidth(1);
    doc.setFillColor(10, 10, 30); doc.roundedRect(14, 35, 182, 32, 3, 3, "FD");
    txt("OVERALL CAMPAIGN PERFORMANCE", 105, 45, 11, COLORS.muted, "center", "bold");
    
    // Clean engagement string (strip emojis for PDF compatibility)
    const cleanEng = String(k.engagement).replace(/[^\x00-\x7F]/g, "").trim().toUpperCase();
    txt(cleanEng, 105, 60, 24, engColor, "center", "bold");

    let kx = 14;
    const kpiList = [
      { L: "Attention Mean", V: k.avg_attention.toFixed(1), S: "/ 100", C: COLORS.purple },
      { L: "Peak Activation", V: k.peak_attention.toFixed(1), S: "/ 100", C: COLORS.cyan },
      { L: "Hook Strength", V: k.hook_score.toFixed(1), S: "/ 100", C: [0, 255, 163] },
      { L: "Brain Steps", V: String(k.n_timesteps), S: "seconds", C: [255, 60, 172] }
    ];
    kpiList.forEach(item => {
      doc.setFillColor(18, 18, 35); doc.roundedRect(kx, 75, 43, 35, 2, 2, "F");
      doc.setDrawColor(...item.C); doc.setLineWidth(0.4); doc.line(kx+8, 75, kx+35, 75);
      txt(item.L.toUpperCase(), kx+21.5, 83, 6, COLORS.muted, "center");
      txt(item.V, kx+21.5, 95, 15, item.C, "center", "bold");
      txt(item.S, kx+21.5, 102, 6, COLORS.muted, "center");
      kx += 46;
    });

    txt("STRATEGIC PREDICTIVE INSIGHTS", 14, 130, 9, COLORS.text, "left", "bold");
    doc.setDrawColor(...COLORS.purple); doc.setLineWidth(0.2); doc.line(14, 133, 196, 133);
    
    let iy = 142;
    const insights = [
      { t: "Viral Potential", b: k.hook_score >= 60 ? "Exceptional initial engagement. The first 3 seconds contain high-frequency visual anchors that trigger immediate dopamine response." : "Initial hook requires reinforcement. Consider increasing visual complexity or audio contrast in the opening 2 seconds." },
      { t: "Retention Pacing", b: "Peak cortical activation detected at " + (k.peak_timestep * 2) + "s. This segment represents the highest cognitive investment window for key message delivery." },
      { t: "Perceptual Alignment", b: "Balanced cross-modal synchronization. The predictive engine indicators suggest high alignment between visual and auditory stimuli." }
    ];
    insights.forEach(ins => {
      doc.setFillColor(8, 8, 25); doc.roundedRect(14, iy, 182, 24, 2, 2, "F");
      txt(ins.t.toUpperCase(), 18, iy+8, 8, COLORS.cyan, "left", "bold");
      const lines = doc.splitTextToSize(String(ins.b), 174);
      txt(lines, 18, iy+15, 8, COLORS.text);
      iy += 28;
    });
    addFooter();


    // ===================== PAGE 3: NEURAL ACTIVATION =====================
    doc.addPage(); fillBG(COLORS.dark);
    if (logo) doc.addImage(logo.data, "PNG", 14, 10, tinyLogo.w, tinyLogo.h);
    txt("Neural Activation Mapping", 196, 18, 14, COLORS.text, "right", "bold");

    const activationCharts = [
      { id: "chart-attention", t: "NEURAL ATTENTION ENVELOPE (TIME-DOMAIN)" },
      { id: "chart-roi", t: "CORTICAL REGION PERFORMANCE (ROI)" }
    ];
    let cy1 = 40;
    activationCharts.forEach(c => {
      const canvas = document.getElementById(c.id);
      if (canvas) {
        doc.setFillColor(10, 10, 25); doc.roundedRect(14, cy1, 182, 105, 3, 3, "F");
        txt(c.t, 20, cy1+10, 8, COLORS.muted, "left", "bold");
        try { 
          doc.addImage(canvas.toDataURL("image/png", 1.0), "PNG", 18, cy1+15, 174, 82); 
        } catch(e) {}
        cy1 += 115;
      }
    });
    addFooter();


    // ===================== PAGE 4: ENGAGEMENT DYNAMICS =====================
    doc.addPage(); fillBG(COLORS.dark);
    if (logo) doc.addImage(logo.data, "PNG", 14, 10, tinyLogo.w, tinyLogo.h);
    txt("Dynamic Engagement Analysis", 196, 18, 14, COLORS.text, "right", "bold");

    const strategicCharts = [
      { id: "chart-segments", t: "SEGMENT ATTENTION INTENSITY" },
      { id: "chart-radar", t: "HOLISTIC BRAIN ENGAGEMENT PROFILE" }
    ];
    let cy2 = 40;
    strategicCharts.forEach(c => {
      const canvas = document.getElementById(c.id);
      if (canvas) {
        doc.setFillColor(10, 10, 25); doc.roundedRect(14, cy2, 182, 105, 3, 3, "F");
        txt(c.t, 20, cy2+10, 8, COLORS.muted, "left", "bold");
        try { 
          doc.addImage(canvas.toDataURL("image/png", 1.0), "PNG", 18, cy2+15, 174, 82); 
        } catch(e) {}
        cy2 += 115;
      }
    });
    addFooter();


    // ===================== PAGE 5: STRATEGIC HOOKS =====================
    doc.addPage(); fillBG(COLORS.dark);
    if (logo) doc.addImage(logo.data, "PNG", 14, 10, tinyLogo.w, tinyLogo.h);
    txt("Key Attention Hooks (Predictive)", 196, 18, 14, COLORS.text, "right", "bold");

    txt("OPTIMAL ENGAGEMENT SEGMENTS (ORDERED BY STRENGTH)", 14, 38, 9, COLORS.muted, "left", "bold");
    doc.setDrawColor(...COLORS.purple); doc.setLineWidth(0.2); doc.line(14, 41, 196, 41);

    let hy = 48;
    (r.top_hooks || []).slice(0, 8).forEach((h, i) => {
      doc.setFillColor(15, 15, 35); doc.roundedRect(14, hy, 182, 22, 2, 2, "F");
      doc.setFillColor(...COLORS.purple); doc.circle(22, hy+11, 4, "F");
      txt(String(i+1), 22, hy+13.5, 10, [255,255,255], "center", "bold");
      
      txt(String(h.text || "(Visual context detected)").substring(0,65), 32, hy+9.5, 9, COLORS.text, "left", "bold");
      txt(`Time Segment: ${h.start.toFixed(1)}s – ${h.end.toFixed(1)}s`, 32, hy+16.5, 7, COLORS.muted);
      
      doc.setFillColor(30,30,55); doc.roundedRect(145, hy+9, 40, 4, 2, 2, "F");
      doc.setFillColor(...COLORS.cyan); doc.roundedRect(145, hy+9, 40 * (h.attention/100), 4, 2, 2, "F");
      txt(h.attention.toFixed(0), 191, hy+12.5, 9, COLORS.cyan, "right", "bold");
      hy += 26;
    });
    addFooter();


    // ===================== PAGE 6: TECHNICAL APPENDIX =====================
    doc.addPage(); fillBG(COLORS.dark);
    if (logo) doc.addImage(logo.data, "PNG", 14, 10, tinyLogo.w, tinyLogo.h);
    txt("Appendix: Signal Event Log", 196, 18, 14, COLORS.text, "right", "bold");

    const hLabels = ["EVENT TYPE", "START", "DURATION", "DATA CONTEXT"];
    const cols = [16, 50, 75, 100];
    doc.setFillColor(25, 25, 50); doc.rect(14, 35, 182, 10, "F");
    hLabels.forEach((lbl, i) => {
      txt(lbl, cols[i], 41.5, 7, COLORS.purple, "left", "bold");
    });

    let ty = 52;
    (r.events_preview || []).forEach((ev, i) => {
      if (ty > 270) { 
        addFooter(); 
        doc.addPage(); 
        fillBG(COLORS.dark); 
        ty = 20; 
        // Redraw headers on new page
        doc.setFillColor(25, 25, 50); doc.rect(14, ty, 182, 10, "F");
        hLabels.forEach((lbl, i) => {
          txt(lbl, cols[i], ty+6.5, 7, COLORS.purple, "left", "bold");
        });
        ty += 15;
      }
      doc.setFillColor(i%2===0 ? 8 : 15, i%2===0 ? 8 : 15, i%2===0 ? 25 : 35);
      doc.rect(14, ty-4, 182, 9, "F");
      txt(ev.type.toUpperCase(), cols[0], ty+1.5, 7, COLORS.text, "left", "bold");
      txt(ev.start.toFixed(2)+"s", cols[1], ty+1.5, 7, COLORS.text);
      txt(ev.duration.toFixed(2)+"s", cols[2], ty+1.5, 7, COLORS.text);
      txt(String(ev.text||"—").substring(0,55), cols[3], ty+1.5, 7, COLORS.muted);
      ty += 9;
    });
    addFooter();

    doc.save(`NeuroAds_Premium_Report_${Date.now()}.pdf`);

  } catch (err) {
    console.error("PDF generation failed:", err);
    alert("An error occurred during PDF generation: " + err.message);
  }
}
