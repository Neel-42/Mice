const ROOT = (() => {
  const m = document.querySelector('meta[name="pages-base"]');
  if (m && m.content) return m.content;
  const parts = window.location.pathname.split('/').filter(Boolean);
  if (parts.length && parts[0].toLowerCase() === 'mice') return '/Mice/';
  return './';
})();

const state = {
  recId: null,
  durationS: 86400,
  events: { true_spans: [], pred_spans: [], featured: [] },
  trace: { t: [], ecog: [] },
  seizureIndex: 0,
};

const els = {
  recordingSelect: document.getElementById('recordingSelect'),
  startSlider: document.getElementById('startSlider'),
  durationSlider: document.getElementById('durationSlider'),
  startLabel: document.getElementById('startLabel'),
  durationLabel: document.getElementById('durationLabel'),
  showTrue: document.getElementById('showTrue'),
  showPred: document.getElementById('showPred'),
  prevBtn: document.getElementById('prevBtn'),
  nextBtn: document.getElementById('nextBtn'),
  featuredList: document.getElementById('featuredList'),
  zoomGallery: document.getElementById('zoomGallery'),
  status: document.getElementById('status'),
};

async function fetchJson(path) {
  const url = `${ROOT}${path}`.replace(/\\/g, '/');
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Request failed: ${url}`);
  return res.json();
}

function formatTime(s) {
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  return `${h}h ${m}m ${sec}s`;
}

function windowRange() {
  const t0 = Number(els.startSlider.value);
  const t1 = t0 + Number(els.durationSlider.value);
  return { t0, t1 };
}

function spansInWindow(spans, t0, t1) {
  return spans
    .filter((sp) => !(sp.end_s < t0 || sp.start_s > t1))
    .map((sp) => ({
      start_s: Math.max(t0, sp.start_s),
      end_s: Math.min(t1, sp.end_s),
    }));
}

function sliceTrace(t0, t1, maxPoints = 12000) {
  const t = state.trace.t;
  const y = state.trace.ecog;
  let i0 = 0;
  while (i0 < t.length && t[i0] < t0) i0 += 1;
  let i1 = i0;
  while (i1 < t.length && t[i1] <= t1) i1 += 1;
  const segT = t.slice(i0, i1);
  const segY = y.slice(i0, i1);
  if (segT.length <= maxPoints) return { t: segT, ecog: segY };
  const step = Math.ceil(segT.length / maxPoints);
  return {
    t: segT.filter((_, i) => i % step === 0),
    ecog: segY.filter((_, i) => i % step === 0),
  };
}

function buildShapes(trueSpans, predSpans) {
  const shapes = [];
  if (els.showTrue.checked) {
    for (const sp of trueSpans) {
      shapes.push({
        type: 'rect', xref: 'x', yref: 'paper',
        x0: sp.start_s, x1: sp.end_s, y0: 0, y1: 1,
        fillcolor: 'rgba(46, 204, 113, 0.25)', line: { width: 0 }, layer: 'below',
      });
    }
  }
  if (els.showPred.checked) {
    for (const sp of predSpans) {
      shapes.push({
        type: 'rect', xref: 'x', yref: 'paper',
        x0: sp.start_s, x1: sp.end_s, y0: 0, y1: 1,
        fillcolor: 'rgba(231, 76, 60, 0.22)', line: { width: 0 }, layer: 'below',
      });
    }
  }
  return shapes;
}

function renderPlot() {
  const { t0, t1 } = windowRange();
  const trace = sliceTrace(t0, t1);
  const trueSpans = spansInWindow(state.events.true_spans, t0, t1);
  const predSpans = spansInWindow(state.events.pred_spans, t0, t1);

  const layout = {
    paper_bgcolor: '#0f1419',
    plot_bgcolor: '#121a24',
    font: { color: '#e8eef7' },
    margin: { l: 56, r: 24, t: 36, b: 48 },
    title: 'ECoG level vs time',
    xaxis: { title: 'Time (s)', gridcolor: '#2a3548', zeroline: false, range: [t0, t1] },
    yaxis: { title: 'ECoG level', gridcolor: '#2a3548', zeroline: false },
    shapes: buildShapes(trueSpans, predSpans),
    legend: { orientation: 'h', y: 1.08 },
  };

  Plotly.react('ecogPlot', [{
    x: trace.t,
    y: trace.ecog,
    type: 'scattergl',
    mode: 'lines',
    name: 'ECoG',
    line: { color: '#4da3ff', width: 1 },
  }], layout, { responsive: true, displayModeBar: true, scrollZoom: true });

  els.status.textContent = `Showing ${trace.t.length.toLocaleString()} points | green=true, red=predicted`;
}

function renderFeatured() {
  els.featuredList.innerHTML = '';
  for (const seg of state.events.featured) {
    const li = document.createElement('li');
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = `${seg.label} (${Math.round(seg.start_s)}s)`;
    btn.addEventListener('click', () => {
      const center = (seg.start_s + seg.end_s) / 2;
      const dur = Number(els.durationSlider.value);
      els.startSlider.value = Math.max(0, Math.floor(center - dur / 2));
      els.startLabel.textContent = els.startSlider.value;
      renderPlot();
    });
    li.appendChild(btn);
    els.featuredList.appendChild(li);
  }

  els.zoomGallery.innerHTML = '';
  for (const seg of state.events.featured) {
    const fig = document.createElement('figure');
    const img = document.createElement('img');
    img.src = `${ROOT}static/zoom_${seg.id}.png`;
    img.alt = seg.label;
    img.addEventListener('click', () => {
      els.startSlider.value = Math.max(0, Math.floor(seg.start_s));
      els.startLabel.textContent = els.startSlider.value;
      renderPlot();
    });
    const cap = document.createElement('figcaption');
    cap.textContent = seg.label;
    fig.appendChild(img);
    fig.appendChild(cap);
    els.zoomGallery.appendChild(fig);
  }
}

function allSeizureCenters() {
  return [...state.events.true_spans, ...state.events.pred_spans]
    .map((s) => (s.start_s + s.end_s) / 2)
    .sort((a, b) => a - b);
}

function jumpSeizure(delta) {
  const centers = allSeizureCenters();
  if (!centers.length) return;
  state.seizureIndex = (state.seizureIndex + delta + centers.length) % centers.length;
  const center = centers[state.seizureIndex];
  const dur = Number(els.durationSlider.value);
  els.startSlider.value = Math.max(0, Math.floor(center - dur / 2));
  els.startLabel.textContent = els.startSlider.value;
  renderPlot();
}

async function loadRecording(recId) {
  state.recId = recId;
  const recs = await fetchJson('data/recordings.json');
  const meta = recs.find((r) => r.id === recId);
  state.durationS = meta.duration_s;
  els.startSlider.max = Math.max(0, Math.floor(meta.duration_s - 10));
  els.startLabel.textContent = els.startSlider.value;
  state.events = await fetchJson(`data/${recId}_events.json`);
  state.trace = await fetchJson(`data/${recId}_trace.json`);
  renderFeatured();
  renderPlot();
}

async function init() {
  const recs = await fetchJson('data/recordings.json');
  for (const rec of recs) {
    const opt = document.createElement('option');
    opt.value = rec.id;
    opt.textContent = `${rec.label} (${rec.duration_h.toFixed(1)} h)`;
    els.recordingSelect.appendChild(opt);
  }

  els.recordingSelect.addEventListener('change', () => loadRecording(els.recordingSelect.value));
  els.startSlider.addEventListener('input', () => { els.startLabel.textContent = els.startSlider.value; });
  els.startSlider.addEventListener('change', renderPlot);
  els.durationSlider.addEventListener('input', () => { els.durationLabel.textContent = els.durationSlider.value; });
  els.durationSlider.addEventListener('change', renderPlot);
  els.showTrue.addEventListener('change', renderPlot);
  els.showPred.addEventListener('change', renderPlot);
  els.prevBtn.addEventListener('click', () => jumpSeizure(-1));
  els.nextBtn.addEventListener('click', () => jumpSeizure(1));

  await loadRecording(recs[0].id);
}

init().catch((err) => {
  els.status.textContent = `Error: ${err.message}`;
  console.error(err);
});
