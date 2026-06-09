"""Export static JSON + site files for GitHub Pages."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .seizure_viewer_data import (
    DISPLAY_HZ,
    PROJECT_ROOT,
    RECORDINGS,
    block_average_downsample,
    get_featured_segments,
    get_predicted_spans,
    get_recording,
    get_true_spans,
    recording_meta,
)

DOCS_DIR = PROJECT_ROOT / "docs"
WEB_DIR = PROJECT_ROOT / "web"


def export_trace_binary(rec_id: str, target_hz: float = DISPLAY_HZ) -> tuple[dict, np.ndarray]:
    rec = get_recording(rec_id)
    ecog_ds, display_hz, block = block_average_downsample(
        rec.ecog, rec.fs_hz, target_hz
    )
    meta = {
        "source_fs_hz": float(rec.fs_hz),
        "display_hz": display_hz,
        "block_size": block,
        "n_samples": int(ecog_ds.shape[0]),
        "duration_s": float(ecog_ds.shape[0] / display_hz),
    }
    return meta, ecog_ds


def export_events_json(rec_id: str) -> dict:
    featured = get_featured_segments(rec_id)
    return {
        "true_spans": get_true_spans(rec_id),
        "pred_spans": get_predicted_spans(rec_id),
        "featured": featured,
    }


def export_zoom_pngs(rec_id: str, static_dir: Path, limit: int = 5) -> None:
    """Render zoom snapshot PNGs from updated postprocessed predictions."""
    rec = get_recording(rec_id)
    featured = get_featured_segments(rec_id, limit=limit)
    true_spans = get_true_spans(rec_id)
    pred_spans = get_predicted_spans(rec_id)

    for seg in featured:
        t0 = float(seg["start_s"])
        t1 = float(seg["end_s"])
        s0 = int(max(0, t0 * rec.fs_hz))
        s1 = int(min(len(rec.ecog), t1 * rec.fs_hz))
        raw = rec.ecog[s0:s1]
        if raw.size == 0:
            continue
        ecog, hz, _ = block_average_downsample(raw, rec.fs_hz, DISPLAY_HZ)
        t = (s0 / rec.fs_hz) + (np.arange(ecog.shape[0]) / hz)

        plt.figure(figsize=(12, 4))
        plt.plot(t, ecog, color="navy", linewidth=0.7)
        for sp in true_spans:
            if sp["end_s"] < t0 or sp["start_s"] > t1:
                continue
            plt.axvspan(
                max(t0, sp["start_s"]),
                min(t1, sp["end_s"]),
                color="green",
                alpha=0.18,
            )
        for sp in pred_spans:
            if sp["end_s"] < t0 or sp["start_s"] > t1:
                continue
            plt.axvspan(
                max(t0, sp["start_s"]),
                min(t1, sp["end_s"]),
                color="red",
                alpha=0.15,
            )
        plt.xlabel("Time (s)")
        plt.ylabel("ECoG level")
        plt.title(f"{seg['label']} ({rec_id}, 100 Hz block average)")
        plt.tight_layout()
        out = static_dir / f"zoom_{seg['id']}.png"
        plt.savefig(out, dpi=160)
        plt.close()
        print(f"  wrote {out.name}")


def write_static_app_js(dest: Path) -> None:
    dest.write_text(
        """const ROOT = (() => {
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
  const url = `${ROOT}${path}`.replace(/\\\\/g, '/');
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
""",
        encoding="utf-8",
    )


def export_site(out_dir: Path, pages_base: str = "/Mice/") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data"
    static_dir = out_dir / "static"
    data_dir.mkdir(exist_ok=True)
    static_dir.mkdir(exist_ok=True)

    recordings = [recording_meta(rid) for rid in RECORDINGS]
    (data_dir / "recordings.json").write_text(
        json.dumps(recordings, indent=2), encoding="utf-8"
    )

    for rec_id in RECORDINGS:
        print(f"Exporting {rec_id}…")
        events = export_events_json(rec_id)
        (data_dir / f"{rec_id}_events.json").write_text(
            json.dumps(events, indent=2), encoding="utf-8"
        )
        meta, ecog_ds = export_trace_binary(rec_id, target_hz=DISPLAY_HZ)
        (data_dir / f"{rec_id}_trace_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        (data_dir / f"{rec_id}_trace.bin").write_bytes(ecog_ds.tobytes())
        old_json = data_dir / f"{rec_id}_trace.json"
        if old_json.exists():
            old_json.unlink()

    shutil.copy2(WEB_DIR / "styles.css", out_dir / "styles.css")
    # Zoom PNGs from updated postprocessed dataset (rec1 featured gallery).
    export_zoom_pngs("rec1", static_dir, limit=5)
    for i in range(1, 6):
        src = static_dir / f"zoom_{i}.png"
        dst_web = WEB_DIR / "static" / f"zoom_{i}.png"
        if src.exists():
            dst_web.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_web)

    if not (out_dir / "index.html").exists():
        index = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        index = index.replace(
            "<head>",
            f'<head>\n    <meta name="pages-base" content="{pages_base}" />',
        )
        index = index.replace('href="styles.css"', f'href="{pages_base}styles.css"')
        index = index.replace('src="app.js"', f'src="{pages_base}app.js"')
        (out_dir / "index.html").write_text(index, encoding="utf-8")
    src_app = (PROJECT_ROOT / "docs" / "app.js").resolve()
    dst_app = (out_dir / "app.js").resolve()
    if src_app.exists() and src_app != dst_app:
        shutil.copy2(src_app, dst_app)
    elif not dst_app.exists():
        write_static_app_js(dst_app)
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")
    print("Wrote", out_dir.resolve())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=DOCS_DIR)
    p.add_argument("--pages-base", default="/Mice/")
    args = p.parse_args()
    export_site(args.out, pages_base=args.pages_base)


if __name__ == "__main__":
    main()
