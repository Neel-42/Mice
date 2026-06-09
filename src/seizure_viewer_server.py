"""Local web server for the interactive seizure ECoG viewer."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from .seizure_viewer_data import (
    PROJECT_ROOT,
    RECORDINGS,
    downsample_trace,
    get_featured_segments,
    get_predicted_spans,
    get_recording,
    get_true_spans,
    recording_meta,
    spans_in_window,
)

WEB_DIR = PROJECT_ROOT / "web"
STATIC_DIR = WEB_DIR / "static"


def _sync_zoom_images() -> None:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(1, 6):
        src = PROJECT_ROOT / "outputs" / f"ecog_predicted_seizure_zoom_{i}.png"
        if src.exists():
            shutil.copy2(src, STATIC_DIR / f"zoom_{i}.png")


def create_app() -> Flask:
    _sync_zoom_images()
    app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="")

    @app.get("/")
    def index() -> object:
        return send_from_directory(WEB_DIR, "index.html")

    @app.get("/<path:asset>")
    def assets(asset: str) -> object:
        return send_from_directory(WEB_DIR, asset)

    @app.get("/api/recordings")
    def api_recordings() -> object:
        return jsonify([recording_meta(rid) for rid in RECORDINGS])

    @app.get("/api/events/<rec_id>")
    def api_events(rec_id: str) -> object:
        return jsonify(
            {
                "true_spans": get_true_spans(rec_id),
                "pred_spans": get_predicted_spans(rec_id),
                "featured": get_featured_segments(rec_id),
            }
        )

    @app.get("/api/trace/<rec_id>")
    def api_trace(rec_id: str) -> object:
        t0 = float(request.args.get("t0", 0))
        t1 = float(request.args.get("t1", 60))
        max_points = int(request.args.get("max_points", 12000))
        rec = get_recording(rec_id)
        trace = downsample_trace(rec.ecog, rec.fs_hz, t0, t1, max_points=max_points)
        true_sp = spans_in_window(get_true_spans(rec_id), t0, t1)
        pred_sp = spans_in_window(get_predicted_spans(rec_id), t0, t1)
        return jsonify(
            {
                "t0": t0,
                "t1": t1,
                "trace": trace,
                "true_spans": true_sp,
                "pred_spans": pred_sp,
            }
        )

    return app


def main() -> None:
    p = argparse.ArgumentParser(description="Run interactive seizure viewer")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    app = create_app()
    print(f"Open http://{args.host}:{args.port}/ in your browser")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
