"""Local YOLO tracker service for the Tracking Set B browser annotator.

Start from the repository root:
    python tools/annotation_tracker_server.py

    The browser sends one frame and the last accepted rectangle(s) per request. This
is deliberately a local, opt-in helper: the annotator still owns the track ID
and the human must verify every generated rectangle.
"""
from __future__ import annotations

import argparse
import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from ultralytics import YOLO


DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "yolov11x" / "best.pt"


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
    right_area = max(0.0, float(right[2] - right[0])) * max(0.0, float(right[3] - right[1]))
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _selection_score(candidate: np.ndarray, seed: np.ndarray) -> float:
    seed_center = (seed[:2] + seed[2:]) / 2.0
    candidate_center = (candidate[:2] + candidate[2:]) / 2.0
    seed_size = np.maximum(seed[2:] - seed[:2], 1.0)
    distance = np.linalg.norm((candidate_center - seed_center) / seed_size)
    return 3.0 * _iou(candidate, seed) - float(distance)


def _match_boxes(boxes: np.ndarray, seeds: np.ndarray) -> list[np.ndarray]:
    """Match distinct detections to seeds, preserving one result per seed."""
    available = set(range(len(boxes)))
    matched = []
    for seed in seeds:
        if not available:
            break
        index = max(available, key=lambda candidate_index:
                    _selection_score(boxes[candidate_index], seed))
        matched.append(boxes[index])
        available.remove(index)
    return matched


def _pad_box(box: np.ndarray, image_width: int, image_height: int,
             padding_ratio: float) -> np.ndarray:
    width = max(1.0, float(box[2] - box[0]))
    height = max(1.0, float(box[3] - box[1]))
    padded = box.astype(np.float32).copy()
    padded[0] -= width * padding_ratio
    padded[1] -= height * padding_ratio
    padded[2] += width * padding_ratio
    padded[3] += height * padding_ratio
    padded[[0, 2]] = np.clip(padded[[0, 2]], 0, image_width)
    padded[[1, 3]] = np.clip(padded[[1, 3]], 0, image_height)
    return padded


class TrackerHandler(BaseHTTPRequestHandler):
    detector: YOLO
    confidence: float
    iou_threshold: float
    padding_ratio: float

    def _write(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/health"):
            self._write(200, {
                "ok": True,
                "tracker": "YOLOv11x",
                "message": "Annotation tracker is running. Use the VVT page and press t.",
            })
            return
        self._write(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/track":
            self._write(404, {"error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            image_data = base64.b64decode(request["image"])
            image = Image.open(BytesIO(image_data)).convert("RGB")
            seed_values = request.get("rois", request.get("roi"))
            if seed_values is None:
                raise ValueError("request must include roi or rois")
            seeds = np.asarray(seed_values, dtype=np.float32)
            if seeds.ndim == 1:
                seeds = seeds.reshape(1, -1)
            if seeds.shape[1] != 4:
                raise ValueError("each ROI must be [x1, y1, x2, y2]")
            result = self.detector(np.asarray(image), conf=self.confidence,
                                   iou=self.iou_threshold, verbose=False)[0]
            if result.boxes is None or len(result.boxes) == 0:
                self._write(200, {"roi": None, "reason": "no detections"})
                return

            boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
            confidences = result.boxes.conf.cpu().numpy().astype(float)
            matched_boxes = _match_boxes(boxes, seeds)
            matched_indices = [next(index for index, box in enumerate(boxes)
                              if np.array_equal(box, matched_box))
                               for matched_box in matched_boxes]
            if len(matched_boxes) != len(seeds):
                self._write(200, {"roi": None, "reason": "ambiguous detection"})
                return

            output_boxes = []
            for seed, matched_box, matched_index in zip(seeds, matched_boxes, matched_indices):
                score = _selection_score(matched_box, seed)
                if confidences[matched_index] < self.confidence or score < -1.5:
                    self._write(200, {"rois": None, "reason": "ambiguous detection"})
                    return
                output_boxes.append(_pad_box(
                    matched_box,
                    image.width,
                    image.height,
                    self.padding_ratio,
                ))
            rois = [{
                "x": float(output_box[0]),
                "y": float(output_box[1]),
                "width": float(output_box[2] - output_box[0]),
                "height": float(output_box[3] - output_box[1]),
            } for output_box in output_boxes]
            self._write(200, {
                "rois": rois,
                "confidence": [float(confidences[index]) for index in matched_indices],
                "padding_ratio": self.padding_ratio,
            })
        except Exception as error:  # Keep failures visible to the annotator.
            self._write(400, {"error": str(error)})

    def log_message(self, format: str, *args: Any) -> None:
        print(format % args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.30)
    parser.add_argument(
        "--padding",
        type=float,
        default=0.06,
        help="Extra box margin per side as a fraction of box size (default: 0.06)",
    )
    args = parser.parse_args()
    if not args.weights.is_file():
        raise SystemExit(f"YOLO weights not found: {args.weights}")

    TrackerHandler.detector = YOLO(str(args.weights))
    TrackerHandler.confidence = args.conf
    TrackerHandler.iou_threshold = args.iou
    TrackerHandler.padding_ratio = max(0.0, args.padding)
    server = ThreadingHTTPServer((args.host, args.port), TrackerHandler)
    print(f"YOLO annotation tracker listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
