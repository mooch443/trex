#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone harness for testing trex_sam3_interface.py outside TRex.

Examples:
  python Application/src/tracker/python/test_trex_sam3_interface.py \
      --weights /path/to/sam3_weights.pt \
      --video /path/to/video.mp4 \
      --text "fish" \
      --text-session-scope

  python Application/src/tracker/python/test_trex_sam3_interface.py \
      --weights /path/to/sam3_weights.pt \
      --images-dir /path/to/frames \
      --box 0:100,120,300,320 \
      --max-frames 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import trex_sam3_interface as sam3


def _load_video_frames(path: Path, max_frames: int) -> List[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    frames: List[np.ndarray] = []
    try:
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(np.ascontiguousarray(frame))
    finally:
        cap.release()
    return frames


def _load_image_frames(path: Path, max_frames: int) -> List[np.ndarray]:
    import cv2

    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    files: List[Path] = []
    for ext in exts:
        files.extend(path.glob(ext))
        files.extend(path.glob(ext.upper()))
    files = sorted(set(files))[:max_frames]
    if not files:
        raise RuntimeError(f"No images found in: {path}")

    frames: List[np.ndarray] = []
    for p in files:
        frame = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Failed to read image: {p}")
        frames.append(np.ascontiguousarray(frame))
    return frames


def _make_synthetic_frames(num_frames: int, h: int = 512, w: int = 768) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    yy, xx = np.mgrid[0:h, 0:w]
    for i in range(num_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[..., 0] = (xx % 255).astype(np.uint8)
        frame[..., 1] = (yy % 255).astype(np.uint8)
        frame[..., 2] = ((xx + yy) % 255).astype(np.uint8)

        x0 = 40 + 12 * i
        y0 = 60 + 8 * i
        x1 = min(x0 + 180, w)
        y1 = min(y0 + 140, h)
        frame[y0:y1, x0:x1, :] = 255
        frames.append(np.ascontiguousarray(frame))
    return frames


def _parse_box_spec(spec: str) -> Tuple[int, List[float]]:
    # Format: frame_index:x1,y1,x2,y2
    if ":" not in spec:
        raise ValueError(f"Invalid --box format: {spec!r}")
    frame_s, coords_s = spec.split(":", 1)
    coords = [float(x.strip()) for x in coords_s.split(",") if x.strip()]
    if len(coords) != 4:
        raise ValueError(f"Invalid --box coordinates in {spec!r}")
    return int(frame_s), coords


def _obj_to_mask(obj: Dict[str, Any]) -> np.ndarray:
    h = int(obj["height"])
    w = int(obj["width"])
    flat = np.zeros((h * w,), dtype=np.uint8)
    fg = np.asarray(obj.get("foreground_indices", []), dtype=np.int64)
    if fg.size:
        flat[fg] = 1
    return flat.reshape((h, w))


def _save_masks_npz(frame_out: Dict[str, Any], output_dir: Path) -> Path:
    frame_index = int(frame_out["frame_index"])
    objects = list(frame_out.get("objects", []))
    if objects:
        masks = np.stack([_obj_to_mask(obj) for obj in objects], axis=0).astype(np.uint8)
        scores = np.asarray([float(obj.get("score", 0.0)) for obj in objects], dtype=np.float32)
        class_ids = np.asarray([int(obj.get("class_id", 0)) for obj in objects], dtype=np.int32)
        obj_ids = np.asarray([int(obj.get("obj_id", i)) for i, obj in enumerate(objects)], dtype=np.int32)
    else:
        masks = np.zeros((0, 0, 0), dtype=np.uint8)
        scores = np.zeros((0,), dtype=np.float32)
        class_ids = np.zeros((0,), dtype=np.int32)
        obj_ids = np.zeros((0,), dtype=np.int32)

    out_path = output_dir / f"frame_{frame_index:06d}_masks.npz"
    np.savez_compressed(out_path, masks=masks, scores=scores, class_ids=class_ids, obj_ids=obj_ids)
    return out_path


def _color_for_obj_id(obj_id: int) -> Tuple[int, int, int]:
    oid = int(obj_id)
    # Deterministic BGR palette for stable visualization across frames.
    return (
        int((37 * oid + 71) % 255),
        int((67 * oid + 131) % 255),
        int((97 * oid + 191) % 255),
    )


def _render_preview_frame(frame: np.ndarray, frame_out: Dict[str, Any], alpha: float) -> np.ndarray:
    import cv2

    vis = np.ascontiguousarray(frame.copy())
    a = float(np.clip(alpha, 0.0, 1.0))

    for obj in frame_out.get("objects", []):
        obj_id = int(obj.get("obj_id", 0))
        score = float(obj.get("score", 0.0))
        class_id = int(obj.get("class_id", 0))
        color = np.asarray(_color_for_obj_id(obj_id), dtype=np.float32)

        mask = _obj_to_mask(obj) > 0
        if np.any(mask):
            src = vis[mask].astype(np.float32)
            vis[mask] = np.clip((1.0 - a) * src + a * color[None, :], 0, 255).astype(np.uint8)

            ys, xs = np.where(mask)
            x0 = int(xs.min())
            y0 = int(ys.min())
            x1 = int(xs.max() + 1)
            y1 = int(ys.max() + 1)
        else:
            x0 = y0 = x1 = y1 = 0

        cv2.rectangle(vis, (x0, y0), (x1, y1), tuple(int(v) for v in color), 2)
        label = f"id={obj_id} cls={class_id} conf={score:.2f}"
        ty = y0 - 8 if y0 > 20 else y0 + 16
        cv2.putText(vis, label, (x0, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    header = f"frame={int(frame_out.get('frame_index', -1))} objects={int(frame_out.get('num_objects', 0))}"
    cv2.putText(vis, header, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return vis


def _run(args: argparse.Namespace) -> int:
    if args.video is None and args.images_dir is None and not args.synthetic:
        args.synthetic = True

    if args.video is not None:
        frames = _load_video_frames(args.video, args.max_frames)
        source_label = f"video={args.video}"
    elif args.images_dir is not None:
        frames = _load_image_frames(args.images_dir, args.max_frames)
        source_label = f"images={args.images_dir}"
    else:
        frames = _make_synthetic_frames(args.max_frames)
        source_label = "synthetic"

    if not frames:
        raise RuntimeError("No frames were loaded.")

    predictor_kwargs: Dict[str, Any] | None = None
    if args.device:
        predictor_kwargs = {"device": args.device}

    create_req: Dict[str, Any] = {
        "weights_path": str(args.weights),
        "imgsz": int(args.imgsz),
        "conf": float(args.conf),
        "half": bool(args.half),
        "verbose": bool(args.verbose),
    }
    if predictor_kwargs is not None:
        create_req["predictor_kwargs"] = predictor_kwargs

    print(f"[test] creating session with {source_label}, frames={len(frames)}")
    print(f"[test] request={json.dumps(create_req, sort_keys=True)}")
    created = sam3.create_session(create_req)
    print(f"[test] create_session -> {created}")

    all_frame_indices = [args.start_index + i for i in range(len(frames))]
    by_frame_boxes: Dict[int, List[List[float]]] = {}
    for spec in args.box:
        fi, box = _parse_box_spec(spec)
        by_frame_boxes.setdefault(fi, []).append(box)

    total_objects = 0
    summary_rows: List[Dict[str, Any]] = []

    output_dir: Path | None = args.output_dir
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    preview_video_writer = None
    preview_window_name = "SAM3 Preview"
    preview_enabled = bool(args.preview or args.preview_video is not None)
    processed_frame_indices: List[int] = []

    if args.preview_video is not None:
        import cv2

        args.preview_video.parent.mkdir(parents=True, exist_ok=True)
        h, w = int(frames[0].shape[0]), int(frames[0].shape[1])
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        preview_video_writer = cv2.VideoWriter(str(args.preview_video), fourcc, float(args.preview_fps), (w, h))
        if not preview_video_writer.isOpened():
            raise RuntimeError(f"Could not open preview video writer: {args.preview_video}")

    if args.preview:
        import cv2

        cv2.namedWindow(preview_window_name, cv2.WINDOW_NORMAL)

    try:
        if args.text is not None:
            text_req = {
                "type": "text",
                "text": args.text,
                "frame_index": int(args.text_frame_index),
                "text_session_scope": bool(args.text_session_scope),
                "text_skip_if_unchanged": True,
            }
            print(f"[test] add_prompt(text) -> {sam3.add_prompt(text_req)}")

        removed_tested = False
        user_stopped_preview = False
        for frame_index, frame in zip(all_frame_indices, frames):
            if frame_index in by_frame_boxes:
                req = {"type": "boxes", "frame_index": int(frame_index), "boxes": by_frame_boxes[frame_index]}
                print(f"[test] add_prompt(boxes@{frame_index}) -> {sam3.add_prompt(req)}")

            # Mirror ModuleProxy::set_variable("sam3_frame", frame) + run("set_frame", ...).
            sam3.sam3_frame = frame
            set_meta = sam3.set_frame({"frame_index": int(frame_index)})
            out = sam3.get_frame({"frame_index": int(frame_index)})

            num_objects = int(out.get("num_objects", 0))
            total_objects += num_objects
            first_areas = [len(o.get("foreground_indices", [])) for o in out.get("objects", [])[:3]]
            row = {"frame_index": frame_index, "num_objects": num_objects, "first_obj_areas": first_areas}
            summary_rows.append(row)
            processed_frame_indices.append(int(frame_index))
            print(f"[test] frame={frame_index} ingest={set_meta['width']}x{set_meta['height']} objects={num_objects}")

            if output_dir is not None:
                saved_path = _save_masks_npz(out, output_dir)
                print(f"[test] saved {saved_path}")

            if preview_enabled:
                preview_frame = _render_preview_frame(frame, out, args.preview_alpha)
                if preview_video_writer is not None:
                    preview_video_writer.write(preview_frame)
                if args.preview:
                    import cv2

                    cv2.imshow(preview_window_name, preview_frame)
                    key = cv2.waitKey(max(1, int(args.preview_delay_ms))) & 0xFF
                    if key in (27, ord("q")):
                        print("[test] preview stopped by user input (q/esc).")
                        user_stopped_preview = True
                        break

            if args.remove_object is not None and not removed_tested and num_objects > 0:
                print(f"[test] remove_object({args.remove_object}) -> {sam3.remove_object({'obj_id': int(args.remove_object)})}")
                out_after = sam3.get_frame({"frame_index": int(frame_index)})
                print(f"[test] frame={frame_index} after remove -> objects={out_after.get('num_objects', 0)}")
                removed_tested = True

        if user_stopped_preview:
            print("[test] ending loop early due to preview stop.")

        # Cache policy may evict older frame entries; do not require get_frames()
        # to succeed for the full historical frame list.
        try:
            batch = sam3.get_frames({"frame_indices": [int(x) for x in processed_frame_indices]})
            print(f"[test] get_frames -> {len(batch['frames'])} frames")
        except Exception as exc:
            print(f"[test] get_frames skipped (likely cache eviction): {exc}")

        if args.output_json is not None:
            payload = {
                "session": created,
                "summary": summary_rows,
                "total_objects": total_objects,
                "frame_indices": processed_frame_indices,
            }
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(payload, indent=2))
            print(f"[test] wrote summary json: {args.output_json}")

        if args.fail_if_empty and total_objects == 0:
            print("[test] FAIL: no objects were produced.")
            return 2

        print(f"[test] PASS: total_objects={total_objects}")
        return 0
    finally:
        if preview_video_writer is not None:
            preview_video_writer.release()
            print(f"[test] saved preview video: {args.preview_video}")
        if args.preview:
            import cv2

            cv2.destroyAllWindows()
        print(f"[test] reset_session -> {sam3.reset_session({'clear_prompts': True})}")
        print(f"[test] close_session -> {sam3.close_session()}")
        print(f"[test] shutdown -> {sam3.shutdown()}")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Standalone test harness for trex_sam3_interface.py")
    p.add_argument("--weights", type=Path, required=True, help="Path to SAM3 weights.")
    p.add_argument("--video", type=Path, default=None, help="Optional video path for frame input.")
    p.add_argument("--images-dir", type=Path, default=None, help="Optional image directory for frame input.")
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic frames (default if --video/--images-dir are not set).",
    )
    p.add_argument("--max-frames", type=int, default=6, help="Maximum frames to process.")
    p.add_argument("--start-index", type=int, default=0, help="Frame index of first input frame.")

    p.add_argument("--imgsz", type=int, default=640, help="Predictor input size.")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    p.add_argument("--half", dest="half", action="store_true", default=True, help="Request fp16 when supported.")
    p.add_argument("--no-half", dest="half", action="store_false", help="Disable fp16 request.")
    p.add_argument("--device", type=str, default=None, help="Optional device override (cpu/cuda/mps).")
    p.add_argument("--verbose", action="store_true", help="Enable verbose SAM3 backend logging.")

    p.add_argument("--text", type=str, default="fish", help="Text prompt. Use empty string to disable.")
    p.add_argument(
        "--text-session-scope",
        action="store_true",
        default=True,
        help="Apply text prompt as a session-global prompt.",
    )
    p.add_argument(
        "--no-text-session-scope",
        dest="text_session_scope",
        action="store_false",
        help="Restrict text prompt to --text-frame-index only.",
    )
    p.add_argument(
        "--text-frame-index",
        type=int,
        default=0,
        help="Frame index for non-session-scope text prompt.",
    )
    p.add_argument(
        "--box",
        action="append",
        default=[],
        help="Box prompt in format frame_index:x1,y1,x2,y2 (repeatable).",
    )
    p.add_argument(
        "--remove-object",
        type=int,
        default=None,
        help="Optional object id to remove after first non-empty frame output.",
    )

    p.add_argument("--output-dir", type=Path, default=None, help="Optional directory for per-frame masks (.npz).")
    p.add_argument("--output-json", type=Path, default=None, help="Optional summary json output path.")
    p.add_argument("--preview", action="store_true", help="Show live OpenCV preview with masks/boxes.")
    p.add_argument(
        "--preview-delay-ms",
        type=int,
        default=1,
        help="Delay per preview frame in milliseconds (q/esc to stop).",
    )
    p.add_argument(
        "--preview-alpha",
        type=float,
        default=0.45,
        help="Mask overlay alpha in [0,1] for preview rendering.",
    )
    p.add_argument(
        "--preview-video",
        type=Path,
        default=None,
        help="Optional output .mp4 path for annotated preview frames.",
    )
    p.add_argument(
        "--preview-fps",
        type=float,
        default=10.0,
        help="FPS for --preview-video output.",
    )
    p.add_argument("--fail-if-empty", action="store_true", help="Exit nonzero if total detected objects == 0.")
    return p


def main() -> int:
    args = _parser().parse_args()
    if args.text == "":
        args.text = None
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
