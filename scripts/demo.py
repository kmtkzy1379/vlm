"""Interactive demo - capture screen, detect changes, track objects.

Runs the pipeline without LLM narration to test capture/detection/tracking.

Usage:
    python scripts/demo.py [--frames N] [--no-detect]
"""

from __future__ import annotations

import argparse
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo")


def run_capture_only(num_frames: int = 20):
    """Test capture + change detection only."""
    from vlm.capture.change_detector import ChangeDetector
    from vlm.capture.screen import ScreenCapture

    capture = ScreenCapture(monitor=1, target_fps=2.0)
    detector = ChangeDetector()

    logger.info("=== Capture + Change Detection Demo ===")
    logger.info("Capturing %d frames from primary monitor...", num_frames)

    for i, frame in enumerate(capture.stream()):
        if i >= num_frames:
            break

        level = detector.evaluate(frame)
        logger.info(
            "Frame %d: %dx%d phash=%016x change=%s",
            frame.metadata.frame_id,
            frame.metadata.source_width,
            frame.metadata.source_height,
            frame.phash,
            level.name,
        )

    logger.info("Done.")


def run_with_detection(num_frames: int = 20):
    """Test capture + change detection + YOLO + tracking."""
    from vlm.capture.change_detector import ChangeDetector
    from vlm.capture.screen import ScreenCapture
    from vlm.common.datatypes import ChangeLevel
    from vlm.common.device import detect_device
    from vlm.detection.yolo_detector import YOLODetector
    from vlm.tracking.id_authority import IDAuthority

    device = detect_device()
    capture = ScreenCapture(monitor=1, target_fps=2.0)
    change_det = ChangeDetector()
    yolo = YOLODetector("yolov8n", device, tier="small")
    tracker = IDAuthority()

    logger.info("=== Full Pipeline Demo (no LLM) ===")
    logger.info("Device: %s", device.device_name)
    logger.info("Capturing %d frames...", num_frames)

    for i, frame in enumerate(capture.stream()):
        if i >= num_frames:
            break

        level = change_det.evaluate(frame)
        if level == ChangeLevel.NONE:
            logger.info("Frame %d: NONE (skipped)", frame.metadata.frame_id)
            continue

        t0 = time.perf_counter()
        detections = yolo.detect(frame)
        state = tracker.update(frame, detections)
        elapsed = (time.perf_counter() - t0) * 1000

        active = [e for e in state.entities.values() if e.is_active]
        logger.info(
            "Frame %d: %s | %d detections | %d active tracks | %.0fms",
            frame.metadata.frame_id,
            level.name,
            len(detections.boxes),
            len(active),
            elapsed,
        )

        for entity in active:
            crop_shape = entity.crop.shape if entity.crop is not None else "N/A"
            logger.info(
                "  E%d [%s] conf=%.2f bbox=(%.0f,%.0f,%.0f,%.0f) crop=%s",
                entity.track_id,
                entity.class_name,
                entity.bbox.confidence,
                entity.bbox.x1, entity.bbox.y1,
                entity.bbox.x2, entity.bbox.y2,
                crop_shape,
            )

        if state.new_ids:
            logger.info("  NEW: %s", state.new_ids)
        if state.lost_ids:
            logger.info("  LOST: %s", state.lost_ids)

    logger.info("Done.")


def main():
    parser = argparse.ArgumentParser(description="VLM Demo")
    parser.add_argument("--frames", type=int, default=20, help="Number of frames")
    parser.add_argument("--no-detect", action="store_true", help="Capture only, no detection")
    args = parser.parse_args()

    if args.no_detect:
        run_capture_only(args.frames)
    else:
        run_with_detection(args.frames)


if __name__ == "__main__":
    main()
