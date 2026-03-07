"""Download required model weights."""

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def download_yolo_models():
    """Download YOLOv8 models (auto-downloads on first use)."""
    from ultralytics import YOLO

    for model_name in ["yolov8n", "yolov8m"]:
        logger.info("Downloading %s...", model_name)
        try:
            model = YOLO(f"{model_name}.pt")
            logger.info("  %s ready.", model_name)
        except Exception as e:
            logger.error("  Failed to download %s: %s", model_name, e)


def download_pose_model():
    """Download MediaPipe Pose Landmarker model."""
    from vlm.analysis.pose import _ensure_model

    for complexity, label in [(0, "lite"), (1, "full"), (2, "heavy")]:
        logger.info("Downloading pose_landmarker_%s...", label)
        try:
            path = _ensure_model(complexity)
            logger.info("  %s ready: %s", label, path)
        except Exception as e:
            logger.error("  Failed to download %s: %s", label, e)


def main():
    logger.info("=== VLM Model Downloader ===")
    download_yolo_models()
    download_pose_model()
    logger.info("Done.")


if __name__ == "__main__":
    main()
