"""YOLOv8 detector with automatic backend selection (CUDA/DirectML/CPU)."""

from __future__ import annotations

import logging
import time

from ultralytics import YOLO

from vlm.common.datatypes import BoundingBox, CapturedFrame, DetectionResult
from vlm.common.device import DeviceInfo, DeviceType
from vlm.detection.base import BaseDetector

logger = logging.getLogger(__name__)


class YOLODetector(BaseDetector):
    """YOLOv8 object detector.

    Args:
        model_name: Model name (e.g. "yolov8n", "yolov8m").
        device_info: Device configuration from detect_device().
        conf_threshold: Minimum confidence for detections.
        nms_threshold: NMS IoU threshold.
        input_size: Input image size for the model.
        tier: "small" or "mid" label for this detector.
    """

    def __init__(
        self,
        model_name: str,
        device_info: DeviceInfo,
        conf_threshold: float = 0.35,
        nms_threshold: float = 0.45,
        input_size: int = 640,
        tier: str = "small",
    ):
        self._tier = tier
        self._conf = conf_threshold
        self._nms = nms_threshold
        self._imgsz = input_size

        # Determine device string for ultralytics
        if device_info.device_type == DeviceType.CUDA:
            self._device = "cuda:0"
        else:
            self._device = "cpu"

        logger.info(
            "Loading %s model=%s device=%s", tier, model_name, self._device
        )
        self._model = YOLO(f"{model_name}.pt")
        self._model.to(self._device)

        # Warmup with dummy inference
        self._warmup()

    def detect(self, frame: CapturedFrame) -> DetectionResult:
        t0 = time.perf_counter()
        results = self._model.predict(
            frame.image,
            conf=self._conf,
            iou=self._nms,
            imgsz=self._imgsz,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        boxes = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0].item())
                cls_name = self._model.names.get(cls_id, f"class_{cls_id}")
                boxes.append(
                    BoundingBox(
                        x1=float(xyxy[0]),
                        y1=float(xyxy[1]),
                        x2=float(xyxy[2]),
                        y2=float(xyxy[3]),
                        confidence=float(box.conf[0].item()),
                        class_id=cls_id,
                        class_name=cls_name,
                    )
                )

        logger.debug(
            "frame=%d tier=%s detections=%d time=%.1fms",
            frame.metadata.frame_id,
            self._tier,
            len(boxes),
            elapsed_ms,
        )

        return DetectionResult(
            frame_id=frame.metadata.frame_id,
            boxes=boxes,
            model_tier=self._tier,
            inference_ms=elapsed_ms,
        )

    @property
    def model_tier(self) -> str:
        return self._tier

    def _warmup(self) -> None:
        """Run a dummy inference to avoid cold-start latency."""
        import numpy as np

        dummy = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
        self._model.predict(dummy, conf=0.5, verbose=False)
        logger.info("%s model warmup complete", self._tier)
