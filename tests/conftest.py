"""Shared test fixtures."""

import numpy as np
import pytest

from vlm.common.datatypes import (
    BoundingBox,
    CapturedFrame,
    DetectionResult,
    FrameMetadata,
)


@pytest.fixture
def blank_frame() -> CapturedFrame:
    return CapturedFrame(
        metadata=FrameMetadata(
            frame_id=0, timestamp_ms=0.0, source_width=640, source_height=480
        ),
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        phash=0,
    )


@pytest.fixture
def noise_frame() -> CapturedFrame:
    rng = np.random.default_rng(42)
    return CapturedFrame(
        metadata=FrameMetadata(
            frame_id=1, timestamp_ms=500.0, source_width=640, source_height=480
        ),
        image=rng.integers(0, 256, (480, 640, 3), dtype=np.uint8),
        phash=0xFFFFFFFFFFFFFFFF,
    )


@pytest.fixture
def sample_detections() -> DetectionResult:
    return DetectionResult(
        frame_id=0,
        boxes=[
            BoundingBox(
                x1=100, y1=100, x2=200, y2=300,
                confidence=0.9, class_id=0, class_name="person",
            ),
            BoundingBox(
                x1=400, y1=150, x2=500, y2=350,
                confidence=0.85, class_id=0, class_name="person",
            ),
        ],
        model_tier="small",
        inference_ms=50.0,
    )
