"""Tests for change detection gate."""

import numpy as np
import pytest

from vlm.capture.change_detector import ChangeDetector
from vlm.common.datatypes import CapturedFrame, ChangeLevel, FrameMetadata


def _make_frame(frame_id: int, image: np.ndarray, phash: int = 0) -> CapturedFrame:
    return CapturedFrame(
        metadata=FrameMetadata(
            frame_id=frame_id,
            timestamp_ms=frame_id * 500.0,
            source_width=image.shape[1],
            source_height=image.shape[0],
        ),
        image=image,
        phash=phash,
    )


class TestChangeDetector:
    def test_first_frame_is_always_major(self):
        detector = ChangeDetector()
        frame = _make_frame(0, np.zeros((100, 100, 3), dtype=np.uint8))
        assert detector.evaluate(frame) == ChangeLevel.MAJOR

    def test_identical_frames_return_none(self):
        detector = ChangeDetector()
        image = np.full((100, 100, 3), 128, dtype=np.uint8)
        f1 = _make_frame(0, image, phash=0xAAAA)
        f2 = _make_frame(1, image.copy(), phash=0xAAAA)

        detector.evaluate(f1)  # first = MAJOR
        assert detector.evaluate(f2) == ChangeLevel.NONE

    def test_completely_different_frames_return_major(self):
        detector = ChangeDetector()
        black = np.zeros((100, 100, 3), dtype=np.uint8)
        white = np.full((100, 100, 3), 255, dtype=np.uint8)
        f1 = _make_frame(0, black, phash=0x0000000000000000)
        f2 = _make_frame(1, white, phash=0xFFFFFFFFFFFFFFFF)

        detector.evaluate(f1)
        assert detector.evaluate(f2) == ChangeLevel.MAJOR

    def test_periodic_check_triggers_moderate(self):
        detector = ChangeDetector(periodic_interval=3)
        image = np.full((100, 100, 3), 128, dtype=np.uint8)

        # First frame
        f0 = _make_frame(0, image, phash=0xAAAA)
        detector.evaluate(f0)

        # Next 2 idle frames
        for i in range(1, 3):
            fi = _make_frame(i, image.copy(), phash=0xAAAA)
            assert detector.evaluate(fi) == ChangeLevel.NONE

        # 3rd idle frame triggers periodic
        f3 = _make_frame(3, image.copy(), phash=0xAAAA)
        assert detector.evaluate(f3) == ChangeLevel.MODERATE

    def test_hamming_distance(self):
        assert ChangeDetector._hamming_distance(0, 0) == 0
        assert ChangeDetector._hamming_distance(0b1111, 0b0000) == 4
        assert ChangeDetector._hamming_distance(0xFF, 0x00) == 8

    def test_force_update_reference(self):
        detector = ChangeDetector()
        image = np.full((100, 100, 3), 128, dtype=np.uint8)
        f1 = _make_frame(0, image, phash=0xAAAA)
        detector.evaluate(f1)

        # Force update with same image
        f2 = _make_frame(1, image.copy(), phash=0xAAAA)
        detector.force_update_reference(f2)
        assert detector._idle_counter == 0
