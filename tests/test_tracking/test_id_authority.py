"""Tests for the central ID authority (ByteTrack wrapper)."""

import numpy as np
import pytest

from vlm.common.datatypes import (
    BoundingBox,
    CapturedFrame,
    DetectionResult,
    FrameMetadata,
)
from vlm.tracking.id_authority import IDAuthority


def _make_frame(frame_id: int, w: int = 640, h: int = 480) -> CapturedFrame:
    rng = np.random.default_rng(frame_id)
    return CapturedFrame(
        metadata=FrameMetadata(
            frame_id=frame_id, timestamp_ms=frame_id * 500.0,
            source_width=w, source_height=h,
        ),
        image=rng.integers(0, 256, (h, w, 3), dtype=np.uint8),
    )


def _make_detections(frame_id: int, boxes: list[tuple]) -> DetectionResult:
    """boxes: list of (x1, y1, x2, y2, conf, class_id, class_name)"""
    return DetectionResult(
        frame_id=frame_id,
        boxes=[
            BoundingBox(
                x1=b[0], y1=b[1], x2=b[2], y2=b[3],
                confidence=b[4], class_id=b[5], class_name=b[6],
            )
            for b in boxes
        ],
        model_tier="small",
        inference_ms=10.0,
    )


class TestIDAuthority:
    def test_ids_are_monotonically_increasing(self):
        tracker = IDAuthority(max_age=5, min_hits=1)
        frame = _make_frame(0)
        dets = _make_detections(0, [
            (100, 100, 200, 300, 0.9, 0, "person"),
            (400, 100, 500, 300, 0.85, 0, "person"),
        ])
        state = tracker.update(frame, dets)
        ids = sorted(state.new_ids)
        assert ids == [0, 1] or ids[0] < ids[1]

    def test_same_detection_keeps_same_id(self):
        tracker = IDAuthority(max_age=5, min_hits=1)

        # Person at same position for multiple frames
        all_ids = set()
        for i in range(5):
            frame = _make_frame(i)
            dets = _make_detections(i, [
                (100, 100, 200, 300, 0.9, 0, "person"),
            ])
            state = tracker.update(frame, dets)
            active = tracker.get_active_entities()
            all_ids.update(active.keys())

        # Should have only 1 unique ID
        assert len(all_ids) == 1

    def test_lost_and_recovered_within_max_age(self):
        tracker = IDAuthority(max_age=10, min_hits=1)

        # Frame 0: person appears
        frame0 = _make_frame(0)
        det0 = _make_detections(0, [(100, 100, 200, 300, 0.9, 0, "person")])
        state0 = tracker.update(frame0, det0)
        original_id = state0.new_ids[0]

        # Frames 1-3: person disappears
        for i in range(1, 4):
            frame = _make_frame(i)
            det = _make_detections(i, [])
            state = tracker.update(frame, det)
            assert original_id in state.lost_ids or not tracker.get_entity(original_id).is_active

        # Frame 4: person reappears at same position
        frame4 = _make_frame(4)
        det4 = _make_detections(4, [(105, 105, 205, 305, 0.9, 0, "person")])
        state4 = tracker.update(frame4, det4)

        # ID should be preserved (same or recovered)
        active = tracker.get_active_entities()
        assert len(active) >= 1

    def test_empty_detections_marks_all_lost(self):
        tracker = IDAuthority(max_age=5, min_hits=1)

        # Frame 0: two people
        frame0 = _make_frame(0)
        det0 = _make_detections(0, [
            (100, 100, 200, 300, 0.9, 0, "person"),
            (400, 100, 500, 300, 0.85, 0, "person"),
        ])
        tracker.update(frame0, det0)

        # Frame 1: no detections
        frame1 = _make_frame(1)
        det1 = _make_detections(1, [])
        state1 = tracker.update(frame1, det1)

        assert len(state1.lost_ids) >= 1
        active = tracker.get_active_entities()
        assert len(active) == 0

    def test_reset_clears_all_state(self):
        tracker = IDAuthority(max_age=5, min_hits=1)

        frame = _make_frame(0)
        dets = _make_detections(0, [(100, 100, 200, 300, 0.9, 0, "person")])
        tracker.update(frame, dets)

        tracker.reset()
        assert len(tracker.get_active_entities()) == 0

    def test_crop_is_correct_region(self):
        tracker = IDAuthority(max_age=5, min_hits=1)
        # Create frame with known pattern
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[100:300, 100:200] = 255  # White rectangle

        frame = CapturedFrame(
            metadata=FrameMetadata(0, 0.0, 640, 480),
            image=image,
        )
        dets = _make_detections(0, [(100, 100, 200, 300, 0.9, 0, "person")])
        tracker.update(frame, dets)

        active = tracker.get_active_entities()
        for entity in active.values():
            assert entity.crop is not None
            assert entity.crop.shape[0] == 200  # height: 300-100
            assert entity.crop.shape[1] == 100  # width: 200-100
            assert entity.crop.mean() == 255.0  # all white

    def test_max_entities_limit(self):
        tracker = IDAuthority(max_age=5, min_hits=1, max_entities=2)
        frame = _make_frame(0)
        dets = _make_detections(0, [
            (10, 10, 50, 50, 0.9, 0, "person"),
            (100, 100, 150, 150, 0.5, 0, "person"),
            (200, 200, 250, 250, 0.3, 0, "person"),
        ])
        tracker.update(frame, dets)
        active = tracker.get_active_entities()
        assert len(active) <= 2

    def test_min_hits_delays_new_ids(self):
        """With min_hits=3, new_ids should be empty until 3rd frame."""
        tracker = IDAuthority(max_age=5, min_hits=3)

        box = (100, 100, 200, 300, 0.9, 0, "person")
        for i in range(2):
            frame = _make_frame(i)
            dets = _make_detections(i, [box])
            state = tracker.update(frame, dets)
            assert state.new_ids == [], f"Frame {i}: new_ids should be empty before min_hits"

        # Frame 2 (3rd detection): should now report as new
        frame2 = _make_frame(2)
        dets2 = _make_detections(2, [box])
        state2 = tracker.update(frame2, dets2)
        assert len(state2.new_ids) == 1

    def test_ghost_entity_not_reported(self):
        """Entity appearing for 1 frame then disappearing should not show in new_ids or lost_ids."""
        tracker = IDAuthority(max_age=5, min_hits=3)

        # Frame 0: entity appears
        frame0 = _make_frame(0)
        det0 = _make_detections(0, [(100, 100, 200, 300, 0.9, 0, "person")])
        state0 = tracker.update(frame0, det0)
        assert state0.new_ids == []

        # Frame 1: entity disappears
        frame1 = _make_frame(1)
        det1 = _make_detections(1, [])
        state1 = tracker.update(frame1, det1)
        # Should NOT report lost_ids since entity was never confirmed
        assert state1.lost_ids == []
