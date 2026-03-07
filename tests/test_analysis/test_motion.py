"""Tests for motion detection."""

from vlm.analysis.motion import MotionDetector
from vlm.common.datatypes import BoundingBox


def _bbox(x1, y1, x2, y2):
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=0.9, class_id=0, class_name="person")


class TestMotionDetector:
    def test_new_entity_returns_zero_velocity(self):
        md = MotionDetector()
        result = md.compute(0, _bbox(100, 100, 200, 300), None)
        assert result.velocity == (0.0, 0.0)
        assert result.action_label == "new"

    def test_stationary_entity(self):
        md = MotionDetector()
        result = md.compute(0, _bbox(100, 100, 200, 300), _bbox(100, 100, 200, 300))
        assert result.action_label == "stationary"
        assert result.displacement_since_last < 3.0

    def test_moving_entity(self):
        md = MotionDetector()
        result = md.compute(
            0, _bbox(150, 100, 250, 300), _bbox(100, 100, 200, 300)
        )
        assert result.velocity[0] == 50.0  # moved 50px right
        assert result.displacement_since_last > 0

    def test_walking_classification(self):
        md = MotionDetector()
        result = md.compute(
            0, _bbox(125, 100, 225, 300), _bbox(100, 100, 200, 300)
        )
        # 25px displacement = walking
        assert result.action_label == "walking"

    def test_running_classification(self):
        md = MotionDetector()
        result = md.compute(
            0, _bbox(200, 100, 300, 300), _bbox(100, 100, 200, 300)
        )
        # 100px displacement = running
        assert result.action_label == "running"
