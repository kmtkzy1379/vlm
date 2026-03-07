"""Tests for optical flow motion detection (brain MT/V5-inspired)."""

import numpy as np
import pytest

from vlm.analysis.optical_flow import OpticalFlowMotion, FlowField
from vlm.common.datatypes import BoundingBox


def _bbox(x1, y1, x2, y2):
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=0.9, class_id=0, class_name="person")


class TestOpticalFlowMotion:
    def test_first_frame_returns_none_flow(self):
        ofm = OpticalFlowMotion()
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = ofm.update_frame(image)
        assert result is None
        assert not ofm.has_flow

    def test_second_frame_returns_flow_field(self):
        ofm = OpticalFlowMotion()
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2[40:60, 40:60] = 200  # Add a bright square
        ofm.update_frame(img1)
        result = ofm.update_frame(img2)
        assert result is not None
        assert isinstance(result, FlowField)
        assert result.flow.shape == (100, 100, 2)
        assert result.magnitude.shape == (100, 100)
        assert ofm.has_flow

    def test_stationary_scene_low_motion(self):
        ofm = OpticalFlowMotion()
        image = np.full((100, 100, 3), 128, dtype=np.uint8)
        ofm.update_frame(image)
        ofm.update_frame(image.copy())
        motion = ofm.compute_entity_motion(0, _bbox(10, 10, 90, 90))
        assert motion.action_label == "stationary"
        assert abs(motion.velocity[0]) < 2.0
        assert abs(motion.velocity[1]) < 2.0

    def test_moving_object_detected(self):
        ofm = OpticalFlowMotion()
        # Frame 1: white square at position A
        img1 = np.zeros((200, 200, 3), dtype=np.uint8)
        img1[50:80, 50:80] = 255
        # Frame 2: white square shifted right by 20px
        img2 = np.zeros((200, 200, 3), dtype=np.uint8)
        img2[50:80, 70:100] = 255

        ofm.update_frame(img1)
        ofm.update_frame(img2)

        motion = ofm.compute_entity_motion(0, _bbox(40, 40, 110, 90))
        # Should detect rightward motion
        assert motion.velocity[0] > 0  # Moving right
        assert motion.displacement_since_last > 0

    def test_entity_with_no_prior_flow_returns_new(self):
        ofm = OpticalFlowMotion()
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        ofm.update_frame(image)
        # No second frame → no flow
        motion = ofm.compute_entity_motion(0, _bbox(10, 10, 50, 50))
        assert motion.action_label == "new"

    def test_reset_clears_state(self):
        ofm = OpticalFlowMotion()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        ofm.update_frame(img)
        ofm.update_frame(img.copy())
        assert ofm.has_flow
        ofm.reset()
        assert not ofm.has_flow

    def test_out_of_bounds_bbox_handled(self):
        ofm = OpticalFlowMotion()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        ofm.update_frame(img)
        ofm.update_frame(img.copy())
        # bbox extends beyond image bounds
        motion = ofm.compute_entity_motion(0, _bbox(-10, -10, 200, 200))
        assert motion.track_id == 0
        assert motion.action_label == "stationary"
