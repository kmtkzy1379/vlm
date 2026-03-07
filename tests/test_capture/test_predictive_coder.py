"""Tests for predictive coding (brain V2-inspired change region detection)."""

import numpy as np
import pytest

from vlm.capture.predictive_coder import ChangedRegion, PredictiveCoder


class TestPredictiveCoder:
    def test_first_frame_returns_empty(self):
        pc = PredictiveCoder()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        regions = pc.compute_change_regions(image)
        assert regions == []

    def test_identical_frames_return_empty(self):
        pc = PredictiveCoder()
        image = np.full((480, 640, 3), 128, dtype=np.uint8)
        pc.compute_change_regions(image)  # first frame
        regions = pc.compute_change_regions(image.copy())
        assert regions == []

    def test_large_change_detected(self):
        pc = PredictiveCoder(min_region_area=100)
        # Frame 1: black
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        pc.compute_change_regions(frame1)

        # Frame 2: white rectangle in center
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2[100:300, 200:400] = 255
        regions = pc.compute_change_regions(frame2)

        assert len(regions) >= 1
        # The changed region should overlap with the white rectangle
        r = regions[0]
        assert r.x <= 200 and r.y <= 100
        assert r.x + r.w >= 400 and r.y + r.h >= 300

    def test_small_change_filtered_by_min_area(self):
        pc = PredictiveCoder(min_region_area=10000)
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        pc.compute_change_regions(frame1)

        # Small 5x5 change (area=25, below min_area=10000)
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2[100:105, 100:105] = 255
        regions = pc.compute_change_regions(frame2)
        assert len(regions) == 0

    def test_multiple_change_regions(self):
        pc = PredictiveCoder(min_region_area=100, merge_distance=10)
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        pc.compute_change_regions(frame1)

        # Two separate white rectangles (far apart)
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2[10:60, 10:60] = 255     # Top-left
        frame2[400:450, 500:550] = 255  # Bottom-right
        regions = pc.compute_change_regions(frame2)

        assert len(regions) >= 2

    def test_regions_sorted_by_magnitude(self):
        pc = PredictiveCoder(min_region_area=100)
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        pc.compute_change_regions(frame1)

        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2[10:60, 10:60] = 100      # Moderate change
        frame2[400:450, 500:550] = 255   # Strong change
        regions = pc.compute_change_regions(frame2)

        if len(regions) >= 2:
            assert regions[0].change_magnitude >= regions[1].change_magnitude

    def test_change_mask_shape(self):
        pc = PredictiveCoder()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        mask = pc.compute_change_mask(image)
        assert mask.shape == (480, 640)
        assert mask.dtype == np.uint8

    def test_reset_clears_state(self):
        pc = PredictiveCoder()
        image = np.full((480, 640, 3), 128, dtype=np.uint8)
        pc.compute_change_regions(image)
        pc.reset()
        # After reset, next frame should return empty (like first frame)
        regions = pc.compute_change_regions(image)
        assert regions == []


class TestChangedRegion:
    def test_area(self):
        r = ChangedRegion(x=10, y=20, w=100, h=50, change_magnitude=128.0)
        assert r.area == 5000

    def test_bbox(self):
        r = ChangedRegion(x=10, y=20, w=100, h=50, change_magnitude=128.0)
        assert r.bbox == (10, 20, 110, 70)
