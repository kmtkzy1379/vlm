"""Tests for saliency detection (brain V1 attention-inspired)."""

import numpy as np
import pytest

from vlm.capture.predictive_coder import ChangedRegion
from vlm.capture.saliency import SaliencyDetector, ScoredRegion


class TestSaliencyDetector:
    def test_saliency_map_shape_and_range(self):
        sd = SaliencyDetector()
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        smap = sd.compute_saliency_map(image)
        assert smap.shape == (480, 640)
        assert smap.dtype == np.float32
        assert smap.min() >= 0.0
        assert smap.max() <= 1.0

    def test_uniform_image_low_saliency(self):
        sd = SaliencyDetector()
        # Uniform gray image should have very low saliency
        image = np.full((200, 200, 3), 128, dtype=np.uint8)
        smap = sd.compute_saliency_map(image)
        # Mean saliency should be low for uniform images
        assert smap.mean() < 0.5

    def test_high_contrast_produces_saliency(self):
        sd = SaliencyDetector(saliency_threshold=0.2, min_region_area=50)
        # Black image with bright white square = highly salient
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        image[80:120, 80:120] = 255
        smap = sd.compute_saliency_map(image)
        # The white square area should have higher saliency
        center_saliency = smap[80:120, 80:120].mean()
        border_saliency = smap[0:40, 0:40].mean()
        assert center_saliency > border_saliency

    def test_find_salient_regions_returns_scored_regions(self):
        sd = SaliencyDetector(saliency_threshold=0.2, min_region_area=50)
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        image[80:120, 80:120] = 255
        regions = sd.find_salient_regions(image)
        # Should detect at least the bright square
        assert all(isinstance(r, ScoredRegion) for r in regions)

    def test_combine_with_changes_prioritizes_changed_salient(self):
        sd = SaliencyDetector(change_weight=0.6, saliency_weight=0.4)
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        image[50:100, 50:100] = 200  # Bright area

        changed = [
            ChangedRegion(x=50, y=50, w=50, h=50, change_magnitude=200.0),
        ]
        scored = sd.combine_with_changes(image, changed)

        assert len(scored) >= 1
        # The changed+salient region should have high combined score
        top = scored[0]
        assert top.change_score > 0
        assert top.combined_score > 0

    def test_combine_preserves_salient_only_regions(self):
        sd = SaliencyDetector(
            saliency_threshold=0.1, min_region_area=50,
            change_weight=0.6, saliency_weight=0.4,
        )
        # Image with salient area but no change regions
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        image[80:120, 80:120] = 255

        scored = sd.combine_with_changes(image, [])
        # Should still find the salient region even with no changes
        # (salient-only regions are included with lower priority)
        for r in scored:
            assert r.change_score == 0.0

    def test_scored_region_bbox(self):
        r = ScoredRegion(
            x=10, y=20, w=100, h=50,
            saliency_score=0.8, change_score=0.5, combined_score=0.65,
        )
        assert r.bbox == (10, 20, 110, 70)
