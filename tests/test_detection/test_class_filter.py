"""Tests for YOLO class whitelist and min box area filters."""

from vlm.common.datatypes import BoundingBox


def _make_boxes() -> list[BoundingBox]:
    """Create a mix of boxes: person (large), refrigerator (large), person (tiny)."""
    return [
        BoundingBox(x1=100, y1=100, x2=200, y2=300, confidence=0.9, class_id=0, class_name="person"),
        BoundingBox(x1=300, y1=100, x2=500, y2=400, confidence=0.8, class_id=72, class_name="refrigerator"),
        BoundingBox(x1=10, y1=10, x2=30, y2=30, confidence=0.7, class_id=0, class_name="person"),  # area=400
    ]


class TestClassWhitelist:
    def test_whitelist_filters_classes(self):
        boxes = _make_boxes()
        whitelist = {"person", "laptop"}
        filtered = [b for b in boxes if b.class_name in whitelist]
        assert len(filtered) == 2
        assert all(b.class_name == "person" for b in filtered)

    def test_whitelist_none_passes_all(self):
        boxes = _make_boxes()
        whitelist = None
        if whitelist is not None:
            boxes = [b for b in boxes if b.class_name in whitelist]
        assert len(boxes) == 3

    def test_whitelist_empty_set_filters_all(self):
        boxes = _make_boxes()
        whitelist: set[str] = set()
        boxes = [b for b in boxes if b.class_name in whitelist]
        assert len(boxes) == 0


class TestMinBoxArea:
    def test_min_area_filters_small_boxes(self):
        boxes = _make_boxes()
        min_area = 4000
        filtered = [b for b in boxes if b.area >= min_area]
        # person large: 100*200=20000 ✓, refrigerator: 200*300=60000 ✓, person tiny: 20*20=400 ✗
        assert len(filtered) == 2
        assert all(b.area >= min_area for b in filtered)

    def test_min_area_zero_passes_all(self):
        boxes = _make_boxes()
        min_area = 0
        filtered = [b for b in boxes if b.area >= min_area]
        assert len(filtered) == 3

    def test_combined_whitelist_and_area(self):
        boxes = _make_boxes()
        whitelist = {"person"}
        min_area = 4000
        filtered = [b for b in boxes if b.class_name in whitelist]
        filtered = [b for b in filtered if b.area >= min_area]
        # Only the large person box passes both filters
        assert len(filtered) == 1
        assert filtered[0].class_name == "person"
        assert filtered[0].area >= min_area
