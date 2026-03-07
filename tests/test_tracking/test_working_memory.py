"""Tests for working memory (brain hippocampus-inspired)."""

import numpy as np
import pytest

from vlm.common.datatypes import BoundingBox, TrackedEntity
from vlm.tracking.working_memory import WorkingMemory, ReIDMatch


def _bbox(x1, y1, x2, y2):
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=0.9, class_id=0, class_name="person")


def _entity(eid, x1, y1, x2, y2, cls="person", color=128):
    """Create entity with a solid-color crop for testing."""
    crop = np.full((y2 - y1, x2 - x1, 3), color, dtype=np.uint8)
    return TrackedEntity(
        track_id=eid,
        class_name=cls,
        bbox=_bbox(x1, y1, x2, y2),
        crop=crop,
        frames_alive=10,
        is_active=True,
    )


class TestWorkingMemory:
    def test_remember_lost_entity(self):
        wm = WorkingMemory()
        e = _entity(0, 100, 100, 200, 300)
        wm.on_entity_lost(e, frame_id=10)
        assert wm.remembered_count == 1

    def test_reid_matches_same_appearance(self):
        wm = WorkingMemory(reid_threshold=0.3)
        # Entity 0 lost with color=128
        e0 = _entity(0, 100, 100, 200, 300, color=128)
        wm.on_entity_lost(e0, frame_id=10)

        # New entity 1 with same color → should match
        e1 = _entity(1, 150, 120, 250, 320, color=128)
        match = wm.on_entity_new(e1, frame_id=20)

        assert match is not None
        assert isinstance(match, ReIDMatch)
        assert match.new_track_id == 1
        assert match.old_track_id == 0
        assert match.similarity > 0.3

    def test_no_reid_for_different_appearance(self):
        wm = WorkingMemory(reid_threshold=0.8)
        # Entity 0: blue-ish
        crop0 = np.zeros((100, 100, 3), dtype=np.uint8)
        crop0[:, :, 0] = 200  # Blue channel
        e0 = TrackedEntity(
            track_id=0, class_name="person", bbox=_bbox(100, 100, 200, 200),
            crop=crop0, frames_alive=10, is_active=True,
        )
        wm.on_entity_lost(e0, frame_id=10)

        # Entity 1: red-ish (very different)
        crop1 = np.zeros((100, 100, 3), dtype=np.uint8)
        crop1[:, :, 2] = 200  # Red channel
        e1 = TrackedEntity(
            track_id=1, class_name="person", bbox=_bbox(100, 100, 200, 200),
            crop=crop1, frames_alive=1, is_active=True,
        )
        match = wm.on_entity_new(e1, frame_id=20)
        # With high threshold, different colors should NOT match
        assert match is None

    def test_no_reid_across_different_classes(self):
        wm = WorkingMemory(reid_threshold=0.3)
        e0 = _entity(0, 100, 100, 200, 300, cls="person", color=128)
        wm.on_entity_lost(e0, frame_id=10)

        # New entity is a car, not a person → should not match
        e1 = _entity(1, 100, 100, 200, 300, cls="car", color=128)
        match = wm.on_entity_new(e1, frame_id=20)
        assert match is None

    def test_reid_mapping_stored(self):
        wm = WorkingMemory(reid_threshold=0.3)
        e0 = _entity(0, 100, 100, 200, 300, color=128)
        wm.on_entity_lost(e0, frame_id=10)

        e1 = _entity(1, 100, 100, 200, 300, color=128)
        wm.on_entity_new(e1, frame_id=20)

        assert wm.get_reid_mapping(1) == 0

    def test_episodic_memory_logs_events(self):
        wm = WorkingMemory()
        e = _entity(0, 100, 100, 200, 300)
        wm.on_entity_lost(e, frame_id=10)

        episodes = wm.get_recent_episodes(5)
        assert len(episodes) == 1
        assert episodes[0].event_type == "disappear"

    def test_episodic_memory_text_format(self):
        wm = WorkingMemory()
        e = _entity(0, 100, 100, 200, 300)
        wm.on_entity_lost(e, frame_id=10)

        text = wm.get_episodes_text()
        assert "MEMORY:" in text
        assert "E0" in text
        assert "disappeared" in text

    def test_reset_clears_remembered_keeps_episodes(self):
        wm = WorkingMemory()
        e = _entity(0, 100, 100, 200, 300)
        wm.on_entity_lost(e, frame_id=10)
        assert wm.remembered_count == 1

        wm.reset()
        assert wm.remembered_count == 0
        # Episodes should include the scene_cut event
        assert wm.episode_count >= 1

    def test_max_remembered_limit(self):
        wm = WorkingMemory(max_remembered=3)
        for i in range(5):
            e = _entity(i, 100, 100, 200, 300, color=100 + i * 20)
            wm.on_entity_lost(e, frame_id=i)
        assert wm.remembered_count <= 3

    def test_entity_without_crop_skipped(self):
        wm = WorkingMemory()
        e = TrackedEntity(
            track_id=0, class_name="person", bbox=_bbox(100, 100, 200, 300),
            crop=None, frames_alive=10, is_active=True,
        )
        wm.on_entity_lost(e, frame_id=10)
        assert wm.remembered_count == 0
