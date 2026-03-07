"""Tests for delta encoder."""

from vlm.aggregation.delta_encoder import DeltaEncoder
from vlm.aggregation.feature_store import FeatureStore
from vlm.common.datatypes import (
    BoundingBox,
    ChangeLevel,
    EntityFeatures,
    ExpressionData,
    MotionData,
    SkeletonData,
    TrackingState,
    TrackedEntity,
)
import numpy as np


def _bbox(x1, y1, x2, y2):
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=0.9, class_id=0, class_name="person")


def _features(track_id, frame_id, bbox, pose="standing", emotion="neutral", action="stationary"):
    return EntityFeatures(
        track_id=track_id,
        frame_id=frame_id,
        bbox=bbox,
        skeleton=SkeletonData(track_id=track_id, keypoints=np.zeros((33, 3)), pose_label=pose),
        expression=ExpressionData(track_id=track_id, dominant_emotion=emotion),
        motion=MotionData(track_id=track_id, velocity=(0, 0), acceleration=(0, 0), action_label=action),
        attributes={"class": "person"},
    )


class TestDeltaEncoder:
    def test_new_entity_emits_all_fields(self):
        store = FeatureStore()
        encoder = DeltaEncoder(store)

        state = TrackingState(
            frame_id=0,
            entities={0: TrackedEntity(track_id=0, class_name="person", bbox=_bbox(100, 100, 200, 300))},
            new_ids=[0], lost_ids=[], recovered_ids=[],
        )
        feats = {0: _features(0, 0, _bbox(100, 100, 200, 300))}
        delta = encoder.encode(state, feats, ChangeLevel.MAJOR)

        assert len(delta.entity_deltas) == 1
        assert delta.entity_deltas[0].is_new
        assert "bbox" in delta.entity_deltas[0].changed_fields

    def test_unchanged_entity_emits_no_delta(self):
        store = FeatureStore()
        encoder = DeltaEncoder(store)

        bbox = _bbox(100, 100, 200, 300)

        # Frame 0: new entity
        state0 = TrackingState(0, {0: TrackedEntity(0, "person", bbox)}, [0], [], [])
        feats0 = {0: _features(0, 0, bbox)}
        encoder.encode(state0, feats0, ChangeLevel.MAJOR)

        # Frame 1: same entity, same features
        state1 = TrackingState(1, {0: TrackedEntity(0, "person", bbox)}, [], [], [])
        feats1 = {0: _features(0, 1, bbox)}
        delta = encoder.encode(state1, feats1, ChangeLevel.NONE)

        # No delta because nothing changed
        updated = [d for d in delta.entity_deltas if not d.is_new and not d.is_lost]
        assert len(updated) == 0

    def test_expression_change_emits_delta(self):
        store = FeatureStore()
        encoder = DeltaEncoder(store)

        bbox = _bbox(100, 100, 200, 300)

        # Frame 0
        state0 = TrackingState(0, {0: TrackedEntity(0, "person", bbox)}, [0], [], [])
        feats0 = {0: _features(0, 0, bbox, emotion="neutral")}
        encoder.encode(state0, feats0, ChangeLevel.MAJOR)

        # Frame 1: expression changed
        state1 = TrackingState(1, {0: TrackedEntity(0, "person", bbox)}, [], [], [])
        feats1 = {0: _features(0, 1, bbox, emotion="happy")}
        delta = encoder.encode(state1, feats1, ChangeLevel.NONE)

        updated = [d for d in delta.entity_deltas if not d.is_new and not d.is_lost]
        assert len(updated) == 1
        assert "expr" in updated[0].changed_fields

    def test_lost_entity_emits_lost_delta(self):
        store = FeatureStore()
        encoder = DeltaEncoder(store)

        bbox = _bbox(100, 100, 200, 300)

        # Frame 0: entity appears
        state0 = TrackingState(0, {0: TrackedEntity(0, "person", bbox)}, [0], [], [])
        feats0 = {0: _features(0, 0, bbox)}
        encoder.encode(state0, feats0, ChangeLevel.MAJOR)

        # Frame 1: entity lost
        state1 = TrackingState(1, {}, [], [0], [])
        delta = encoder.encode(state1, {}, ChangeLevel.NONE)

        assert len(delta.entity_deltas) == 1
        assert delta.entity_deltas[0].is_lost
        assert delta.entity_deltas[0].track_id == 0

    def test_compact_text_format(self):
        store = FeatureStore()
        encoder = DeltaEncoder(store)

        bbox = _bbox(100, 100, 200, 300)
        state = TrackingState(0, {0: TrackedEntity(0, "person", bbox)}, [0], [], [])
        feats = {0: _features(0, 0, bbox)}
        delta = encoder.encode(state, feats, ChangeLevel.MAJOR, scene_label="office")

        text = encoder.to_compact_text(delta)
        assert "SCENE: office" in text
        assert "+E0[person]:" in text
        assert "change=major" in text


class TestTemporalText:
    def test_multiple_frames_timeline_format(self):
        store = FeatureStore()
        encoder = DeltaEncoder(store)

        bbox0 = _bbox(100, 100, 200, 300)
        bbox1 = _bbox(130, 100, 230, 300)  # moved right by 30

        # Frame 0: new entity
        state0 = TrackingState(0, {0: TrackedEntity(0, "person", bbox0)}, [0], [], [])
        feats0 = {0: _features(0, 0, bbox0)}
        delta0 = encoder.encode(state0, feats0, ChangeLevel.MAJOR)

        # Frame 1: entity moved
        state1 = TrackingState(1, {0: TrackedEntity(0, "person", bbox1)}, [], [], [])
        feats1 = {0: _features(0, 1, bbox1)}
        delta1 = encoder.encode(state1, feats1, ChangeLevel.MINOR)

        text = encoder.to_temporal_text([delta0, delta1])
        assert "TIMELINE" in text
        assert "f0→f1" in text
        assert "[f0]" in text
        assert "[f1]" in text
        assert "+E0[person]:" in text

    def test_empty_deltas_skipped(self):
        store = FeatureStore()
        encoder = DeltaEncoder(store)

        bbox = _bbox(100, 100, 200, 300)

        # Frame 0: new entity
        state0 = TrackingState(0, {0: TrackedEntity(0, "person", bbox)}, [0], [], [])
        feats0 = {0: _features(0, 0, bbox)}
        delta0 = encoder.encode(state0, feats0, ChangeLevel.MAJOR)

        # Frame 1: no changes → empty entity_deltas
        state1 = TrackingState(1, {0: TrackedEntity(0, "person", bbox)}, [], [], [])
        feats1 = {0: _features(0, 1, bbox)}
        delta1 = encoder.encode(state1, feats1, ChangeLevel.MINOR)

        # Frame 2: expression changed
        state2 = TrackingState(2, {0: TrackedEntity(0, "person", bbox)}, [], [], [])
        feats2 = {0: _features(0, 2, bbox, emotion="happy")}
        delta2 = encoder.encode(state2, feats2, ChangeLevel.MINOR)

        text = encoder.to_temporal_text([delta0, delta1, delta2])
        # delta1 has no entity_deltas, should be skipped
        assert "[f1]" not in text
        assert "[f0]" in text
        assert "[f2]" in text

    def test_single_delta_falls_back_to_compact(self):
        store = FeatureStore()
        encoder = DeltaEncoder(store)

        bbox = _bbox(100, 100, 200, 300)
        state = TrackingState(0, {0: TrackedEntity(0, "person", bbox)}, [0], [], [])
        feats = {0: _features(0, 0, bbox)}
        delta = encoder.encode(state, feats, ChangeLevel.MAJOR, scene_label="office")

        text = encoder.to_temporal_text([delta])
        # Should use compact format (has SCENE: header)
        assert "SCENE: office" in text
        assert "TIMELINE" not in text

    def test_empty_list_returns_empty(self):
        store = FeatureStore()
        encoder = DeltaEncoder(store)
        assert encoder.to_temporal_text([]) == ""
