"""Tests for scene graph builder (brain spatial cognition-inspired)."""

from vlm.aggregation.scene_graph import SceneGraphBuilder, SpatialRelation
from vlm.common.datatypes import BoundingBox, TrackedEntity


def _entity(eid, x1, y1, x2, y2, cls="person"):
    return TrackedEntity(
        track_id=eid,
        class_name=cls,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=0.9, class_id=0, class_name=cls),
        is_active=True,
    )


class TestSceneGraphBuilder:
    def test_above_below_relation(self):
        sg = SceneGraphBuilder()
        entities = {
            0: _entity(0, 100, 50, 200, 150),   # Top
            1: _entity(1, 100, 300, 200, 400),   # Bottom
        }
        rels = sg.build(entities)
        rel_types = {(r.subject_id, r.relation, r.object_id) for r in rels}
        assert (0, "above", 1) in rel_types

    def test_left_right_relation(self):
        sg = SceneGraphBuilder()
        entities = {
            0: _entity(0, 50, 100, 150, 200),    # Left
            1: _entity(1, 400, 100, 500, 200),    # Right
        }
        rels = sg.build(entities)
        rel_types = {(r.subject_id, r.relation, r.object_id) for r in rels}
        assert (0, "left_of", 1) in rel_types

    def test_near_relation(self):
        sg = SceneGraphBuilder(near_threshold=200)
        entities = {
            0: _entity(0, 100, 100, 150, 150),
            1: _entity(1, 200, 100, 250, 150),  # Close but not overlapping
        }
        rels = sg.build(entities)
        rel_types = {r.relation for r in rels}
        assert "near" in rel_types

    def test_containment_relation(self):
        sg = SceneGraphBuilder(containment_threshold=0.7)
        entities = {
            0: _entity(0, 120, 120, 180, 180),   # Small box inside
            1: _entity(1, 100, 100, 200, 200),    # Large box
        }
        rels = sg.build(entities)
        rel_types = {(r.subject_id, r.relation, r.object_id) for r in rels}
        assert (0, "inside", 1) in rel_types

    def test_no_relations_for_single_entity(self):
        sg = SceneGraphBuilder()
        entities = {0: _entity(0, 100, 100, 200, 200)}
        rels = sg.build(entities)
        assert rels == []

    def test_inactive_entities_excluded(self):
        sg = SceneGraphBuilder()
        e0 = _entity(0, 100, 50, 200, 150)
        e1 = _entity(1, 100, 300, 200, 400)
        e1.is_active = False
        entities = {0: e0, 1: e1}
        rels = sg.build(entities)
        assert rels == []

    def test_delta_detects_new_relation(self):
        sg = SceneGraphBuilder()
        # First: one entity
        entities1 = {0: _entity(0, 100, 50, 200, 150)}
        sg.build_delta(entities1)

        # Second: two entities
        entities2 = {
            0: _entity(0, 100, 50, 200, 150),
            1: _entity(1, 100, 300, 200, 400),
        }
        added, removed = sg.build_delta(entities2)
        assert len(added) > 0
        assert len(removed) == 0

    def test_delta_detects_removed_relation(self):
        sg = SceneGraphBuilder()
        # First: two entities
        entities1 = {
            0: _entity(0, 100, 50, 200, 150),
            1: _entity(1, 100, 300, 200, 400),
        }
        sg.build_delta(entities1)

        # Second: only one entity
        entities2 = {0: _entity(0, 100, 50, 200, 150)}
        added, removed = sg.build_delta(entities2)
        assert len(removed) > 0
        assert len(added) == 0

    def test_compact_text_format(self):
        sg = SceneGraphBuilder()
        rels = [
            SpatialRelation(0, "above", 1),
            SpatialRelation(0, "near", 2),
        ]
        text = sg.to_compact_text(rels)
        assert "RELATIONS:" in text
        assert "E0 above E1" in text
        assert "E0 near E2" in text

    def test_delta_text_format(self):
        added = [SpatialRelation(0, "above", 1)]
        removed = [SpatialRelation(2, "near", 3)]
        sg = SceneGraphBuilder()
        text = sg.to_delta_text(added, removed)
        assert "+E0 above E1" in text
        assert "-E2 near E3" in text

    def test_empty_relations_produce_empty_text(self):
        sg = SceneGraphBuilder()
        assert sg.to_compact_text([]) == ""
        assert sg.to_delta_text([], []) == ""
