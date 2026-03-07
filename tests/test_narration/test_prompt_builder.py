"""Tests for prompt builder with scene graph and memory integration."""

from unittest.mock import MagicMock

import numpy as np

from vlm.narration.prompt_builder import PromptBuilder


def _make_builder():
    """Create PromptBuilder with a mock DeltaEncoder."""
    mock_encoder = MagicMock()
    mock_encoder.to_compact_text.return_value = (
        "SCENE: office | change=major\n"
        "ENTITIES(1 active, 1 new, 0 lost):\n"
        "+E0[person]: (100,100,200,300) standing neutral"
    )
    return PromptBuilder(delta_encoder=mock_encoder)


def _make_delta():
    """Create a mock FrameDelta."""
    return MagicMock()


class TestPromptBuilder:
    def test_basic_build_produces_system_and_user(self):
        pb = _make_builder()
        msgs = pb.build(_make_delta(), "")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_system_prompt_contains_relation_instructions(self):
        pb = _make_builder()
        msgs = pb.build(_make_delta(), "")
        system_text = msgs[0]["content"]
        assert "RELATIONS" in system_text
        assert "MEMORY" in system_text

    def test_relations_text_included_in_observation(self):
        pb = _make_builder()
        rel_text = "RELATIONS: E0 above E1 | E0 near E2"
        msgs = pb.build(_make_delta(), "", relations_text=rel_text)
        user_content = msgs[1]["content"]
        text_parts = [p["text"] for p in user_content if p.get("type") == "text"]
        combined = " ".join(text_parts)
        assert "RELATIONS: E0 above E1" in combined

    def test_memory_text_included_in_observation(self):
        pb = _make_builder()
        mem_text = "MEMORY:\n  f10: E0[person] disappeared"
        msgs = pb.build(_make_delta(), "", memory_text=mem_text)
        user_content = msgs[1]["content"]
        text_parts = [p["text"] for p in user_content if p.get("type") == "text"]
        combined = " ".join(text_parts)
        assert "MEMORY:" in combined
        assert "disappeared" in combined

    def test_both_relations_and_memory_included(self):
        pb = _make_builder()
        rel_text = "RELATIONS: E0 left_of E1"
        mem_text = "MEMORY:\n  f5: E2[car] appeared (new)"
        msgs = pb.build(_make_delta(), "", relations_text=rel_text, memory_text=mem_text)
        user_content = msgs[1]["content"]
        text_parts = [p["text"] for p in user_content if p.get("type") == "text"]
        combined = " ".join(text_parts)
        assert "RELATIONS:" in combined
        assert "MEMORY:" in combined

    def test_empty_relations_and_memory_not_added(self):
        pb = _make_builder()
        msgs = pb.build(_make_delta(), "", relations_text="", memory_text="")
        user_content = msgs[1]["content"]
        text_parts = [p["text"] for p in user_content if p.get("type") == "text"]
        combined = " ".join(text_parts)
        assert "RELATIONS:" not in combined
        assert "MEMORY:" not in combined

    def test_context_text_appears_as_separate_message(self):
        pb = _make_builder()
        msgs = pb.build(_make_delta(), "前回: E0が歩いていた")
        assert len(msgs) == 3  # system + context + current
        assert "前回" in msgs[1]["content"]

    def test_crop_image_included(self):
        pb = _make_builder()
        crop = np.zeros((50, 50, 3), dtype=np.uint8)
        msgs = pb.build(_make_delta(), "", key_crops=[(0, crop)])
        user_content = msgs[1]["content"]
        types = [p.get("type") for p in user_content]
        assert "image_url" in types

    def test_multiple_deltas_uses_temporal_text(self):
        mock_encoder = MagicMock()
        mock_encoder.to_temporal_text.return_value = (
            "TIMELINE (2 frames, f0→f1):\n"
            "[f0] change=major\n"
            "  +E0[person]: (100,100,200,300) standing neutral\n"
            "[f1] change=minor\n"
            "  ~E0[person]: move(+30,+0)"
        )
        pb = PromptBuilder(delta_encoder=mock_encoder)
        deltas = [_make_delta(), _make_delta()]
        msgs = pb.build(deltas, "")
        mock_encoder.to_temporal_text.assert_called_once_with(deltas)
        user_content = msgs[1]["content"]
        text_parts = [p["text"] for p in user_content if p.get("type") == "text"]
        combined = " ".join(text_parts)
        assert "TIMELINE" in combined

    def test_single_delta_uses_compact_text(self):
        mock_encoder = MagicMock()
        mock_encoder.to_compact_text.return_value = "SCENE: office | change=major"
        pb = PromptBuilder(delta_encoder=mock_encoder)
        single_delta = _make_delta()
        msgs = pb.build(single_delta, "")
        mock_encoder.to_compact_text.assert_called_once_with(single_delta)

    def test_system_prompt_contains_timeline_instructions(self):
        pb = _make_builder()
        msgs = pb.build(_make_delta(), "")
        system_text = msgs[0]["content"]
        assert "TIMELINE" in system_text

    def test_screenshot_included_as_first_image(self):
        pb = _make_builder()
        screenshot = np.zeros((1080, 1920, 3), dtype=np.uint8)
        msgs = pb.build(_make_delta(), "", screenshot=screenshot)
        user_content = msgs[1]["content"]
        # First part should be the screenshot image
        assert user_content[0]["type"] == "image_url"
        assert "base64" in user_content[0]["image_url"]["url"]
        # Second part should be the screenshot label
        assert user_content[1]["type"] == "text"
        assert "スクリーンショット" in user_content[1]["text"]

    def test_no_screenshot_when_none(self):
        pb = _make_builder()
        msgs = pb.build(_make_delta(), "", screenshot=None)
        user_content = msgs[1]["content"]
        # First part should be the observation text, not an image
        assert user_content[0]["type"] == "text"
        assert "現在の観測" in user_content[0]["text"]

    def test_system_prompt_contains_screenshot_instructions(self):
        pb = _make_builder()
        msgs = pb.build(_make_delta(), "")
        system_text = msgs[0]["content"]
        assert "スクリーンショット" in system_text
