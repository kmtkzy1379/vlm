"""Tests for narrative context manager."""

from vlm.narration.context_manager import ContextManager


class TestContextManager:
    def test_empty_context(self):
        cm = ContextManager()
        assert cm.get_context() == []
        assert cm.get_context_text() == ""

    def test_append_and_retrieve(self):
        cm = ContextManager(max_entries=3)
        cm.append("Event 1")
        cm.append("Event 2")
        assert cm.get_context() == ["Event 1", "Event 2"]

    def test_sliding_window(self):
        cm = ContextManager(max_entries=2)
        cm.append("A")
        cm.append("B")
        cm.append("C")
        assert cm.get_context() == ["B", "C"]

    def test_clear(self):
        cm = ContextManager()
        cm.append("Something")
        cm.clear()
        assert cm.get_context() == []

    def test_context_text_formatting(self):
        cm = ContextManager()
        cm.append("First")
        cm.append("Second")
        text = cm.get_context_text()
        assert "First" in text
        assert "Second" in text
        assert "---" in text
