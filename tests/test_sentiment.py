"""
tests/test_sentiment.py

Tests for the LLM sentiment parsing layer.

We test _parse_and_validate directly - no real API calls.
_call_llm is isolated precisely so we can do this.
"""

import pytest

from llm.sentiment import _parse_and_validate


class TestValidOutput:
    def test_parses_correct_json(self):
        raw = '{"score": 0.75, "reasoning": "Strong earnings beat expectations."}'
        score, reasoning = _parse_and_validate(raw)
        assert score == 0.75
        assert reasoning == "Strong earnings beat expectations."

    def test_score_at_boundaries(self):
        for boundary in [-1.0, 0.0, 1.0]:
            raw = f'{{"score": {boundary}, "reasoning": "Test."}}'
            score, _ = _parse_and_validate(raw)
            assert score == boundary

    def test_reasoning_is_stripped(self):
        raw = '{"score": 0.1, "reasoning": "  Slight positive tone.  "}'
        _, reasoning = _parse_and_validate(raw)
        assert reasoning == "Slight positive tone."


class TestInvalidJSON:
    def test_non_json_raises_value_error(self):
        with pytest.raises(ValueError, match="non-JSON"):
            _parse_and_validate("not json at all")

    def test_truncated_json_raises(self):
        with pytest.raises(ValueError):
            _parse_and_validate('{"score": 0.5')


class TestMissingFields:
    def test_missing_score_raises(self):
        raw = '{"reasoning": "Missing score field."}'
        with pytest.raises(ValueError, match="Missing required fields"):
            _parse_and_validate(raw)

    def test_missing_reasoning_raises(self):
        raw = '{"score": 0.5}'
        with pytest.raises(ValueError, match="Missing required fields"):
            _parse_and_validate(raw)

    def test_empty_object_raises(self):
        with pytest.raises(ValueError):
            _parse_and_validate("{}")


class TestInvalidScore:
    def test_score_above_range_raises(self):
        raw = '{"score": 1.5, "reasoning": "Too high."}'
        with pytest.raises(ValueError, match="out of range"):
            _parse_and_validate(raw)

    def test_score_below_range_raises(self):
        raw = '{"score": -1.5, "reasoning": "Too low."}'
        with pytest.raises(ValueError, match="out of range"):
            _parse_and_validate(raw)

    def test_score_as_string_range_raises(self):
        # LLM sometimes returns "0.4-0.6" despite prompt instructions
        raw = '{"score": "0.4-0.6", "reasoning": "Mixed signals."}'
        with pytest.raises(ValueError, match="not a valid float"):
            _parse_and_validate(raw)

    def test_score_as_null_raises(self):
        raw = '{"score": null, "reasoning": "Nothing."}'
        with pytest.raises(ValueError, match="not a valid float"):
            _parse_and_validate(raw)


class TestEmptyReasoning:
    def test_empty_string_raises(self):
        raw = '{"score": 0.5, "reasoning": ""}'
        with pytest.raises(ValueError, match="empty"):
            _parse_and_validate(raw)

    def test_whitespace_only_raises(self):
        raw = '{"score": 0.5, "reasoning": "   "}'
        with pytest.raises(ValueError, match="empty"):
            _parse_and_validate(raw)