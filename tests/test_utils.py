"""Tests for cogitator.utils module."""

import pytest

from cogitator.utils import count_steps, approx_token_length, exact_match, accuracy


class TestCountSteps:
    """Tests for count_steps function."""

    def test_numbered_steps_with_period(self):
        """Test counting steps with numbered format (1. 2. 3.)"""
        cot = """1. First step
2. Second step
3. Third step"""
        assert count_steps(cot) == 3

    def test_numbered_steps_with_paren(self):
        """Test counting steps with numbered format (1) 2) 3))"""
        cot = """1) First step
2) Second step
3) Third step"""
        assert count_steps(cot) == 3

    def test_bullet_points_dash(self):
        """Test counting steps with dash bullet points."""
        cot = """- First step
- Second step
- Third step"""
        assert count_steps(cot) == 3

    def test_bullet_points_asterisk(self):
        """Test counting steps with asterisk bullet points."""
        cot = """* First step
* Second step"""
        assert count_steps(cot) == 2

    def test_bullet_points_bullet(self):
        """Test counting steps with bullet character (•)."""
        cot = """• First step
• Second step"""
        assert count_steps(cot) == 2

    def test_mixed_no_steps(self):
        """Test text with no step markers."""
        cot = """This is just regular text.
No steps here.
Just explanation."""
        assert count_steps(cot) == 0

    def test_empty_string(self):
        """Test with empty string."""
        assert count_steps("") == 0

    def test_mixed_formats(self):
        """Test mixed number and bullet formats."""
        cot = """1. First step
- Also a step
2. Another numbered
* Asterisk step"""
        assert count_steps(cot) == 4

    def test_indented_steps(self):
        """Test that indented steps are still counted."""
        cot = """  1. Indented step one
  2. Indented step two"""
        assert count_steps(cot) == 2


class TestApproxTokenLength:
    """Tests for approx_token_length function."""

    def test_simple_sentence(self):
        """Test token counting for simple sentence."""
        text = "Hello world"
        assert approx_token_length(text) == 2

    def test_punctuation_counted(self):
        """Test that punctuation is counted as separate tokens."""
        text = "Hello, world!"
        # "Hello" + "," + "world" + "!" = 4 tokens
        assert approx_token_length(text) == 4

    def test_empty_string(self):
        """Test with empty string."""
        assert approx_token_length("") == 0

    def test_numbers_and_words(self):
        """Test with numbers mixed with words."""
        text = "I have 5 apples"
        assert approx_token_length(text) == 4

    def test_special_characters(self):
        """Test various special characters."""
        text = "test@example.com:8080"
        # "test" + "@" + "example" + "." + "com" + ":" + "8080"
        assert approx_token_length(text) == 7


class TestExactMatch:
    """Tests for exact_match function."""

    def test_identical_strings(self):
        """Test identical strings match."""
        assert exact_match("hello", "hello") is True

    def test_case_insensitive(self):
        """Test case insensitivity."""
        assert exact_match("Hello", "hello") is True
        assert exact_match("HELLO", "hello") is True

    def test_whitespace_handling(self):
        """Test whitespace is stripped."""
        assert exact_match("  hello  ", "hello") is True
        assert exact_match("hello", "  hello  ") is True

    def test_different_strings(self):
        """Test different strings don't match."""
        assert exact_match("hello", "world") is False

    def test_empty_strings(self):
        """Test empty strings match."""
        assert exact_match("", "") is True

    def test_numeric_strings(self):
        """Test numeric string matching."""
        assert exact_match("42", "42") is True
        assert exact_match("42", "43") is False


class TestAccuracy:
    """Tests for accuracy function."""

    def test_perfect_accuracy(self):
        """Test 100% accuracy."""
        preds = ["a", "b", "c"]
        golds = ["a", "b", "c"]
        assert accuracy(preds, golds) == 1.0

    def test_zero_accuracy(self):
        """Test 0% accuracy."""
        preds = ["x", "y", "z"]
        golds = ["a", "b", "c"]
        assert accuracy(preds, golds) == 0.0

    def test_partial_accuracy(self):
        """Test partial accuracy."""
        preds = ["a", "x", "c"]
        golds = ["a", "b", "c"]
        # 2/3 = 0.666...
        assert accuracy(preds, golds) == pytest.approx(2/3)

    def test_empty_golds(self):
        """Test with empty gold list."""
        preds = ["a", "b"]
        golds = []
        assert accuracy(preds, golds) == 0.0

    def test_empty_both(self):
        """Test with both empty."""
        assert accuracy([], []) == 0.0

    def test_case_insensitive_accuracy(self):
        """Test accuracy is case insensitive."""
        preds = ["A", "B", "C"]
        golds = ["a", "b", "c"]
        assert accuracy(preds, golds) == 1.0

    def test_mismatched_lengths(self):
        """Test with different length lists (uses shorter)."""
        preds = ["a", "b", "c", "d"]
        golds = ["a", "b", "c"]
        # Only compares first 3, all match, but accuracy based on len(golds)
        assert accuracy(preds, golds) == 1.0
