import pytest
import sys
import src.word_count as wc


def test_simple_text_count():
    """Tests the word len and word count of a simple text"""
    event = {"body": "Hello World"}
    expected = {
        "len": 11,
        "words": 2,
    }

    # Test if return is as expected
    assert wc.word_count_handler(event, None) == expected
