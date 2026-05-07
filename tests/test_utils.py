import pytest
from src.utils import validate_output, estimate_tokens, UUID_PATTERN

TXN_A = "e1021ab7-c2de-4791-994b-bab86e6fbe3e"
TXN_B = "8830a720-ff34-4dce-a578-e5b8006b2976"
TXN_C = "1c6db202-22d8-443f-86e7-fb1a8df05e84"


def test_validate_output_finds_present_ids():
    text = f"{TXN_A}: 1 | high | suspicious\n{TXN_B}: 0 | low | normal"
    found, missing = validate_output(text, {TXN_A, TXN_B})
    assert found == {TXN_A, TXN_B}
    assert missing == set()


def test_validate_output_detects_missing_ids():
    text = f"{TXN_A}: 1 | high | suspicious"
    found, missing = validate_output(text, {TXN_A, TXN_B})
    assert found == {TXN_A}
    assert missing == {TXN_B}


def test_validate_output_case_insensitive():
    text = f"{TXN_A.upper()}: 1 | high | reason"
    found, missing = validate_output(text, {TXN_A})
    assert found == {TXN_A}
    assert missing == set()


def test_estimate_tokens_rough_heuristic():
    text = "a" * 400
    assert estimate_tokens(text) == 100


def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0
