from src.orchestrator import parse_classifications

TXN_A = "e1021ab7-c2de-4791-994b-bab86e6fbe3e"
TXN_B = "8830a720-ff34-4dce-a578-e5b8006b2976"


def test_parse_classifications_fraud():
    output = f"{TXN_A}: 1 | high | Suspicious transfer at 3am to new recipient"
    result = parse_classifications(output)
    assert result == {TXN_A: 1}


def test_parse_classifications_legitimate():
    output = f"{TXN_B}: 0 | low | Normal salary payment"
    result = parse_classifications(output)
    assert result == {TXN_B: 0}


def test_parse_classifications_mixed_batch():
    output = (
        f"{TXN_A}: 1 | high | Circular transfer detected\n"
        f"{TXN_B}: 0 | medium | Slightly unusual hour but known recipient"
    )
    result = parse_classifications(output)
    assert result == {TXN_A: 1, TXN_B: 0}


def test_parse_classifications_case_insensitive():
    output = f"{TXN_A.upper()}: 1 | high | reason"
    result = parse_classifications(output)
    assert result == {TXN_A: 1}


def test_parse_classifications_empty():
    assert parse_classifications("No output here") == {}
