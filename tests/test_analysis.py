import pandas as pd
import pytest
from src.analysis import (
    load_transactions,
    compute_user_baselines,
    compute_graph_signals,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {
            "transaction_id": "e1021ab7-c2de-4791-994b-bab86e6fbe3e",
            "sender_id": "USER_A",
            "recipient_id": "USER_B",
            "transaction_type": "transfer",
            "amount": "100.0",
            "location": "",
            "payment_method": "",
            "sender_iban": "IT1234",
            "recipient_iban": "DE5678",
            "balance_after": "900.0",
            "description": "Rent",
            "timestamp": "2087-01-01T10:00:00",
        },
        {
            "transaction_id": "8830a720-ff34-4dce-a578-e5b8006b2976",
            "sender_id": "USER_A",
            "recipient_id": "USER_B",
            "transaction_type": "transfer",
            "amount": "200.0",
            "location": "",
            "payment_method": "",
            "sender_iban": "IT1234",
            "recipient_iban": "DE5678",
            "balance_after": "700.0",
            "description": "Rent 2",
            "timestamp": "2087-01-02T11:00:00",
        },
        {
            "transaction_id": "1c6db202-22d8-443f-86e7-fb1a8df05e84",
            "sender_id": "USER_C",
            "recipient_id": "USER_B",
            "transaction_type": "transfer",
            "amount": "5000.0",
            "location": "",
            "payment_method": "",
            "sender_iban": "FR9999",
            "recipient_iban": "DE5678",
            "balance_after": "200.0",
            "description": "Suspicious",
            "timestamp": "2087-01-02T03:00:00",
        },
    ])


def test_compute_user_baselines_mean(sample_df):
    from src.analysis import _coerce_types
    df = _coerce_types(sample_df)
    baselines = compute_user_baselines(df)
    assert "USER_A" in baselines
    assert baselines["USER_A"]["mean_amount"] == pytest.approx(150.0)


def test_compute_user_baselines_known_recipients(sample_df):
    from src.analysis import _coerce_types
    df = _coerce_types(sample_df)
    baselines = compute_user_baselines(df)
    assert "USER_B" in baselines["USER_A"]["known_recipients"]


def test_compute_graph_signals_shared_iban(sample_df):
    from src.analysis import _coerce_types
    df = _coerce_types(sample_df)
    graph = compute_graph_signals(df)
    # DE5678 is recipient of both USER_A and USER_C — shared_iban_count = 2
    txn_c = "1c6db202-22d8-443f-86e7-fb1a8df05e84"
    assert graph[txn_c]["shared_iban_count"] == 2


def test_compute_graph_signals_circular(sample_df):
    from src.analysis import _coerce_types
    df = _coerce_types(sample_df)
    graph = compute_graph_signals(df)
    # No circular transfers in sample data
    for txn_id, sig in graph.items():
        assert sig["circular_transfer"] is False


def test_compute_statistical_signals_amount_zscore(sample_df):
    from src.analysis import _coerce_types, compute_user_baselines, compute_graph_signals, compute_statistical_signals
    df = _coerce_types(sample_df)
    baselines = compute_user_baselines(df)
    graph = compute_graph_signals(df)
    signals = compute_statistical_signals(df, baselines, graph, [])
    txn_c = "1c6db202-22d8-443f-86e7-fb1a8df05e84"
    # USER_C only has 1 transaction — std is 1.0 (default), z = (5000 - 5000) / 1 = 0
    assert signals[txn_c]["amount"] == pytest.approx(5000.0)
    assert "amount_zscore" in signals[txn_c]


def test_compute_statistical_signals_new_recipient(sample_df):
    from src.analysis import _coerce_types, compute_user_baselines, compute_graph_signals, compute_statistical_signals
    df = _coerce_types(sample_df)
    baselines = compute_user_baselines(df)
    graph = compute_graph_signals(df)
    signals = compute_statistical_signals(df, baselines, graph, [])
    # USER_A has transacted with USER_B before — not a new recipient on 2nd txn
    txn_b = "8830a720-ff34-4dce-a578-e5b8006b2976"
    assert signals[txn_b]["new_recipient"] is False


def test_format_txn_summary_contains_flags(sample_df):
    from src.analysis import (
        _coerce_types, compute_user_baselines, compute_graph_signals,
        compute_statistical_signals, format_txn_summary,
    )
    df = _coerce_types(sample_df)
    baselines = compute_user_baselines(df)
    graph = compute_graph_signals(df)
    signals = compute_statistical_signals(df, baselines, graph, [])
    txn_c = "1c6db202-22d8-443f-86e7-fb1a8df05e84"
    summary = format_txn_summary(txn_c, signals, baselines)
    assert "1c6db202" in summary  # UUID prefix shown
    assert "USER_C" in summary
    assert "SHARED_IBAN" in summary  # shared_iban_count == 2 triggers this flag
