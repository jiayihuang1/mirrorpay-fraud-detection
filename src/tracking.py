"""
Langfuse trace checker — query and analyze traces by session ID.

Run standalone to inspect a completed session's token usage, cost, and latency.

Usage:
    python -m src.tracking <session_id>
"""

import sys
from datetime import datetime
from collections import defaultdict

from src.utils import make_langfuse_client


def get_trace_info(session_id: str) -> dict | None:
    """Fetch and aggregate trace data for a Langfuse session.

    Args:
        session_id: The Langfuse session ID to query.

    Returns:
        Dict with keys: counts, costs, time, input, output.
        None if no traces found.
    """
    client = make_langfuse_client()
    traces = _fetch_all_traces(client, session_id)
    if not traces:
        return None

    observations = _collect_observations(client, traces)
    if not observations:
        return None

    return _aggregate_observations(observations)


def _fetch_all_traces(client, session_id: str) -> list:
    """Paginate through all traces for a session.

    Args:
        client: Langfuse client instance.
        session_id: Session ID to query.

    Returns:
        List of trace objects.
    """
    traces = []
    page = 1
    while True:
        response = client.api.trace.list(
            session_id=session_id, limit=100, page=page
        )
        if not response.data:
            break
        traces.extend(response.data)
        if len(response.data) < 100:
            break
        page += 1
    return traces


def _collect_observations(client, traces: list) -> list:
    """Gather all observations across traces.

    Args:
        client: Langfuse client instance.
        traces: List of trace objects.

    Returns:
        List of observation objects sorted by start time.
    """
    observations = []
    for trace in traces:
        detail = client.api.trace.get(trace.id)
        if detail and hasattr(detail, "observations"):
            observations.extend(detail.observations)

    return sorted(
        observations,
        key=lambda o: (
            o.start_time
            if hasattr(o, "start_time") and o.start_time
            else datetime.min
        ),
    )


def _aggregate_observations(observations: list) -> dict:
    """Compute summary stats from observations.

    Args:
        observations: Sorted list of observation objects.

    Returns:
        Dict with counts, costs, time, input preview, output preview.
    """
    counts: dict[str, int] = defaultdict(int)
    costs: dict[str, float] = defaultdict(float)
    total_time = 0.0

    for obs in observations:
        if not (hasattr(obs, "type") and obs.type == "GENERATION"):
            continue

        model = getattr(obs, "model", "unknown") or "unknown"
        counts[model] += 1

        cost = getattr(obs, "calculated_total_cost", None)
        if cost:
            costs[model] += cost

        start = getattr(obs, "start_time", None)
        end = getattr(obs, "end_time", None)
        if start and end:
            total_time += (end - start).total_seconds()

    first_input = ""
    if observations and hasattr(observations[0], "input"):
        inp = observations[0].input
        if inp:
            first_input = str(inp)[:100]

    last_output = ""
    if observations and hasattr(observations[-1], "output"):
        out = observations[-1].output
        if out:
            last_output = str(out)[:100]

    return {
        "counts": dict(counts),
        "costs": dict(costs),
        "time": total_time,
        "input": first_input,
        "output": last_output,
    }


def print_results(info: dict | None) -> None:
    """Pretty-print trace summary to stdout.

    Args:
        info: Aggregated trace info dict, or None if no traces found.
    """
    if not info:
        print("\nNo traces found for this session\n")
        return

    print("\nTrace Count by Model:")
    for model, count in info["counts"].items():
        print(f"  {model}: {count}")

    print("\nCost by Model:")
    total = 0.0
    for model, cost in info["costs"].items():
        print(f"  {model}: ${cost:.6f}")
        total += cost
    if total > 0:
        print(f"  Total: ${total:.6f}")

    print(f"\nTotal Time: {info['time']:.2f}s")

    if info["input"]:
        print(f"\nInitial Input:\n  {info['input']}")
    if info["output"]:
        print(f"\nFinal Output:\n  {info['output']}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.tracking <session_id>")
        sys.exit(1)

    session_id = sys.argv[1]
    print(f"\nQuerying session: {session_id}")

    try:
        info = get_trace_info(session_id)
        print_results(info)
    except Exception as e:
        print(f"\nError: {e}\n")
        sys.exit(1)
