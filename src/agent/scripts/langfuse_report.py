#!/usr/bin/env python
"""
Summarise the agent's latency, per-step time, tokens and cost from Langfuse.

    python scripts/langfuse_report.py --since 30m            # everything in the last 30 minutes
    python scripts/langfuse_report.py --since 1d --session <thread_id>   # one conversation

Reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL from the
environment or .env. Uses Langfuse's Metrics API (percentiles are computed
by Langfuse) and, with --session, the Observations API for a per-turn table.

For a session to be findable the agent must tag its traces with the
conversation: chat.py sets metadata {"langfuse_session_id": thread_id}.
Traces are exported in the background, so wait ~10 seconds after sending
traffic before running this.
"""
import argparse
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

def parse_since(text: str) -> timedelta:
    m = re.fullmatch(r"(\d+)([mhd])", text.strip())
    if not m:
        raise SystemExit("--since must look like 30m, 2h or 1d")
    n, unit = int(m.group(1)), m.group(2)
    return timedelta(minutes=n) if unit == "m" else timedelta(hours=n) if unit == "h" else timedelta(days=n)

def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")

def make_client():
    from dotenv import load_dotenv
    load_dotenv()
    from langfuse import Langfuse
    try:
        public, secret = os.environ["LANGFUSE_PUBLIC_KEY"], os.environ["LANGFUSE_SECRET_KEY"]
    except KeyError as e:
        raise SystemExit(f"{e.args[0]} is not set (put it in .env or export it)")
    host = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST") or "https://us.cloud.langfuse.com"
    return Langfuse(public_key=public, secret_key=secret, host=host)

# ---- Metrics API ---------------------------------------------------------------

def metrics_query(client, *, metrics, dimensions=(), filters=(), start, end, order_by=None, row_limit=100,
                  debug=False) -> List[Dict[str, Any]]:
    query = {
        "view": "observations",
        "dimensions": [{"field": d} for d in dimensions],
        "metrics": [{"measure":m, "aggregation":a} for m, a in metrics],
        "filters": list(filters),
        "fromTimestamp": iso(start),
        "toTimestamp": iso(end),
        "config": {"row_limit": row_limit},
    }
    if order_by:
        query["orderBy"] = [{"field": order_by, "direction": "desc"}]
    rows = client.api.metrics.metrics(query=json.dumps(query)).data
    if debug:
        print("QUERY", json.dumps(query)); print("ROWS", json.dumps(rows[:3], default=str))
    return rows

def pick(row: Dict[str, Any], measure: str, agg: str) -> Optional[float]:
    for key in (f"{agg}_{measure}", f"{measure}_{agg}", measure):
        if key in row and row[key] is not None:
            try:
                return float(row[key])
            except (TypeError, ValueError):
                return None
    return None

def f_ms(v: Optional[float]) -> str:
    return "      -" if v is None else f"{v:>7.0f}"

def report_aggregate(client, start, end, base_filters, debug) -> None:
    # 1. End-to-end: the root observation of each trace is the whole request.
    root_filters = base_filters + [{"column": "isRootObservation", "operator": "=", "value": True, "type": "boolean"}]
    rows = metrics_query(client, start=start, end=end, filters=root_filters, debug=debug, metrics=[
        ("count", "count"), ("latency", "p50"), ("latency", "p95"), ("latency", "p99"),
        ("latency", "avg"), ("latency", "max")])
    if not rows or not pick(rows[0], "count", "count"):
        print("No traces found in this window. Check the keys, the time window (--since), and that the agent "
              "has Langfuse configured; wait ~10s after sending traffic.")
        return
    r = rows[0]
    n_requests = int(pick(r, "count", "count"))
    print(f"\nRequests: {n_requests}")
    print("End-to-end latency (ms)")
    print(f"  p50 {f_ms(pick(r,'latency','p50'))}   p95 {f_ms(pick(r,'latency','p95'))}   "
          f"p99 {f_ms(pick(r,'latency','p99'))}   avg {f_ms(pick(r,'latency','avg'))}   "
          f"max {f_ms(pick(r,'latency','max'))}")
    if n_requests < 100:
        print(f"  (only {n_requests} requests: p99 is effectively the slowest one)")

    # 2. Where the time goes: latency by step name (graph nodes, LLM calls, tools).
    rows = metrics_query(client, start=start, end=end, filters=base_filters, dimensions=["name"],
                         order_by="p95_latency", row_limit=30, debug=debug, metrics=[
                             ("count", "count"), ("latency", "p50"), ("latency", "p95"), ("latency", "avg")])
    print("\nPer step (ms). Steps are nested, so rows overlap and do not add up to the total.")
    print(f"  {'step':<34}{'calls':>7}{'p50':>9}{'p95':>9}{'avg':>9}")
    for row in rows:
        print(f"  {str(row.get('name'))[:33]:<34}{int(pick(row,'count','count') or 0):>7}"
              f"{f_ms(pick(row,'latency','p50')):>9}{f_ms(pick(row,'latency','p95')):>9}"
              f"{f_ms(pick(row,'latency','avg')):>9}")

    # 3. LLM calls: tokens and cost per model.
    gen_filters = base_filters + [{"column": "type", "operator": "=", "value": "GENERATION", "type": "string"}]
    rows = metrics_query(client, start=start, end=end, filters=gen_filters, dimensions=["providedModelName"],
                         debug=debug, metrics=[("count", "count"), ("inputTokens", "sum"), ("outputTokens", "sum"),
                                               ("totalCost", "sum"), ("latency", "p50"), ("latency", "p95")])
    print("\nLLM calls by model")
    total_cost = total_in = total_out = total_calls = 0.0
    for row in rows:
        calls = pick(row, "count", "count") or 0
        tin, tout = pick(row, "inputTokens", "sum") or 0, pick(row, "outputTokens", "sum") or 0
        cost = pick(row, "totalCost", "sum") or 0
        total_calls += calls; total_in += tin; total_out += tout; total_cost += cost
        print(f"  {str(row.get('providedModelName')):<28} calls {int(calls):>5}   in {int(tin):>9}   out {int(tout):>8}"
              f"   cost ${cost:.4f}   latency p50 {f_ms(pick(row,'latency','p50')).strip()} "
              f"p95 {f_ms(pick(row,'latency','p95')).strip()} ms")
    if n_requests:
        print("\nPer request")
        print(f"  LLM calls {total_calls / n_requests:.1f}   input tokens {total_in / n_requests:.0f}   "
              f"output tokens {total_out / n_requests:.0f}")
        print(f"  cost ${total_cost / n_requests:.5f}   per 1,000 requests ${total_cost / n_requests * 1000:.2f}")

# ---- Observations API: one conversation, turn by turn -------------------------------

def fetch_session_observations(client, session_id: str) -> List[Any]:
    out, cursor = [], None
    while True:
        resp = client.api.observations.get_many(
            session_id=session_id, fields="core,basic,time,model,usage", limit=1000, cursor=cursor)
        out.extend(resp.data)
        cursor = getattr(resp.meta, "cursor", None) if resp.meta else None
        if not cursor:
            return out

def _secs(o) -> float:
    return (o.end_time - o.start_time).total_seconds() if o.end_time else 0.0

def report_session(client, session_id: str) -> None:
    obs = fetch_session_observations(client, session_id)
    if not obs:
        print(f"\nNo observations found for session {session_id}.")
        return
    by_trace = defaultdict(list)
    for o in obs:
        by_trace[o.trace_id].append(o)

    print(f"\nSession {session_id}: {len(by_trace)} turn(s)")
    print(f"  {'started (UTC)':<20}{'total s':>9}{'llm s':>8}{'tools s':>9}{'LLM calls':>11}{'in tok':>9}{'out tok':>9}{'cost $':>10}")
    for trace_id, items in sorted(by_trace.items(), key=lambda kv: min(o.start_time for o in kv[1])):
        started = min(o.start_time for o in items)
        root = next((o for o in items if o.is_root_observation), None)
        total = _secs(root) if root else (max(o.end_time or o.start_time for o in items) - started).total_seconds()
        gens = [o for o in items if o.type == "GENERATION"]
        llm_s = sum(_secs(o) for o in gens)
        tools_s = sum(_secs(o) for o in items if o.name == "tools")
        tin = sum((o.usage_details or {}).get("input", 0) for o in gens)
        tout = sum((o.usage_details or {}).get("output", 0) for o in gens)
        cost = sum(o.total_cost or 0 for o in gens)
        print(f"  {started.strftime('%Y-%m-%d %H:%M:%S'):<20}{total:>9.2f}{llm_s:>8.2f}{tools_s:>9.2f}"
              f"{len(gens):>11}{tin:>9}{tout:>9}{cost:>10.5f}")

def main() -> None:
    ap = argparse.ArgumentParser(description="Latency, per-step time, tokens and cost from Langfuse")
    ap.add_argument("--since", default="1h", help="time window ending now, e.g. 30m, 2h, 1d (default 1h)")
    ap.add_argument("--session", default=None, help="only this conversation (the thread_id), plus a per-turn table")
    ap.add_argument("--environment", default=None, help="Langfuse environment name, if you use one")
    ap.add_argument("--debug", action="store_true", help="print the raw queries and first rows")
    args = ap.parse_args()

    end = datetime.now(timezone.utc)
    start = end - parse_since(args.since)
    filters: List[Dict[str, Any]] = []
    if args.session:
        filters.append({"column": "sessionId", "operator": "=", "value": args.session, "type": "string"})
    if args.environment:
        filters.append({"column": "environment", "operator": "=", "value": args.environment, "type": "string"})

    client = make_client()
    print(f"Langfuse report: last {args.since}" + (f", session {args.session}" if args.session else ""))
    report_aggregate(client, start, end, filters, args.debug)
    if args.session:
        report_session(client, args.session)


if __name__ == "__main__":
    main()

