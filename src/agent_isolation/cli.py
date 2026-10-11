"""Command line interface."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .anomalies import run_all
from .bench import (
    run_decision_suite,
    run_one,
    run_suite,
    think_time_sweep,
)
from .store import Isolation


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max([len(h)] + [len(r[i]) for r in rows]) for i, h in enumerate(headers)]
    out = [
        "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True)) + " |",
        "|" + "|".join("-" * (w + 2) for w in widths) + "|",
    ]
    out += [
        "| " + " | ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)) + " |" for r in rows
    ]
    return "\n".join(out)


def cmd_anomalies(args) -> int:
    """The four anomalies at every isolation level."""
    results = run_all()
    rows = [
        [
            r.name,
            r.isolation.value,
            "OCCURS" if r.anomaly_occurred else "prevented",
            r.detail,
        ]
        for r in results
    ]
    print("\nfour classical anomalies, each driven through an exact interleaving\n")
    print(_table(["anomaly", "isolation", "result", "what happened"], rows))
    occurred = sum(1 for r in results if r.anomaly_occurred)
    print(f"\n{occurred} of {len(results)} combinations allow the anomaly.")
    print(
        "snapshot stops agents overwriting each other; only serializable stops an agent "
        "acting on a premise that changed.\n"
    )
    return 0


def cmd_bench(args) -> int:
    """Throughput and correctness across isolation levels."""
    results = run_suite(
        agents=args.agents,
        txns_per_agent=args.txns,
        keys=args.keys,
        think_time_ms=args.think_ms,
        seed=args.seed,
    )
    rows = [
        [
            r.isolation.value,
            str(r.committed),
            str(r.attempts),
            f"{r.abort_rate:.1%}",
            str(r.lost_updates),
            f"{r.throughput:,.0f}",
            f"{r.p50_ms:.1f}",
            f"{r.p99_ms:.1f}",
            "yes" if r.serializable else "NO",
        ]
        for r in results
    ]
    print(
        f"\ncounter workload · {args.agents} agents · {args.txns} transactions each · "
        f"{args.keys} shared keys · {args.think_ms:.0f}ms inference window\n"
    )
    print(
        _table(
            [
                "isolation",
                "commits",
                "attempts",
                "abort",
                "lost",
                "txn/s",
                "p50 ms",
                "p99 ms",
                "serializable",
            ],
            rows,
        )
    )
    baseline = results[0]
    print(
        f"\nwithout concurrency control, {baseline.lost_updates} of {baseline.committed} "
        "committed increments were silently overwritten."
    )
    print("'lost' is measured, not inferred: every commit added one, so the counters must add up.")
    print("'serializable' is checked on the dependency graph of the recorded history.\n")

    if args.json:
        payload = [
            {
                "isolation": r.isolation.value,
                "committed": r.committed,
                "attempts": r.attempts,
                "abort_rate": r.abort_rate,
                "lost_updates": r.lost_updates,
                "throughput": r.throughput,
                "p50_ms": r.p50_ms,
                "p99_ms": r.p99_ms,
                "serializable": r.serializable,
            }
            for r in results
        ]
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"raw results written to {args.json}\n")
    return 0


def cmd_decisions(args) -> int:
    """The workload where agents read shared state and write elsewhere."""
    results = run_decision_suite(
        agents=args.agents,
        decisions_per_agent=args.txns,
        think_time_ms=args.think_ms,
        seed=args.seed,
    )
    rows = [
        [
            r.isolation.value,
            str(r.committed),
            f"{r.abort_rate:.1%}",
            str(r.stale_decisions),
            f"{r.stale_rate:.1%}",
            f"{r.throughput:,.0f}",
        ]
        for r in results
    ]
    print(
        f"\ndecision workload · {args.agents} agents reading a budget another agent revises · "
        f"{args.think_ms:.0f}ms inference window\n"
    )
    print(_table(["isolation", "decisions", "abort", "stale", "stale rate", "txn/s"], rows))
    print(
        "\nNobody writes the key they read, so there is no write-write conflict to find "
        "and snapshot rejects nothing."
    )
    print(
        "A stale decision committed against a budget that had already changed. "
        "Read validation is what removes them, and the throughput column is its price.\n"
    )
    return 0


def cmd_sweep(args) -> int:
    """Abort rate as the inference window grows."""
    results = think_time_sweep(
        agents=args.agents,
        txns_per_agent=args.txns,
        keys=args.keys,
        isolation=Isolation(args.isolation),
        seed=args.seed,
    )
    rows = [
        [
            f"{r.think_time_ms:.0f}ms",
            str(r.committed),
            str(r.attempts),
            f"{r.abort_rate:.1%}",
            f"{r.throughput:,.0f}",
        ]
        for r in results
    ]
    print(f"\ninference window sweep · {args.isolation} · {args.agents} agents\n")
    print(_table(["think time", "commits", "attempts", "abort", "txn/s"], rows))
    print(
        "\nThe longer an agent holds a read, the more often the world moves underneath it. "
        "This is why agent workloads need this and ordinary services mostly do not.\n"
    )
    return 0


def cmd_check(args) -> int:
    """Run a workload and check its history for serializability."""
    result = run_one(
        Isolation(args.isolation),
        agents=args.agents,
        txns_per_agent=args.txns,
        keys=args.keys,
        think_time_ms=args.think_ms,
        seed=args.seed,
    )
    print(f"\nhistory check · {args.isolation} · {result.committed} committed transactions\n")
    if result.serializable:
        print("  serializable: no cycle in the dependency graph")
    else:
        path = " -> ".join(f"T{t}" for t in result.cycle)
        print(f"  NOT serializable: cycle {path}")
    print(f"\n  lost updates: {result.lost_updates}")
    print(f"  abort rate:   {result.abort_rate:.1%}\n")
    return 0 if result.serializable else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aiso",
        description="Concurrency control for multi-agent shared state.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def workload_args(p, *, keys: bool = True, think: float = 2.0):
        p.add_argument("--agents", type=int, default=8)
        p.add_argument("--txns", type=int, default=30)
        if keys:
            p.add_argument("--keys", type=int, default=4)
        p.add_argument("--think-ms", type=float, default=think, dest="think_ms")
        p.add_argument("--seed", type=int, default=0x5EED)
        return p

    anomalies = sub.add_parser("anomalies", help="the four anomalies at every isolation level")
    anomalies.set_defaults(func=cmd_anomalies)

    bench = workload_args(sub.add_parser("bench", help="throughput and lost updates"))
    bench.add_argument("--json", help="write raw results here")
    bench.set_defaults(func=cmd_bench)

    decisions = workload_args(
        sub.add_parser("decisions", help="agents deciding from shared state"),
        keys=False,
        think=5.0,
    )
    decisions.set_defaults(func=cmd_decisions)

    sweep = workload_args(sub.add_parser("sweep", help="abort rate vs inference window"))
    sweep.add_argument(
        "--isolation",
        default="serializable",
        choices=[i.value for i in Isolation],
    )
    sweep.set_defaults(func=cmd_sweep)

    checker = workload_args(sub.add_parser("check", help="verify a history is serializable"))
    checker.add_argument(
        "--isolation",
        default="none",
        choices=[i.value for i in Isolation],
    )
    checker.set_defaults(func=cmd_check)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except BrokenPipeError:
        # `aiso bench | head` closes stdout mid-write. Redirect what remains so
        # Python does not report the same failure again while flushing at exit.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0
    except KeyboardInterrupt:
        return 130


__all__ = ["main"]

if __name__ == "__main__":
    raise SystemExit(main())
