"""Concurrency control for multi-agent systems.

Agents sharing state hit anomalies databases solved decades ago. Long
inference windows make them frequent rather than rare: an agent holds a read
across a model call while other agents commit underneath it.

Snapshot reads, optimistic validation at commit, and a side-effect model for
what databases never faced — tool calls that already happened when the
transaction aborts.

    from agent_isolation import Coordinator, Isolation

    with Coordinator(isolation=Isolation.SERIALIZABLE) as coord:
        coord.set("budget", 100)

        def spend(tx):
            remaining = tx.get("budget")
            tx.put("budget", remaining - 10)
            tx.defer("notify", {"spent": 10})
            return remaining - 10

        result = coord.run(spend, agent_id="agent-1")
"""

from .anomalies import ScenarioResult, run_all
from .coordinator import Coordinator, Stats, TxnResult
from .effects import CompensableEffect, DeferredEffect, EffectRunner
from .errors import (
    ConcurrencyControlError,
    ConflictError,
    RetriesExhausted,
    TransactionClosed,
    UnsafeEffectError,
)
from .serializability import CheckResult, build_graph, check
from .store import MISSING, CommitRecord, Isolation, Store
from .transaction import Transaction

__version__ = "0.1.0"

__all__ = [
    "CheckResult",
    "CommitRecord",
    "CompensableEffect",
    "ConcurrencyControlError",
    "ConflictError",
    "Coordinator",
    "DeferredEffect",
    "EffectRunner",
    "MISSING",
    "Isolation",
    "RetriesExhausted",
    "ScenarioResult",
    "Stats",
    "Store",
    "Transaction",
    "TransactionClosed",
    "TxnResult",
    "UnsafeEffectError",
    "build_graph",
    "check",
    "run_all",
    "__version__",
]
