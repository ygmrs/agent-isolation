"""The four anomalies, each as a runnable scenario.

Every scenario runs the same agent code at every isolation level, driving an
exact interleaving by hand rather than launching threads and hoping. A
concurrency test that depends on timing passes on a fast machine and fails in
CI.

The names come from the database literature because that is the argument: what
multi-agent frameworks report as coordination failure was characterized
decades ago and has known remedies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import ConflictError
from .store import Isolation, Store
from .transaction import Transaction


@dataclass
class ScenarioResult:
    name: str
    isolation: Isolation
    anomaly_occurred: bool
    detail: str
    final_state: dict[str, Any]


def _commit(store: Store, tx: Transaction) -> int | None:
    """Commit a hand-driven transaction, returning None if it was rejected."""
    try:
        ts = store.commit(
            agent_id=tx.agent_id,
            snapshot_ts=tx.snapshot_ts,
            read_set=tx.read_set,
            writes=tx.writes,
            deletes=tx.deletes,
            isolation=tx.isolation,
        )
    except ConflictError:
        tx.close()
        return None
    tx.close()
    return ts


def lost_update(isolation: Isolation) -> ScenarioResult:
    """Two agents increment the same counter. One increment disappears.

    Both read 10, both compute 11, both write 11. Nothing errors, no tool
    fails, and the second agent's work is gone. This is the anomaly most often
    reported as "the agents overwrote each other".

    The rejected agent retries here, because that is what the coordinator does
    in real use. Without the retry the scenario would show a missing increment
    under every level and prove nothing: being rejected is not the same as
    losing the work.
    """
    store = Store()
    store.commit(
        agent_id="setup",
        snapshot_ts=0,
        read_set={},
        writes={"tickets:open": 10},
        deletes=set(),
        isolation=Isolation.NONE,
    )

    a = Transaction(store, "agent-a", isolation)
    b = Transaction(store, "agent-b", isolation)

    a.put("tickets:open", a.get("tickets:open") + 1)
    b.put("tickets:open", b.get("tickets:open") + 1)

    _commit(store, a)
    rejected = _commit(store, b) is None

    if rejected:
        retry = Transaction(store, "agent-b", isolation)
        retry.put("tickets:open", retry.get("tickets:open") + 1)
        _commit(store, retry)

    final = store.snapshot_of_all()
    value = final.get("tickets:open")
    lost = value != 12
    if lost:
        detail = f"both agents read 10 and wrote 11; final value {value}, one increment lost"
    else:
        detail = f"agent-b was rejected, retried against 11, and wrote 12; final value {value}"
    store.close()
    return ScenarioResult("lost update", isolation, lost, detail, final)


def stale_read(isolation: Isolation) -> ScenarioResult:
    """An agent decides on a value that changed while it was thinking.

    The agent reads a budget, spends twenty seconds in a model call, and writes
    an approval based on a number that is no longer true. Snapshot isolation
    does not catch this, because the agent never wrote the key it read.
    """
    store = Store()
    for key, value in (("budget:remaining", 100), ("approval:status", "none")):
        store.commit(
            agent_id="setup",
            snapshot_ts=0,
            read_set={},
            writes={key: value},
            deletes=set(),
            isolation=Isolation.NONE,
        )

    reader = Transaction(store, "agent-approver", isolation)
    budget = reader.get("budget:remaining")  # sees 100

    spender = Transaction(store, "agent-spender", isolation)
    spender.put("budget:remaining", budget - 95)  # another agent spends 95
    _commit(store, spender)

    # The approver is still mid-inference and decides on the stale 100.
    reader.put("approval:status", f"approved-against-{budget}")
    committed = _commit(store, reader)

    final = store.snapshot_of_all()
    occurred = committed is not None
    detail = (
        f"approved against a budget of {budget} while only {final.get('budget:remaining')} remained"
        if occurred
        else "approver rejected at commit: the budget it read had changed"
    )
    store.close()
    return ScenarioResult("stale read", isolation, occurred, detail, final)


def write_skew(isolation: Isolation) -> ScenarioResult:
    """Two agents each check an invariant, each act, and together break it.

    Two on-call engineers, at least one must stay on duty. Each agent reads
    both records, sees the other is on duty, and takes itself off. Neither
    wrote a key the other wrote, so snapshot isolation sees no conflict and
    both commit. Nobody is on call.

    This is the scenario that justifies serializable isolation existing, and
    the one a write-set check cannot catch by construction.
    """
    store = Store()
    for key in ("oncall:ana", "oncall:ben"):
        store.commit(
            agent_id="setup",
            snapshot_ts=0,
            read_set={},
            writes={key: "on"},
            deletes=set(),
            isolation=Isolation.NONE,
        )

    a = Transaction(store, "agent-ana", isolation)
    b = Transaction(store, "agent-ben", isolation)

    # Each reads both, confirming the other covers the rota.
    a_sees_ben = a.get("oncall:ben")
    b_sees_ana = b.get("oncall:ana")
    a.get("oncall:ana")
    b.get("oncall:ben")

    if a_sees_ben == "on":
        a.put("oncall:ana", "off")
    if b_sees_ana == "on":
        b.put("oncall:ben", "off")

    _commit(store, a)
    _commit(store, b)

    final = store.snapshot_of_all()
    on_duty = [k for k, v in final.items() if v == "on"]
    broken = not on_duty
    detail = (
        "both agents went off call; the rota is empty"
        if broken
        else f"invariant held, on duty: {', '.join(sorted(on_duty)) or 'none'}"
    )
    store.close()
    return ScenarioResult("write skew", isolation, broken, detail, final)


def inconsistent_outcome(isolation: Isolation) -> ScenarioResult:
    """An agent reads two keys that are only meaningful together.

    Five items sit across two queues and one agent moves an item between them
    in a single transaction, so the total is five at every committed state. A
    second agent reads one queue, waits, then reads the other, and lands on a
    total that was never true at any instant.

    This is the scenario where snapshot isolation earns its place rather than
    being a weaker serializable: reading at a fixed timestamp makes the
    observation consistent without rejecting anything.
    """
    store = Store()
    store.commit(
        agent_id="setup",
        snapshot_ts=0,
        read_set={},
        writes={"queue:inbox": 5, "queue:working": 0},
        deletes=set(),
        isolation=Isolation.NONE,
    )

    observer = Transaction(store, "agent-observer", isolation)
    inbox = observer.get("queue:inbox")

    # One atomic move: the total never changes at any committed timestamp.
    mover = Transaction(store, "agent-mover", isolation)
    mover.put("queue:inbox", 4)
    mover.put("queue:working", 1)
    _commit(store, mover)

    working = observer.get("queue:working")
    total = inbox + working
    observer.put("report:total", total)
    committed = _commit(store, observer)

    final = store.snapshot_of_all()
    occurred = committed is not None and total != 5
    if occurred:
        detail = f"observer recorded a total of {total}; the real total was 5 throughout"
    elif committed is None:
        detail = "observer rejected at commit: a key it read had changed"
    else:
        detail = f"observer read at a fixed snapshot and saw the true total of {total}"
    store.close()
    return ScenarioResult("inconsistent outcome", isolation, occurred, detail, final)


SCENARIOS = {
    "lost update": lost_update,
    "stale read": stale_read,
    "write skew": write_skew,
    "inconsistent outcome": inconsistent_outcome,
}


def run_all() -> list[ScenarioResult]:
    """Every scenario at every isolation level, in table order."""
    return [
        scenario(isolation)
        for scenario in SCENARIOS.values()
        for isolation in (Isolation.NONE, Isolation.SNAPSHOT, Isolation.SERIALIZABLE)
    ]
