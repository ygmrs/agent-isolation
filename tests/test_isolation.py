"""Tests for the isolation guarantees.

The load-bearing ones are ``test_history_is_serializable_under_serializable``
and its negative twin. They do not assert on values the implementation
produced, they reconstruct the dependency graph from the recorded history and
ask whether any serial order could have produced it. A test that only checked
final values would pass on a run where the anomaly happened and the arithmetic
happened to come out even.
"""

from __future__ import annotations

import threading

import pytest

from agent_isolation import (
    ConflictError,
    Coordinator,
    Isolation,
    RetriesExhausted,
    Store,
    Transaction,
    UnsafeEffectError,
    check,
)
from agent_isolation.anomalies import (
    inconsistent_outcome,
    lost_update,
    stale_read,
    write_skew,
)
from agent_isolation.serializability import build_graph


# ---------------------------------------------------------------- snapshots
def test_a_snapshot_does_not_move_under_the_reader():
    """The property the whole design rests on.

    An agent holds a read across an inference call. Whatever else commits in
    that window, the value it already saw must not change, or every decision
    made after the read rests on something that silently moved.
    """
    with Coordinator(isolation=Isolation.SNAPSHOT) as coord:
        coord.set("k", "first")
        tx = Transaction(coord.store, "reader", Isolation.SNAPSHOT)
        assert tx.get("k") == "first"

        coord.set("k", "second", agent_id="other")

        assert tx.get("k") == "first", "the snapshot moved under an open transaction"
        assert coord.get("k") == "second"


def test_reads_see_this_transactions_own_writes():
    with Coordinator() as coord:

        def body(tx):
            tx.put("k", 1)
            assert tx.get("k") == 1
            tx.put("k", 2)
            return tx.get("k")

        assert coord.run(body).value == 2


def test_absence_is_recorded_in_the_read_set():
    """ "This key did not exist" is a premise a decision can rest on.

    An agent that checks for a lock file and proceeds because there was none
    has read something, even though it got nothing back. Leaving that out of
    the read set would make the most common agent guard invisible to
    validation.
    """
    with Coordinator(isolation=Isolation.SERIALIZABLE) as coord:
        tx = Transaction(coord.store, "reader", Isolation.SERIALIZABLE)
        assert tx.get("lock:job") is None
        assert "lock:job" in tx.read_set
        assert tx.read_set["lock:job"] == 0

        coord.set("lock:job", "held", agent_id="other")

        with pytest.raises(ConflictError) as excinfo:
            coord.store.commit(
                agent_id="reader",
                snapshot_ts=tx.snapshot_ts,
                read_set=tx.read_set,
                writes={"job:started": True},
                deletes=set(),
                isolation=Isolation.SERIALIZABLE,
            )
        assert excinfo.value.kind == "read-write"


# ---------------------------------------------------------------- anomalies
@pytest.mark.parametrize(
    ("scenario", "allowed_at"),
    [
        (lost_update, {Isolation.NONE}),
        (stale_read, {Isolation.NONE, Isolation.SNAPSHOT}),
        (write_skew, {Isolation.NONE, Isolation.SNAPSHOT}),
        (inconsistent_outcome, {Isolation.NONE}),
    ],
)
def test_each_anomaly_is_prevented_at_exactly_the_right_level(scenario, allowed_at):
    """The isolation matrix, asserted rather than described.

    Each anomaly must occur at the weaker levels and stop at the one that is
    supposed to stop it. Asserting only that serializable prevents everything
    would pass for an implementation that rejected every transaction.
    """
    for isolation in Isolation:
        result = scenario(isolation)
        expected = isolation in allowed_at
        assert result.anomaly_occurred is expected, (
            f"{result.name} at {isolation.value}: "
            f"expected occurred={expected}, got {result.anomaly_occurred}. {result.detail}"
        )


def test_write_skew_is_invisible_to_write_set_validation():
    """Why read validation has to exist.

    Neither agent writes a key the other writes, so no write-set check can
    find this however carefully it is implemented. The test pins the reason
    rather than the symptom.
    """
    result = write_skew(Isolation.SNAPSHOT)
    assert result.anomaly_occurred
    assert write_skew(Isolation.SERIALIZABLE).anomaly_occurred is False


# --------------------------------------------------------- serializability
def test_history_is_serializable_under_serializable():
    """Checked on the dependency graph, not on the final values."""
    from agent_isolation.bench import run_one

    result = run_one(Isolation.SERIALIZABLE, agents=6, txns_per_agent=12, keys=3, think_time_ms=1.0)
    assert result.serializable, f"cycle found: {result.cycle}"
    assert result.lost_updates == 0


def test_an_uncontrolled_history_is_caught_by_the_checker():
    """The checker must be able to fail, or passing means nothing."""
    from agent_isolation.bench import run_one

    result = run_one(Isolation.NONE, agents=6, txns_per_agent=12, keys=2, think_time_ms=1.0)
    assert not result.serializable, "the checker accepted a history with lost updates"
    assert result.lost_updates > 0


def test_checker_finds_the_write_skew_cycle():
    """Two anti-dependencies pointing opposite ways is the cycle."""
    store = Store()
    for key in ("x", "y"):
        store.commit(
            agent_id="setup",
            snapshot_ts=0,
            read_set={},
            writes={key: "on"},
            deletes=set(),
            isolation=Isolation.NONE,
        )

    a = Transaction(store, "a", Isolation.SNAPSHOT)
    b = Transaction(store, "b", Isolation.SNAPSHOT)
    a.get("x"), a.get("y")
    b.get("x"), b.get("y")
    a.put("x", "off")
    b.put("y", "off")
    for tx in (a, b):
        store.commit(
            agent_id=tx.agent_id,
            snapshot_ts=tx.snapshot_ts,
            read_set=tx.read_set,
            writes=tx.writes,
            deletes=tx.deletes,
            isolation=Isolation.SNAPSHOT,
        )

    verdict = check(store.history())
    assert not verdict.serializable
    assert any(edge.kind == "rw" for edge in verdict.cycle_edges)
    store.close()


def test_a_serial_history_has_no_cycle():
    """A sanity floor: transactions run one at a time cannot be unserializable."""
    with Coordinator(isolation=Isolation.SERIALIZABLE) as coord:
        for i in range(10):
            coord.run(lambda tx, i=i: tx.put(f"k{i % 3}", i), agent_id="solo")
        verdict = check(coord.history())
        assert verdict.serializable, verdict.explain()
        assert build_graph(coord.history())  # there are dependencies, just no cycle


# --------------------------------------------------------------- conflicts
def test_write_write_conflict_is_reported_as_write_write():
    """The two kinds are distinguished because they mean different things.

    A write-write conflict means two agents changed the same thing. A
    read-write conflict means one acted on something the other changed. Mixing
    them in the benchmark would hide which problem the workload actually has.
    """
    with Coordinator(isolation=Isolation.SERIALIZABLE) as coord:
        coord.set("k", 0)
        tx = Transaction(coord.store, "a", Isolation.SERIALIZABLE)
        tx.get("k")
        tx.put("k", 1)
        coord.set("k", 99, agent_id="b")

        with pytest.raises(ConflictError) as excinfo:
            coord.store.commit(
                agent_id="a",
                snapshot_ts=tx.snapshot_ts,
                read_set=tx.read_set,
                writes=tx.writes,
                deletes=set(),
                isolation=Isolation.SERIALIZABLE,
            )
        assert excinfo.value.kind == "write-write"


def test_retry_eventually_succeeds_and_is_counted():
    with Coordinator(isolation=Isolation.SNAPSHOT) as coord:
        coord.set("k", 0)
        attempts: list[int] = []

        def body(tx):
            attempts.append(1)
            value = tx.get("k")
            if len(attempts) == 1:
                # Commit from another agent while this one is mid-flight.
                coord.set("k", value + 100, agent_id="interloper")
            tx.put("k", value + 1)

        result = coord.run(body, agent_id="a")
        assert result.attempts == 2
        assert result.retried
        assert coord.get("k") == 101


def test_retries_are_bounded():
    with Coordinator(isolation=Isolation.SNAPSHOT, max_attempts=3) as coord:
        coord.set("k", 0)

        def body(tx):
            tx.get("k")
            coord.set("k", 1, agent_id="interloper")  # conflicts every time
            tx.put("k", 2)

        with pytest.raises(RetriesExhausted) as excinfo:
            coord.run(body, agent_id="a")
        assert excinfo.value.attempts == 3


# ----------------------------------------------------------------- effects
def test_deferred_effects_do_not_run_before_commit():
    with Coordinator() as coord:
        fired: list[dict] = []
        coord.effects.register("notify", fired.append)
        coord.set("k", 0)

        def body(tx):
            tx.put("k", 1)
            tx.defer("notify", {"k": 1})
            assert fired == [], "the effect ran while the transaction could still abort"

        coord.run(body)
        assert fired == [{"k": 1}]


def test_deferred_effects_do_not_run_when_the_transaction_aborts():
    """The guarantee that makes deferral the safe default."""
    with Coordinator(isolation=Isolation.SNAPSHOT, max_attempts=1) as coord:
        fired: list[dict] = []
        coord.effects.register("notify", fired.append)
        coord.set("k", 0)

        def body(tx):
            tx.get("k")
            coord.set("k", 99, agent_id="interloper")
            tx.put("k", 1)
            tx.defer("notify", {"sent": True})

        with pytest.raises(RetriesExhausted):
            coord.run(body, agent_id="a")
        assert fired == [], "an effect fired for a transaction that never committed"


def test_compensation_runs_when_the_transaction_aborts():
    """The answer for effects that cannot wait.

    A reservation has to happen during the transaction, because the agent
    needs to know whether it succeeded. If the transaction then aborts, the
    inverse is the only thing standing between the agent and a hold nobody
    will ever release.
    """
    with Coordinator(isolation=Isolation.SNAPSHOT, max_attempts=1) as coord:
        reservations: list[str] = []
        coord.set("k", 0)

        def body(tx):
            tx.get("k")
            tx.compensable(
                "reserve-seat",
                do=lambda: reservations.append("held"),
                undo=lambda: reservations.remove("held"),
            )
            assert reservations == ["held"]
            coord.set("k", 99, agent_id="interloper")
            tx.put("k", 1)

        with pytest.raises(RetriesExhausted):
            coord.run(body, agent_id="a")
        assert reservations == [], "the reservation outlived the transaction that made it"
        assert coord.stats.compensations_run == 1


def test_compensations_run_in_reverse_order():
    """Inverses must unwind, not replay. Later effects may depend on earlier ones."""
    with Coordinator(isolation=Isolation.SNAPSHOT, max_attempts=1) as coord:
        order: list[str] = []
        coord.set("k", 0)

        def body(tx):
            tx.get("k")
            tx.compensable("first", do=lambda: None, undo=lambda: order.append("undo-first"))
            tx.compensable("second", do=lambda: None, undo=lambda: order.append("undo-second"))
            coord.set("k", 99, agent_id="interloper")
            tx.put("k", 1)

        with pytest.raises(RetriesExhausted):
            coord.run(body, agent_id="a")
        assert order == ["undo-second", "undo-first"]


def test_irreversible_effects_are_refused():
    """The refusal is the feature, not a missing capability."""
    with Coordinator() as coord:

        def body(tx):
            tx.immediate("wire-funds", do=lambda: "sent")

        with pytest.raises(UnsafeEffectError) as excinfo:
            coord.run(body)
        assert "defer()" in str(excinfo.value)
        assert "compensable()" in str(excinfo.value)


def test_irreversible_effects_can_be_forced_explicitly():
    with Coordinator() as coord:

        def body(tx):
            return tx.immediate("wire-funds", do=lambda: "sent", allow_unsafe=True)

        assert coord.run(body).value == "sent"


def test_deferred_effects_are_idempotent_by_key():
    """At-least-once delivery means a handler can see the same effect twice."""
    with Coordinator() as coord:
        fired: list[dict] = []
        coord.effects.register("notify", fired.append)

        coord.run(lambda tx: tx.defer("notify", {"n": 1}, key="fixed-key"))
        coord.run(lambda tx: tx.defer("notify", {"n": 2}, key="fixed-key"))

        assert len(fired) == 1, "the same idempotency key was delivered twice"


# -------------------------------------------------------------- concurrency
def test_concurrent_agents_do_not_lose_updates():
    """The end-to-end claim, with real threads."""
    with Coordinator(isolation=Isolation.SNAPSHOT, max_attempts=50) as coord:
        coord.set("counter", 0)
        per_thread, threads = 25, 8

        def worker():
            for _ in range(per_thread):
                coord.run(
                    lambda tx: tx.put("counter", tx.get("counter", 0) + 1),
                    agent_id="worker",
                )

        workers = [threading.Thread(target=worker) for _ in range(threads)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()

        assert coord.get("counter") == per_thread * threads


def test_uncontrolled_concurrent_agents_do_lose_updates():
    """The baseline has to actually fail, or the comparison is theatre."""
    with Coordinator(isolation=Isolation.NONE) as coord:
        coord.set("counter", 0)
        per_thread, threads = 40, 8

        def worker():
            for _ in range(per_thread):
                coord.run(
                    lambda tx: tx.put("counter", tx.get("counter", 0) + 1),
                    agent_id="worker",
                )

        workers = [threading.Thread(target=worker) for _ in range(threads)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()

        assert coord.get("counter") < per_thread * threads
