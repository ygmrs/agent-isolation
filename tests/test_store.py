"""Tests for storage, durability, and recovery."""

from __future__ import annotations

import pytest

from agent_isolation import MISSING, Coordinator, Isolation, Store, Transaction
from agent_isolation.effects import EffectRunner


def test_read_at_returns_the_version_current_at_that_timestamp():
    """Multi-version reads, which is what makes a snapshot possible."""
    store = Store()
    timestamps = []
    for value in ("a", "b", "c"):
        timestamps.append(
            store.commit(
                agent_id="w",
                snapshot_ts=store.now(),
                read_set={},
                writes={"k": value},
                deletes=set(),
                isolation=Isolation.NONE,
            )
        )

    for ts, expected in zip(timestamps, ("a", "b", "c"), strict=True):
        assert store.read_at("k", ts)[0] == expected
    assert store.read_at("k", timestamps[0] - 1)[0] is MISSING
    assert store.read_at("k", 10_000)[0] == "c"
    store.close()


def test_read_returns_the_version_timestamp_not_the_value_identity():
    """Validation compares versions, so a rewrite of the same value conflicts.

    An agent that read x=5 and then saw another agent write x=5 has still had
    its premise rewritten: it read a version that no longer exists. Comparing
    values would call that no conflict and quietly weaken the guarantee.
    """
    store = Store()
    first = store.commit(
        agent_id="w",
        snapshot_ts=0,
        read_set={},
        writes={"k": 5},
        deletes=set(),
        isolation=Isolation.NONE,
    )
    second = store.commit(
        agent_id="w",
        snapshot_ts=first,
        read_set={},
        writes={"k": 5},
        deletes=set(),
        isolation=Isolation.NONE,
    )
    assert first != second
    _, version = store.read_at("k", second)
    assert version == second
    store.close()


def test_deletes_are_tombstones_not_erasures():
    """A snapshot taken before a delete must still see the value."""
    store = Store()
    before = store.commit(
        agent_id="w",
        snapshot_ts=0,
        read_set={},
        writes={"k": "here"},
        deletes=set(),
        isolation=Isolation.NONE,
    )
    after = store.commit(
        agent_id="w",
        snapshot_ts=before,
        read_set={},
        writes={},
        deletes={"k"},
        isolation=Isolation.NONE,
    )
    assert store.read_at("k", before)[0] == "here"
    assert store.read_at("k", after)[0] is MISSING
    store.close()


def test_state_survives_reopening(tmp_path):
    """Durability. The commit log and the data come back together."""
    path = tmp_path / "state.db"
    with Coordinator(path, isolation=Isolation.SERIALIZABLE) as coord:
        coord.set("k", "written")
        committed = len(coord.history())

    with Coordinator(path, isolation=Isolation.SERIALIZABLE) as reopened:
        assert reopened.get("k") == "written"
        assert len(reopened.history()) == committed


def test_a_pending_effect_survives_a_restart(tmp_path):
    """Why deferred effects are rows rather than callbacks.

    A crash between commit and dispatch must leave a durable record of what is
    owed. If the queue lived in memory, the data would be committed and the
    effect that was supposed to follow it would be gone with no trace.
    """
    path = tmp_path / "state.db"
    with Coordinator(path) as coord:
        # No handler registered, so the effect is enqueued and stays pending.
        coord.run(lambda tx: tx.defer("notify", {"n": 1}, key="k1"))
        assert coord.effects.pending() == 1

    with Coordinator(path) as reopened:
        fired: list[dict] = []
        reopened.effects.register("notify", fired.append)
        assert reopened.effects.pending() == 1
        assert reopened.effects.drain() == 1
        assert fired == [{"n": 1}]
        assert reopened.effects.pending() == 0


def test_a_failing_handler_does_not_mark_the_effect_done():
    with Coordinator() as coord:

        def explode(_payload):
            raise RuntimeError("downstream is down")

        coord.effects.register("notify", explode)
        coord.run(lambda tx: tx.defer("notify", {"n": 1}))
        row = coord.store.conn.execute("SELECT status, attempts FROM effects").fetchone()
        assert row[0] == "failed"
        assert row[1] == 1


def test_validation_failure_writes_nothing():
    """An aborted transaction must leave no trace in the versions table."""
    store = Store()
    store.commit(
        agent_id="w",
        snapshot_ts=0,
        read_set={},
        writes={"k": 1},
        deletes=set(),
        isolation=Isolation.NONE,
    )
    stale_snapshot = 0
    before = store.conn.execute("SELECT COUNT(*) FROM versions").fetchone()[0]

    from agent_isolation.errors import ConflictError

    with pytest.raises(ConflictError):
        store.commit(
            agent_id="w",
            snapshot_ts=stale_snapshot,
            read_set={},
            writes={"k": 2, "other": 3},
            deletes=set(),
            isolation=Isolation.SNAPSHOT,
        )
    after = store.conn.execute("SELECT COUNT(*) FROM versions").fetchone()[0]
    assert after == before, "a rejected transaction left rows behind"
    assert store.read_at("other", store.now())[0] is MISSING
    store.close()


def test_commit_timestamps_are_strictly_increasing():
    store = Store()
    stamps = [
        store.commit(
            agent_id="w",
            snapshot_ts=store.now(),
            read_set={},
            writes={f"k{i}": i},
            deletes=set(),
            isolation=Isolation.NONE,
        )
        for i in range(20)
    ]
    assert stamps == sorted(stamps)
    assert len(set(stamps)) == len(stamps)
    store.close()


def test_history_records_the_read_set():
    """Without it the checker cannot reconstruct anti-dependencies."""
    with Coordinator(isolation=Isolation.SERIALIZABLE) as coord:
        coord.set("a", 1)
        coord.run(lambda tx: tx.put("b", tx.get("a") + 1), agent_id="reader")
        last = coord.history()[-1]
        assert "a" in last.read_set
        assert last.write_set == ["b"]


def test_transaction_cannot_be_used_after_it_closes():
    from agent_isolation import TransactionClosed

    store = Store()
    tx = Transaction(store, "a", Isolation.SNAPSHOT)
    tx.close()
    with pytest.raises(TransactionClosed):
        tx.get("k")
    with pytest.raises(TransactionClosed):
        tx.put("k", 1)
    store.close()


def test_effect_runner_ignores_unknown_names():
    """An effect whose handler is not registered stays pending, not lost."""
    store = Store()
    runner = EffectRunner(store)
    store.conn.execute(
        "INSERT INTO effects (id, commit_ts, name, payload, status, attempts, updated_at) "
        "VALUES ('x', 1, 'unregistered', '{}', 'pending', 0, 0)"
    )
    assert runner.drain() == 0
    assert runner.pending() == 1
    store.close()


def test_concurrent_drains_deliver_each_effect_once():
    """Regression: every agent drains after its own commit.

    Selecting pending rows and then dispatching is a race. Two threads read
    the same row, both call the handler, and the downstream system sees a
    duplicate that no idempotency key was going to stop, because each delivery
    carried a different one. Found by an example that reported nine
    confirmations for eight bookings.
    """
    import threading as _threading

    from agent_isolation import Coordinator

    with Coordinator(max_attempts=60) as coord:
        delivered: list[dict] = []
        guard = _threading.Lock()

        def handler(payload):
            with guard:
                delivered.append(payload)

        coord.effects.register("confirm", handler)
        coord.set("k", 0)

        agents, per_agent = 8, 6

        def worker(index: int):
            for n in range(per_agent):
                coord.run(
                    lambda tx, i=index, n=n: tx.defer("confirm", {"agent": i, "n": n}),
                    agent_id=f"agent-{index}",
                )

        threads = [_threading.Thread(target=worker, args=(i,)) for i in range(agents)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        coord.effects.drain()

        assert len(delivered) == agents * per_agent, (
            f"expected {agents * per_agent} deliveries, got {len(delivered)}"
        )
        assert len({(d["agent"], d["n"]) for d in delivered}) == len(delivered)


def test_a_claimed_effect_is_returned_to_the_queue_on_restart(tmp_path):
    """A process that dies mid-handler must not strand the effect."""
    from agent_isolation import Coordinator

    path = tmp_path / "state.db"
    with Coordinator(path) as coord:
        coord.run(lambda tx: tx.defer("notify", {"n": 1}, key="k1"))
        # Simulate a crash after claiming and before recording the outcome.
        coord.store.conn.execute("UPDATE effects SET status = 'running' WHERE id = 'k1'")

    with Coordinator(path) as reopened:
        fired: list[dict] = []
        reopened.effects.register("notify", fired.append)
        assert reopened.effects.pending() == 1
        assert reopened.effects.drain() == 1
        assert fired == [{"n": 1}]


def test_a_stored_null_is_not_an_absent_key():
    """Regression: ``get`` returned the default for a value that was there.

    An agent can store ``None`` deliberately, and a guard that checks whether
    a key exists has to tell that apart from a key holding null. Collapsing
    the two made a real value invisible.
    """
    from agent_isolation import Coordinator

    with Coordinator() as coord:
        coord.set("stored-null", None)
        assert coord.get("stored-null", "DEFAULT") is None
        assert coord.get("never-written", "DEFAULT") == "DEFAULT"
        assert "stored-null" in coord.store.snapshot_of_all()

        def body(tx):
            return (tx.get("stored-null", "DEFAULT"), tx.get("never-written", "DEFAULT"))

        assert coord.run(body).value == (None, "DEFAULT")


def test_a_deleted_key_reads_as_absent_not_as_null():
    from agent_isolation import Coordinator

    with Coordinator() as coord:
        coord.set("k", "here")
        coord.run(lambda tx: tx.delete("k"))
        assert coord.get("k", "DEFAULT") == "DEFAULT"


def test_a_read_only_transaction_takes_no_timestamp():
    """Regression: every read advanced the clock and logged an empty row.

    A transaction that writes nothing changes no state, so allocating a
    commit timestamp for it both polluted the history the checker reads and
    made readers contend for the write lock they never needed.
    """
    from agent_isolation import Coordinator, Isolation

    with Coordinator(isolation=Isolation.SERIALIZABLE) as coord:
        coord.set("k", 1)
        clock_before = coord.store.now()
        logged_before = len(coord.history())

        result = coord.run(lambda tx: tx.get("k"), agent_id="reader")

        assert result.value == 1
        assert coord.store.now() == clock_before, "a read-only transaction moved the clock"
        assert len(coord.history()) == logged_before, "a read-only transaction was logged"
        assert result.commit_ts == clock_before


def test_a_read_only_transaction_is_still_validated():
    """Writing nothing is not a reason to skip the check.

    An agent that read stale state should be told, even when it decided not to
    write. The fast path has to stay a performance change, not a weakening of
    the guarantee.
    """
    from agent_isolation import Coordinator, Isolation, RetriesExhausted

    with Coordinator(isolation=Isolation.SERIALIZABLE, max_attempts=1) as coord:
        coord.set("k", 1)

        def body(tx):
            tx.get("k")
            coord.set("k", 2, agent_id="interloper")
            return tx.get("k")

        with pytest.raises(RetriesExhausted):
            coord.run(body, agent_id="reader")


def test_a_transaction_with_only_effects_is_committed():
    """Queuing an effect is a state change, even with no writes."""
    from agent_isolation import Coordinator

    with Coordinator() as coord:
        fired: list[dict] = []
        coord.effects.register("notify", fired.append)
        clock_before = coord.store.now()

        coord.run(lambda tx: tx.defer("notify", {"n": 1}), agent_id="notifier")

        assert coord.store.now() > clock_before
        assert fired == [{"n": 1}]


def test_effects_are_queued_inside_the_commit():
    """The queue row and the data land together or not at all.

    Written afterwards, a crash in between would commit the data and lose the
    effect it owed, with nothing left to say it was ever promised.
    """
    from agent_isolation import Coordinator, Isolation, RetriesExhausted

    with Coordinator(isolation=Isolation.SNAPSHOT, max_attempts=1) as coord:
        coord.set("k", 0)

        def body(tx):
            tx.get("k")
            coord.set("k", 99, agent_id="interloper")
            tx.put("k", 1)
            tx.defer("notify", {"sent": True}, key="never")

        with pytest.raises(RetriesExhausted):
            coord.run(body, agent_id="a")

        row = coord.store.conn.execute("SELECT COUNT(*) FROM effects WHERE id = 'never'").fetchone()
        assert row[0] == 0, "an effect was queued for a transaction that never committed"
