"""Measuring what isolation costs and what it buys.

Agents increment shared counters: read, add one, write. Small enough that the
result cannot be explained by anything but the concurrency control, and it has
an exact oracle — every commit added one, so the counters must add up. Any
shortfall is a lost update, counted rather than argued about.

``think_time_ms`` holds each transaction open without doing work, standing in
for an inference call. It is the parameter that makes this about agents rather
than about databases.

``keys`` sets how many counters the agents share. Fewer keys means more agents
landing on the same one.

Results are reported per isolation level, including ``NONE``, which is both
the fastest and the one that silently loses work.
"""

from __future__ import annotations

import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .coordinator import Coordinator, Stats
from .errors import RetriesExhausted
from .serializability import check
from .store import Isolation
from .transaction import Transaction

DEFAULT_SEED = 0x5EED


@dataclass
class BenchResult:
    """One isolation level under one workload."""

    isolation: Isolation
    agents: int
    txns_per_agent: int
    keys: int
    think_time_ms: float

    committed: int = 0
    attempts: int = 0
    write_write: int = 0
    read_write: int = 0
    exhausted: int = 0
    wall_s: float = 0.0
    latencies_ms: list[float] = field(default_factory=list)

    expected_total: int = 0
    actual_total: int = 0
    serializable: bool = True
    cycle: list[int] = field(default_factory=list)

    @property
    def lost_updates(self) -> int:
        """Increments that committed and then vanished."""
        return max(0, self.expected_total - self.actual_total)

    @property
    def abort_rate(self) -> float:
        return 0.0 if self.attempts == 0 else 1.0 - (self.committed / self.attempts)

    @property
    def throughput(self) -> float:
        return 0.0 if self.wall_s == 0 else self.committed / self.wall_s

    @property
    def p50_ms(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p99_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        ordered = sorted(self.latencies_ms)
        index = min(len(ordered) - 1, int(0.99 * len(ordered)))
        return ordered[index]


def _increment_body(key: str):
    def body(tx: Transaction) -> int:
        current = tx.get(key, 0)
        tx.put(key, current + 1)
        return current + 1

    return body


def run_one(
    isolation: Isolation,
    *,
    agents: int = 8,
    txns_per_agent: int = 40,
    keys: int = 4,
    think_time_ms: float = 2.0,
    max_attempts: int = 24,
    seed: int = DEFAULT_SEED,
) -> BenchResult:
    """Run the counter workload at one isolation level."""
    result = BenchResult(
        isolation=isolation,
        agents=agents,
        txns_per_agent=txns_per_agent,
        keys=keys,
        think_time_ms=think_time_ms,
    )
    coord = Coordinator(isolation=isolation, max_attempts=max_attempts)
    key_names = [f"counter:{i}" for i in range(keys)]
    for name in key_names:
        coord.set(name, 0)
    # Seeding the counters costs one transaction per key. Leaving those in the
    # totals would report an abort rate for an arm that never aborted.
    coord.stats = Stats()

    think_s = think_time_ms / 1000.0
    latencies: list[float] = []
    committed = 0
    lock = threading.Lock()

    def agent(agent_index: int) -> None:
        nonlocal committed
        # Each agent draws its key sequence from its own seeded generator, so
        # every isolation level faces the identical sequence of collisions.
        # Without that, a slower arm would also be a differently-contended one.
        rng = random.Random(seed + agent_index)
        for _ in range(txns_per_agent):
            key = key_names[rng.randrange(keys)]
            started = time.perf_counter()
            try:
                coord.run(
                    _increment_body(key),
                    agent_id=f"agent-{agent_index}",
                    think_time_s=think_s,
                )
            except RetriesExhausted:
                continue
            elapsed = (time.perf_counter() - started) * 1000.0
            with lock:
                latencies.append(elapsed)
                committed += 1

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=agents) as pool:
        list(pool.map(agent, range(agents)))
    result.wall_s = time.perf_counter() - started

    result.latencies_ms = latencies
    result.committed = committed
    result.attempts = coord.stats.attempts
    result.total_committed = coord.stats.committed
    result.write_write = coord.stats.write_write_conflicts
    result.read_write = coord.stats.read_write_conflicts
    result.exhausted = coord.stats.exhausted

    # The oracle: one increment per commit, so the counters must add up.
    state = coord.store.snapshot_of_all()
    result.actual_total = sum(v for k, v in state.items() if k.startswith("counter:"))
    result.expected_total = committed

    verdict = check(coord.history())
    result.serializable = verdict.serializable
    result.cycle = verdict.cycle

    coord.close()
    return result


def run_suite(
    *,
    agents: int = 8,
    txns_per_agent: int = 40,
    keys: int = 4,
    think_time_ms: float = 2.0,
    seed: int = DEFAULT_SEED,
) -> list[BenchResult]:
    """Every isolation level against the same workload."""
    return [
        run_one(
            isolation,
            agents=agents,
            txns_per_agent=txns_per_agent,
            keys=keys,
            think_time_ms=think_time_ms,
            seed=seed,
        )
        for isolation in (Isolation.NONE, Isolation.SNAPSHOT, Isolation.SERIALIZABLE)
    ]


def think_time_sweep(
    *,
    agents: int = 8,
    txns_per_agent: int = 20,
    keys: int = 4,
    think_times_ms: tuple[float, ...] = (0.0, 2.0, 10.0, 50.0),
    isolation: Isolation = Isolation.SERIALIZABLE,
    seed: int = DEFAULT_SEED,
) -> list[BenchResult]:
    """Abort rate as the inference window grows.

    This is the central claim of the project reduced to one table: the longer
    an agent holds a read, the more often the world moves underneath it.
    """
    return [
        run_one(
            isolation,
            agents=agents,
            txns_per_agent=txns_per_agent,
            keys=keys,
            think_time_ms=think_time,
            seed=seed,
        )
        for think_time in think_times_ms
    ]


@dataclass
class DecisionResult:
    """A workload where agents decide from shared state they do not write."""

    isolation: Isolation
    agents: int
    decisions_per_agent: int
    think_time_ms: float

    committed: int = 0
    attempts: int = 0
    total_committed: int = 0
    stale_decisions: int = 0
    wall_s: float = 0.0

    @property
    def abort_rate(self) -> float:
        # Measured over every transaction, including the policy agent's
        # revisions. Dividing decisions by all attempts would report aborts
        # for an arm where nothing was ever rejected.
        if self.attempts == 0:
            return 0.0
        return 1.0 - (self.total_committed / self.attempts)

    @property
    def stale_rate(self) -> float:
        return 0.0 if self.committed == 0 else self.stale_decisions / self.committed

    @property
    def throughput(self) -> float:
        return 0.0 if self.wall_s == 0 else self.committed / self.wall_s


BUDGET_KEY = "policy:budget"


def run_decision_workload(
    isolation: Isolation,
    *,
    agents: int = 8,
    decisions_per_agent: int = 20,
    think_time_ms: float = 5.0,
    revisions: int = 12,
    seed: int = DEFAULT_SEED,
) -> DecisionResult:
    """Agents approve spending against a budget another agent keeps changing.

    Nobody writes the key they read, so there is never a write-write conflict
    and snapshot isolation never rejects anything. Every agent still records a
    decision, and some of those decisions rest on a budget that had already
    changed by the time they committed. Those are counted here.

    This is the shape most agent work actually takes. Read shared context,
    think, write your own result somewhere else. A write-set check cannot see
    it, which is the argument for paying for read validation.
    """
    result = DecisionResult(
        isolation=isolation,
        agents=agents,
        decisions_per_agent=decisions_per_agent,
        think_time_ms=think_time_ms,
    )
    coord = Coordinator(isolation=isolation, max_attempts=24)
    coord.set(BUDGET_KEY, 100)
    coord.stats = Stats()

    think_s = think_time_ms / 1000.0
    lock = threading.Lock()
    committed = 0

    def decide(agent_index: int) -> None:
        nonlocal committed
        rng = random.Random(seed + agent_index)
        for n in range(decisions_per_agent):

            def body(tx: Transaction, n: int = n, agent_index: int = agent_index) -> None:
                budget = tx.get(BUDGET_KEY, 0)
                tx.put(f"decision:{agent_index}:{n}", budget)

            try:
                coord.run(body, agent_id=f"agent-{agent_index}", think_time_s=think_s)
            except RetriesExhausted:
                continue
            with lock:
                committed += 1
            if rng.random() < 0.05:
                time.sleep(think_s / 2)

    def revise() -> None:
        # A separate agent moves the budget underneath everyone else.
        for i in range(revisions):
            time.sleep(think_s * 2)
            try:
                coord.run(
                    lambda tx, i=i: tx.put(BUDGET_KEY, 100 - i),
                    agent_id="agent-policy",
                )
            except RetriesExhausted:
                continue

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=agents + 1) as pool:
        futures = [pool.submit(decide, i) for i in range(agents)]
        futures.append(pool.submit(revise))
        for future in futures:
            future.result()
    result.wall_s = time.perf_counter() - started

    result.committed = committed
    result.attempts = coord.stats.attempts
    result.total_committed = coord.stats.committed

    # A decision is stale when the budget it recorded was already superseded
    # at the moment it committed. Computed from the log rather than guessed:
    # the read set names the version seen, and the versions table says which
    # version was current at that commit timestamp.
    conn = coord.store.conn
    stale = 0
    for record in coord.history():
        if BUDGET_KEY not in record.read_set:
            continue
        if not any(k.startswith("decision:") for k in record.write_set):
            continue
        row = conn.execute(
            "SELECT MAX(commit_ts) FROM versions WHERE key = ? AND commit_ts <= ?",
            (BUDGET_KEY, record.commit_ts),
        ).fetchone()
        current = int(row[0]) if row and row[0] is not None else 0
        if current != record.read_set[BUDGET_KEY]:
            stale += 1
    result.stale_decisions = stale

    coord.close()
    return result


def run_decision_suite(**kwargs) -> list[DecisionResult]:
    """The decision workload at every isolation level."""
    return [
        run_decision_workload(isolation, **kwargs)
        for isolation in (Isolation.NONE, Isolation.SNAPSHOT, Isolation.SERIALIZABLE)
    ]
