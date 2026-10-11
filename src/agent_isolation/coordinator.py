"""Running agent work under concurrency control.

The coordinator owns the retry loop. A conflict means the transaction's
premise went stale, so the only correct response is to discard the work and
redo it against fresh state.

``run`` takes a function rather than exposing begin/commit, because anything
that may be retried has to be re-runnable. Handing out a transaction object
invites setup outside it that silently does not get redone.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .effects import EffectRunner
from .errors import ConflictError, RetriesExhausted
from .store import MISSING, Isolation, Store
from .transaction import Transaction


@dataclass
class TxnResult:
    """What happened to one unit of agent work."""

    value: Any
    commit_ts: int
    attempts: int
    conflicts: list[ConflictError] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def retried(self) -> bool:
        return self.attempts > 1


@dataclass
class Stats:
    """Counters across every transaction this coordinator has run."""

    committed: int = 0
    attempts: int = 0
    write_write_conflicts: int = 0
    read_write_conflicts: int = 0
    exhausted: int = 0
    compensations_run: int = 0

    @property
    def abort_rate(self) -> float:
        """Share of attempts that were thrown away."""
        return 0.0 if self.attempts == 0 else 1.0 - (self.committed / self.attempts)

    @property
    def conflicts(self) -> int:
        return self.write_write_conflicts + self.read_write_conflicts


class Coordinator:
    """Entry point. Runs agent transactions under a chosen isolation level."""

    def __init__(
        self,
        path: str | Path = ":memory:",
        isolation: Isolation = Isolation.SNAPSHOT,
        *,
        max_attempts: int = 8,
        backoff_base_s: float = 0.001,
        backoff_cap_s: float = 0.25,
        rng: random.Random | None = None,
    ):
        self.store = Store(path)
        self.isolation = isolation
        self.max_attempts = max_attempts
        self.backoff_base_s = backoff_base_s
        self.backoff_cap_s = backoff_cap_s
        self.effects = EffectRunner(self.store)
        self.stats = Stats()
        # Counter updates and jitter draws are not atomic under the GIL, and
        # the benchmark runs many agents at once. An unguarded ``+= 1`` loses
        # counts, which would understate the abort rate exactly where it
        # matters most.
        self._lock = threading.Lock()
        # Seeded separately from the workload so that jitter does not shift
        # which transactions conflict, only how long they wait.
        self._rng = rng or random.Random(0xACC)

    def run(
        self,
        body: Callable[[Transaction], Any],
        *,
        agent_id: str = "agent",
        isolation: Isolation | None = None,
        think_time_s: float = 0.0,
    ) -> TxnResult:
        """Execute ``body`` in a transaction, retrying on conflict.

        ``think_time_s`` holds the transaction open without doing work, which
        is what an inference call looks like to the store. It is a parameter
        rather than an accident of the caller because the whole premise of the
        project is that this duration is what turns a theoretical anomaly into
        a frequent one, and it needs to be something a benchmark can vary.
        """
        level = isolation or self.isolation
        started = time.perf_counter()
        conflicts: list[ConflictError] = []

        for attempt in range(1, self.max_attempts + 1):
            tx = Transaction(self.store, agent_id, level)
            with self._lock:
                self.stats.attempts += 1
            try:
                value = body(tx)
                if think_time_s:
                    time.sleep(think_time_s)
                commit_ts = self._commit(tx)
            except ConflictError as conflict:
                conflicts.append(conflict)
                with self._lock:
                    if conflict.kind == "write-write":
                        self.stats.write_write_conflicts += 1
                    else:
                        self.stats.read_write_conflicts += 1
                self._abort(tx)
                if attempt == self.max_attempts:
                    with self._lock:
                        self.stats.exhausted += 1
                    raise RetriesExhausted(attempt, conflict) from conflict
                self._sleep_backoff(attempt)
                continue
            except BaseException:
                self._abort(tx)
                raise

            tx.close()
            with self._lock:
                self.stats.committed += 1
            self.effects.drain()
            return TxnResult(
                value=value,
                commit_ts=commit_ts,
                attempts=attempt,
                conflicts=conflicts,
                elapsed_s=time.perf_counter() - started,
            )

        raise AssertionError("unreachable: the loop either returns or raises")

    def _commit(self, tx: Transaction) -> int:
        # Effects go in with the data, inside one exclusive section, so there
        # is no window where the data lands and the effect it owed is lost.
        return self.store.commit(
            agent_id=tx.agent_id,
            snapshot_ts=tx.snapshot_ts,
            read_set=tx.read_set,
            writes=tx.writes,
            deletes=tx.deletes,
            isolation=tx.isolation,
            effects=EffectRunner.rows(tx.deferred),
        )

    def _abort(self, tx: Transaction) -> None:
        count = len(tx.compensations)
        try:
            tx.rollback_effects()
        finally:
            with self._lock:
                self.stats.compensations_run += count
            tx.close()

    def _sleep_backoff(self, attempt: int) -> None:
        """Exponential backoff with full jitter.

        Without jitter, agents that collide once collide again on the same
        schedule, which is how a transient conflict becomes a standing one.
        """
        ceiling = min(self.backoff_cap_s, self.backoff_base_s * (2 ** (attempt - 1)))
        with self._lock:
            delay = self._rng.uniform(0, ceiling)
        time.sleep(delay)

    # ------------------------------------------------------------ utilities
    def get(self, key: str, default: Any = None) -> Any:
        """Read one key at the current clock, outside any transaction.

        ``default`` is returned for an absent or deleted key. A key holding
        ``None`` returns ``None``.
        """
        value, _ = self.store.read_at(key, self.store.now())
        return default if value is MISSING else value

    def set(self, key: str, value: Any, *, agent_id: str = "setup") -> int:
        """Write one key. For fixtures and setup, not for agent work."""
        return self.run(lambda tx: tx.put(key, value), agent_id=agent_id).commit_ts

    def history(self):
        return self.store.history()

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> Coordinator:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
