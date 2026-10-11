"""Side effects that outlive the transaction.

Withholding a write until commit is easy. Recalling a sent email is not, and a
transaction can always abort. So an effect is either **deferred** until commit
or **compensable** with an inverse that runs on abort. Anything else is
refused, because an agent that must act irreversibly mid-transaction is
describing a transaction boundary in the wrong place.

Effects are queued inside the same exclusive section that commits the data.
Delivery is at-least-once: a crash during a handler leaves the row claimed,
and recovery runs it again.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .store import Store


@dataclass
class DeferredEffect:
    """An effect queued to run after the transaction commits."""

    name: str
    payload: dict[str, Any]
    idempotency_key: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class CompensableEffect:
    """An effect already performed, with the inverse that undoes it."""

    name: str
    undo: Callable[[], None]


class EffectRunner:
    """Dispatches deferred effects to registered handlers.

    Handlers are looked up by name rather than held as closures, because the
    queue outlives the process that enqueued it. On recovery there is a row in
    a table, not a Python object, and a name is the only thing that survives
    that trip.
    """

    def __init__(self, store: Store):
        self.store = store
        self._handlers: dict[str, Callable[[dict[str, Any]], None]] = {}
        self.recover()

    def recover(self) -> None:
        """Return claimed-but-unfinished effects to the queue.

        A row left in ``running`` is one whose process died between claiming
        it and recording the outcome. On startup it has to go back, or the
        effect is owed forever and nothing will ever notice.

        This assumes one runner per database at a time. A second runner
        starting while the first has effects in flight would hand them back to
        the queue and cause a duplicate delivery, which is the one case the
        idempotency key exists to absorb.
        """
        self.store.conn.execute("UPDATE effects SET status = 'pending' WHERE status = 'running'")

    def register(self, name: str, handler: Callable[[dict[str, Any]], None]) -> None:
        self._handlers[name] = handler

    @staticmethod
    def rows(effects: list[DeferredEffect]) -> list[tuple[str, str, str]]:
        """Flatten effects into rows the store can insert during a commit.

        The store stays free of this module: it takes plain tuples, which is
        what keeps the dependency pointing one way.
        """
        return [
            (effect.idempotency_key, effect.name, json.dumps(effect.payload)) for effect in effects
        ]

    def _claim(self) -> list[tuple[str, str, str]]:
        """Take exclusive ownership of the pending effects, atomically.

        Selecting and then dispatching is a race: every agent drains after its
        own commit, so two threads can read the same pending row and both call
        the handler. Claiming flips the rows to ``running`` inside the same
        exclusive section that read them, so exactly one caller wins
        and the rest see nothing to do.
        """
        conn = self.store.conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            rows = conn.execute(
                "SELECT id, name, payload FROM effects WHERE status = 'pending' ORDER BY commit_ts"
            ).fetchall()
            claimable = [r for r in rows if r[1] in self._handlers]
            if claimable:
                placeholders = ",".join("?" * len(claimable))
                conn.execute(
                    "UPDATE effects SET status = 'running', attempts = attempts + 1, "
                    f"updated_at = ? WHERE id IN ({placeholders})",
                    (time.time(), *[r[0] for r in claimable]),
                )
            conn.execute("COMMIT")
            return claimable
        except BaseException:
            conn.execute("ROLLBACK")
            raise

    def drain(self) -> int:
        """Run every pending effect. Returns how many succeeded.

        An effect is marked done only after its handler returns, so a process
        that dies mid-handler leaves the row claimed and :meth:`recover`
        returns it to the queue on restart. Handlers must therefore tolerate
        being called twice with the same idempotency key: this is
        at-least-once, and no layer can promise exactly-once against a world
        that does not deduplicate.
        """
        conn = self.store.conn
        done = 0
        for effect_id, name, payload in self._claim():
            handler = self._handlers[name]
            try:
                handler(json.loads(payload))
            except Exception:
                conn.execute(
                    "UPDATE effects SET status = 'failed', updated_at = ? WHERE id = ?",
                    (time.time(), effect_id),
                )
                continue
            conn.execute(
                "UPDATE effects SET status = 'done', updated_at = ? WHERE id = ?",
                (time.time(), effect_id),
            )
            done += 1
        return done

    def pending(self) -> int:
        """Effects still owed, including any claimed but not yet finished."""
        row = self.store.conn.execute(
            "SELECT COUNT(*) FROM effects WHERE status IN ('pending', 'running')"
        ).fetchone()
        return int(row[0])
