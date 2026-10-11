"""The handle an agent works through.

A transaction accumulates the versions it read, the values it intends to
write, and the effects it wants to happen. Nothing touches storage until
commit, which is what makes an abort free.

The read set records the *version* behind every value seen, including the
absence of a key, because "this did not exist" is a premise a decision can
rest on.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .effects import CompensableEffect, DeferredEffect
from .errors import TransactionClosed, UnsafeEffectError
from .store import MISSING, Isolation, Store


class Transaction:
    """A unit of agent work, isolated from other agents until it commits."""

    def __init__(self, store: Store, agent_id: str, isolation: Isolation):
        self.store = store
        self.agent_id = agent_id
        self.isolation = isolation
        self.snapshot_ts = store.now()
        self.read_set: dict[str, int] = {}
        self.writes: dict[str, Any] = {}
        self.deletes: set[str] = set()
        self.deferred: list[DeferredEffect] = []
        self._compensable: list[CompensableEffect] = []
        self._closed = False

    def _check_open(self) -> None:
        if self._closed:
            raise TransactionClosed("this transaction has already finished")

    # ----------------------------------------------------------------- data
    def get(self, key: str, default: Any = None) -> Any:
        """Read at the snapshot, recording the version seen.

        ``default`` is returned only when the key is absent or deleted. A key
        holding ``None`` returns ``None``.

        Reads of keys this transaction has already written come from its own
        buffer. An agent that writes a value and reads it back must see what it
        wrote, and under ``NONE`` that is the only read-your-writes guarantee
        there is.
        """
        self._check_open()
        if key in self.deletes:
            return default
        if key in self.writes:
            return self.writes[key]

        if self.isolation is Isolation.NONE:
            # The baseline reads whatever is newest, which is exactly how an
            # agent framework backed by a plain table behaves.
            value, version = self.store.read_at(key, self.store.now())
        else:
            value, version = self.store.read_at(key, self.snapshot_ts)

        # Only the first read of a key is recorded. A later read returns the
        # same version by construction, and overwriting the entry would let a
        # repeat read quietly refresh the premise validation is meant to check.
        self.read_set.setdefault(key, version)
        # MISSING, not None: an agent may have stored a null deliberately, and
        # returning the default for it would hide a value that is really there.
        return default if value is MISSING else value

    def put(self, key: str, value: Any) -> None:
        self._check_open()
        self.deletes.discard(key)
        self.writes[key] = value

    def delete(self, key: str) -> None:
        self._check_open()
        self.writes.pop(key, None)
        self.deletes.add(key)

    def keys_written(self) -> set[str]:
        return set(self.writes) | self.deletes

    # -------------------------------------------------------------- effects
    def defer(self, name: str, payload: dict[str, Any], *, key: str | None = None) -> None:
        """Queue an effect to run after this transaction commits."""
        self._check_open()
        effect = DeferredEffect(name=name, payload=payload)
        if key is not None:
            effect.idempotency_key = key
        self.deferred.append(effect)

    def compensable(self, name: str, do: Callable[[], Any], undo: Callable[[], None]) -> Any:
        """Perform an effect now and register the inverse to run on abort.

        ``do`` runs immediately and its result is returned, so the agent can
        act on it. If the transaction later aborts, ``undo`` runs. Register the
        inverse before performing the action, so a failure inside ``do`` that
        has already changed something is still covered.
        """
        self._check_open()
        self._compensable.append(CompensableEffect(name=name, undo=undo))
        return do()

    def immediate(self, name: str, do: Callable[[], Any], *, allow_unsafe: bool = False) -> Any:
        """Perform an irreversible effect now. Refused unless forced.

        The refusal is the feature. A transaction can abort after this point,
        and an effect with no inverse and no deferral leaves the world ahead of
        the state that justified it.
        """
        self._check_open()
        if not allow_unsafe:
            raise UnsafeEffectError(
                f"effect {name!r} is irreversible and cannot run inside a transaction. "
                "Use defer() to run it after commit, or compensable() to register an "
                "inverse. Pass allow_unsafe=True only if a duplicate or orphaned effect "
                "is acceptable here."
            )
        return do()

    @property
    def compensations(self) -> list[CompensableEffect]:
        """Inverses registered so far, oldest first."""
        return list(self._compensable)

    def rollback_effects(self) -> None:
        """Run registered inverses in reverse order. Called on abort."""
        for effect in reversed(self._compensable):
            effect.undo()
        self._compensable.clear()

    def close(self) -> None:
        self._closed = True
