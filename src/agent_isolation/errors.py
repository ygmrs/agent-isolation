"""Failure modes callers are expected to handle.

``ConflictError`` is not an error in the usual sense. Under optimistic
concurrency a conflict is the normal outcome of contention, and the
coordinator retries rather than surfacing it. It reaches the caller only once
the retry budget is spent.
"""

from __future__ import annotations


class ConcurrencyControlError(Exception):
    """Base class for everything raised by this package."""


class ConflictError(ConcurrencyControlError):
    """Validation rejected the transaction at commit time.

    ``kind`` is ``"write-write"`` or ``"read-write"``. The distinction matters
    when reading benchmark output: write-write conflicts mean two agents
    changed the same key, while read-write conflicts mean one agent decided
    based on a value another agent has since changed. Only the second kind is
    detected under serializable isolation, and it is the one that long
    inference windows make common.
    """

    def __init__(self, key: str, kind: str, snapshot_ts: int, conflicting_ts: int):
        self.key = key
        self.kind = kind
        self.snapshot_ts = snapshot_ts
        self.conflicting_ts = conflicting_ts
        super().__init__(
            f"{kind} conflict on {key!r}: snapshot at {snapshot_ts}, "
            f"conflicting commit at {conflicting_ts}"
        )


class RetriesExhausted(ConcurrencyControlError):
    """The transaction conflicted more times than the retry budget allowed."""

    def __init__(self, attempts: int, last: ConflictError):
        self.attempts = attempts
        self.last = last
        super().__init__(f"gave up after {attempts} attempts; last was {last}")


class UnsafeEffectError(ConcurrencyControlError):
    """An irreversible side effect was attempted inside a transaction.

    A transaction can always abort, so an effect that can neither be deferred
    until after commit nor compensated on abort has no correct place to run.
    Rather than let it through and hope, the layer refuses it and names the two
    ways to express it safely.
    """


class TransactionClosed(ConcurrencyControlError):
    """The transaction object was used after it committed or aborted."""
