"""Multi-version storage and the commit protocol.

Every version of every key is kept, which is what makes a snapshot possible: a
reader fixes a timestamp and sees the same state for the length of its run,
however long that run takes.

Validation is optimistic. Locking would mean an agent holding a row lock for
the length of a model call, which turns contention into a queue and serializes
every agent behind the slowest one.

Validation, append, and the effect queue run in one exclusive section holding
SQLite's write lock, so a commit is atomic across processes and not only
across threads.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .errors import ConflictError


class _Missing:
    """The absence of a key, which is not the same as a stored null.

    An agent can legitimately store ``None``, and a guard that checks whether
    a key exists has to be able to tell that apart from a key holding null.
    Collapsing the two made ``get(key, default)`` return the default for a
    value that was really there.
    """

    _instance: _Missing | None = None

    def __new__(cls) -> _Missing:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "MISSING"

    def __bool__(self) -> bool:
        return False


MISSING = _Missing()

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    id    INTEGER PRIMARY KEY CHECK (id = 0),
    clock INTEGER NOT NULL
);
INSERT OR IGNORE INTO meta (id, clock) VALUES (0, 0);

-- Append-only. A write never updates a row, it adds one at a higher timestamp.
CREATE TABLE IF NOT EXISTS versions (
    key       TEXT    NOT NULL,
    commit_ts INTEGER NOT NULL,
    value     TEXT,
    deleted   INTEGER NOT NULL DEFAULT 0,
    writer    TEXT,
    PRIMARY KEY (key, commit_ts)
);
CREATE INDEX IF NOT EXISTS idx_versions_key_ts ON versions (key, commit_ts DESC);

-- The commit log is what the serializability checker reads. Without the read
-- sets recorded here, anti-dependencies cannot be reconstructed after the fact
-- and the only available evidence of an anomaly is the corrupted value itself.
CREATE TABLE IF NOT EXISTS transactions (
    commit_ts   INTEGER PRIMARY KEY,
    agent_id    TEXT    NOT NULL,
    snapshot_ts INTEGER NOT NULL,
    read_set    TEXT    NOT NULL,
    write_set   TEXT    NOT NULL,
    isolation   TEXT    NOT NULL,
    wall_clock  REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS effects (
    id         TEXT PRIMARY KEY,
    commit_ts  INTEGER,
    name       TEXT    NOT NULL,
    payload    TEXT    NOT NULL,
    status     TEXT    NOT NULL,
    attempts   INTEGER NOT NULL DEFAULT 0,
    updated_at REAL    NOT NULL
);
"""


class Isolation(StrEnum):
    """What the store checks before letting a transaction commit.

    ``NONE`` is the baseline the benchmark compares against, and it is what a
    multi-agent framework does today when it hands agents a shared dictionary
    or a table: read the newest value, write whenever you finish. It is
    included so the anomalies have something to happen to.

    ``SNAPSHOT`` validates the write set, which stops two agents overwriting
    each other. It does not stop two agents reading the same state and each
    writing somewhere the other did not look.

    ``SERIALIZABLE`` validates the read set as well, so a transaction that
    decided based on a value that has since changed is rejected. This is the
    level that costs throughput, and the benchmark reports how much.
    """

    NONE = "none"
    SNAPSHOT = "snapshot"
    SERIALIZABLE = "serializable"


@dataclass(frozen=True)
class CommitRecord:
    """One committed transaction, as the checker sees it."""

    commit_ts: int
    agent_id: str
    snapshot_ts: int
    read_set: dict[str, int]
    write_set: list[str]
    isolation: str
    wall_clock: float


class Store:
    """Versioned key-value storage with an optimistic commit protocol."""

    def __init__(self, path: str | Path = ":memory:"):
        self._commit_lock = threading.Lock()
        self._local = threading.local()
        self._tempdir: tempfile.TemporaryDirectory | None = None

        if str(path) == ":memory:":
            # Not a real in-memory database. SQLite's ``:memory:`` is private
            # to one connection, and the shared-cache workaround takes
            # table-level locks that make a concurrent reader fail outright
            # while another connection writes. A temporary file in WAL mode
            # gives the behaviour the name promises: many readers alongside
            # one writer, which is the whole access pattern here. It is
            # removed on close.
            self._tempdir = tempfile.TemporaryDirectory(prefix="agent-isolation-")
            self.path = str(Path(self._tempdir.name) / "store.db")
        else:
            self.path = str(path)

        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        # WAL is what lets a reader proceed while a committer holds the write
        # lock. Without it every snapshot read would queue behind commits and
        # the benchmark would measure SQLite's lock contention rather than the
        # concurrency control being tested.
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        """One connection per thread. SQLite connections are not thread-safe."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
        return conn

    # ---------------------------------------------------------------- reads
    def now(self) -> int:
        """The current logical clock, which is the snapshot a reader gets."""
        row = self.conn.execute("SELECT clock FROM meta WHERE id = 0").fetchone()
        return int(row[0])

    def read_at(self, key: str, snapshot_ts: int) -> tuple[Any, int]:
        """Newest version of ``key`` at or below ``snapshot_ts``.

        Returns the value and the timestamp of the version actually read, or
        ``MISSING`` when the key has no version at that point or its newest
        one is a tombstone. ``MISSING`` rather than ``None`` because a stored
        null is a value an agent may have meant.

        The caller records the timestamp rather than the value, because
        validation asks whether the version changed. Comparing values would
        call a write that happened to restore the same value no conflict, when
        it replaced the version the read rested on.
        """
        row = self.conn.execute(
            "SELECT value, deleted, commit_ts FROM versions "
            "WHERE key = ? AND commit_ts <= ? ORDER BY commit_ts DESC LIMIT 1",
            (key, snapshot_ts),
        ).fetchone()
        if row is None:
            return MISSING, 0
        value, deleted, commit_ts = row
        if deleted:
            return MISSING, int(commit_ts)
        return json.loads(value), int(commit_ts)

    def latest_ts(self, key: str, conn: sqlite3.Connection | None = None) -> int:
        """Timestamp of the newest version of ``key``, or 0 if it has none."""
        cur = (conn or self.conn).execute(
            "SELECT commit_ts FROM versions WHERE key = ? ORDER BY commit_ts DESC LIMIT 1",
            (key,),
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0

    @contextmanager
    def _exclusive(self, conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
        """Hold the database write lock for the body of the block.

        ``IMMEDIATE`` rather than ``EXCLUSIVE``: in WAL mode an exclusive
        transaction also locks out readers, which would make every snapshot
        read queue behind every commit. ``IMMEDIATE`` takes the write lock and
        leaves readers running, which is the access pattern here.
        """
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")

    # --------------------------------------------------------------- commit
    def commit(
        self,
        *,
        agent_id: str,
        snapshot_ts: int,
        read_set: dict[str, int],
        writes: dict[str, Any],
        deletes: set[str],
        isolation: Isolation,
        effects: Sequence[tuple[str, str, str]] = (),
    ) -> int:
        """Validate and append, exclusively. Returns the commit timestamp.

        ``effects`` are ``(id, name, payload_json)`` rows queued in the same
        exclusive section as the data. Writing them afterwards would leave a
        window where a crash commits the data and loses the effect that was
        owed, with nothing left to say it was ever promised.

        A transaction that writes nothing and queues nothing changes no state,
        so it takes no timestamp and is not logged. It is still validated,
        because an agent that read stale data should be told even when it
        decided not to write. Without this, every read-only transaction would
        advance the clock and append an empty row to the commit log.

        Raises :class:`ConflictError` and writes nothing if validation fails.
        """
        write_keys = set(writes) | deletes
        read_only = not write_keys and not effects

        with self._commit_lock, self._exclusive(self.conn) as conn:
            if isolation is not Isolation.NONE:
                self._validate(conn, snapshot_ts, read_set, write_keys, isolation)
            if read_only:
                return snapshot_ts

            clock = int(conn.execute("SELECT clock FROM meta WHERE id = 0").fetchone()[0])
            commit_ts = clock + 1

            for key, value in writes.items():
                conn.execute(
                    "INSERT INTO versions (key, commit_ts, value, deleted, writer) "
                    "VALUES (?, ?, ?, 0, ?)",
                    (key, commit_ts, json.dumps(value), agent_id),
                )
            for key in deletes:
                conn.execute(
                    "INSERT INTO versions (key, commit_ts, value, deleted, writer) "
                    "VALUES (?, ?, NULL, 1, ?)",
                    (key, commit_ts, agent_id),
                )
            for effect_id, name, payload in effects:
                conn.execute(
                    "INSERT OR IGNORE INTO effects "
                    "(id, commit_ts, name, payload, status, attempts, updated_at) "
                    "VALUES (?, ?, ?, ?, 'pending', 0, ?)",
                    (effect_id, commit_ts, name, payload, time.time()),
                )
            conn.execute("UPDATE meta SET clock = ? WHERE id = 0", (commit_ts,))
            conn.execute(
                "INSERT INTO transactions "
                "(commit_ts, agent_id, snapshot_ts, read_set, write_set, isolation, wall_clock)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    commit_ts,
                    agent_id,
                    snapshot_ts,
                    json.dumps(read_set),
                    json.dumps(sorted(write_keys)),
                    isolation.value,
                    time.time(),
                ),
            )
            return commit_ts

    def _validate(
        self,
        conn: sqlite3.Connection,
        snapshot_ts: int,
        read_set: dict[str, int],
        write_keys: set[str],
        isolation: Isolation,
    ) -> None:
        """Reject the transaction if anything it depended on has moved.

        Write validation is checked first. When a transaction both read and
        wrote a key, the write-write conflict is the more precise description
        of what went wrong, and reporting it as read-write would send whoever
        reads the benchmark looking for an anti-dependency that is not the
        cause.
        """
        for key in sorted(write_keys):
            latest = self.latest_ts(key, conn)
            if latest > snapshot_ts:
                raise ConflictError(key, "write-write", snapshot_ts, latest)

        if isolation is Isolation.SERIALIZABLE:
            for key in sorted(read_set):
                if key in write_keys:
                    continue
                latest = self.latest_ts(key, conn)
                if latest > snapshot_ts:
                    raise ConflictError(key, "read-write", snapshot_ts, latest)

    # ------------------------------------------------------------- history
    def history(self) -> list[CommitRecord]:
        """Every committed transaction, oldest first."""
        rows = self.conn.execute(
            "SELECT commit_ts, agent_id, snapshot_ts, read_set, write_set, isolation, wall_clock "
            "FROM transactions ORDER BY commit_ts"
        ).fetchall()
        return [
            CommitRecord(
                commit_ts=int(r[0]),
                agent_id=r[1],
                snapshot_ts=int(r[2]),
                read_set={k: int(v) for k, v in json.loads(r[3]).items()},
                write_set=list(json.loads(r[4])),
                isolation=r[5],
                wall_clock=float(r[6]),
            )
            for r in rows
        ]

    def snapshot_of_all(self) -> dict[str, Any]:
        """Current committed value of every live key. For assertions and demos."""
        ts = self.now()
        keys = [r[0] for r in self.conn.execute("SELECT DISTINCT key FROM versions").fetchall()]
        out: dict[str, Any] = {}
        for key in keys:
            value, _ = self.read_at(key, ts)
            if value is not MISSING:
                out[key] = value
        return out

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
        if self._tempdir is not None:
            self._tempdir.cleanup()
            self._tempdir = None
