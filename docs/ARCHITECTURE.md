# Architecture

## The problem

Two agents read the same state, think for twenty seconds, and write. Neither
sees the other. The framework reports a coordination failure, but nothing
failed: both agents did exactly what they were told against a world that
changed while they were deciding.

These are the anomalies databases characterized decades ago. What makes them
urgent for agents is duration. A service transaction holds a read for
microseconds and overlaps with a competitor rarely. An agent holds a read
across an inference call, so overlap is the normal case rather than the
unlucky one. The `sweep` command measures exactly this: the same workload at a
0ms and a 50ms inference window moves the abort rate from roughly 16% to
roughly 58%.

Locking is the obvious fix and the wrong one. A lock held for the length of a
model call turns contention into a queue, and the agents serialize behind
whichever one is slowest. This layer uses optimistic concurrency instead:
agents run unblocked against a consistent snapshot, and the conflict is
discovered at commit, when it costs a retry rather than a wait.

## Shape

```
          agent code
              │  body(tx)
              ▼
    ┌───────────────────┐      retry on conflict
    │    Coordinator    │◄──────────────────────┐
    │  retry · backoff  │                       │
    └─────────┬─────────┘                       │
              │ begin                           │
              ▼                                 │
    ┌───────────────────┐                       │
    │    Transaction    │   read set  ──────────┤
    │  snapshot_ts      │   write buffer        │
    │  read set         │   effect queue        │
    │  write buffer     │                       │
    └─────────┬─────────┘                       │
              │ commit                          │
              ▼                                 │
    ┌───────────────────┐   ConflictError ──────┘
    │       Store       │
    │  validate         │
    │  append versions  │
    │  log transaction  │
    └─────────┬─────────┘
              │
      ┌───────┴────────┐
      ▼                ▼
┌───────────┐   ┌──────────────┐
│  SQLite   │   │ EffectRunner │
│  WAL      │   │ post-commit  │
└───────────┘   └──────────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ serializability  │
              │ checker (offline)│
              └──────────────────┘
```

`Coordinator` owns the retry loop. `Transaction` accumulates intent and touches
no storage. `Store` validates and appends atomically. `EffectRunner` dispatches
what was deferred. The checker reads the commit log afterwards and is never in
the request path.

## The commit protocol

```
agent A                         store                        agent B
   │                              │                              │
   ├── begin ────────────────────►│                              │
   │   snapshot_ts = 7            │                              │
   │◄── snapshot ─────────────────┤                              │
   │                              │◄──────────── begin ──────────┤
   ├── get("budget") ────────────►│              snapshot_ts = 7 │
   │◄── 100 @ v5 ─────────────────┤                              │
   │                              │◄──── get("budget") ──────────┤
   │   [inference, 20s]           ├───── 100 @ v5 ──────────────►│
   │                              │                              │
   │                              │◄──── commit ─────────────────┤
   │                              │  validate: ok, append @ 8    │
   │                              ├───── committed @ 8 ─────────►│
   │                              │                              │
   ├── commit ───────────────────►│                              │
   │   validate: budget @ 8 > 7   │                              │
   │◄── ConflictError ────────────┤                              │
   │                              │                              │
   ├── retry from snapshot 8 ────►│                              │
```

Validation, append, and the effect queue run inside one **exclusive** section
(`Store._exclusive`), which holds the database write lock from the first read
to the last insert. Two committers therefore cannot validate against the same
state and both append, and that holds across processes rather than only across
threads. The in-process lock around it avoids contending for that write lock
and retrying on `SQLITE_BUSY`; it is an optimization, not the thing that makes
commits correct.

The section opens with `IMMEDIATE` rather than `EXCLUSIVE`, because in WAL
mode an exclusive transaction also locks out readers, and every snapshot read
would then queue behind every commit. `IMMEDIATE` takes the write lock and
leaves readers running.

A transaction that writes nothing and queues no effect changes no state, so it
is validated and then returns without taking a timestamp or appending to the
log. Allocating one would advance the clock and add an empty row to the
history the checker reads, for a transaction that did nothing.

## Data model

Storage is append-only. A write never updates a row.

| Table | Purpose |
|---|---|
| `meta` | The logical clock. A commit takes `clock + 1`; a reader takes `clock` as its snapshot. |
| `versions` | `(key, commit_ts) → value`. Every version of every key, with tombstones for deletes. |
| `transactions` | The commit log: read set, write set, snapshot, isolation. |
| `effects` | Deferred side effects with status and idempotency key. |

A read at snapshot `s` is the newest version of the key at or below `s`. The
reader records the *timestamp* of the version it saw, not the value. Comparing
values would treat a rewrite of the same value as no conflict, when it has in
fact replaced the version the decision rested on.

Absence is a version too. A key that has never been written reads as version
`0`, and that goes in the read set. An agent that checks for a lock file and
proceeds because there was none has read something, and leaving that out would
make the most common agent guard invisible to validation.

## Isolation levels

Validation at commit, given snapshot `s`:

| Level | Write set checked | Read set checked |
|---|---|---|
| `NONE` | no | no |
| `SNAPSHOT` | yes | no |
| `SERIALIZABLE` | yes | yes |

A check fails when the key's newest version is above `s`.

`NONE` is the baseline, and it is what a framework does today when it hands
agents a shared table: read the newest value, write when you finish.

`SNAPSHOT` stops two agents overwriting each other, and gives consistent
multi-key reads for free, because every read comes from one timestamp.

`SERIALIZABLE` adds read validation, which is the only way to catch an agent
that acted on a premise that changed. Write-set validation cannot find write
skew by construction: neither agent wrote a key the other wrote.

Write conflicts are checked before read conflicts. When a transaction both
read and wrote a key, write-write is the more precise description, and
reporting read-write would send whoever reads the benchmark looking for an
anti-dependency that is not the cause.

## Side effects

Databases never had to solve this. A write can be withheld until commit; a
sent email cannot be recalled. Since any transaction can abort, an effect has
to be expressible in one of two ways:

| Kind | When it runs | On abort | Use for |
|---|---|---|---|
| `defer` | after commit | never ran | notifications, downstream dispatch |
| `compensable` | immediately | inverse runs | reservations, holds, drafts |
| `immediate` | immediately | **refused by default** | nothing, unless forced |

The refusal is deliberate. An agent that must take an irreversible action
mid-transaction is describing a transaction boundary in the wrong place.
Allowing it silently would make the layer's guarantee true for storage and
false for everything the agent actually did.

Deferred effects are rows, not callbacks, because the queue outlives the
process. They are inserted inside the same exclusive section that commits the
data, so either both land or neither does, and there is no window where the
data is durable and the effect it owed is gone.

Delivery is **at-least-once**. The remaining cause is a crash during a
handler: the row stays claimed, `recover()` returns it to the queue on
startup, and the handler runs again. Handlers get an idempotency key for that.
No layer can promise exactly-once against a world that does not deduplicate,
and claiming otherwise would push the hard part onto the caller without
telling them.

## Checking instead of asserting

Asserting final values is weak. It catches the anomalies you thought of, in
the interleavings you happened to produce, and says nothing about a run where
the damage happened and the arithmetic came out even.

The checker reconstructs the dependency graph from the commit log and searches
for a cycle. A history is conflict-serializable exactly when that graph is
acyclic, so the result is a proof obligation rather than an opinion, and when
it fails the evidence is the cycle itself.

Edges from `T1` to `T2`:

- **ww** — both wrote key *k*, and `T1` committed first
- **wr** — `T2` read the version of *k* that `T1` wrote
- **rw** — `T1` read a version of *k* that `T2` then overwrote

The anti-dependency is why this exists. Write skew produces no write-write
conflict and no stale read that snapshot isolation can see, but it produces
two anti-dependencies pointing opposite ways, which is a cycle.

The checker never asks the store whether it behaved. It asks whether the
history could have come from running those transactions one at a time.

## Concurrency inside the implementation

- One SQLite connection per thread, held in thread-local storage. Connections
  are not thread-safe.
- WAL mode, so snapshot reads proceed while a committer holds the write lock.
  Without it every read would queue behind commits and the benchmark would
  measure SQLite's lock contention rather than the concurrency control.
- `Coordinator` guards its counters and its jitter generator with a lock.
  `+= 1` is not atomic under the GIL, and an unguarded counter understates the
  abort rate exactly where it matters.
- Backoff is exponential with full jitter. Without jitter, agents that collide
  once collide again on the same schedule, and a transient conflict becomes a
  standing one.
- `Coordinator.run` takes a function rather than exposing begin/commit,
  because a transaction that may be retried must be re-runnable. An API that
  hands out a transaction object invites setup outside it that silently does
  not get redone.

`Store(":memory:")` is a temporary file, not SQLite's in-memory database.
SQLite's `:memory:` is private to one connection, and the shared-cache
workaround takes table-level locks that fail a concurrent reader outright. The
temporary file gives what the name promises: many readers alongside one
writer.

## Boundaries

**This is a coordination layer, not a storage engine.** It sits on SQLite and
delegates durability. Swapping in Postgres means reimplementing `Store`
against `SELECT ... FOR UPDATE` or its native snapshot isolation, and nothing
above it changes.

**Point reads only.** There are no range scans, so there is no phantom
protection. Validating a predicate requires tracking the predicate, not the
keys, and that is a materially different design. For the key-addressed state
agents actually share, point reads are the access pattern.

**Validation is backward-oriented.** A long transaction competing with many
short ones can starve, and the retry budget bounds the damage rather than
removing it. Forward-oriented validation would abort the short transactions
instead, which for agents is the wrong trade: the long transaction is the one
holding an expensive inference call.

**Single writer per commit.** Commits serialize on the SQLite write lock,
which caps write throughput at what one machine can sequence. The workloads
here run far below that ceiling, and correctness, not throughput, is what this
layer is measured on.
