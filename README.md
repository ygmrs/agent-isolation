# agent-isolation

Concurrency control for multi-agent systems: snapshot isolation, write-set conflict detection, and compensating rollback for side effects already taken.

Two agents read the same state, think, and write. Neither sees the other. The framework reports a coordination failure, but nothing failed: both agents did what they were told against a world that changed while they were deciding.

These are the anomalies databases characterized decades ago. What makes them urgent for agents is **duration**. A service transaction holds a read for microseconds and rarely overlaps. An agent holds a read across an inference call, so overlap is the normal case.

```bash
uv sync
uv run aiso anomalies     # the four anomalies at every isolation level
uv run aiso bench         # throughput, abort rate, lost updates
uv run aiso decisions     # agents deciding from shared state they do not write
uv run aiso sweep         # abort rate as the inference window grows
uv run pytest             # 43 tests
```

No runtime dependencies. Storage is SQLite from the standard library.

## The four anomalies

```
uv run aiso anomalies
```

| Anomaly | `none` | `snapshot` | `serializable` |
|---|---|---|---|
| **lost update** — two agents increment a counter, one increment vanishes | occurs | prevented | prevented |
| **inconsistent outcome** — an agent reads two keys mid-move and totals a state that never existed | occurs | prevented | prevented |
| **stale read** — an agent approves against a budget that changed while it was thinking | occurs | **occurs** | prevented |
| **write skew** — two on-call agents each see the other is covering, both go off duty | occurs | **occurs** | prevented |

Each scenario drives an exact interleaving by hand rather than launching threads and hoping. A concurrency test that depends on timing passes on a fast machine and fails in CI.

The two bold cells are the argument for read validation. In both, neither agent writes a key the other writes, so no write-set check can find them however carefully it is implemented.

## What it costs and what it buys

8 agents, 30 transactions each, 4 shared counters, 2ms inference window.

| Isolation | Commits | Attempts | Abort | **Lost** | txn/s | p99 ms | Serializable |
|---|---|---|---|---|---|---|---|
| none | 240 | 240 | 0.0% | **147** | 1,918 | 9.5 | **NO** |
| snapshot | 240 | 495 | 51.5% | **0** | 697 | 76.8 | yes |
| serializable | 240 | 434 | 44.7% | **0** | 674 | 98.5 | yes |

**Without concurrency control, 147 of 240 committed increments were silently overwritten.** Nothing errored. No tool failed. The agents reported success.

`lost` is measured, not inferred: every commit added one, so the counters must add up. `serializable` is decided by the dependency-graph checker, not by inspecting values.

On this workload both levels validate identically, since the only key read is also written, so the gap between their abort rates is scheduling noise. The decision workload below is where they diverge.

### The workload agents actually run

Most agent work reads shared context, thinks, and writes its result somewhere else. Nobody writes the key they read, so there is no write-write conflict to find.

| Isolation | Decisions | Abort | **Stale** | txn/s |
|---|---|---|---|---|
| none | 240 | 0.0% | **95 (39.6%)** | 1,045 |
| snapshot | 240 | 0.0% | **96 (40.0%)** | 619 |
| serializable | 240 | 27.4% | **0 (0.0%)** | 526 |

Snapshot isolation rejects nothing here and two in five decisions commit against a budget that had already changed. Read validation removes them, and the throughput column is its price.

### Why agents and not ordinary services

| Inference window | Commits | Attempts | Abort rate | txn/s |
|---|---|---|---|---|
| 0ms | 240 | 338 | 29.0% | 2,197 |
| 2ms | 240 | 461 | 47.9% | 682 |
| 10ms | 240 | 490 | 51.0% | 174 |
| 50ms | 240 | 582 | 58.8% | 47 |

The longer an agent holds a read, the more often the world moves underneath it. A real inference call is seconds, not milliseconds.

Thread scheduling moves these numbers between runs; the shape does not. Lost updates and stale decisions are exact in every run.

## Every run has to prove itself

The checker rebuilds the dependency graph from the commit log and looks for a cycle. No cycle means the run could have happened one transaction at a time. A cycle means it couldn't, and you get the cycle.

Checking final values only finds what you thought to look for. It says nothing about the run where the damage happened and the numbers still added up.

```
$ uv run aiso check --isolation none

history check · none · 240 committed transactions

NOT serializable: cycle T214 -> T219 -> T214

lost updates: 151
abort rate:   0.0%
```

**The checker never asks the store whether it behaved.** It asks whether the history could have come from running those transactions one at a time.

## Side effects

Databases never had to solve this. A write can be withheld until commit; a sent email cannot be recalled. Since any transaction can abort, an effect has to be one of two things:

| Kind | Runs | On abort | For |
|---|---|---|---|
| `tx.defer(...)` | after commit | never ran | notifications, downstream dispatch |
| `tx.compensable(do, undo)` | immediately | inverse runs | reservations, holds, drafts |
| `tx.immediate(...)` | immediately | **refused by default** | nothing, unless forced |

The refusal is the feature. An agent that must take an irreversible action mid-transaction is describing a transaction boundary in the wrong place, and allowing it silently would make the guarantee true for storage and false for everything the agent did.

Deferred effects are queued inside the same exclusive section that commits the data, so either both land or neither does. Delivery is at-least-once, because a crash during a handler leaves the row claimed and recovery runs it again, and every effect carries an idempotency key.

## Using it

```python
from agent_isolation import Coordinator, Isolation

with Coordinator("state.db", isolation=Isolation.SERIALIZABLE) as coord:
    coord.effects.register("notify", lambda payload: send_email(**payload))

    def approve(tx):
        budget = tx.get("policy:budget", 0)
        if budget < 50:
            return "declined"
        tx.put("policy:budget", budget - 50)
        tx.put("approval:1042", {"amount": 50})
        tx.defer("notify", {"approval": 1042})
        return "approved"

    result = coord.run(approve, agent_id="approver", think_time_s=0.0)
    print(result.value, result.attempts)
```

`run` takes a function rather than handing out a transaction, because a transaction that may be retried has to be re-runnable. Conflicts are retried with exponential backoff and full jitter; without jitter, agents that collide once collide again on the same schedule.

A worked example with eight agents competing for four seats:

```bash
uv run python examples/booking_agent.py
```

```
  none          booked=8  remaining=3  confirmations=8  oversold=4
  snapshot      booked=4  remaining=0  confirmations=4  oversold=0
  serializable  booked=4  remaining=0  confirmations=4  oversold=0
```

## Scope

This is **isolation**, not durability. It makes concurrent agent runs correct
against shared state: snapshot reads, conflict detection, compensation. Making
a single run survive a crash — leases, retries, recovery — is a separate
problem, and [`agent-execution-plane`](https://github.com/ygmrs/agent-execution-plane)
is where that lives.

## Design

[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) covers the commit protocol, the data model, the validation rules per isolation level, the effect model, the serializability checker, and the boundaries of the design.

```
src/agent_isolation/
  store.py            multi-version storage and the commit protocol
  transaction.py      read set, write buffer, effect queue
  coordinator.py      retry loop, backoff, statistics
  effects.py          deferred and compensating side effects
  serializability.py  dependency graph and cycle detection
  anomalies.py        the four anomalies as runnable scenarios
  bench.py            workloads and measurement
  cli.py              anomalies | bench | decisions | sweep | check
```

Validation, append, and the effect queue run in one **exclusive** section holding SQLite's write lock, so a commit is atomic across processes and not only across threads. Reads run in WAL mode and proceed while a committer holds that lock.

## License

MIT — see [LICENSE](LICENSE).
