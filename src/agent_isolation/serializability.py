"""Checking a history instead of trusting the implementation.

Asserting final values catches the anomalies you thought of, in the
interleaving you happened to produce, and says nothing about a run where the
damage happened and the numbers came out even.

This reconstructs the dependency graph from the commit log and searches for a
cycle. A history is conflict-serializable exactly when that graph is acyclic,
so the evidence on failure is the cycle itself.

Edges are ww (both wrote k), wr (read what the other wrote), and rw (read a
version the other than overwrote). The rw anti-dependency is why this exists:
write skew produces no write conflict and no stale read, but it produces two
of them pointing opposite ways, which is a cycle.

Three kinds of edge, from T1 to T2:

- **write-write**: both wrote key k, and T1 committed first.
- **write-read**: T2 read the version of k that T1 wrote.
- **read-write** (anti-dependency): T1 read a version of k that T2 then
  overwrote. T1 must be ordered first because it saw the world before T2's
  change.
"""

from __future__ import annotations

from dataclasses import dataclass

from .store import CommitRecord

_NO_RECORD = CommitRecord(
    commit_ts=-1,
    agent_id="",
    snapshot_ts=-1,
    read_set={},
    write_set=[],
    isolation="",
    wall_clock=0.0,
)


@dataclass(frozen=True)
class Edge:
    frm: int
    to: int
    kind: str
    key: str

    def __str__(self) -> str:
        return f"T{self.frm} -{self.kind}({self.key})-> T{self.to}"


@dataclass
class CheckResult:
    """Whether the history could have come from running one at a time."""

    serializable: bool
    edges: list[Edge]
    cycle: list[int]
    cycle_edges: list[Edge]

    def explain(self) -> str:
        if self.serializable:
            return f"serializable: {len(self.edges)} dependencies across the history, no cycle"
        path = " -> ".join(f"T{t}" for t in self.cycle)
        detail = "\n".join(f"    {edge}" for edge in self.cycle_edges)
        return f"NOT serializable: cycle {path}\n{detail}"


def build_graph(history: list[CommitRecord]) -> list[Edge]:
    """Dependency edges between committed transactions."""
    edges: list[Edge] = []
    by_ts = {record.commit_ts: record for record in history}

    writers: dict[str, list[int]] = {}
    for record in history:
        for key in record.write_set:
            writers.setdefault(key, []).append(record.commit_ts)
    for key in writers:
        writers[key].sort()

    for key, timestamps in writers.items():
        # write-write: consecutive writers of the same key are ordered by the
        # store, so one edge per adjacent pair captures the chain. Every pair
        # would add transitive edges that cannot create a cycle the adjacent
        # ones miss.
        for earlier, later in zip(timestamps, timestamps[1:], strict=False):
            edges.append(Edge(earlier, later, "ww", key))

    for record in history:
        for key, version_read in record.read_set.items():
            # write-read: whoever wrote the version this transaction saw.
            if (
                version_read
                and version_read != record.commit_ts
                and key in by_ts.get(version_read, _NO_RECORD).write_set
            ):
                edges.append(Edge(version_read, record.commit_ts, "wr", key))

            # read-write: anyone who overwrote that version afterwards. The
            # reader is ordered first because it saw the earlier state.
            for writer_ts in writers.get(key, ()):
                if writer_ts > version_read and writer_ts != record.commit_ts:
                    edges.append(Edge(record.commit_ts, writer_ts, "rw", key))

    return edges


def find_cycle(edges: list[Edge]) -> tuple[list[int], list[Edge]]:
    """Depth-first search for a cycle. Returns the nodes and the edges on it."""
    adjacency: dict[int, list[Edge]] = {}
    for edge in edges:
        adjacency.setdefault(edge.frm, []).append(edge)

    WHITE, GREY, BLACK = 0, 1, 2
    colour: dict[int, int] = {}
    stack: list[Edge] = []

    def visit(node: int) -> tuple[list[int], list[Edge]] | None:
        colour[node] = GREY
        for edge in adjacency.get(node, ()):
            if colour.get(edge.to, WHITE) == GREY:
                # Found a back edge. Walk the current path back to the target
                # to recover just the cycle rather than the whole descent.
                stack.append(edge)
                cycle_edges: list[Edge] = []
                for candidate in reversed(stack):
                    cycle_edges.append(candidate)
                    if candidate.frm == edge.to:
                        break
                cycle_edges.reverse()
                nodes = [e.frm for e in cycle_edges] + [cycle_edges[-1].to]
                return nodes, cycle_edges
            if colour.get(edge.to, WHITE) == WHITE:
                stack.append(edge)
                found = visit(edge.to)
                if found:
                    return found
                stack.pop()
        colour[node] = BLACK
        return None

    nodes = {edge.frm for edge in edges} | {edge.to for edge in edges}
    for node in sorted(nodes):
        if colour.get(node, WHITE) == WHITE:
            stack.clear()
            found = visit(node)
            if found:
                return found
    return [], []


def check(history: list[CommitRecord]) -> CheckResult:
    """Is this history conflict-serializable?"""
    edges = build_graph(history)
    cycle, cycle_edges = find_cycle(edges)
    return CheckResult(
        serializable=not cycle,
        edges=edges,
        cycle=cycle,
        cycle_edges=cycle_edges,
    )
