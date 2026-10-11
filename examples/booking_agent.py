"""Two agents booking against shared inventory.

Run it:

    uv run python examples/booking_agent.py

The same code runs at each isolation level. Without concurrency control the
agents oversell, because both read the same remaining count before either
writes. The reservation shows the other half of the problem: it has to happen
during the transaction, so it needs an inverse for when the transaction aborts.
"""

from __future__ import annotations

import threading

from agent_isolation import Coordinator, Isolation, RetriesExhausted

SEATS = 4
AGENTS = 8


def book(coord: Coordinator, agent: int, holds: list[str], lock: threading.Lock) -> bool:
    def body(tx) -> bool:
        remaining = tx.get("seats:remaining", 0)
        if remaining <= 0:
            return False

        # The hold has to happen now: the agent needs to know it succeeded
        # before it commits. If this transaction aborts, the inverse releases
        # it, which is the only thing between here and a hold nobody frees.
        tx.compensable(
            "hold-seat",
            do=lambda: holds.append(f"agent-{agent}"),
            undo=lambda: holds.remove(f"agent-{agent}"),
        )

        tx.put("seats:remaining", remaining - 1)
        tx.put(f"booking:{agent}", {"seat": remaining})
        # Safe to send only once the booking is real.
        tx.defer("confirm", {"agent": agent, "seat": remaining})
        return True

    try:
        return coord.run(body, agent_id=f"agent-{agent}", think_time_s=0.01).value
    except RetriesExhausted:
        return False


def run(isolation: Isolation) -> None:
    confirmations: list[dict] = []
    holds: list[str] = []
    lock = threading.Lock()

    with Coordinator(isolation=isolation, max_attempts=40) as coord:
        coord.effects.register("confirm", confirmations.append)
        coord.set("seats:remaining", SEATS)

        threads = [
            threading.Thread(target=book, args=(coord, i, holds, lock)) for i in range(AGENTS)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        remaining = coord.get("seats:remaining")
        booked = sum(1 for k in coord.store.snapshot_of_all() if k.startswith("booking:"))
        oversold = max(0, booked - SEATS)

        print(
            f"  {isolation.value:13} booked={booked}  remaining={remaining}  "
            f"confirmations={len(confirmations)}  oversold={oversold}  "
            f"dangling holds={len(holds) - booked}"
        )


if __name__ == "__main__":
    print(f"\n{AGENTS} agents competing for {SEATS} seats\n")
    for level in Isolation:
        run(level)
    print(
        "\nOnly committed bookings send a confirmation, and every abandoned hold "
        "was released by its compensation.\n"
    )
