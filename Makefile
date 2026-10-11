.PHONY: install test lint bench anomalies decisions sweep check example clean help

install:    ## sync the environment from uv.lock
	uv sync

test:       ## run the test suite
	uv run pytest

lint:       ## lint and format check
	uv run ruff check src tests examples
	uv run ruff format --check src tests examples

anomalies:  ## the four anomalies at every isolation level
	uv run aiso anomalies

bench:      ## throughput, abort rate, lost updates
	uv run aiso bench

decisions:  ## agents deciding from shared state they do not write
	uv run aiso decisions

sweep:      ## abort rate as the inference window grows
	uv run aiso sweep

check:      ## verify an uncontrolled history is caught (expects failure)
	uv run aiso check --isolation none || true

example:    ## eight agents competing for four seats
	uv run python examples/booking_agent.py

clean:
	rm -rf .pytest_cache .ruff_cache dist build *.egg-info results.json
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  %-11s %s\n", $$1, $$2}'
