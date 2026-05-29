# Contributing to Weather Edge

Thanks for your interest. Before anything else, please read the status notice in
the [README](README.md): this project was **sunset after losing money** and is
published as a post-mortem and a clean reference implementation. It is not
maintained as a production product, and it is **not** meant to be pointed at real
funds. That shapes what contributions make sense.

## What's welcome

- Correctness fixes in the reference implementation (resolution logic, consensus
  math, backtester honesty, METAR parsing).
- Documentation improvements and clarifications.
- Bug fixes, ideally with a regression test.
- Portability and dependency fixes.

## What's out of scope

- Features whose goal is to make the bot profitable or to trade live. The README
  explains why there was no durable edge; PRs chasing alpha will be closed.
- Anything that adds new ways to spend real funds without strong justification.

## How to submit a pull request

1. Fork the repo and create a branch.
2. Make your change, with a test where it makes sense.
3. Run the tests and linter locally (see below).
4. Open a PR against `main` with a clear description of what changed and why.

Only the maintainer can merge. A PR is a proposal; please keep diffs focused and
reviewable.

## Dev setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,dashboard]"     # add ,execution for the trading code
python -m pytest -q                   # no network or API keys needed
ruff check .                          # lint
```

The test suite needs no API keys or network access.

## Code style

- `ruff` is the source of truth (line length 100, target py311). Run
  `ruff check .` before submitting.
- Match the surrounding code. Keep changes minimal and well scoped.

## High-scrutiny areas

Because this code can place orders that move real money, changes to these paths
get extra review and may take longer to merge:

- `trading/executor.py` and anything touching wallet keys, order signing, or
  fund movement.
- The dashboard (`dashboard/`), especially auth, CSRF, the Host allowlist, and
  any new endpoint.
- Dependency additions or version bumps.

## Security

Do not report security issues in a public issue or pull request. See
[SECURITY.md](SECURITY.md) for how to report privately.
