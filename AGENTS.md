# Agent Instructions: Network Optimizer

## Setup
- **Python**: 3.9+ | **Venv**: `.venv/` (pre-created)
- **Activate**: `source .venv/bin/activate`
- **Install deps**: `pip install -e ".[dev]"`

## Commands
- **Run**: `run-optimizer` or `python -m network_optimizer` (entry point: `src/network_optimizer/main.py`)
- **CLI**: `--pool`, `--members`, `--thresholds` required; `--network` optional (starts empty). See README for full option list.
- **Quick smoke test**: `run-optimizer --pool data/mi_market_data.csv --members data/mi_members_sample.csv --thresholds data/thresholds.json --quick`
- **Real data test (fast)**: `run-optimizer --pool data/mi_market_data.csv --members data/mi_members_sample.csv --thresholds data/thresholds.json --min-entity-size 20 --max-rounds 2`
- **Lint**: `ruff check .` (selects: E F W I N UP B SIM; use `--fix --unsafe-fixes`)
- **Type check**: `mypy src/network_optimizer` (note: `disallow_untyped_defs = false`)

## Architecture
`src/network_optimizer/` — deterministic two-phase local search (steepest ascent). No LLM, pure algorithmic.

| File | Purpose |
|---|---|
| `__init__.py` | Exports: `OptimizerConfig`, `NetworkOptimizer`, `SearchResult`, `adequacy_score`, `compute_coverage`, `load_*` |
| `__main__.py` | Enables `python -m network_optimizer` |
| `main.py` | CLI entry point (argparse) |
| `config.py` | `OptimizerConfig` dataclass with `__post_init__` validation |
| `data.py` | Load/validate pool, members, thresholds; raw-format normalization (column renames, coordinate scaling, string lowercasing, NaN filtering) |
| `scoring.py` | BallTree coverage queries + adequacy scoring (`adequacy_score`, `compute_coverage`, `miles_to_radians`) |
| `search.py` | `NetworkOptimizer` — phase 1 (greedy adds), phase 2 (swaps), supports custom `objective` callable |

## Data Schema
- **Pool / Network** (provider-level, grouped by `entity`): `id`, `entity`, `specialty`, `lat`, `lon`, `state`, `effectiveness`, `efficiency` (+ optional `county`, city, location_confidence, claims)
- **Members**: `id`, `state`, `county`, `lat`, `lon`
- **Thresholds** (JSON): `{state: {county: {specialty: distance_miles}}}` — binary coverage (member covered if any matching provider within distance)

## Key Patterns
- **Two-phase search**: Phase 1 adds entities greedily until convergence. Phase 2 swaps entities at fixed network size. No standalone removals (never improve pure adequacy).
- **BallTree coverage**: Per-specialty BallTree on provider coords. Per-county threshold via `query_radius`. Avoids O(M × P) cross join. 14x speedup on large networks vs per-county rebuild.
- **Custom objectives**: `NetworkOptimizer` accepts `objective: Callable[[pd.DataFrame], float] | None`. Default is `adequacy_score`; pass any `(network) -> float` function for custom scoring.
- **Python 3.9**: Use `from __future__ import annotations` for `X | None`, `list[X]` syntax.
- **Ruff isort**: `scripts` is `known-first-party` alongside `network_optimizer`.

## Performance Notes
- `adequacy_score` on full pool (124K providers, 10K members, 100 thresholds): ~0.79s
- One Phase 1 round with 479 entities (20+ providers): ~58s
- One Phase 1 round with 2,122 entities (5+ providers): ~28 min
- Use `--min-entity-size` to reduce search space
