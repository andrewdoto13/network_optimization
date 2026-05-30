# Agent Instructions: Network Optimizer

## Setup
- **Python**: 3.9+ | **Venv**: `.venv/` (pre-created)
- **Activate**: `source .venv/bin/activate`
- **Install deps**: `pip install -e ".[dev]"`

## Commands
- **Run**: `run-optimizer` or `python -m network_optimizer` (entry point: `src/network_optimizer/main.py`)
- **CLI**: `--pool`, `--members`, `--thresholds` required; `--network` optional (starts empty). See README for full option list.
- **Run with real data**: `run-optimizer --pool <pool_csv> --members <members_csv> --thresholds <thresholds_json>`
- **With weights**: `echo '{"efficiency": 0.3}' > weights.json && run-optimizer --pool <pool_csv> --members <members_csv> --thresholds <thresholds_json> --weights weights.json`
- **Lint**: `ruff check .` (selects: E F W I N UP B SIM; use `--fix --unsafe-fixes`)
- **Type check**: `mypy src/network_optimizer` (note: `disallow_untyped_defs = false`)

## Architecture
`src/network_optimizer/` — deterministic two-phase local search with configurable strategy. No LLM, pure algorithmic.

| File | Purpose |
|---|---|
| `__init__.py` | Exports: `OptimizerConfig`, `NetworkOptimizer`, `SearchResult`, `adequacy_score`, `compute_coverage`, `weighted_objective`, `CandidateRanker`, `load_*` |
| `__main__.py` | Enables `python -m network_optimizer` |
| `main.py` | CLI entry point (argparse) |
| `config.py` | `OptimizerConfig` dataclass with `__post_init__` validation |
| `data.py` | Load/validate pool, members, thresholds, weights; raw-format normalization (column renames, coordinate scaling, string lowercasing, NaN filtering) |
| `scoring.py` | BallTree coverage queries + adequacy scoring (`adequacy_score`, `compute_coverage`, `weighted_objective`, `normalize_to_100`, `miles_to_radians`) |
| `ranking.py` | `CandidateRanker` — BallTree-based prefiltering + weighted ranking of candidates |
| `search.py` | `NetworkOptimizer` — phase 1 (greedy adds), phase 2 (swaps), first-improvement or steepest ascent |

## Data Schema
- **Pool / Network** (provider-level, grouped by `entity`): `id`, `entity`, `specialty`, `lat`, `lon`, `state`, `effectiveness`, `efficiency` (+ optional `county`, city, location_confidence, claims)
- **Members**: `id`, `state`, `county`, `lat`, `lon`
- **Thresholds** (JSON): `{state: {county: {specialty: distance_miles}}}` — binary coverage (member covered if any matching provider within distance)
- **Weights** (JSON): `{column_name: weight}` — e.g. `{"efficiency": 0.3, "effectiveness": 0.2}`. Values in [0, 1], sum < 1.0. Adequacy gets remaining weight.

## Key Patterns
- **Two-phase search**: Phase 1 adds entities greedily until convergence. Phase 2 swaps entities at fixed network size. No standalone removals (never improve pure adequacy).
- **Search modes**: `first_improvement` (default) accepts first improving candidate. `steepest` evaluates all, picks best. First-improvement is faster due to ranked candidate order.
- **Candidate prefiltering**: `CandidateRanker` uses BallTree on uncovered members to filter out entities that cannot improve coverage. Two checks: specialty gap (entity has a specialty in a sub-100% coverage pair) and geographic reach (entity has a provider within threshold of an uncovered member). Typically reduces candidates by 80-90%. Lossless — never eliminates a candidate that could improve the score.
- **BallTree coverage**: Per-specialty BallTree on provider coords. Per-county threshold via `query_radius`. Avoids O(M × P) cross join. 14x speedup on large networks vs per-county rebuild.
- **Weighted objectives**: `weighted_objective()` combines adequacy with column metrics. Columns normalized to 0-100 via pool min/max. Configured via `OptimizerConfig.metric_weights` or `--weights` CLI flag.
- **Python 3.9**: Use `from __future__ import annotations` for `X | None`, `list[X]` syntax.
- **Ruff isort**: `scripts` is `known-first-party` alongside `network_optimizer`.

## Performance Notes
- `adequacy_score` on full pool (124K providers, 10K members, 100 thresholds): ~0.79s
- One Phase 1 round with 479 entities (20+ providers): ~58s
- One Phase 1 round with 2,122 entities (5+ providers): ~28 min
- Prefilter reduces 13,842 entities → ~1,658 candidates (88% reduction)
- Use `--min-entity-size` to reduce search space further
