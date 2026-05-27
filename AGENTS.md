# Agent Instructions: Network Optimizer

## Setup
- **Python**: 3.9.6 (system)
- **Venv**: `.venv/` — already created and configured
- **Activate**: `source .venv/bin/activate`
- **Install deps**: `pip install -e ".[dev]"` (editable install with dev tools)

## Developer Commands
- **Activate venv**: `source .venv/bin/activate`
- **Run Optimizer**: `run-optimizer` (or `python -m network_optimizer`)
- **CLI Options**:
    - `--pool <path>`: Provider pool CSV (required)
    - `--members <path>`: Members CSV (required)
    - `--adequacy-reqs <path>`: Adequacy requirements CSV (required)
    - `--network <path>`: Initial network CSV (optional, starts empty if omitted)
    - `--max-rounds <N>`: Max rounds per phase
    - `--patience <N>`: Consecutive no-improvement rounds to stop early
    - `--enable-swaps`: Run phase 2 (swap refinement)
    - `--time-budget <seconds>`: Max runtime in seconds
    - `--output <path>`: Write results to JSON file
- **Run Tests**: `pytest`
- **Lint**: `ruff check .`
- **Type Check**: `mypy src/network_optimizer`
- **Install**: `pip install -e ".[dev]"`

## Architecture & Key Files
- **Approach**: Deterministic local search (steepest ascent) with two-phase search.
- **Core Logic**:
    - `src/network_optimizer/__init__.py`: Package exports, version `0.1.0`.
    - `src/network_optimizer/__main__.py`: Enables `python -m network_optimizer` invocation.
    - `src/network_optimizer/config.py`: `OptimizerConfig` dataclass (max_rounds, patience, time_budget, enable_swaps).
    - `src/network_optimizer/data.py`: Data loading, validation, normalization.
    - `src/network_optimizer/distance.py`: BallTree-based haversine distance queries for coverage computation.
    - `src/network_optimizer/adequacy.py`: Adequacy scoring functions and objective wrappers.
    - `src/network_optimizer/search.py`: `NetworkOptimizer` — two-phase local search (additions then swaps).
    - `src/network_optimizer/main.py`: CLI entry point with argparse.
- **Data**: Synthetic test data in `data/synth/`. Real data: Medicare Sample Census for members.
- **Interactive Dev**: `notebooks/demo.ipynb`.
- **Reports**: Benchmark results in `reports/` directory.

## Key Logic & Patterns
- **Two-Phase Search**: Phase 1 (construct) — greedy additions until convergence. Phase 2 (refine) — fixed-size swap refinement. No standalone removals.
- **BallTree Coverage**: Build BallTree on provider locations. For each member, query providers within county/specialty distance threshold via `query_radius`. O(M log P) instead of O(M × P) cross join.
- **Adequacy Scoring**: Per (county, specialty) — what fraction of members have ≥ `provider_count` accessible providers? Weighted average across all county/specialty combinations.
- **Multi-Region**: Adequacy requirements have per-county rules (`distance_req`, `provider_count`, `min_providers`). BallTree queries use per-county distance thresholds.

## Important Notes
- **No LLM**: This is a deterministic optimizer — no language model involved. Pure algorithmic approach.
- **Move Types**: Additions (phase 1), swaps (phase 2). Removals alone never improve a pure adequacy objective.
- **Incremental Evaluation**: TODO — cache baseline adequacy, compute deltas for successors instead of full recalculation.
