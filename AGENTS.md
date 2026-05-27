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
    - `--thresholds <path>`: Distance thresholds JSON (required)
    - `--network <path>`: Initial network CSV (optional, starts empty if omitted)
    - `--max-rounds <N>`: Max rounds per phase (default: 100)
    - `--patience <N>`: Consecutive no-improvement rounds to stop early (default: 5)
    - `--enable-swaps`: Run phase 2 (swap refinement)
    - `--swap-only`: Skip phase 1, only do swaps
    - `--time-budget <seconds>`: Max runtime in seconds
    - `--convergence <0-1>`: Relative convergence threshold (default: 0)
    - `--verbosity <0|1|2>`: 0=silent, 1=summary, 2=per-round (default: 1)
    - `--output <path>`: Write results to JSON file
    - `--quick`: Quick test mode (10 rounds, no swaps)
- **Run Tests**: `pytest`
- **Lint**: `ruff check .`
- **Type Check**: `mypy src/network_optimizer`
- **Install**: `pip install -e ".[dev]"`

## Architecture & Key Files
- **Approach**: Deterministic local search (steepest ascent) with two-phase search.
- **Core Logic**:
    - `src/network_optimizer/__init__.py`: Package exports, version `0.1.0`.
    - `src/network_optimizer/config.py`: `OptimizerConfig` dataclass with `__post_init__` validation.
    - `src/network_optimizer/data.py`: Load pool/members/thresholds, validate columns, normalize state/county.
    - `src/network_optimizer/distance.py`: BallTree-based haversine distance queries for coverage computation.
    - `src/network_optimizer/adequacy.py`: Score = mean coverage % across all (county, specialty) thresholds.
    - `src/network_optimizer/search.py`: `NetworkOptimizer` — two-phase local search (additions then swaps).
    - `src/network_optimizer/main.py`: CLI entry point with argparse.
- **Data**: Synthetic test data in `data/synth/`. Schema matches `network_manager_agent` project.
- **Scripts**: `scripts/generate_data.py` — generate synthetic data matching agent schema.
- **Interactive Dev**: `notebooks/demo.ipynb` (TODO).
- **Reports**: Benchmark results in `reports/` directory (TODO).

## Data Schema (agent-compatible)

### Pool / Network (provider-level)
- Grouping key: `entity` (string name, e.g. "Beaumont Health")
- Required columns: `id`, `entity`, `specialty`, `lat`, `lon`, `state`, `county`, `effectiveness`, `efficiency`
- Optional: `city`, `location_confidence`, claims volume/amount columns

### Members
- Required columns: `id`, `state`, `county`, `lat`, `lon`

### Thresholds (JSON)
- Nested format: `{state: {county: {specialty: distance_miles}}}`
- Binary coverage: member has access if **any** provider of matching specialty is within distance

## Key Logic & Patterns
- **Two-Phase Search**: Phase 1 (construct) — greedy entity additions until convergence. Phase 2 (refine) — fixed-size swap refinement. No standalone removals.
- **BallTree Coverage**: Build BallTree on provider locations per specialty. For each member, query providers within distance threshold via `query_radius`. O(M log P) instead of O(M × P) cross join.
- **Adequacy Scoring**: Score = mean coverage % across all (county, specialty) thresholds. Coverage = fraction of members with at least one provider in range.
- **Multi-Region**: Thresholds JSON has per-county distance rules. BallTree queries use per-county thresholds.
- **Python 3.9**: Use `from __future__ import annotations` for modern type hint syntax (`X | None`, `list[X]`).

## Important Notes
- **No LLM**: This is a deterministic optimizer — no language model involved. Pure algorithmic approach.
- **Move Types**: Additions (phase 1), swaps (phase 2). Removals alone never improve a pure adequacy objective.
- **Incremental Evaluation**: TODO — cache baseline coverage per entity, compute deltas for swap evaluation instead of full recalculation.
- **Lint**: `ruff` with E, F, W, I, N, UP, B, SIM selects. Use `--fix --unsafe-fixes` for auto-fixes.
- **Type hints**: Use `from __future__ import annotations` + modern syntax (`X | None`, `list[X]`, `tuple[X, Y]`).
