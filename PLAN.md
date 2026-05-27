# Network Optimization: Review & Refactoring Plan

**Date:** May 27, 2026
**Repo:** andrewdoto13/network_optimization
**Target standard:** Match the structure and quality of `network_manager_agent`

---

## Current State

### What exists
- `src/network_optimizer.py` — single monolithic file (244 lines), all logic in one class
- `src/synthetic_data_generators.py` — data generation functions + hardcoded `synth_reqs`
- `src/test_optimizer.ipynb` — demo/test notebook
- `src/dev.ipynb` — development notebook (class duplicated inline, 971 lines)
- `data/` — synthetic CSVs (pool, members, network)
- `README.md` — empty (2 lines)
- No `pyproject.toml`, no tests, no linting, no type checking, no entry point

### Issues identified

**Structural**
1. No package structure — everything is loose files in `src/`
2. No `pyproject.toml` — no dependency management, no installable package
3. No test suite — notebooks are the only "tests"
4. No linting or type checking configuration
5. No CLI entry point
6. No AGENTS.md for developer onboarding
7. README is empty

**Code quality**
8. Monolithic class — 244 lines in one file mixing data loading, distance computation, adequacy scoring, and optimization logic
9. Row-wise haversine via `.apply(lambda row: ...)` — O(n) Python loop, sklearn's vectorized `haversine_distances` is imported but unused
10. No incremental evaluation — each successor state recalculates adequacy from scratch (full merge-groupby pipeline)
11. `successor()` only generates additions — docstring mentions removals and swaps but they're not implemented
12. `create_state()` only handles additions — same gap
13. Hardcoded haversine function nested inside `__init__` — should be a module-level utility
14. No input validation — no checks for required columns, coordinate ranges, or empty DataFrames
15. `performance_history` and `time_tracker` are numpy arrays — mutable state that persists across `optimize()` calls
16. No logging — uses `print()` statements
17. Unused import: `sklearn.metrics.pairwise.haversine_distances`
18. `dev.ipynb` duplicates the entire class — source of truth ambiguity

**Algorithmic**
19. Pure constructive (additions only) — greedy choices lock in, can't escape local optima
20. No multi-region support — synthetic data is single county, adequacy_reqs has county column but code doesn't leverage it
21. No configurable neighborhood size — always evaluates all remaining groups
22. No early termination beyond "no improvement" — no time budget or convergence threshold

---

## Target Structure

```
network_optimization/
├── AGENTS.md                          # Developer commands and architecture
├── README.md                          # Project overview, usage, results
├── pyproject.toml                     # Dependencies, scripts, lint config
├── data/
│   └── synth/                         # Synthetic test data
│       ├── pool.csv
│       ├── members.csv
│       └── network.csv
├── src/
│   └── network_optimizer/             # Package
│       ├── __init__.py                # Package exports, version
│       ├── __main__.py                # Enables `python -m network_optimizer`
│       ├── config.py                  # Config dataclass, constants
│       ├── data.py                    # Data loading, validation, normalization
│       ├── distance.py                # Haversine utilities, BallTree integration
│       ├── adequacy.py                # Adequacy scoring, objective functions
│       ├── search.py                  # Local search algorithm (steepest ascent)
│       └── main.py                    # CLI entry point
├── tests/
│   ├── conftest.py                    # Shared fixtures
│   ├── test_data.py                   # Data loading and validation
│   ├── test_distance.py               # Distance calculations
│   ├── test_adequacy.py               # Adequacy scoring
│   ├── test_search.py                 # Search algorithm
│   └── test_main.py                   # CLI integration
├── notebooks/
│   └── demo.ipynb                     # Interactive demo (replaces test_optimizer.ipynb)
├── scripts/
│   └── generate_data.py               # CLI data generator (replaces synthetic_data_generators.py)
└── reports/
    └── eval_report.md                 # Benchmark results
```

---

## Phases

### Phase 1: Review and baseline (current code works)

**Goal:** Verify the existing code runs, document current behavior, establish baseline metrics.

1. Create venv, install dependencies, run existing optimizer on synthetic data
2. Record baseline: adequacy score, runtime, number of rounds
3. Run `dev.ipynb` cells to verify no divergence from `network_optimizer.py`
4. Document current data schema (pool, members, network, adequacy_reqs columns)

**Deliverable:** Baseline metrics in `reports/eval_report.md`

### Phase 2: Project scaffolding

**Goal:** Establish proper project structure matching `network_manager_agent` standards.

1. Create `pyproject.toml`
   - Dependencies: pandas, numpy, scikit-learn
   - Dev: pytest, ruff, mypy, ipykernel, matplotlib, seaborn
   - Scripts: `run-optimizer` entry point
   - Ruff config (E, F, W, I, N, UP, B, SIM selects, E501 ignore)
   - Mypy config (warn_return_any, warn_unused_configs)
2. Create package structure: `src/network_optimizer/__init__.py` with version `0.1.0`
3. Create `AGENTS.md` with developer commands
4. Write `README.md` with project overview, installation, usage
5. Move synthetic data to `data/synth/`
6. Move notebooks to `notebooks/`

**Deliverable:** Installable package, `pip install -e ".[dev]"` works

### Phase 3: Module extraction

**Goal:** Split monolithic class into focused modules. Each module has a single responsibility.

1. **`config.py`** — `OptimizerConfig` dataclass
   - `max_rounds`, `distance_threshold`, `convergence_threshold`
   - Constants: default values, column names
2. **`data.py`** — Data loading and validation
   - `DataManager` class or functions: load pool/members/network/adequacy_reqs
   - Column validation (required fields, types)
   - Coordinate normalization (reuse pattern from agent if useful)
   - Move `synthetic_data_generators.py` logic to `scripts/generate_data.py`
3. **`distance.py`** — Distance computation
   - Module-level `haversine()` function (extract from `__init__`)
   - Vectorized `compute_distances()` using sklearn or numpy
   - `build_access_listing()` — members × providers cross join with distances
4. **`adequacy.py`** — Scoring functions
   - `compute_adequacy(network, members, adequacy_reqs, access_listing)` — pure function
   - `default_objective()` — wraps adequacy
   - Support for custom objective functions via callable
   - Store `adequacy_detail` as return value, not instance state
5. **`search.py`** — Local search algorithm
   - `NetworkOptimizer` class — only the search logic
   - `successor()` — generate candidate moves
   - `evaluate_successors()` — batch evaluation
   - `optimize()` — main loop
   - Track: performance history, move history, timing
6. **`main.py`** — CLI entry point
   - argparse: `--pool`, `--members`, `--adequacy-reqs`, `--network`, `--max-rounds`, `--objective`
   - Load data, create optimizer, run, print results
   - JSON output option for programmatic use

**Deliverable:** All tests pass, same output as baseline

### Phase 4: Algorithm improvements

**Goal:** Address the algorithmic limitations identified in the review.

1. **Add swap successors** — remove one group from network, add one from pool
   - `successor()` returns additions, removals, and swaps
   - `create_state()` handles all three move types
   - Enables escaping local optima from greedy additions
2. **Incremental adequacy evaluation** — cache baseline, compute deltas
   - Pre-compute member access per provider group
   - When evaluating "add group X", only recompute affected county/specialty buckets
   - Significant speedup when pool is large
3. **Vectorized haversine** — replace `.apply(lambda row: ...)` with vectorized computation
4. **Convergence criteria** — configurable early termination
   - Max rounds (existing)
   - No improvement for N consecutive rounds (patience)
   - Time budget (max seconds)
5. **Multi-region support** — adequacy_reqs already has county column
   - Ensure `compute_adequacy()` correctly handles multiple counties
   - Per-county adequacy breakdown in output

**Deliverable:** Improved optimizer with documented speedup and quality gains vs baseline

### Phase 5: Test suite

**Goal:** Comprehensive tests matching the standard of `network_manager_agent` (111 tests).

1. **`conftest.py`** — Shared fixtures
   - Small synthetic datasets (3 groups, 10 members, 2 specialties)
   - Larger synthetic datasets (15 groups, 1000 members, 5 specialties)
   - Edge cases: empty pool, single provider, all same location
2. **`test_data.py`** — Data module
   - Column validation (missing columns, wrong types)
   - Coordinate normalization
   - Data loading from CSV
3. **`test_distance.py`** — Distance module
   - Haversine correctness (known distances)
   - Vectorized vs row-wise equivalence
   - Access listing construction
4. **`test_adequacy.py`** — Adequacy module
   - Empty network → 0 adequacy
   - Full coverage → 1.0 adequacy
   - Adequacy index calculation correctness
   - Custom objective function passthrough
5. **`test_search.py`** — Search algorithm
   - Addition-only optimization (baseline behavior preserved)
   - Swap optimization improves over additions-only
   - Convergence: stops when no improvement
   - Move tracker correctness
   - Performance history monotonic increase
   - Time budget enforcement
6. **`test_main.py`** — CLI
   - Help text
   - Run with synthetic data
   - JSON output format
   - Error on missing files

**Deliverable:** pytest passes, ruff clean, mypy clean

### Phase 6: Polish

**Goal:** Production-ready repo.

1. Update README with architecture diagram, usage examples, benchmark results
2. Write `reports/eval_report.md` — compare baseline vs refactored vs improved
3. Create `notebooks/demo.ipynb` — clean demo replacing the two existing notebooks
4. Pin the repo on GitHub profile
5. Add topics: `network-optimization`, `healthcare`, `local-search`, `provider-network`

---

## Priority order

1. **Phase 2** (scaffolding) — enables everything else, low risk
2. **Phase 3** (module extraction) — core refactoring, preserves behavior
3. **Phase 5** (tests) — confidence in refactoring, matches agent standard
4. **Phase 4** (algorithm improvements) — adds new capability
5. **Phase 6** (polish) — presentation

Phase 1 runs in parallel with Phase 2.

---

## Open questions

- Keep the existing synthetic data or regenerate with more realistic distributions?
- Add ILP comparison (PuLP/ortools) as a second optimizer for benchmarking?
- Add visualization module (coverage maps, convergence plots)?

---

## Design Decisions

### Move types: two-phase search (construct then refine)

**Decision:** Additions in phase 1, swaps in phase 2. No standalone removals.

**Rationale:** With a pure adequacy objective (no size penalty), the greedy algorithm will never choose a removal — removing a group always reduces coverage. Swaps alone also won't fire because at the addition local optimum, adding group B didn't help, and removing A first makes the baseline worse so adding B is even less likely to improve.

Two-phase approach solves this:
- **Phase 1 (construct):** Greedy additions until convergence. Same behavior as current code.
- **Phase 2 (refine):** Fixed network size. Try every swap (remove A, add B). Accept improving swaps. Repeat until stable. Because network size is fixed, swaps trade coverage patterns without changing the group count — the algorithm can find better spatial coverage by swapping a poorly-placed group for a well-placed one.

No size penalty λ to tune. Each phase has a clear stopping criterion.

**Search space:**
- Phase 1: N candidates per round, ≤N rounds → O(N) adequacy evaluations
- Phase 2: K × (N-K) swap candidates per round (K = groups in network, N-K = groups in pool). For 500/500 split: 250K evaluations/round. Requires BallTree + incremental evaluation.

### BallTree for distance computation

**Decision:** Replace members × providers cross join with BallTree-based radius queries.

**Current approach:** Cross join all members with all providers, compute haversine row-by-row. O(M × P) rows. For 100K members × 5K providers = 500M rows.

**New approach:** Build BallTree on provider locations. For each member, query all providers within their county/specialty distance threshold via `query_radius`. O(M log P) per query.

**Implementation in `distance.py`:**
1. Build BallTree on provider lat/lon (once per optimization run, rebuilt when network changes)
2. `compute_access(network, members, adequacy_reqs)` → per-member access counts per (county, specialty)
   - For each (county, specialty) in adequacy_reqs: filter providers by specialty, build BallTree, query each member in that county within distance threshold, count accessible providers
3. Returns a DataFrame: `member_id`, `county`, `specialty`, `accessible_provider_count`
4. Adequacy calculation consumes this directly — no cross join needed

**When network changes** (add/remove/swap a group): rebuild BallTree with updated provider set. BallTree construction is O(P log P), fast for thousands of providers.

### Multi-region support

**Decision:** Build multi-region from the start. Adequacy requirements already have a `county` column with per-county rules.

**Requirements:**
- Different counties have different `distance_req`, `provider_count`, `min_providers`, `min_access_pct`
- Members distributed across counties
- Providers distributed across counties
- Adequacy calculation groups by county × specialty
- BallTree queries use per-county distance thresholds

**Synthetic data:** Regenerate with 2-3 counties, different adequacy rules per county, realistic geographic clustering.

### Python version

**Decision:** Latest stable (3.12+). Match the agent project.

### Component specifications

**`config.py`** — `OptimizerConfig` dataclass
- `max_rounds`: int — max rounds per phase
- `patience`: int — consecutive no-improvement rounds to stop early
- `time_budget`: Optional[float] — max seconds total
- `enable_swaps`: bool — run phase 2 (swap refinement)
- Constants: default column names, coordinate bounds

**`data.py`** — Loading + validation
- `load_pool(path)`, `load_members(path)`, `load_network(path)`, `load_adequacy_reqs(path)`
- Validate: required columns present, non-empty, coordinate ranges valid (lat -90 to 90, lon -180 to 180)
- Normalize: lowercase specialties and counties, detect/fix coordinate scaling
- `validate_pool_network_separation(pool, network)` — warn if groups overlap

**`distance.py`** — BallTree-based coverage
- `build_balltree(providers)` → BallTree instance
- `compute_access(providers, members, adequacy_reqs)` → DataFrame with `member_id`, `county`, `specialty`, `accessible_provider_count`
- `haversine(lat1, lon1, lat2, lon2)` — module-level utility (for tests and reference)

**`adequacy.py`** — Pure scoring functions
- `compute_adequacy(access_df, members, adequacy_reqs)` → `(score: float, detail: DataFrame)`
  - `access_df`: output of `compute_access()`
  - `detail`: per (county, specialty) breakdown with `pct_with_access`, `servicing_providers`, `adequacy_index`
- `default_objective(network, data_ctx)` → wraps `compute_adequacy`
- Custom objective support: user passes callable `(optimizer, network) -> float`

**`search.py`** — Local search algorithm
- `NetworkOptimizer` class: holds config, data refs, BallTree cache
- `successors_additions(network, pool)` → `List[Move]` — one per pool group
- `successors_swaps(network, pool)` → `List[Move]` — one per (network_group, pool_group) pair
- `apply_move(network, move)` → new network DataFrame
- `evaluate(network)` → float score (uses cached BallTree or recomputes)
- `optimize()` → two phases:
  1. Additions until convergence (no improvement or pool exhausted)
  2. Swaps if enabled, until convergence (no improving swap found)
- Returns `OptimizationResult`: best_network, performance_history, moves, timing, adequacy_detail

**`main.py`** — CLI entry point
- argparse: `--pool`, `--members`, `--adequacy-reqs`, `--network` (optional), `--max-rounds`, `--patience`, `--enable-swaps`, `--output` (JSON path), `--time-budget`
- Load → create optimizer → run → print summary → optionally write JSON

### Data schema

**Pool / Network (same schema):**
| Column | Type | Description |
|---|---|---|
| npi | int | National Provider Identifier |
| specialty | str | Provider specialty (lowercase) |
| group_id | int | Contracting entity ID |
| efficiency | int | 1-5 rating |
| effectiveness | int | 1-5 rating |
| location_id | int | Physical location ID |
| county | str | County name (lowercase) |
| latitude | float | WGS84 latitude |
| longitude | float | WGS84 longitude |

**Members:**
| Column | Type | Description |
|---|---|---|
| member_id | int | Unique member identifier |
| county | str | County name (lowercase) |
| latitude | float | WGS84 latitude |
| longitude | float | WGS84 longitude |

**Adequacy requirements:**
| Column | Type | Description |
|---|---|---|
| specialty | str | Specialty name (lowercase) |
| county | str | County name (lowercase) |
| provider_count | int | Min providers of this specialty a member needs access to |
| distance_req | float | Max distance in miles |
| min_access_pct | float | Target % of members with access (informational, not used in scoring) |
| min_providers | int | Min total providers of this specialty in the network |
