# Network Optimization: Plan & Status

**Date:** May 27, 2026
**Repo:** andrewdoto13/network_optimization
**Status:** Phase 2-3 complete, Phase 4 in progress

---

## Current State

### Completed
- [x] Phase 2: Project scaffolding (pyproject.toml, package structure, AGENTS.md, README)
- [x] Phase 3: Module extraction (config, data, distance, adequacy, search, main)
- [x] Data schema refactored to match agent project (entity-based, JSON thresholds)
- [x] Synthetic data generator (`scripts/generate_data.py`)
- [x] Lint clean (ruff), Python 3.9 compatible
- [x] CLI entry point (`run-optimizer`)
- [x] Two-phase search: greedy additions + swap refinement
- [x] BallTree-based coverage queries (O(M log P))
- [x] Convergence: patience, time budget, relative threshold
- [x] GitHub pushed (3 commits ahead)

### In Progress
- [ ] Phase 4: Algorithm improvements
  - [x] Swap successors (Phase 2 refinement)
  - [x] BallTree distance computation
  - [x] Convergence criteria (patience, time budget, threshold)
  - [x] Multi-region support (3 counties, different thresholds)
  - [ ] Incremental evaluation (cache baseline, compute deltas for swaps)
- [ ] Phase 5: Test suite
- [ ] Phase 6: Polish (README benchmarks, demo notebook, GitHub topics)

### Performance baseline
- Synthetic data: 233 providers, 50 entities, 1000 members, 3 counties, 30 thresholds
- Empty network → 31.52% (10 entities, 16s)
- 5-entity start → 38.62% (15 entities, 17s)
- Score = mean coverage % across all (county, specialty) thresholds

---

## Data Schema (agent-compatible)

### Pool / Network (provider-level)
| Column | Type | Description |
|---|---|---|
| id | int | Provider ID |
| entity | str | Contracting entity name (grouping key) |
| specialty | str | Provider specialty (lowercase) |
| lat | float | WGS84 latitude |
| lon | float | WGS84 longitude |
| state | str | State code (lowercase) |
| county | str | County name (lowercase) |
| city | str | City name |
| effectiveness | int | 1-5 rating |
| efficiency | int | 1-5 rating |
| location_confidence | float | 0-1 confidence score |
| total_claims_volume | int | Total claims count |
| medicare_claims_volume | int | Medicare claims count |
| total_claims_amount | float | Total claims dollar amount |
| medicare_total_claims_amount | float | Medicare claims dollar amount |
| new_patient_claims | int | New patient claims count |

### Members
| Column | Type | Description |
|---|---|---|
| id | int | Member ID |
| state | str | State code (lowercase) |
| county | str | County name (lowercase) |
| lat | float | WGS84 latitude |
| lon | float | WGS84 longitude |

### Thresholds (JSON)
```json
{
  "mi": {
    "wayne": {
      "general practice": 20.0,
      "cardiology": 15.0
    },
    "oakland": { ... },
    "kalamazoo": { ... }
  }
}
```

Binary coverage: member has access if **any** provider of matching specialty is within distance threshold.

---

## Target Structure

```
network_optimization/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── data/
│   └── synth/
│       ├── candidates.csv       # Provider pool (agent schema)
│       ├── members.csv          # Member locations
│       ├── network.csv          # Initial network (subset of candidates)
│       └── thresholds.json      # County/specialty distance thresholds
├── src/network_optimizer/
│   ├── __init__.py
│   ├── config.py                # OptimizerConfig dataclass + validation
│   ├── data.py                  # Load pool/members/thresholds, validate columns
│   ├── distance.py              # BallTree coverage queries
│   ├── adequacy.py              # Score = mean coverage %
│   ├── search.py                # Two-phase local search
│   └── main.py                  # CLI entry point
├── tests/                       # [TODO] pytest suite
├── notebooks/                   # [TODO] demo.ipynb
├── scripts/
│   └── generate_data.py         # Synthetic data generator
└── reports/                     # [TODO] eval_report.md
```

---

## Remaining Work

### Phase 4: Algorithm improvements
1. **Incremental evaluation** — cache coverage per entity, compute delta for swaps
   - Pre-compute per-entity contribution to coverage
   - Swap evaluation = remove entity A's contribution, add entity B's contribution
   - Reduces O(K × (N-K)) full evaluations to O(K × (N-K)) delta computations
   - Critical for large pools (500+ entities × 250K swap combinations)

### Phase 5: Test suite
1. `conftest.py` — Small/large synthetic fixtures
2. `test_data.py` — Column validation, coordinate normalization
3. `test_distance.py` — BallTree coverage correctness
4. `test_adequacy.py` — Score = mean coverage %, edge cases
5. `test_search.py` — Phase 1 additions, Phase 2 swaps, convergence
6. `test_main.py` — CLI args, JSON output, error handling

### Phase 6: Polish
1. Update README with benchmarks and usage examples
2. Write `reports/eval_report.md`
3. Create `notebooks/demo.ipynb`
4. Add GitHub topics: `network-optimization`, `healthcare`, `local-search`, `provider-network`

---

## Open Questions
- Add ILP comparison (PuLP/ortools) as a second optimizer for benchmarking?
- Add visualization module (coverage maps, convergence plots)?
- Real data integration: Medicare Sample Census + state provider directories
- Target user: health plans, consultants, hospital admins, or portfolio piece?

---

## Design Decisions

### Move types: two-phase search (construct then refine)
**Decision:** Additions in phase 1, swaps in phase 2. No standalone removals.

**Rationale:** With a pure adequacy objective (no size penalty), the greedy algorithm will never choose a removal — removing an entity always reduces coverage. Swaps alone also won't fire at the addition local optimum.

Two-phase approach:
- **Phase 1 (construct):** Greedy additions until convergence. Same behavior as original code.
- **Phase 2 (refine):** Fixed network size. Try every swap (remove A, add B). Accept improving swaps. Repeat until stable.

No size penalty λ to tune. Each phase has a clear stopping criterion.

**Search space:**
- Phase 1: N candidates per round, ≤N rounds → O(N) adequacy evaluations
- Phase 2: K × (N-K) swap candidates per round (K = entities in network, N-K = entities in pool). For 15/35 split: 525 evaluations/round.

### BallTree for distance computation
**Decision:** Replace members × providers cross join with BallTree-based radius queries.

**Current approach:** Build BallTree on provider lat/lon per specialty. For each member, query all providers within distance threshold via `query_radius`. O(M log P) per query.

### Multi-region support
**Decision:** Built from the start. Thresholds JSON has per-county distance rules.

**Implementation:**
- Different counties have different distance thresholds per specialty
- Members and providers distributed across counties
- Coverage computed per (state, county, specialty) threshold
- BallTree queries use per-county distance thresholds

### Python version
**Decision:** Python 3.9.6 (system default). Use `from __future__ import annotations` for modern type hint syntax.
