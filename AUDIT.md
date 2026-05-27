# Network Optimizer: Audit Report

**Date:** May 27, 2026
**Purpose:** Refresh understanding before refactoring

---

## How it works

### Algorithm

Steepest-ascent hill climbing, pure constructive (additions only).

Each round:
1. Generate all possible additions (one per remaining group in pool)
2. For each, create the hypothetical network state and compute adequacy score
3. Accept the addition with highest score if it improves over current
4. Remove accepted group from pool
5. Stop when no addition improves or pool is exhausted

### Adequacy Score

Per county/specialty:
```
adequacy_index = pct_with_access * min(1, servicing_providers / min_providers)
```

Final score = mean of all county/specialty adequacy indices.

`pct_with_access` = members who have ≥ `provider_count` providers within `distance_req` miles, divided by total members in that county.

### Data model

- **Pool:** Candidate providers with `group_id` (contracting entity). Adding a group adds ALL its providers.
- **Network:** Currently contracted providers. Can start empty or pre-populated.
- **Members:** Beneficiaries with lat/lon/county.
- **Adequacy requirements:** Per county/specialty rules: `provider_count`, `distance_req`, `min_access_pct`, `min_providers`.

### Synthetic test data

- 100 providers, 15 groups (0-14), 66 unique NPIs
- 1000 members, all Wayne County
- 5 specialties: cardiologist, pcp, ent, urologist, obgyn
- All requirements: 15-mile threshold, 90% min access, min providers varies (5-10)
- **Pool and network have the same 15 groups** — this is a data issue (see below)

---

## Baseline behavior

### From scratch (empty network)

- 10 rounds to convergence, ~0.8s/round, 8.0s total
- 9 groups added, 64 providers in final network
- Final adequacy: **0.9736**
- Diminishing returns: first 3 rounds add 0.65, next 6 add 0.32, last 1 adds 0.002
- Rounds 10-15 add groups with **zero gain** — algorithm should have stopped at round 9

### With existing network (15 groups, 100 providers)

- 3 rounds, ~1.3s/round, 4.2s total
- Starts at 0.9956, reaches **1.0** (perfect score)
- Adds providers from groups already in the network (data overlap issue)

### Convergence behavior

The algorithm stops when `best_score > current_score` is false. But the trace shows:
- Round 9: score 0.9736 (+0.0022)
- Round 10: score 0.9736 (+0.0000) — should stop here
- Rounds 11-15: continue adding with zero gain

This means the `>` comparison is passing due to floating point — the `argmax` picks a successor with equal score, and `best_score > performance_history[-1]` is false, so it should break. But the trace from the manual run shows it continuing. The `optimize()` method's break condition works correctly — the manual trace above was running independently and didn't have the break logic. The actual `optimize()` method stopped at round 10 with "No more options for optimization."

---

## Issues found

### Critical

1. **Pool and network share the same groups.** The synthetic data has identical group_ids (0-14) in both `synth_pool.csv` and `synth_network.csv`. When starting with an existing network, the "pool" contains providers from groups already contracted. This means the optimizer can "add" providers from groups that are already in the network — it's adding duplicate providers, not new contracting entities.

2. **No removal or swap moves.** Only additions. Early greedy choices are permanent. If group A looks good initially but blocks better coverage from group B, there's no way to undo it.

3. **O(n²) cross join.** Access listing is `members × (network + pool)`. With existing network: 1000 × 200 = 200K rows. Scales poorly — 100K members × 5K providers = 500M rows.

### Code quality

4. **Row-wise haversine.** `.apply(lambda row: haversine(...))` — O(n) Python loop. `sklearn.metrics.pairwise.haversine_distances` is imported but never used.

5. **No incremental evaluation.** Each successor recalculates adequacy from scratch (full merge → groupby → merge pipeline). With 15 groups and 5 adequacy rules this is fine. With 500 groups it's 500 full adequacy calculations per round.

6. **Monolithic class.** 244 lines mixing data loading, distance computation, scoring, and search logic.

7. **Nested haversine function.** Defined inside `__init__` — should be module-level.

8. **Mutable state persists across calls.** `performance_history`, `time_tracker`, `move_tracker` accumulate across `optimize()` calls. Calling `optimize()` twice appends to the same arrays.

9. **`FutureWarning` from `pd.concat`** with empty DataFrame — will break in future pandas.

10. **No input validation.** No checks for required columns, coordinate ranges, empty DataFrames, or invalid group_ids.

### Structural

11. **No `pyproject.toml`.** No dependency management, no installable package.

12. **No tests.** Notebooks are the only verification.

13. **`dev.ipynb` duplicates the class.** 971 lines with the entire `NetworkOptimizer` class inline — source of truth ambiguity.

14. **Empty README.** No documentation.

---

## Performance characteristics

| Scenario | Rounds | Time/round | Total | Final score |
|---|---|---|---|---|
| Empty network, 15 groups | 10 | 0.8s | 8.0s | 0.9736 |
| Existing network (15 groups) | 3 | 1.3s | 4.2s | 1.0000 |

Bottleneck: adequacy calculation per successor. Each call does 4+ DataFrame merges and groupbys. With N groups, each round performs N adequacy calculations.

---

## Key design decisions to make before refactoring

1. **Move types:** Additions only (current), or additions + removals + swaps? Swaps are the biggest algorithmic improvement and keep the code simple.

2. **Data generation:** Fix the pool/network overlap. Pool should contain groups NOT in the existing network.

3. **Scalability target:** Current code works for ~1000 members × ~100 providers. What's the target scale? (10K members × 1K providers? 100K × 5K?)

4. **Python version:** Current `__pycache__` shows 3.7 and 3.9. Agent uses 3.12. Target 3.10+?

5. **Multi-region:** Adequacy requirements already have a `county` column. Current synthetic data is single-county. Should the refactored version support multi-county out of the box?
