# Status Sheet — Codex handoff → current

**Base at handoff:** `8557e8d` (`agentv1.0`), everything uncommitted
**Now:** `152cb70`, working tree clean, 66 files changed since base
**Deadline:** 1 Sep 2026, 12:00 SGT

---

## 1. State at handoff vs now

| | at handoff | now |
|---|---|---|
| Test suite | **17 fail + 2 error** of 110 | **166 pass, 0 fail** |
| Ruff | 4 errors | clean |
| mypy | 9 errors | clean (34 files) |
| Commits | none — all uncommitted | 3 commits, tree clean |
| Organizer parity | unverified | 4/4 byte-identical |
| Finalizer | **retrained at finalization** | inference-only from a bundle |
| Test leakage | **present** | fixed + 6 proving tests |
| Champion | `exp_008`, no bundle | **none certified** — recertifying |
| Submission for champion | none | none |
| Test evaluation | unspent | **unspent** |

---

## 2. DONE

### 2.1 Blocking defects fixed

| defect | detail |
|---|---|
| `_file_sha256` undefined | called at 4 sites in `runner.py`, never defined; every completed run raised `NameError`. All 19 test failures cascaded from it. |
| **Test leakage** | finalization passed test rows as the *validation* argument, so every member early-stopped on test labels |
| **Bundle replay defect** | pruning zero-weight members + renormalizing shifted the validation primary by 1.7e-4; a bundle could not reproduce its own certified score |
| Unregistered candidate | runner allowlist correctly refused the dispatcher; registering it exposed that a router's hash ignores the model it routes to |
| Timeouts | bundle serialization pushed full-fidelity runs past 900s, killing a promotion; budgets raised to 2400s |
| Pause-aware runtime | elapsed counted paused hours as research budget |

### 2.2 Structural work

- **`PreparedData` has three splits** — `train_rows` fits, `validation_rows` selects, `prediction_rows` is scored and never fitted on. A separate field, not a calling convention.
- **Schema-v2 inference bundles** — member parameters, fitted encoders, vocabularies, bin edges, statistic tables; `allow_pickle=False`; per-array dtype/shape/SHA-256.
- **Lineage threaded end to end** — `PreparedData` → runner → worker job → bundle manifest. 8 fields plus Python/NumPy/Torch versions. Kept separate from experiment config.
- **Certificates bind bundle content** — sha256 + size + schema + validation hash, not a mutable path.
- **Model-code drift rejected before test access**; routed hash covers `dispatch`, `ensemble_fm`, `torch_fm`.
- **Stale recovery lock** — dead owner archived and reclaimed with hashed evidence; live/malformed/PID-less fail closed.
- **12-field staged rows**, schema v2, tolerant loader.

### 2.3 Verified on real data

```
124,909 validation rows replayed from the bundle alone
replay primary 0.6038910352389726 == bundle primary
hash match: True          →  REPLAY VERIFIED
```

Also verified: replay in a **fresh subprocess**, replay without training rows,
array-mutation detection, all 4 member states exact to 1e-12.

### 2.4 Science

**Both organizer top-recommendations refuted:**

| direction | primary | delta |
|---|---|---|
| pairwise BPR | 0.598824 | −0.00278 |
| listwise T=1.0 | 0.598970 | −0.00247 |
| listwise + BCE | 0.599678 | −0.00176 |
| DIN (3-seed) | 0.601280 | −0.00016 |

The listwise rejection at **0.599678** has now replicated in three independent
runs under three different code states.

**What worked:** rank ensembling, +0.00251 (3-seed, agent-accepted in the old
run). Members that individually lose improve the blend.

**Also refuted:** popularity member (weight 0), DIN as member (−0.00013),
z-score blending (−0.00020), finer step-0.1 weights (−0.00005).

### 2.5 Test coverage: 110 → 166

| file | tests | file | tests |
|---|---|---|---|
| controller | 18 | finalization_recovery | 15 |
| finalize | 14 | ensemble_checkpoint | 11 |
| bundle_reproducibility | 9 | torch_fm | 9 |
| contracts | 9 | runner / store | 8 / 8 |
| din (leakage) | 7 | loop / search | 7 / 7 |
| no_test_leakage | 6 | llm_planner | 6 |
| shadow_finalization | 5 | safety | 5 |
| others | 24 | | |

### 2.6 Documents

`AUDIT_REPORT.md`, `ANALYSIS.md`, `manual/HANDOFF.md`, `SESSION_STATUS.md`,
plus two manual interventions recorded in `runs_ensemble` that were unlogged.

---

## 3. NOT DONE

| # | item | status |
|---|---|---|
| 1 | **Recertification** | **in progress** — `runs_ensemble_v3` on cycle 2, 2 bundles written, no promotion yet |
| 2 | **Champion certificate** | none valid; `exp_012` invalidated by the pruning defect |
| 3 | **Submission for a champion** | none — only the baseline floor (0.5953) exists |
| 4 | **Final test evaluation** | not crossed |
| 5 | Full 25-check pre-boundary sequence | ~8 of 25 implemented |
| 6 | Clean autonomy replay as a separate artifact | not run |
| 7 | Independent architecture/code review | not run |
| 8 | Push / PR | blocked — no GitHub credential with write access |
| 9 | Devpost, demo material | not started |
| 10 | Unbiased evaluation on 288k random-exposure rows | not run |
| 11 | `runs_ensemble/finalization.json` | still `test_access_started`; directory abandoned for certification |

---

## 4. Open risks

1. **No submittable champion.** If everything stopped now we would score the
   baseline 0.5953.
2. **Validation-selection risk.** Members, member sets and weights were all
   chosen on validation. The weight fit is protected by a user-half split;
   member selection is not. Baseline's own validation→test gap is −0.006, so the
   test delta may be below +0.0025.
3. **The result is modest.** ~1% of the headroom above baseline.
4. **The LLM never ran.** No API key; deterministic planner throughout. Recorded
   honestly as `LLM input tokens: 0`, fallback count 1.
5. **Recertification may find another defect.** It already found two.

---

## 5. Next actions, in order

1. `runs_ensemble_v3` completes promotion + three-seed confirmation
2. **Replay the champion bundle on real validation data and require an exact
   hash match** before accepting the certificate
3. Shadow-finalize with validation as fake test
4. Rerun all gates, commit
5. Request authorization for the single real finalization
6. `submit.py --check`, then publication if a credential appears

**Cut line, unchanged:** do not spend the test boundary until the certified
validation model can be reconstructed solely by loading its immutable bundle.
