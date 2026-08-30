# Final Audit Report — KuaiRand-Pure Autonomous Research Agent

**Frozen commit:** `2629af4` on `codex/competition-valid-agent`
**Base:** `8557e8d` (`agentv1.0`)

This report states what is verified, what is not, and what went wrong. Numbers
are traceable to files in this repository.

---

## 1. Headline

A bounded autonomous research loop operating over a human-approved,
evidence-informed direction catalogue. It reproduced the organizer FM baseline
exactly, tested and rejected the organizer's two top-recommended directions with
measured evidence, and accepted a rank-ensemble candidate under three-seed
matched confirmation.

**This was not an LLM-invented result.** No API key was configured; the run used
the deterministic planner and the log records `LLM input tokens: 0` with an
explicit fallback event. The ensemble direction was identified by human-run
manual harnesses and then added to the approved catalogue.

---

## 2. Benchmark fidelity

| check | result |
|---|---|
| Organizer `evaluate.py` / `baseline.py` / `data.py` / `submit.py` | byte-identical to the starter kit after newline normalization |
| Evaluator SHA-256 | `ecfde28392eb14fec4f488083251df50624e1af2b86278b962daecfb42d195de` |
| Harness reproduces organizer FM | `base` config → 0.601470, 11 epochs, seed 0 — exact match |
| Organizer FM 3-seed validation mean | 0.6014399039 ± 0.0002746908 |

---

## 3. Scientific results

### Rejected directions (organizer's own top recommendations)

| direction | validation primary | delta |
|---|---|---|
| pairwise BPR | 0.598824 | −0.00278 |
| listwise softmax T=1.0 | 0.598970 | −0.00247 |
| listwise + 0.25 BCE | 0.599678 | −0.00176 |
| DIN target attention (3-seed mean) | 0.601280 | −0.00016 |

Both of the organizer's stated headroom directions — a better ranking objective,
and user behaviour sequences — lose on this benchmark. DIN was implemented with
seven passing leakage tests, so this is a result about the benchmark rather than
an implementation failure.

### Feature findings

| configuration | primary | delta |
|---|---|---|
| 5-field baseline | 0.601470 | +0.00003 |
| + watch-ratio pair | 0.602741 | +0.00130 (seed 0) |
| + watch, 3 seeds | 0.602436 | +0.00100, **0/3 seeds cleared** |
| 13 engineered fields | 0.600287 | −0.00115 |

`u_watch` alone gives +0.0002 and `v_watch` alone **−0.0010**, but together
+0.0013: the signal is the FM cross, not either field.

### What worked

Rank ensembling. Members that individually lose to the baseline improve the
blend, because the metric rewards decorrelated errors.

| ensemble | primary | delta | seeds ≥ +0.002 |
|---|---|---|---|
| 6-way (agent champion) | 0.603945 | +0.00251 | 3/3 |
| 8-way + k8 + k32 (manual, unshipped) | 0.604007 | +0.00257 | 3/3 |

Refuted while searching: item-popularity member (weight 0.0), DIN member
(−0.00013), z-score blending (−0.00020), finer step-0.1 weights (−0.00005).

---

## 4. The leakage incident — disclosed in full

An earlier finalization path constructed `PreparedData(train_rows, test_rows)`.
The ensemble candidate treats its second split as validation, so **every member
selected its best epoch using test labels**.

A finalization process was started under that defective code and killed once the
defect was found.

**Precise statement:** the flawed finalization process accessed test labels
during training but was terminated before a submission, model output, or test
metric was persisted. `runs_ensemble/finalization.json` remains at
`test_access_started` and that transaction will never certify a model. The
validation-selected champion predates the incident and is uncontaminated.

`runs_ensemble` is preserved as historical evidence and is abandoned for
certification purposes.

### Structural fix

`PreparedData` now has three distinct splits — `train_rows` fits,
`validation_rows` selects, `prediction_rows` is scored and never fitted on.
Making it a separate field rather than a calling convention is what prevents the
mistake from recurring.

---

## 5. Reproducibility

Schema-version-2 inference bundles persist member parameters, fitted encoders,
vocabularies, bin edges and statistic tables, loaded with `allow_pickle=False`.

| property | evidence |
|---|---|
| bundle replays validation scores | `rtol=0, atol=0` — bit-identical |
| replay reproduces the certified hash | match |
| replay in a **fresh subprocess** | match |
| replay requires no training rows | encoders load from the bundle |
| array mutation detected | raises |
| lineage + runtime recorded | 7 lineage fields, Python/NumPy/Torch versions |

The finalizer contains no reference to any training entry point; it loads a
bundle and performs inference.

---

## 6. Honest limitations

1. **The LLM never ran.** No `OPENAI_API_KEY`; deterministic planner used
   throughout. Recorded as `LLM input tokens: 0`, fallback count 1.
2. **The result is modest.** +0.0025 validation against an oracle ceiling of
   0.8645 primary — roughly 1% of the available headroom above the baseline.
3. **Validation selection risk.** Members, member sets and blend weights were all
   chosen on validation. The weight fit is protected by a user-half split; member
   selection is not. The baseline's own validation→test gap is −0.006, so the
   test delta may be smaller than +0.0025.
4. **The 8-way configuration measured better but was never agent-certified** and
   is not shipped.
5. **Two manual interventions are recorded** in `runs_ensemble`: a mid-run
   search-space widening, and the killed finalization.
6. **Not published.** No GitHub credential with write access to
   `dkjw75/Tiktok-Techjam-Larpers` is available; nothing has been pushed.

---

## 7. Engineering gates

| gate | status |
|---|---|
| Unit tests | 164 passing |
| Ruff | clean |
| mypy | clean, 34 source files |
| `git diff --check` | clean |
| Organizer parity | 4/4 byte-identical |
| Secret scan | no credentials committed |

---

## 8. Status of the final test boundary

**Not crossed.** No valid submission for the champion exists yet; the only
submission on disk is the organizer FM baseline floor
(`submission_baseline_floor.csv`, 170,588 rows, `submit.py --check` passed).

Recertification under the frozen commit is in progress in `runs_ensemble_v2`.
Finalization requires that recertification to complete and explicit human
authorization.
