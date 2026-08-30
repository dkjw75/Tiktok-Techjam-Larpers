# Manual Experiment Handoff — KuaiRand-Pure

**Owner: you (manual track). Written: 30 Aug 2026. Deadline: 1 Sep 12:00 SGT.**

You run this track by hand. I run the agent track. We do not touch each other's
files. Everything here is validation-only — the test split is never loaded by
any script in `manual/`.

---

## 1. Where we actually stand (verified, not claimed)

| Thing | Value | Source |
|---|---|---|
| Organizer FM, 3-seed **valid** mean | **0.6014399** | `runs_baseline_calibration/baseline_calibration.json` |
| Organizer FM **test** (reference repro) | 0.5953 | `docs/environment.md` |
| Best candidate the agent ever produced | 0.598824 (**−0.0028**) | `runs_agent_real/metrics.csv` |
| Champion right now | `"baseline"` | `runs_agent_real/state.json` |

**We have never beaten the baseline.** Everything below exists to change that.

**Target: 3-seed mean valid primary ≥ 0.604440** (baseline + 0.003). Minimum
evidence threshold is +0.002; we aim past it because per-seed noise is ~0.0008.

### Already fixed today (don't redo)
- `research_agent/runner.py` was missing `_file_sha256` → every completed run
  raised `NameError`. Fixed. **Test suite now 110/110 green** (was 17 fail + 2 error).
- `submission_baseline_floor.csv` written and `--check` passed (170,588 rows).
  That is our guaranteed floor. Do not delete it.

### Live results from this harness (seed 0, full fidelity 40/4)

| experiment | fields | valid primary | delta | note |
|---|---|---|---|---|
| `base` | 5 | 0.601470 | +0.00003 | reproduces organizer FM exactly — harness is trustworthy |
| `base,uwatch` | 6 | 0.601640 | +0.00020 | user watch ratio alone: nothing |
| `base,vwatch` | 6 | 0.600393 | −0.00105 | video watch ratio alone: **negative** |
| `base,watch` (= u+v) | 7 | 0.602741 | +0.00130 | best at seed 0 |
| `base,watch` **3 seeds** | 7 | **0.602436 +/- 0.00048** | **+0.00100** | **FAILS gate: 0/3 seeds over +0.002** |
| `base,watch,awatch` | 8 | 0.600790 | −0.00065 | third watch field kills it |
| `base,item,author,ua` | 11 | 0.601877 | +0.00044 | target-rate buckets ~redundant with ID crosses |
| `base,item,author,ua,ut,um,watch,time` | 13 | 0.600287 | −0.00115 | more fields actively hurt |
| loss `pairwise` (agent, earlier) | 5 | 0.598824 | −0.00278 | rejected |
| loss `listwise:t1` | 5 | 0.598970 | −0.00247 | rejected |
| loss `listwise:t1_bce25` | 5 | 0.599678 | −0.00176 | rejected |

### Ensemble results (3 seeds, within-user rank blend)

| ensemble | valid primary | vs baseline | vs best member | seeds ≥ +0.002 |
|---|---|---|---|---|
| `fm+watch` | 0.602271 | +0.00083 | **−0.00050** | 0/2 |
| `fm+listwise` | 0.602140 | +0.00070 | +0.00053 | 0/2 |
| `watch+listwise` | 0.602807 | +0.00137 | +0.00037 | 0/3 |
| `fm+watch+listwise` | 0.602926 | +0.00149 | +0.00049 | 0/3 |
| `fm+watch+item+listwise` | 0.603362 | +0.00192 | +0.00093 | 1/3 |
| `+watchtime` (5-way) | 0.603809 | +0.00237 | +0.00137 | **3/3** |
| **`+pairwise` (6-way)** | **0.603945 +/- 0.00027** | **+0.00251** | **+0.00151** | **3/3** |
| 6-way, equal weights | 0.603426 | +0.00199 | +0.00099 | 1/3 |

**The mechanism is confirmed and it is diversity, not count.** `fm+watch` blends
two models that share a backbone and 5 of 7 fields — it lands *below* its own
best member. Every blend containing `listwise`, whose objective differs, beats
its best member. Each additional decorrelated member adds roughly +0.0004, and
the 4-way is now within 0.00008 of the +0.002 gate.

This is the only lever in the whole project that has behaved monotonically, and
the **6-way blend is the first candidate to clear the +0.002 gate on all three
seeds**: 0.603945 +/- 0.00027, +0.00251 over baseline.

Per-member gains diminish (+0.00045 for member 5, +0.00014 for member 6), so
roughly 6 members is where this saturates. It clears the minimum evidence
threshold but sits under the +0.003 operational target.

**Fitted weights beat equal weights** (+0.00251 / 3-of-3 vs +0.00199 / 1-of-3),
so the half-user weight sweep is not decoration — it correctly down-weights the
weak members (`watchtime` 0.5984, `pairwise` 0.5978). Keep it.

### The current champion

```
fm + watch + item + listwise + watchtime + pairwise
within-user percentile rank blend, weights fitted on half the validation users
valid primary 0.603945 +/- 0.000269   (+0.00251 vs baseline, 3/3 seeds)
```

Reproduce with:

```bash
./.venv/Scripts/python.exe manual/exp_ensemble.py blend   --members fm,watch,item,listwise,watchtime,pairwise --seeds 0,1,2 --step 0.2
```

### Final ensemble evidence (all 3 seeds, rank blend, step 0.2)

| configuration | primary | delta | seeds >= +0.002 | spread |
|---|---|---|---|---|
| 4-way | 0.603362 | +0.00192 | 1/3 | - |
| 5-way | 0.603809 | +0.00237 | 3/3 | - |
| **6-way (agent champion exp_008)** | **0.603945** | **+0.00251** | **3/3** | 0.000269 |
| **8-way + fm_k8 + fm_k32 (best measured)** | **0.604007** | **+0.00257** | **3/3** | **0.000186** |
| 6-way + pop | 0.603945 | +0.00251 | 3/3 | 0.000269 |
| 6-way + din | 0.603811 | +0.00237 | 2/3 | - |
| 6-way, z-score space | 0.603750 | +0.00231 | 2/3 | - |
| 6-way, step 0.1 weights | 0.603899 | +0.00246 | 3/3 | 0.000410 |

### Five hypotheses tested, four refuted

1. **Item popularity as a member** (0.5807, maximally decorrelated) -> weight 0.0,
   blend bit-identical. Decorrelation alone is not sufficient; a member must
   also carry signal.
2. **DIN target attention** -> -0.00013 in-blend, -0.00016 standalone. Cut.
3. **Score-space (z-score) blending instead of rank fusion** -> -0.00020 and
   2/3 seeds. Rank fusion retained.
4. **Finer step-0.1 blend weights** -> -0.00005 and spread nearly doubled
   (0.000269 -> 0.000410). The finer grid overfits the weight-fitting half.
   Coarse weights generalize better.
5. **Capacity-variant members k=8/k=32** -> +0.00006. The only positive, and
   negligible.

Per-member gains across the whole series: +0.00045, +0.00014, +0.00006. That is
a converged sequence. **More members will not reach the +0.003 target.**

### Reproduce the winner

```bash
./.venv/Scripts/python.exe manual/train_winner.py --seeds 0,1,2
```

### Two findings that should redirect the whole plan

**1. The loss branch is dead.** The organizer's #1 recommendation was "change the
objective". Pairwise scored −0.0028 and listwise-softmax −0.0025. Both are well
outside the plan's own abandon threshold of −0.001. Pointwise BCE gives a far
denser gradient — every one of 1.14M rows contributes — while a per-user slate
softmax only draws signal from users with mixed labels, one term per slate.
**Phase 1 is closed as a documented negative result.** Do not spend hours tuning
its learning rate.

**2. Feature count is the enemy, feature quality is the lever.** 2 watch fields
beat 6 target-rate fields; 8 and 13 fields both lost to the 5-field baseline
outright. Every extra field adds noise to the FM's second-order interaction sum.
**Search narrow, not wide.**

**3. The `watch` gain lives in the interaction, not the fields.** Isolated,
`u_watch` gives +0.0002 and `v_watch` gives **−0.0010** — but together they give
+0.0013. Neither field is predictive alone; the FM cross "does this user's
typical watch depth match this video's typical watch depth" is what carries the
signal. That is a real and defensible mechanism, and it is a good line for the
writeup: it is exactly what a factorization machine is *for*.

**4. The watch lead did not survive three seeds.** Seed 0 said +0.00130. The
3-seed mean is **+0.000996 +/- 0.000479, with 0 of 3 seeds clearing +0.002**.
The effect is real and consistently positive — all three seeds beat baseline —
but it is roughly 2 sigma and less than half the target. **`base,watch` does not
clear the gate and must not be shipped as the champion on its own.**

This is exactly why the 3-seed gate exists. Had we promoted on seed 0 we would
have shipped a 0.0013 "win" that is mostly noise.

---

## 1b. Revised plan — what the evidence changed

The original 5-phase plan was sound when written. Six hours of real runs have
since invalidated two of its phases. Updated:

| Original phase | Status | Why |
|---|---|---|
| Phase 0 freeze contract | **done** | 3-seed baseline frozen, harness reproduces FM exactly |
| Phase 1 listwise objective | **dead** | t1 −0.0025, t1_bce25 −0.0018, pairwise −0.0028 |
| Phase 2 history features | **live, narrow** | only `watch` wins; every wide combination loses |
| Phase 3 multi-task | **deprioritised** | it adds heads to a backbone whose loss changes all failed |
| Phase 4 time features | **tested, negative** | `time` is in the all-groups run that scored −0.0012 |
| Phase 5 ensemble | **still the best remaining lever** | see below |

**The honest read:** this dataset is stubborn, and the organizer was right that
the bottleneck is neither features nor capacity — but they were wrong that the
loss is the answer. We have tested their #1 recommendation three ways and it
loses every time.

**Where that leaves the remaining bets, in priority order:**

1. **Rank ensembling (Phase 5) — now the top bet.** We have several models that
   fail in *different* directions: pointwise FM (0.6014), listwise FM (0.5990),
   watch-FM (0.6024). Within-user rank averaging is perfectly aligned to the
   metric, cannot leak, needs no retraining, and is the classic way to turn a set
   of mediocre-but-decorrelated models into a real gain. It is currently unbuilt
   and it is the highest expected value work left. **Start here.**
2. **Squeeze `watch` past the gate.** It is +0.001 and positive on all 3 seeds —
   a real effect that is simply too small. The knobs in Step 2b (bin count,
   smoothing prior, watch-ratio cap) have not been touched at all, and they
   change the signal without adding fields. A 2x on this effect clears the gate.
3. **`watch` as an ensemble member** rather than as the champion. Its errors are
   likely decorrelated from plain FM, which makes it more valuable inside (1)
   than on its own.

**Expectation setting:** we are realistically playing for +0.002 to +0.005, not
+0.02. Say so in the writeup. The "research process" half of the judging rewards
a rigorous negative result far more than a number nobody can reproduce.

---

## 2. Division of labour

| Me (agent track) | You (manual track) |
|---|---|
| `research_agent/`, `data.py`, `tests/` | `manual/` only |
| Phase 1 listwise via the agent loop | Phase 2 feature search (below) |
| Widening the direction catalogue | Deciding which feature groups actually win |
| The 6h autonomous run + finalization | Handing me the winning group list |

**Do not edit** `evaluate.py`, `data.py`, `baseline.py`, `submit.py`, or anything
in `research_agent/`. If you want a feature in the agent, tell me the group name
and I port it into `data.py` with lineage re-calibration.

---

## 3. Setup (once)

```bash
cd "C:/Users/NewName/Documents/ChatGPT/Titok techjem'/audit_repo"
```

**Environment gotcha — you will hit this.** The organizer scripts print Chinese,
and the Windows console is cp1252. Without this, `submit.py` crashes *after*
doing its work, which looks like a failure but isn't:

```bash
export PYTHONIOENCODING=utf-8
```

Always use the repo venv, never global Python:

```bash
./.venv/Scripts/python.exe --version
```

---

## 4. Your two commands

### Feature search (your main job)

```bash
./.venv/Scripts/python.exe manual/exp_features.py --groups base,watch --seeds 0
```

Available groups — combine them comma-separated, always keeping `base`:

| group | fields added | what it is |
|---|---|---|
| `item` | `vid_rate`, `vid_cnt` | smoothed video long-view rate + exposure count |
| `author` | `auth_rate`, `auth_cnt` | same, per author |
| `ua` | `ua_rate`, `ua_cnt` | this user's history with this author |
| `ut` | `ut_rate` | this user's history with this tab |
| `um` | `um_rate`, `utag_rate` | this user's history with this music / tag |
| `watch` | `u_watch`, `v_watch` | mean watch ratio, per user and per video |
| `time` | `hour`, `video_age` | hour-of-day bucket, days since upload |

### Rank ensemble (the top remaining bet)

Two steps. `dump` trains a member once and caches its validation predictions;
`blend` then sweeps weights for free, so you only pay training cost once.

```bash
./.venv/Scripts/python.exe manual/exp_ensemble.py dump  --member fm       --seeds 0,1,2
./.venv/Scripts/python.exe manual/exp_ensemble.py dump  --member watch    --seeds 0,1,2
./.venv/Scripts/python.exe manual/exp_ensemble.py dump  --member listwise --seeds 0,1,2
./.venv/Scripts/python.exe manual/exp_ensemble.py blend --members fm,watch --seeds 0,1,2
```

Members: `fm` (5-field baseline), `watch` (base+watch), `item`
(base+item/author/ua), `watchtime`, `listwise` (torch listwise-t1).

Three design points that keep this defensible:

- **Blending is on within-user percentile ranks**, not raw scores. The metric is
  a within-user ranking metric, the members are on wildly different score
  scales, and rank averaging cannot leak.
- **The weight is fitted on validation days 22–25 and verified on days 26–28**
  before full validation is reported. The blend weight is therefore not fitted
  on the number we report. Watch for `early` and `late` disagreeing — that means
  the weight is overfitting the slice, and the blend should be treated as noise.
- **Row alignment is asserted**, so a NumPy member and the torch member cannot
  be silently blended out of order.

The line that matters in the output is `delta vs best single member`. An
ensemble that only matches its best member is not worth shipping.

### Loss search (run these if I get blocked)

```bash
./.venv/Scripts/python.exe manual/exp_loss.py --loss listwise --variant t1 --seeds 0
```

Variants: `t1` (T=1.0), `t05` (T=0.5), `t1_bce25` (0.75·listwise + 0.25·BCE).
These are exactly candidates A/B/C from the plan; the code already exists in
`research_agent/models/torch_fm.py`.

---

## 5. The search order — do it in this sequence

Each run is ~1 minute at seed 0. **Use seed 0 only while searching.** Only spend
3 seeds on something that already clears +0.002 at seed 0.

**Step 1 — find out which half of `watch` is doing the work.** I have these
three running; take whatever I have not finished:

```bash
./.venv/Scripts/python.exe manual/exp_features.py --groups base,vwatch --seeds 0
./.venv/Scripts/python.exe manual/exp_features.py --groups base,uwatch --seeds 0
./.venv/Scripts/python.exe manual/exp_features.py --groups base,watch,awatch --seeds 0
```

`vwatch` (video mean watch ratio) varies *inside* a user's slate, so it can move
the ranking directly. `uwatch` is constant per user and can only act through the
FM's second-order crosses. Knowing which one carries the gain tells you where to
dig next — that is worth more than another blind combination.

**Step 2 — add exactly one group at a time to the Step-1 winner.** Given that 13
fields lost to 5, the bar is high: if a group does not add ≥ +0.0003 at seed 0,
drop it permanently and write that down. Do not re-test dropped groups in new
combinations "just in case" — that is how a day disappears.

**Step 2b — if the watch thread stalls, vary the signal not the field count.**
Cheap edits inside `exp_features.py`, one at a time:
- `N_BINS` 20 → 40 or 10 (bin resolution may be throttling a continuous signal)
- `SMOOTH_PRIOR` 20 → 5 or 50 (how fast a rare video earns its own estimate)
- the `min(..., 3.0)` cap in `watch_ratio` → 1.0 or 5.0 (replay behaviour)

These are one-line changes with real upside and they keep the field count flat.

**Step 3 — the moment any combination reaches +0.002 at seed 0**, stop searching
and run it on three seeds:

```bash
./.venv/Scripts/python.exe manual/exp_features.py --groups <winner> --seeds 0,1,2
```

**Step 4 — hand me the winning group string.** That is the deliverable. I port
it into `data.py`, re-run `baseline_calibration.py` (the lineage hashes change
when preprocessing changes, so the agent will reject every comparison until I
do), and hand it to the agent.

### If nothing reaches +0.002 by mid-afternoon

Stop the feature search and switch to `exp_loss.py`. Report what you have. A
negative result cleanly documented is worth real marks on the "research process"
half of the judging — the run log showing we tested and rejected six directions
with evidence is a better story than one lucky number.

---

## 6. Leakage rules — non-negotiable

These are already enforced in `exp_features.py`. Do not weaken them; a leak
means disqualification, and it always looks like a fantastic result first.

1. **Train rows are prequential.** A train row dated `d` sees statistics built
   only from rows dated `< d`. A row can never contribute to its own feature.
   Day-1 rows fall back to the global prior. This is the `build_statistics`
   sweep — read it before you change anything.
2. **Validation rows see the whole train split and nothing else.** Never a
   validation label, never another validation row.
3. **Quantile bin edges are fitted on train only** (`bucketize`).
4. **The test split is never loaded.** `load_rich` drops any row dated past
   20220428 at read time.
5. **`play_time_ms` of the row being scored is forbidden.** It is same-impression
   information and is effectively the label. Only *historical aggregates* of it
   are legal — that is what `watch` uses.

**Red flag:** if a feature group jumps you past ~0.65, you have leaked. Stop and
tell me. The oracle ceiling is 0.8645 and the realistic target is 0.605–0.62.

### One judgment call to be aware of

`video_features_statistic_pure.csv` (60+ engagement columns) is unused and legal
— it ships with the required dataset. But it is a platform-wide aggregate with
no time window, so it may encode post-train information. **Do not use it in the
main candidate.** If you want to test it, do it as a clearly separate labelled
experiment so we can disclose it honestly. I would rather win by +0.004 we can
defend than +0.02 a judge can question.

---

## 7. Gates a candidate must clear

1. **Screen** — completes, GAUC and nDCG@5 both move sensibly, no contamination.
2. **Full run** — early-stops inside the official 40 epochs / patience 4. If it
   reports `max_epochs_truncated` the result is **not comparable** and does not
   count. The harness prints this on every line.
3. **Seeds** — 3-seed mean delta > +0.002, target ≥ +0.003, and at least 2 of 3
   seeds individually over +0.002.
4. **Robustness** — does not collapse on the later validation days.
5. **Submission** — `submit.py --check` passes before the single final test call.

`exp_features.py` prints the gate verdict on every run. `FAIL` means keep going.

---

## 8. Log every run here

Fill this in as you go. This table is evidence for the writeup, not bookkeeping.

| # | groups / loss | seeds | primary mean | delta | epochs / stop | keep? |
|---|---|---|---|---|---|---|
| 1 | `base` | 0 | 0.601470 | +0.00003 | 11, early | reference |
| 2 | `base,item,author,ua` | 0 | 0.601877 | +0.00044 | 8, early | no |
| 3 | `base,watch` | 0 | 0.602741 | **+0.00130** | 7, early | **pending 3-seed** |
| 4 | `base,…,watch,time` (all) | 0 | 0.600287 | −0.00115 | 6, early | no |
| 5 | `base,vwatch` | 0 | 0.600393 | −0.00105 | 9, early | no |
| 6 | `base,uwatch` | 0 | 0.601640 | +0.00020 | 9, early | no |
| 7 | `base,watch,awatch` | 0 | 0.600790 | −0.00065 | 7, early | no |
| 8 | `listwise:t1` | 0 | 0.598970 | −0.00247 | 9, early | no |
| 9 | `listwise:t1_bce25` | 0 | 0.599678 | −0.00176 | 6, early | no |
| 10 | | | | | | |
| 11 | | | | | | |

---

## 9. If something breaks

| Symptom | Cause | Fix |
|---|---|---|
| `UnicodeEncodeError: charmap` | cp1252 console vs Chinese output | `export PYTHONIOENCODING=utf-8` — the work already completed |
| `No module named pytest` | not installed in the venv | `./.venv/Scripts/python.exe -m pip install pytest` |
| `FileNotFoundError` on a CSV | wrong `--data_dir` | pass `--data_dir ./KuaiRand-Pure/data` |
| Result identical across two different configs | you changed something that isn't wired in | check the `fields=N (5 base + M added: [...])` line |
| `max_epochs_truncated` | hit the 40-epoch ceiling | not comparable — lower the LR or report it as non-comparable |

Anything else, or any result that looks too good: stop and ping me rather than
building on top of it.
