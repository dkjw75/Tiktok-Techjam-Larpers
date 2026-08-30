"""Manual Phase-5 harness: within-user rank ensembling.

Two modes.

  dump   train one member at one seed, cache its validation predictions
  blend  load cached members, sweep blend weights, report the gate

Blending happens on within-user percentile ranks, not raw scores: the metric is
a within-user ranking metric, members are on wildly different score scales, and
rank averaging cannot leak.

Weights are fitted on one half of the validation USERS and verified on the other
half before full validation is reported, so the blend weight is never fitted on
the number we report. The split is by user, not by date: see `user_half` for why
date-slicing silently measures a different population on this metric.

Validation-only. The test split is never loaded.

Usage
-----
  python manual/exp_ensemble.py dump --member fm       --seeds 0,1,2
  python manual/exp_ensemble.py dump --member watch    --seeds 0,1,2
  python manual/exp_ensemble.py dump --member listwise --seeds 0,1,2
  python manual/exp_ensemble.py blend --members fm,watch --seeds 0,1,2
"""
import argparse
import itertools
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import evaluate

import exp_features as FX  # noqa: E402  reuse the validated feature builder

PRED_DIR = Path(__file__).parent / "preds"
EARLY_DAYS = (20220422, 20220425)
LATE_DAYS = (20220426, 20220428)

# member name -> feature groups for the NumPy FM path; "listwise" uses torch.
NUMPY_MEMBERS = {
    "fm": "",
    "watch": "watch",
    "item": "item,author,ua",
    "watchtime": "watch,time",
    "fm_k8": "",
    "fm_k32": "",
}
# Organizer measured embedding size as flat standalone (0.5895/0.5902/0.5887).
# A different k is still a different error surface, and decorrelation is what
# the blend actually pays for.
MEMBER_K = {"fm_k8": 8, "fm_k32": 32}


POP_MEMBERS = {"pop": {}}


def dump_pop_member(seed, data_dir):
    """Smoothed item popularity from train only. Untrained, seconds to build."""
    import collections

    splits = FX.load_rich(data_dir)
    pos, imp = collections.Counter(), collections.Counter()
    for row in splits["train"]:
        imp[row[FX.D_VID]] += 1
        pos[row[FX.D_VID]] += row[FX.D_Y]
    grand = sum(pos.values()) / sum(imp.values())
    prior = 20.0

    def score(video):
        if not imp[video]:
            return grand
        return (pos[video] + prior * grand) / (imp[video] + prior)

    valid = splits["valid"]
    scores = np.asarray([score(r[FX.D_VID]) for r in valid], dtype=np.float64)
    labels = np.asarray([r[FX.D_Y] for r in valid], dtype=np.float32)
    users = [r[FX.D_USER] for r in valid]
    dates = np.asarray([r[FX.D_DATE] for r in valid], dtype=np.int32)
    primary = evaluate(users, labels, scores)["primary"]
    return scores, labels, users, dates, primary, 0


def dump_numpy_member(member, seed, data_dir):
    groups = [g for g in NUMPY_MEMBERS[member].split(",") if g]
    splits = FX.load_rich(data_dir)
    stats, names = FX.build_statistics(splits, groups)
    buckets = FX.bucketize(stats, names)
    enc, dim, _width = FX.encode(splits, buckets, groups)
    Xtr, ytr, _ = enc["train"]
    Xva, yva, uva = enc["valid"]

    import baseline as B

    model = B.FM(dim, k=MEMBER_K.get(member, 16), lr=0.001, seed=seed)
    rng = np.random.default_rng(seed)
    best, state, bad, epochs = -1.0, None, 0, 0
    for _ in range(40):
        epochs += 1
        order = rng.permutation(len(ytr))
        for i in range(0, len(order), 8192):
            model.step(Xtr[order[i:i + 8192]], ytr[order[i:i + 8192]])
        primary = evaluate(uva, yva, model.predict(Xva))["primary"]
        if primary > best + 1e-5:
            best, bad = primary, 0
            state = (model.V.copy(), model.W.copy(), np.float32(model.b))
        else:
            bad += 1
            if bad >= 4:
                break
    model.V, model.W, model.b = state
    scores = model.predict(Xva)
    dates = np.asarray([r[FX.D_DATE] for r in splits["valid"]], dtype=np.int32)
    return scores, np.asarray(yva, dtype=np.float32), uva, dates, best, epochs


TORCH_MEMBERS = {
    # name -> (loss, listwise variant)
    "listwise": ("listwise", "t1"),
    "listwise05": ("listwise", "t05"),
    "pairwise": ("pairwise", "t1"),
}


def dump_torch_member(member, seed, data_dir):
    from research_agent.data_boundary import load_research_splits
    from research_agent.models.torch_fm import run_torch_fm_candidate
    from research_agent.runner import PreparedData

    loss, variant = TORCH_MEMBERS[member]
    splits = load_research_splits(data_dir, ("train", "valid"))
    prepared = PreparedData(splits["train"], splits["valid"])
    config = {
        "loss": loss, "objective_variant": variant,
        "learning_rate": 0.001, "l2": 1e-6, "embedding_dim": 16,
        "batch_size": 8192, "epochs": 40, "patience": 4, "seed": seed,
    }
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        run_dir.mkdir()
        out = run_torch_fm_candidate(prepared, config, run_dir)
    dates = np.asarray([r[0] for r in splits["valid"]], dtype=np.int32)
    return (np.asarray(out.scores, dtype=np.float64),
            np.asarray(out.labels, dtype=np.float32),
            list(out.user_ids), dates,
            float(out.metadata["best_metrics"]["primary"]),
            int(out.metadata["epochs_run"]))


DIN_MEMBERS = {"din": {}, "din_lr3e4": {"learning_rate": 0.0003}}


def dump_din_member(member, seed, data_dir):
    """DIN is architecturally orthogonal to every FM member -- ideal ensemble fuel."""
    from research_agent.data_boundary import load_research_splits
    from research_agent.models.din import run_din_candidate
    from research_agent.runner import PreparedData

    splits = load_research_splits(data_dir, ("train", "valid"))
    prepared = PreparedData(splits["train"], splits["valid"])
    config = {"seed": seed, "epochs": 40, "patience": 4, "max_len": 20,
              **DIN_MEMBERS[member]}
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        run_dir.mkdir()
        out = run_din_candidate(prepared, config, run_dir)
    dates = np.asarray([r[0] for r in splits["valid"]], dtype=np.int32)
    return (np.asarray(out.scores, dtype=np.float64),
            np.asarray(out.labels, dtype=np.float32),
            list(out.user_ids), dates,
            float(out.metadata["best_primary"]),
            int(out.metadata["epochs_run"]))


def do_dump(args):
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    for seed in [int(s) for s in args.seeds.split(",")]:
        tick = time.time()
        if args.member in POP_MEMBERS:
            scores, labels, users, dates, primary, epochs = dump_pop_member(
                seed, args.data_dir)
        elif args.member in DIN_MEMBERS:
            scores, labels, users, dates, primary, epochs = dump_din_member(
                args.member, seed, args.data_dir)
        elif args.member in TORCH_MEMBERS:
            scores, labels, users, dates, primary, epochs = dump_torch_member(
                args.member, seed, args.data_dir)
        else:
            scores, labels, users, dates, primary, epochs = dump_numpy_member(
                args.member, seed, args.data_dir)
        np.savez_compressed(
            PRED_DIR / f"{args.member}_seed{seed}.npz",
            scores=scores, labels=labels,
            users=np.asarray(users, dtype=np.str_), dates=dates,
        )
        print(f"  {args.member} seed {seed}: primary {primary:.6f} "
              f"({epochs} ep, {time.time() - tick:.0f}s) -> cached")


def within_user_percentile(scores, users):
    """Map scores to [0,1] percentile rank inside each user, average ties."""
    scores = np.asarray(scores, dtype=np.float64)
    users = np.asarray(users)
    out = np.empty(len(scores), dtype=np.float64)
    order = np.argsort(users, kind="stable")
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and users[order[end]] == users[order[start]]:
            end += 1
        idx = order[start:end]
        n = len(idx)
        if n == 1:
            out[idx[0]] = 0.5
        else:
            local = scores[idx]
            ranks = np.empty(n, dtype=np.float64)
            sub = np.argsort(local, kind="stable")
            i = 0
            while i < n:
                j = i
                while j + 1 < n and local[sub[j + 1]] == local[sub[i]]:
                    j += 1
                ranks[sub[i:j + 1]] = (i + j) / 2.0
                i = j + 1
            out[idx] = ranks / (n - 1)
        start = end
    return out


def load_members(members, seed, space="rank"):
    ranks, labels, users, dates = [], None, None, None
    for name in members:
        path = PRED_DIR / f"{name}_seed{seed}.npz"
        if not path.is_file():
            raise SystemExit(f"missing cache: {path}\nrun: dump --member {name} --seeds {seed}")
        blob = np.load(path, allow_pickle=False)
        if labels is None:
            labels, users, dates = blob["labels"], blob["users"], blob["dates"]
        else:
            # Members must be row-aligned or an elementwise blend is meaningless.
            if not np.array_equal(users, blob["users"]) or not np.array_equal(labels, blob["labels"]):
                raise SystemExit(f"row misalignment between members at seed {seed}: {name}")
        transform = within_user_zscore if space == "zscore" else within_user_percentile
        ranks.append(transform(blob["scores"], blob["users"]))
    return np.vstack(ranks), labels, users, dates


def within_user_zscore(scores, users):
    """Standardize scores inside each user, keeping magnitude information.

    Rank fusion discards how *far* apart two candidates were; z-scoring keeps
    that while still removing per-user scale differences between members.
    Tested against rank fusion rather than assumed better.
    """
    scores = np.asarray(scores, dtype=np.float64)
    users = np.asarray(users)
    out = np.zeros(len(scores), dtype=np.float64)
    order = np.argsort(users, kind="stable")
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and users[order[end]] == users[order[start]]:
            end += 1
        idx = order[start:end]
        local = scores[idx]
        spread = local.std()
        out[idx] = (local - local.mean()) / spread if spread > 1e-12 else 0.0
        start = end
    return out


def user_half(users, parity):
    """Split by USER, not by date.

    Date-slicing looked natural but is wrong for this metric: GAUC only counts
    users with 0 < positives < impressions, and nDCG@5 saturates on all-positive
    or all-negative slates. Cutting a user's slate across days changes which
    users qualify at all, so a date slice measures a different population than
    full validation and its scores are not comparable. Splitting on user keeps
    every slate intact, so both halves measure the same quantity.
    """
    return np.asarray([int(u) % 2 == parity for u in users])


def score_half(blended, labels, users, parity):
    mask = user_half(users, parity)
    return evaluate(list(users[mask]), labels[mask], blended[mask])["primary"]


def do_blend(args):
    members = [m.strip() for m in args.members.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",")]
    grid = [w for w in np.round(np.arange(0.0, 1.0001, args.step), 4)]

    per_seed = []
    for seed in seeds:
        ranks, labels, users, dates = load_members(members, seed, args.blend_space)
        # Weight vectors over the simplex at the requested resolution.
        if len(members) == 2:
            candidates = [(w, round(1.0 - w, 4)) for w in grid]
        else:
            candidates = [c for c in itertools.product(grid, repeat=len(members))
                          if abs(sum(c) - 1.0) < 1e-9]
        if args.equal:
            # ponytail: uniform weights need no fitting and cannot overfit the
            # fit-half. Only keep the sweep if it actually beats this.
            candidates = [tuple([1.0 / len(members)] * len(members))]
        best_w, best_early = None, -1.0
        for weights in candidates:
            blended = np.tensordot(np.asarray(weights), ranks, axes=(0, 0))
            fit = score_half(blended, labels, users, 0)
            if fit > best_early:
                best_early, best_w = fit, weights
        blended = np.tensordot(np.asarray(best_w), ranks, axes=(0, 0))
        late = score_half(blended, labels, users, 1)
        full = evaluate(list(users), labels, blended)["primary"]
        singles = {
            name: evaluate(list(users), labels, ranks[i])["primary"]
            for i, name in enumerate(members)
        }
        per_seed.append((best_w, best_early, late, full, singles))
        parts = "  ".join(f"{n} {v:.6f}" for n, v in singles.items())
        print(f"  seed {seed}: w={best_w}  fit-half {best_early:.6f}  held-half {late:.6f}  "
              f"full {full:.6f}   [members: {parts}]")

    mean = statistics.mean(item[3] for item in per_seed)
    spread = statistics.pstdev([item[3] for item in per_seed]) if len(per_seed) > 1 else 0.0
    wins = sum(item[3] > args.baseline_primary + 0.002 for item in per_seed)
    delta = mean - args.baseline_primary
    best_single = max(
        statistics.mean(item[4][name] for item in per_seed) for name in members
    )
    print(f"\nensemble={'+'.join(members)} step={args.step}")
    print(f"  best single member (3-seed mean): {best_single:.6f}")
    print(f"  ensemble valid primary mean {mean:.6f} +/- {spread:.6f}")
    print(f"  delta vs frozen baseline {args.baseline_primary:.6f}: {delta:+.6f}")
    print(f"  delta vs best single member:      {mean - best_single:+.6f}")
    print(f"  seed wins over +0.002: {wins}/{len(per_seed)}")
    if delta >= 0.003 and wins >= 2:
        verdict = "PASS (>= +0.003 target)"
    elif delta > 0.002 and wins >= 2:
        verdict = "MARGINAL (> +0.002 but under target)"
    else:
        verdict = "FAIL"
    print(f"  GATE: {verdict}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("dump", "blend"))
    parser.add_argument("--data_dir", default="./KuaiRand-Pure/data")
    parser.add_argument("--member", default="fm",
                        choices=tuple(NUMPY_MEMBERS) + tuple(TORCH_MEMBERS)
                                + tuple(DIN_MEMBERS) + tuple(POP_MEMBERS))
    parser.add_argument("--members", default="fm,watch")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--blend_space", default="rank", choices=("rank", "zscore"))
    parser.add_argument("--equal", action="store_true",
                        help="use uniform weights instead of fitting them")
    parser.add_argument("--baseline_primary", type=float, default=0.6014399)
    args = parser.parse_args()
    (do_dump if args.mode == "dump" else do_blend)(args)


if __name__ == "__main__":
    main()
