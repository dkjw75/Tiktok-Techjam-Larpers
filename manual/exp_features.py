"""Manual Phase-2 experiment harness: leakage-safe behavioural history features.

Standalone. Does NOT import or modify research_agent/ or data.py, so it can run
in parallel with agent work. Validation-only: the test split is never loaded.

Row order and split boundaries replicate data.py exactly, so any winning feature
group ports straight into data.py's `raw()` for the agent.

Usage
-----
  python manual/exp_features.py --groups base
  python manual/exp_features.py --groups base,item
  python manual/exp_features.py --groups base,item,author,ua --seeds 0,1,2
"""
import argparse
import collections
import csv
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import baseline as B
from evaluate import evaluate

SPLITS = {"train": (20220408, 20220421), "valid": (20220422, 20220428)}
SMOOTH_PRIOR = 20.0   # same prior the organizer's `pop` baseline uses
N_BINS = 40           # quantile bins per continuous statistic

D_DATE, D_USER, D_VID, D_AUTH, D_TAB, D_DUR, D_Y, D_HOUR, D_PLAY, D_MUSIC, D_TAG, D_UP = range(12)
TIME_FEATURES = ["hour", "video_age"]


# ---------------------------------------------------------------- data loading
def load_rich(data_dir):
    """data.py's split logic, carrying the extra columns history features need."""
    vmeta = {}
    with open(os.path.join(data_dir, "video_features_basic_pure.csv")) as fh:
        for r in csv.DictReader(fh):
            dt = r["upload_dt"].replace("-", "")
            vmeta[r["video_id"]] = (
                r["author_id"], r["music_id"], r["tag"].split(",")[0],
                int(dt) if dt.isdigit() else 0,
            )
    unk = ("UNK", "UNK", "UNK", 0)

    rows = []
    for f in ("log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv"):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                date = int(r["date"])
                if date > SPLITS["valid"][1]:
                    continue                      # test rows are never loaded
                author, music, tag, upload = vmeta.get(r["video_id"], unk)
                rows.append((
                    date, r["user_id"], r["video_id"], author, r["tab"],
                    float(r["duration_ms"]),
                    1 if r["long_view"] != "0" else 0,
                    int(r["hourmin"]) // 100, float(r["play_time_ms"]),
                    music, tag, upload,
                ))
    return {n: [x for x in rows if lo <= x[0] <= hi] for n, (lo, hi) in SPLITS.items()}


# ------------------------------------------------------- leakage-safe counters
class RateTable:
    """Smoothed positive-rate lookup with strict prequential train semantics.

    Train rows on date d see only rows with date < d, so a row can never
    contribute to its own feature. Validation rows see the whole train split.
    """

    def __init__(self, keyfn):
        self.keyfn = keyfn
        self.pos = collections.Counter()
        self.n = collections.Counter()
        self.total_pos = 0
        self.total_n = 0

    def observe(self, row):
        key = self.keyfn(row)
        self.n[key] += 1
        self.pos[key] += row[D_Y]
        self.total_n += 1
        self.total_pos += row[D_Y]

    def rate(self, row):
        prior = self.total_pos / self.total_n if self.total_n else 0.0
        key = self.keyfn(row)
        return (self.pos[key] + SMOOTH_PRIOR * prior) / (self.n[key] + SMOOTH_PRIOR)

    def count(self, row):
        return self.n[self.keyfn(row)]


class MeanTable:
    """Same prequential discipline for a continuous quantity (watch ratio)."""

    def __init__(self, keyfn, valfn):
        self.keyfn = keyfn
        self.valfn = valfn
        self.total = collections.defaultdict(float)
        self.n = collections.Counter()
        self.grand_total = 0.0
        self.grand_n = 0

    def observe(self, row):
        key, value = self.keyfn(row), self.valfn(row)
        self.total[key] += value
        self.n[key] += 1
        self.grand_total += value
        self.grand_n += 1

    def rate(self, row):
        prior = self.grand_total / self.grand_n if self.grand_n else 0.0
        key = self.keyfn(row)
        return (self.total[key] + SMOOTH_PRIOR * prior) / (self.n[key] + SMOOTH_PRIOR)


def watch_ratio(row):
    return min(row[D_PLAY] / row[D_DUR], 3.0) if row[D_DUR] > 0 else 0.0


# Each group -> list of (name, table factory, which accessor to read).
GROUPS = {
    "item": [("vid_rate", lambda: RateTable(lambda r: r[D_VID]), "rate"),
             ("vid_cnt", lambda: RateTable(lambda r: r[D_VID]), "count")],
    "author": [("auth_rate", lambda: RateTable(lambda r: r[D_AUTH]), "rate"),
               ("auth_cnt", lambda: RateTable(lambda r: r[D_AUTH]), "count")],
    "ua": [("ua_rate", lambda: RateTable(lambda r: (r[D_USER], r[D_AUTH])), "rate"),
           ("ua_cnt", lambda: RateTable(lambda r: (r[D_USER], r[D_AUTH])), "count")],
    "ut": [("ut_rate", lambda: RateTable(lambda r: (r[D_USER], r[D_TAB])), "rate")],
    "um": [("um_rate", lambda: RateTable(lambda r: (r[D_USER], r[D_MUSIC])), "rate"),
           ("utag_rate", lambda: RateTable(lambda r: (r[D_USER], r[D_TAG])), "rate")],
    "watch": [("u_watch", lambda: MeanTable(lambda r: r[D_USER], watch_ratio), "rate"),
              ("v_watch", lambda: MeanTable(lambda r: r[D_VID], watch_ratio), "rate")],
    # Isolations of `watch`: v_watch varies inside a user's slate directly;
    # u_watch is constant per user and can only act through FM second-order crosses.
    "vwatch": [("v_watch", lambda: MeanTable(lambda r: r[D_VID], watch_ratio), "rate")],
    "uwatch": [("u_watch", lambda: MeanTable(lambda r: r[D_USER], watch_ratio), "rate")],
    "awatch": [("a_watch", lambda: MeanTable(lambda r: r[D_AUTH], watch_ratio), "rate")],
}
# "time" needs no history table; it is derived per row inside encode().


def build_statistics(splits, groups):
    """Return {split: [[float per feature] per row]} under prequential rules."""
    specs = [spec for group in groups if group in GROUPS for spec in GROUPS[group]]
    if not specs:
        return {name: [[] for _ in rows] for name, rows in splits.items()}, []

    names = [name for name, _factory, _kind in specs]
    tables = [factory() for _name, factory, _kind in specs]
    kinds = [kind for _name, _factory, kind in specs]

    def read(row):
        return [table.rate(row) if kind == "rate" else float(table.count(row))
                for table, kind in zip(tables, kinds)]

    train = splits["train"]
    ordered = sorted(range(len(train)), key=lambda i: train[i][D_DATE])
    out_train = [None] * len(train)

    # Prequential sweep: score every row for date d, only then fold day d in.
    i = 0
    while i < len(ordered):
        j = i
        while j < len(ordered) and train[ordered[j]][D_DATE] == train[ordered[i]][D_DATE]:
            j += 1
        for index in ordered[i:j]:
            out_train[index] = read(train[index])
        for index in ordered[i:j]:
            for table in tables:
                table.observe(train[index])
        i = j

    # Validation sees the complete train split and nothing else.
    out_valid = [read(row) for row in splits["valid"]]
    return {"train": out_train, "valid": out_valid}, names


def bucketize(stats, names):
    """Quantile-bin each continuous statistic; edges fitted on train only."""
    if not names:
        return {name: [[] for _ in rows] for name, rows in stats.items()}
    train = np.asarray(stats["train"], dtype=np.float64)
    edges = [np.unique(np.quantile(train[:, c], np.linspace(0, 1, N_BINS + 1)[1:-1]))
             for c in range(train.shape[1])]
    out = {}
    for split, rows in stats.items():
        arr = np.asarray(rows, dtype=np.float64)
        columns = [np.searchsorted(edges[c], arr[:, c]) for c in range(arr.shape[1])]
        out[split] = [[str(int(v)) for v in row] for row in zip(*columns)]
    return out


def encode(splits, buckets, groups):
    """Categorical encoding identical in spirit to data.py's, with extra fields."""
    train = splits["train"]
    edges = np.quantile([x[D_DUR] for x in train], np.linspace(0, 1, 11)[1:-1])
    use_time = "time" in groups

    def raw(x, extra):
        fields = [x[D_USER], x[D_VID], x[D_AUTH], x[D_TAB],
                  str(int(np.searchsorted(edges, x[D_DUR])))]
        fields += list(extra)
        if use_time:
            age = x[D_DATE] - x[D_UP] if x[D_UP] else -1
            fields += [str(x[D_HOUR]), str(min(max(age, -1), 400) // 7)]
        return fields

    width = len(raw(train[0], buckets["train"][0]))
    vocabs = [dict() for _ in range(width)]
    for x, extra in zip(train, buckets["train"]):
        for i, value in enumerate(raw(x, extra)):
            if value not in vocabs[i]:
                vocabs[i][value] = len(vocabs[i])
    unk = [len(v) for v in vocabs]
    dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + dims[:-1]).astype(np.int32)

    enc = {}
    for name, rows in splits.items():
        X = np.empty((len(rows), width), dtype=np.int32)
        y = np.empty(len(rows), dtype=np.float32)
        users = []
        for j, (x, extra) in enumerate(zip(rows, buckets[name])):
            for i, value in enumerate(raw(x, extra)):
                X[j, i] = vocabs[i].get(value, unk[i]) + offsets[i]
            y[j] = x[D_Y]
            users.append(x[D_USER])
        enc[name] = (X, y, users)
    return enc, int(sum(dims)), width


def run_fm(enc, dim, seed, k=16, lr=0.001, max_epochs=40, patience=4, batch=8192):
    """Organizer FM rule, unchanged: 40 epochs max, patience 4, best-epoch restore."""
    Xtr, ytr, _ = enc["train"]
    Xva, yva, uva = enc["valid"]
    model = B.FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, state, bad, epochs_run = -1.0, None, 0, 0
    for _ in range(max_epochs):
        epochs_run += 1
        order = rng.permutation(len(ytr))
        for i in range(0, len(order), batch):
            model.step(Xtr[order[i:i + batch]], ytr[order[i:i + batch]])
        primary = evaluate(uva, yva, model.predict(Xva))["primary"]
        if primary > best + 1e-5:
            best, bad = primary, 0
            state = (model.V.copy(), model.W.copy(), np.float32(model.b))
        else:
            bad += 1
            if bad >= patience:
                break
    model.V, model.W, model.b = state
    stopped = "early_stopping" if epochs_run < max_epochs else "max_epochs_truncated"
    return evaluate(uva, yva, model.predict(Xva)), epochs_run, stopped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./KuaiRand-Pure/data")
    parser.add_argument("--groups", default="base",
                        help="comma list of: base,item,author,ua,ut,um,watch,time")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--baseline_primary", type=float, default=0.6014399,
                        help="frozen 3-seed organizer FM valid mean (runs_baseline_calibration)")
    args = parser.parse_args()

    groups = [g.strip() for g in args.groups.split(",") if g.strip() and g.strip() != "base"]
    seeds = [int(s) for s in args.seeds.split(",")]

    start = time.time()
    splits = load_rich(args.data_dir)
    print("rows: " + ", ".join(f"{k}={len(v):,}" for k, v in splits.items()))
    stats, names = build_statistics(splits, groups)
    buckets = bucketize(stats, names)
    enc, dim, width = encode(splits, buckets, groups)
    added = names + (TIME_FEATURES if "time" in groups else [])
    print(f"fields={width} (5 base + {width - 5} added: {added})")
    print(f"feature build: {time.time() - start:.1f}s")

    results = []
    for seed in seeds:
        tick = time.time()
        metrics, epochs, stopped = run_fm(enc, dim, seed)
        results.append(metrics)
        print(f"  seed {seed}: GAUC {metrics['GAUC']:.4f}  nDCG@5 {metrics['nDCG@5']:.4f}  "
              f"primary {metrics['primary']:.6f}  "
              f"({epochs} ep, {stopped}, {time.time() - tick:.0f}s)")

    mean = statistics.mean(r["primary"] for r in results)
    spread = statistics.pstdev([r["primary"] for r in results]) if len(results) > 1 else 0.0
    wins = sum(r["primary"] > args.baseline_primary + 0.002 for r in results)
    delta = mean - args.baseline_primary
    print(f"\ngroups={args.groups}")
    print(f"  valid primary mean {mean:.6f} +/- {spread:.6f}")
    print(f"  delta vs frozen baseline {args.baseline_primary:.6f}: {delta:+.6f}")
    print(f"  seed wins over +0.002: {wins}/{len(results)}")
    if delta >= 0.003 and wins >= 2:
        verdict = "PASS (>= +0.003 target)"
    elif delta > 0.002 and wins >= 2:
        verdict = "MARGINAL (> +0.002 but under target)"
    else:
        verdict = "FAIL"
    print(f"  GATE: {verdict}")


if __name__ == "__main__":
    main()
