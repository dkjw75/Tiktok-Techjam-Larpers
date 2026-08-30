"""Train and evaluate the winning model in one command.

Reproduces the champion accepted by the autonomous agent as `exp_008`:
a within-user rank ensemble of six FM members that differ by feature set and
training objective. Validation-only -- the test split is never loaded.

  python manual/train_winner.py                # seed 0, ~6 min
  python manual/train_winner.py --seeds 0,1,2  # full 3-seed evidence, ~18 min
"""
import argparse
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research_agent.data_boundary import load_research_splits
from research_agent.metrics import evaluate_predictions
from research_agent.models.ensemble_fm import MEMBER_SETS, run_ensemble_fm_candidate
from research_agent.runner import PreparedData

BASELINE_PRIMARY = 0.6014399   # organizer FM, 3-seed validation mean
THRESHOLD = 0.002              # minimum evidence threshold
TARGET = 0.003                 # operational target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./KuaiRand-Pure/data")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--member_set", default="core6", choices=sorted(MEMBER_SETS))
    parser.add_argument("--epochs", type=int, default=40, help="40 = organizer full fidelity")
    parser.add_argument("--patience", type=int, default=4)
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    print("=" * 72)
    print("WINNING MODEL - within-user rank ensemble of six FM members")
    print("=" * 72)
    print(f"members        : {', '.join(MEMBER_SETS[args.member_set])}")
    print(f"fidelity       : {args.epochs} epochs max, patience {args.patience}")
    print("selection split: valid  (test is never loaded)")
    print(f"baseline        : {BASELINE_PRIMARY:.6f}  (organizer FM, 3-seed mean)")
    print()

    start = time.time()
    splits = load_research_splits(args.data_dir, ("train", "valid"))
    prepared = PreparedData(splits["train"], splits["valid"])
    print(f"train {len(splits['train']):,} rows | valid {len(splits['valid']):,} rows "
          f"({time.time() - start:.0f}s)\n")

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for seed in seeds:
            run_dir = Path(tmp) / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            tick = time.time()
            output = run_ensemble_fm_candidate(
                prepared,
                {
                    "seed": seed,
                    "epochs": args.epochs,
                    "patience": args.patience,
                    "member_set": args.member_set,
                },
                run_dir,
            )
            metrics = evaluate_predictions(output.user_ids, output.labels, output.scores)
            meta = output.metadata
            results.append(metrics.primary)

            print(f"--- seed {seed} ---  ({time.time() - tick:.0f}s, "
                  f"{meta['epochs_run']} epochs, {meta['stopped_by']})")
            print("  member validation primaries (the work behind the blend):")
            for name, value in meta["member_primaries"].items():
                flag = "beats baseline" if value > BASELINE_PRIMARY else "below baseline"
                print(f"    {name:32s} {value:.6f}   {flag}")
            print("  blend weights fitted on half the validation users:")
            for name, weight in meta["blend_weights"].items():
                print(f"    {name:32s} {weight:.2f}")
            print(f"  weight fit-half   : {meta['blend_weight_fit_half_primary']:.6f}")
            print(f"  weight held-half  : {meta['blend_weight_held_half_primary']:.6f}")
            print(f"  ENSEMBLE  GAUC {metrics.gauc:.6f}   nDCG@5 {metrics.ndcg_at_5:.6f}   "
                  f"primary {metrics.primary:.6f}")
            print(f"  delta vs baseline : {metrics.primary - BASELINE_PRIMARY:+.6f}\n")

    mean = statistics.mean(results)
    spread = statistics.pstdev(results) if len(results) > 1 else 0.0
    delta = mean - BASELINE_PRIMARY
    wins = sum(value - BASELINE_PRIMARY > THRESHOLD for value in results)
    print("=" * 72)
    print(f"RESULT over seeds {seeds}")
    print(f"  ensemble mean primary : {mean:.6f} +/- {spread:.6f}")
    print(f"  baseline              : {BASELINE_PRIMARY:.6f}")
    print(f"  mean delta            : {delta:+.6f}")
    print(f"  seed wins over +{THRESHOLD}  : {wins}/{len(results)}")
    if len(results) >= 3:
        if delta >= TARGET and wins >= 2:
            verdict = f"PASS - clears the +{TARGET} operational target"
        elif delta > THRESHOLD and wins >= 2:
            verdict = f"PASS - clears the +{THRESHOLD} evidence threshold, under the +{TARGET} target"
        else:
            verdict = "FAIL - does not clear the evidence threshold"
        print(f"  PROMOTION GATE        : {verdict}")
    else:
        print("  PROMOTION GATE        : needs --seeds 0,1,2 to judge")
    print("=" * 72)


if __name__ == "__main__":
    main()
