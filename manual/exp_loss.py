"""Manual Phase-1 experiment harness: ranking objectives on the FM backbone.

Calls the PyTorch candidate directly, bypassing the whole agent stack, so a
listwise/pairwise/pointwise number lands in minutes instead of a research cycle.
Validation-only: `load_research_splits` refuses to materialize test rows.

Usage
-----
  python manual/exp_loss.py --loss pointwise --seeds 0
  python manual/exp_loss.py --loss listwise --variant t1  --seeds 0,1,2
  python manual/exp_loss.py --loss listwise --variant t05 --epochs 4 --patience 2
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
from research_agent.models.torch_fm import run_torch_fm_candidate
from research_agent.runner import PreparedData


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./KuaiRand-Pure/data")
    parser.add_argument("--loss", default="listwise",
                        choices=("pointwise", "pairwise", "listwise"))
    parser.add_argument("--variant", default="t1", choices=("t1", "t05", "t1_bce25"),
                        help="listwise only: t1=T1.0, t05=T0.5, t1_bce25=0.75*listwise+0.25*BCE")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--l2", type=float, default=1e-6)
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=40, help="40 = full fidelity")
    parser.add_argument("--patience", type=int, default=4, help="4 = full fidelity")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--baseline_primary", type=float, default=0.6014399)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    start = time.time()
    splits = load_research_splits(args.data_dir, ("train", "valid"))
    prepared = PreparedData(splits["train"], splits["valid"])
    print(f"rows: train={len(splits['train']):,}, valid={len(splits['valid']):,} "
          f"({time.time() - start:.1f}s)")

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for seed in seeds:
            config = {
                "loss": args.loss,
                "objective_variant": args.variant,
                "learning_rate": args.learning_rate,
                "l2": args.l2,
                "embedding_dim": args.embedding_dim,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "patience": args.patience,
                "seed": seed,
            }
            run_dir = Path(tmp) / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            tick = time.time()
            output = run_torch_fm_candidate(prepared, config, run_dir)
            meta = output.metadata
            best = meta["best_metrics"]
            results.append(best)
            print(f"  seed {seed}: GAUC {best['GAUC']:.4f}  nDCG@5 {best['nDCG@5']:.4f}  "
                  f"primary {best['primary']:.6f}  "
                  f"({meta['epochs_run']} ep, best {meta['best_epoch']}, "
                  f"{meta['stopped_by']}, {time.time() - tick:.0f}s)")

    mean = statistics.mean(r["primary"] for r in results)
    spread = statistics.pstdev([r["primary"] for r in results]) if len(results) > 1 else 0.0
    wins = sum(r["primary"] > args.baseline_primary + 0.002 for r in results)
    delta = mean - args.baseline_primary
    label = args.loss + (f":{args.variant}" if args.loss == "listwise" else "")
    print(f"\nloss={label} lr={args.learning_rate} l2={args.l2} "
          f"epochs={args.epochs} patience={args.patience}")
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
