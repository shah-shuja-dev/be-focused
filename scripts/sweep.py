"""
sweep.py -- Grid search, every combination is its own MLflow run.

Usage:
    python scripts\sweep.py
"""

import subprocess, itertools, sys

SEARCH_SPACE = {
    "lr":         [1e-3, 5e-4, 1e-4],
    "batch_size": [32, 64],
    "dropout":    [0.3, 0.5],
    "augment":    [True, False],
    "epochs":     [40],
}

def run_combination(combo: dict):
    cmd, parts = [sys.executable, r"scripts\train.py"], []
    for k, v in combo.items():
        if k == "augment":
            if v:
                cmd.append("--augment")
                parts.append("aug")
        else:
            cmd += [f"--{k}", str(v)]
            parts.append(f"{k}={v}")
    cmd += ["--run_name", "_".join(parts)]
    print(f"\n{'='*60}\nRunning: {' '.join(parts)}\n{'='*60}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    combos = [dict(zip(SEARCH_SPACE.keys(), v))
              for v in itertools.product(*SEARCH_SPACE.values())]
    print(f"Total combinations: {len(combos)}")
    for i, combo in enumerate(combos, 1):
        print(f"\n[{i}/{len(combos)}]")
        run_combination(combo)
    print("\nAll done. Open MLflow UI to compare.")