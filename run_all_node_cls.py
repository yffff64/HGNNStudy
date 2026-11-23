"""
Full node-classification re-run for DHG-Bench.

- Organized BY DATASET: for each dataset, run all 20 methods in order.
- The --is_default flag is NEVER passed (the parser default is False, so the
  per-dataset tuned hyperparameters in lib_yamls/node_yamls are used).
- The paper's UniGNN row is reproduced with UniGCNII (paper Sec. 3.2).
- Each method-dataset combination uses 5 seeds.
- Child output is NOT captured: every epoch / progress bar is shown live in
  this console. Failures are skipped and the loop continues.
- Result JSON files are saved automatically by the framework under
  dhgbench/results/node_cls/<dataset>/<method>.json
"""

import subprocess
import json
import os
import sys
import time
from datetime import datetime

# 20 methods in paper order; UniGNN -> UniGCNII
METHODS = [
    'MLP', 'CEGCN', 'CEGAT', 'HGNN', 'HyperGCN', 'HCHA', 'LEGCN',
    'HyperND', 'PhenomNN', 'SheafHyperGNN', 'HJRL', 'DPHGNN', 'TFHNN',
    'HNHN', 'UniGCNII', 'AllSetformer', 'EDHNN', 'HyperGT', 'EHNN', 'TMPHN'
]

# 13 datasets (CLI names)
DATASETS = [
    'cora', 'coauthor_cora', 'NTU2012', 'ModelNet40', 'actor', 'twitch',
    'pubmed', 'amazon', 'coauthor_dblp', 'walmart-trips-100',
    'yelp', 'pokec', 'trivago'
]

PYTHON = r"D:\anaconda3\envs\dhgbench\python.exe"
# This script lives at the repo root; main.py is under dhgbench/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = os.path.join(SCRIPT_DIR, 'dhgbench')
STATUS_FILE = os.path.join(SCRIPT_DIR, 'run_node_cls_status.json')
NUM_SEEDS = 5

# Resume support
status = {}
if os.path.exists(STATUS_FILE):
    with open(STATUS_FILE, 'r', encoding='utf-8') as f:
        status = json.load(f)

def save_status():
    with open(STATUS_FILE, 'w', encoding='utf-8') as f:
        json.dump(status, f, indent=1, ensure_ascii=False)

total = len(METHODS) * len(DATASETS)
print("=" * 74)
print(f" FULL RE-RUN BY DATASET: {len(DATASETS)} datasets x {len(METHODS)} methods = {total}")
print(f" Seeds: {NUM_SEEDS} | --is_default: NOT passed (per-dataset tuned config)")
print(f" Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 74)
sys.stdout.flush()

done = 0
n_success = 0
n_fail = 0

for di, ds in enumerate(DATASETS):
    print("\n")
    print("#" * 74)
    print(f"# DATASET {di + 1}/{len(DATASETS)}: {ds}")
    print("#" * 74)
    sys.stdout.flush()

    for mi, method in enumerate(METHODS):
        key = f"{ds}/{method}"
        done += 1

        if status.get(key, {}).get('state') == 'success':
            n_success += 1
            print(f"[{done:3}/{total}] {key:32} -> already done, skip")
            sys.stdout.flush()
            continue

        print("")
        print(f"[{done:3}/{total}] {key:32} start ...")
        sys.stdout.flush()
        start = time.time()

        # No --is_default argument on purpose.
        cmd = [
            PYTHON, "main.py",
            f"--dname={ds}",
            "--task_type=node_cls",
            f"--method={method}",
            f"--num_seeds={NUM_SEEDS}"
        ]

        try:
            # Inherit stdout/stderr -> fully live output in this window.
            result = subprocess.run(cmd, cwd=WORK_DIR)
            elapsed = time.time() - start

            if result.returncode == 0:
                json_path = os.path.join(
                    WORK_DIR, 'results', 'node_cls', ds, f'{method}.json')
                mean_val = std_val = None
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as jf:
                        rd = json.load(jf)
                    if rd.get('num_seeds') == NUM_SEEDS:
                        mean_val, std_val = rd['summary']['acc']

                if mean_val is not None:
                    status[key] = {
                        'state': 'success',
                        'mean': round(mean_val, 2),
                        'std': round(std_val, 2),
                        'time_s': round(elapsed, 1)
                    }
                    n_success += 1
                    print(f"  >> SUCCESS {key}: {mean_val:.2f} +/- {std_val:.2f} ({elapsed:.0f}s)")
                else:
                    status[key] = {'state': 'failed', 'reason': 'incomplete seeds'}
                    n_fail += 1
                    print(f"  >> FAILED {key}: incomplete seeds")
            else:
                status[key] = {'state': 'failed', 'reason': f'exit {result.returncode}'}
                n_fail += 1
                print(f"  >> SKIP {key}: process exited with code {result.returncode}")

        except Exception as e:
            status[key] = {'state': 'failed', 'reason': str(e)[:150]}
            n_fail += 1
            print(f"  >> SKIP {key}: exception {str(e)[:100]}")

        save_status()
        sys.stdout.flush()

print("\n")
print("=" * 74)
print(f" ALL FINISHED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f" Success: {n_success} | Failed/Skipped: {n_fail} | Total: {total}")
print("=" * 74)
save_status()
