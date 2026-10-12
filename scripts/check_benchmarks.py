import glob
import json
import os
from pathlib import Path

def parse_coco_log(path):
    best = {}
    last = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "test_coco_eval_bbox" in line:
                try:
                    d = json.loads(line)
                    stats = d["test_coco_eval_bbox"]
                    ep = d.get("epoch")
                    last = {
                        "epoch": ep,
                        "stats": stats,
                        "n_parameters": d.get("n_parameters"),
                    }
                    if not best or stats[0] > best["stats"][0]:
                        best = {
                            "epoch": ep,
                            "stats": stats,
                            "n_parameters": d.get("n_parameters"),
                        }
                except Exception:
                    pass
    return best, last

all_logs = glob.glob("**/*log*.txt", recursive=True)
all_logs += glob.glob("**/*log.txt", recursive=True)
all_logs = sorted(list(set(all_logs)))

print(f"{'Path':<65} | {'Best mAP':<8} | {'mAP50':<7} | {'mAP75':<7} | {'mAP_s':<7} | {'mAP_m':<7} | {'mAP_l':<7} | {'Ep':<3}")
print("-" * 125)

results = []
for p in all_logs:
    if any(x in p for x in ["test_err", "error_log", "venv", ".git"]):
        continue
    best, last = parse_coco_log(p)
    if best:
        s = best["stats"]
        results.append((p, best, last))

results.sort(key=lambda x: x[1]["stats"][0], reverse=True)

for p, best, last in results:
    s = best["stats"]
    print(f"{p:<65} | {s[0]*100:6.2f}% | {s[1]*100:5.2f}% | {s[2]*100:5.2f}% | {s[3]*100:5.2f}% | {s[4]*100:5.2f}% | {s[5]*100:5.2f}% | {best['epoch']:<3}")
