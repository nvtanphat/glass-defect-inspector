import json
import os
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

ACCOUNTS = json.loads((Path(__file__).parent / ".kaggle_accounts.json").read_text(encoding="utf-8"))

KERNELS_TO_CHECK = [
    # latest runs
    "jyang10/scale-error-dfine-m-baseline-resume69",
    "baohuynh05/scale-error-dfine-fbdnet-m-resume46",
    "phtnguyn1ytj/scale-error-dfine-official-clone",
    # previous runs
    "jyang10/scale-error-dfine-m-baseline-resume47",
    "baohuynh05/scale-error-dfine-m-baseline-resume23",
    "leppan/scale-error-dfine-fbdnet-m-resume23",
    "leppan/scale-error-dfine-fbdnet-m",
    "baohuynh05/scale-error-dfine-m-baseline",
    "jyang10/scale-error-dfine-frequency-official",
    "jyang10/scale-error-dfine-exp-frequency-screening",
    "jyang10/scale-error-dfine-m-resume-71-100",
    "leppan/scale-error-dfine-m-resume-71-100",
    "leppan/scale-error-dfine-exp-p2-screening",
    "baohuynh05/scale-error-dfine-m-resume-47-70",
    "baohuynh05/scale-error-dfine-m-resume-24-46",
    "jyang10/scale-error-dfine-m-evaluate",
]

apis = {}
for user, creds in ACCOUNTS.items():
    os.environ["KAGGLE_USERNAME"] = creds["username"]
    os.environ["KAGGLE_KEY"] = creds["key"]
    api = KaggleApi()
    api.authenticate()
    apis[user] = api

print(f"{'Kernel Ref':52s} | {'Status':15s} | Detail / Error", flush=True)
print("-" * 90, flush=True)

for k in KERNELS_TO_CHECK:
    owner = k.split("/")[0]
    api = apis.get(owner)
    if not api:
        print(f"{k:52s} | Unknown owner: {owner}", flush=True)
        continue
    try:
        st = api.kernels_status(k)
        status_val = str(getattr(st, "status", st))
        fail_msg = getattr(st, "failureMessage", "") or ""
        print(f"{k:52s} | {status_val:15s} | {fail_msg}", flush=True)
    except Exception as e:
        print(f"{k:52s} | ERROR           | {e}", flush=True)
