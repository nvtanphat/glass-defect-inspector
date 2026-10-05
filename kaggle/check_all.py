import json
import os
import sys
from kaggle.api.kaggle_api_extended import KaggleApi

with open('kaggle/.kaggle_accounts.json') as f:
    accounts = json.load(f)

for user, creds in accounts.items():
    os.environ['KAGGLE_USERNAME'] = creds['username']
    os.environ['KAGGLE_KEY'] = creds['key']
    api = KaggleApi()
    api.authenticate()
    print(f"\n=== User: {user} ===", flush=True)
    try:
        kernels = api.kernels_list(mine=True, page_size=10)
        if not kernels:
            print("  (No kernels found)", flush=True)
        for k in kernels:
            ref = getattr(k, 'ref', None) or f"{user}/{getattr(k, 'slug', '')}"
            if not ref or ref.endswith('/'):
                continue
            try:
                st = api.kernels_status(ref)
                status_val = getattr(st, 'status', str(st))
                msg = getattr(st, 'failureMessage', '') or ''
                print(f"  {ref:52s} | Status: {status_val:12s} | {msg}", flush=True)
            except Exception as ex:
                print(f"  {ref:52s} | Status: unknown ({ex})", flush=True)
    except Exception as e:
        print(f"  Error: {e}", flush=True)
