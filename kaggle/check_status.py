import json
import os
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

ACCOUNTS_FILE = Path(__file__).parent / ".kaggle_accounts.json"
if ACCOUNTS_FILE.exists():
    ACCOUNTS = json.loads(ACCOUNTS_FILE.read_text(encoding="utf-8"))
else:
    ACCOUNTS = {
        'baohuynh05': {'username': 'baohuynh05', 'key': os.getenv('KAGGLE_KEY_BAOHUYNH05', '')},
        'jyang10': {'username': 'jyang10', 'key': os.getenv('KAGGLE_KEY_JYANG10', '')},
        'leppan': {'username': os.getenv('KAGGLE_USERNAME', 'leppan'), 'key': os.getenv('KAGGLE_KEY', '')},
    }


TARGET_KERNELS = [
    'jyang10/scale-error-dfine-m-evaluate',
    'baohuynh05/scale-error-dfine-m-baseline',
    'leppan/scale-error-dfine-fbdnet-m',
    'jyang10/scale-error-dfine-frequency-official',
    'jyang10/scale-error-dfine-exp-frequency-screening',
    'jyang10/scale-error-dfine-m-resume-71-100',
    'leppan/scale-error-dfine-m-resume-71-100',
    'leppan/scale-error-dfine-exp-p2-screening',
    'baohuynh05/scale-error-dfine-m-resume-47-70',
    'baohuynh05/scale-error-dfine-m-resume-24-46',
]

for k in TARGET_KERNELS:
    owner = k.split('/')[0]
    os.environ['KAGGLE_USERNAME'] = ACCOUNTS[owner]['username']
    os.environ['KAGGLE_KEY'] = ACCOUNTS[owner]['key']
    try:
        api = KaggleApi()
        api.authenticate()
        st = api.kernels_status(k)
        status_val = getattr(st, 'status', str(st))
        fail_msg = getattr(st, 'failureMessage', '')
        print(f"[{owner}] {k:48s} | Status: {status_val} | Msg: {fail_msg}")
    except Exception as e:
        print(f"[{owner}] {k:48s} | Error: {e}")
