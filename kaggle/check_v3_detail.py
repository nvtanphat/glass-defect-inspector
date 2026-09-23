import os
import requests
import json

auth = ("phtnguyn1ytj", "16975e4a229dab67962c3f165b86feb1")

def inspect_versions():
    # Let's check kernel list with search or detail
    url = "https://www.kaggle.com/api/v1/kernels/list"
    params = {"user": "phtnguyn1ytj", "search": "sahi"}
    r = requests.get(url, auth=auth, params=params)
    print("List status:", r.status_code)
    try:
        items = r.json()
        for it in items:
            print(it.get("ref"), "totalVersions:", it.get("totalVersions"), "currentVersionNumber:", it.get("currentVersionNumber"))
            print("  lastRunTime:", it.get("lastRunTime"))
    except Exception as e:
        print("Error:", e, r.text[:200])

if __name__ == "__main__":
    inspect_versions()
