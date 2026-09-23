import json
import os
import requests

auth = ("phtnguyn1ytj", "16975e4a229dab67962c3f165b86feb1")

def check_kernel_versions():
    url = "https://www.kaggle.com/api/v1/kernels/list"
    # Or let's inspect the kernel details endpoint
    # On Kaggle, GET https://www.kaggle.com/api/v1/kernels/output
    for slug in ["scale-error-yolo26-sahi-train"]:
        print(f"=== Checking {slug} ===")
        # Check status
        st_url = f"https://www.kaggle.com/api/v1/kernels/status?userName=phtnguyn1ytj&kernelSlug={slug}"
        r = requests.get(st_url, auth=auth)
        print("Status:", r.text)

if __name__ == "__main__":
    check_kernel_versions()
