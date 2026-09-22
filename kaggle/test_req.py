import os
import requests

os.environ["KAGGLE_USERNAME"] = "phtnguyn1ytj"
os.environ["KAGGLE_KEY"] = "16975e4a229dab67962c3f165b86feb1"

def get_kernel_output(user_name, kernel_slug):
    url = f"https://www.kaggle.com/api/v1/kernels/output"
    auth = (os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"])
    params = {"userName": user_name, "kernelSlug": kernel_slug}
    r = requests.get(url, auth=auth, params=params, timeout=30)
    print("STATUS CODE:", r.status_code)
    try:
        data = r.json()
        print("JSON KEYS:", list(data.keys()))
        if "log" in data:
            print("LOG CONTENT:\n", data["log"])
        if "error" in data:
            print("ERROR CONTENT:\n", data["error"])
        for k, v in data.items():
            if k not in ["log", "files"]:
                print(f"{k}: {v}")
    except Exception as e:
        print("Response text:\n", r.text[:2000])

if __name__ == "__main__":
    get_kernel_output("phtnguyn1ytj", "scale-error-yolo26-sahi-train")
