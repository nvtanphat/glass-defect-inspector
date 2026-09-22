import os
import sys

os.environ["KAGGLE_USERNAME"] = "phtnguyn1ytj"
os.environ["KAGGLE_KEY"] = "16975e4a229dab67962c3f165b86feb1"

from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    kernel = sys.argv[1] if len(sys.argv) > 1 else "phtnguyn1ytj/scale-error-yolo26-sahi-train"
    api = KaggleApi()
    api.authenticate()
    status = api.kernels_status(kernel)
    print("STATUS:", getattr(status, "status", status))
    print("FAILURE_MESSAGE:", getattr(status, "failure_message", getattr(status, "failureMessage", None)))

if __name__ == "__main__":
    main()
