import os
os.environ["KAGGLE_USERNAME"] = "phtnguyn1ytj"
os.environ["KAGGLE_KEY"] = "16975e4a229dab67962c3f165b86feb1"

from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    api = KaggleApi()
    api.authenticate()
    kernel = "phtnguyn1ytj/scale-error-yolo26-sahi-train"
    print("Testing kernels_logs:")
    try:
        logs = api.kernels_logs(kernel)
        print("logs:", logs)
    except Exception as e:
        print("kernels_logs error:", e)

    print("\nTesting kernels_list_files:")
    try:
        files = api.kernels_list_files(kernel)
        print("files:", files)
    except Exception as e:
        print("kernels_list_files error:", e)

if __name__ == "__main__":
    main()
