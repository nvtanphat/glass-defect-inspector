import os
import subprocess
import sys

def main():
    kernel = sys.argv[1] if len(sys.argv) > 1 else "phtnguyn1ytj/scale-error-yolo26-sahi-train"
    res = subprocess.run(["kaggle", "kernels", "output", kernel], capture_output=True, text=True, encoding="utf-8", errors="replace")
    print("STDOUT:")
    print(res.stdout)
    print("STDERR:")
    print(res.stderr)

if __name__ == "__main__":
    main()
