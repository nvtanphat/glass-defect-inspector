
# Comprehensive Validation Tuning & Test Evaluation Benchmark

| Fusion Strategy | Reference | Split | mAP@50:95 (%) | AP50 (%) | AP75 (%) | AP_small (%) | DiVat | DiVatLoiLom | LoiChi | LoiNhua | LoiTray |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline WBF | Solovyev et al., 2021 | **Val** | **43.46** | 76.69 | 41.75 | 27.53 | 28.01 | 34.58 | 62.94 | 50.76 | 41.04 |
| Baseline WBF | Solovyev et al., 2021 | **Test** | **43.85** | 76.07 | 42.8 | 33.55 | 24.39 | 35.97 | 65.53 | 53.17 | 40.18 |
| Heterogeneous Triad WBF | Current Paper | **Val** | **43.61** | 75.7 | 42.01 | 25.44 | 29.03 | 34.37 | 66.01 | 47.51 | 41.11 |
| Heterogeneous Triad WBF | Current Paper | **Test** | **46.17** | 77.64 | 46.88 | 35.09 | 26.28 | 35.81 | 67.22 | 57.79 | 43.73 |
| SBF (Selective Box Fusion) | Han et al., 2025 [16] | **Val** | **44.91** | 77.71 | 43.37 | 32.32 | 29.32 | 35.08 | 64.95 | 52.22 | 42.96 |
| SBF (Selective Box Fusion) | Han et al., 2025 [16] | **Test** | **44.3** | 76.69 | 43.51 | 34.32 | 25.84 | 35.68 | 66.2 | 53.89 | 39.87 |
| AR-WBF (Adaptive Reliability) | Daoud & Alqasir, 2026 [17] | **Val** | **44.35** | 77.26 | 42.41 | 28.83 | 29.19 | 34.84 | 64.55 | 51.12 | 42.07 |
| AR-WBF (Adaptive Reliability) | Daoud & Alqasir, 2026 [17] | **Test** | **44.19** | 76.39 | 43.05 | 34.63 | 25.8 | 35.8 | 66.13 | 53.29 | 39.93 |
