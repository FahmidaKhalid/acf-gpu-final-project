
# Accelerating Angular Correlation Function Calculations with GPU for Astronomical Data

## Project Overview

This project aims to develop a program to calculate the Angular Correlation Function (ACF) for large astronomical datasets. The ACF is essential in astrophysics to analyze the spatial distribution of galaxies or celestial objects. However, calculating the ACF sequentially becomes very slow for large datasets due to the need to compute angular distances between every pair of points.

To address this, the project uses GPU acceleration with CUDA to significantly speed up the calculations by performing many computations in parallel. The CPU version provides a baseline for comparison.

## Files

- `acf_cpu.cpp`: CPU implementation of the ACF calculation  
- `acf_gpu.cu`: GPU implementation of the ACF calculation using CUDA  
- `datasets/` – Folder containing input datasets:  
  - `small_data.txt` – Small dataset for testing correctness  
  - `data_100k_degrees.txt` – Large dataset for performance evaluation  
- `acf_results_cpu.txt`: Sample CPU results  
- `acf_results_gpu.txt`: Sample GPU results  
- `Run_Logs/`: Folder containing screenshots and logs from running the code  
- `README.md`: This file

## How to Run

### CPU Version (Windows)
```bash
g++ acf_cpu.cpp -o acf_cpu
acf_cpu small_data.txt
acf_cpu data_100k_degrees.txt
```

### GPU Version (Google Colab with CUDA)
```bash
!nvcc --std=c++11 acf_gpu.cu -o acf_gpu -arch=sm_75
!./acf_gpu small_data.txtt
!./acf_gpu data_100k_degrees.txt
```
| Dataset                 | CPU Time (seconds) | GPU Time (seconds) | Speedup (CPU / GPU)   |
| ----------------------- | ------------------ | ------------------ | --------------------- |
| `small_data.txt`        | 0.0000069          | 0.00026811         | GPU slightly slower   |
| `data_100k_degrees.txt` | 1183.32            | 6.89996            | 171x faster on GPU    |

## Explanation of Results

For the small dataset (`small_data.txt`), both CPU and GPU produce identical ACF histograms with 4 pairs counted within angular bins. The total pairs possible for 5 points is 10, but only pairs falling into the angular bins are counted, explaining the difference.

The GPU is slightly slower on this small dataset due to overhead from kernel launch and data transfer.

For the large dataset (`data_100k_degrees.txt`), both implementations match in results, counting around 357 million pairs out of the theoretical 5 billion possible pairs within angular bins.

The GPU version is ~171 times faster, reducing the time from ~1183 seconds (CPU) to ~7 seconds (GPU). This confirms the significant advantage of parallel computing with GPUs for processing large astronomical datasets.
