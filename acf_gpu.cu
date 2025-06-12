#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>

const int NUM_BINS = 10;
const double MAX_DISTANCE = 10.0;  // degrees
const double DEGREE_TO_RAD = 3.14159265358979323846 / 180.0;

__device__ double angularDistance(double ra1_deg, double dec1_deg, double ra2_deg, double dec2_deg) {
    double ra1 = ra1_deg * DEGREE_TO_RAD;
    double dec1 = dec1_deg * DEGREE_TO_RAD;
    double ra2 = ra2_deg * DEGREE_TO_RAD;
    double dec2 = dec2_deg * DEGREE_TO_RAD;

    double cos_angle = sin(dec1) * sin(dec2) + cos(dec1) * cos(dec2) * cos(ra1 - ra2);
    cos_angle = fmin(fmax(cos_angle, -1.0), 1.0); // Clamp to [-1, 1]

    double angle_rad = acos(cos_angle);
    return angle_rad / DEGREE_TO_RAD;  // convert back to degrees
}

__global__ void computeACF(const double* ra, const double* dec, int num_points, int* histogram, double bin_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points) return;

    double ra1 = ra[i];
    double dec1 = dec[i];

    for (int j = i + 1; j < num_points; ++j) {
        double dist = angularDistance(ra1, dec1, ra[j], dec[j]);
        if (dist < MAX_DISTANCE) {
            int bin_idx = static_cast<int>(dist / bin_size);
            if (bin_idx < NUM_BINS) {
                atomicAdd(&histogram[bin_idx], 1);
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " datafile.txt\n";
        return 1;
    }

    std::ifstream infile(argv[1]);
    if (!infile) {
        std::cerr << "Error opening file " << argv[1] << "\n";
        return 1;
    }

    std::vector<double> ra_vec, dec_vec;
    double ra, dec;
    while (infile >> ra >> dec) {
        ra_vec.push_back(ra);
        dec_vec.push_back(dec);
    }
    infile.close();

    int num_points = ra_vec.size();
    std::cout << "Calculating ACF for " << num_points << " points\n";

    double bin_size = MAX_DISTANCE / NUM_BINS;

    // memory allocation
    int* d_histogram;
    double* d_ra;
    double* d_dec;

    cudaMalloc(&d_ra, num_points * sizeof(double));
    cudaMalloc(&d_dec, num_points * sizeof(double));
    cudaMalloc(&d_histogram, NUM_BINS * sizeof(int));
    cudaMemset(d_histogram, 0, NUM_BINS * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_ra, ra_vec.data(), num_points * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dec, dec_vec.data(), num_points * sizeof(double), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_points + threadsPerBlock - 1) / threadsPerBlock;

    // Timing Start
    auto start = std::chrono::high_resolution_clock::now();

    computeACF<<<blocksPerGrid, threadsPerBlock>>>(d_ra, d_dec, num_points, d_histogram, bin_size);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Copy histogram back to host
    std::vector<int> histogram(NUM_BINS);
    cudaMemcpy(histogram.data(), d_histogram, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);

    // Output to console
    std::cout << "Angular Correlation Function Histogram:\n";
    for (int i = 0; i < NUM_BINS; ++i) {
        std::cout << i * bin_size << "-" << (i + 1) * bin_size << " deg: " << histogram[i] << "\n";
    }
    std::cout << "Time taken (GPU): " << elapsed.count() << " seconds\n";

    // Output to file in append mode
    std::ofstream outfile("acf_results_gpu.txt", std::ios::app);
    outfile << "=== Results for file: " << argv[1] << " ===\n";
    outfile << "Angular Correlation Function Histogram:\n";
    for (int i = 0; i < NUM_BINS; ++i) {
        outfile << i * bin_size << "-" << (i + 1) * bin_size << " deg: " << histogram[i] << "\n";
    }
    outfile << "Time taken (GPU): " << elapsed.count() << " seconds\n\n";
    outfile.close();

    // Cleanup
    cudaFree(d_ra);
    cudaFree(d_dec);
    cudaFree(d_histogram);

    return 0;
}
