#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <utility>
#include <chrono>

const double DEGREE_TO_RAD = M_PI / 180.0;

// Calculate angular distance between two points on a sphere in degrees
double angularDistance(double ra1_deg, double dec1_deg, double ra2_deg, double dec2_deg) {
    double ra1 = ra1_deg * DEGREE_TO_RAD;
    double dec1 = dec1_deg * DEGREE_TO_RAD;
    double ra2 = ra2_deg * DEGREE_TO_RAD;
    double dec2 = dec2_deg * DEGREE_TO_RAD;

    double cos_angle = sin(dec1) * sin(dec2) + cos(dec1) * cos(dec2) * cos(ra1 - ra2);

    // Clamp to [-1,1] to avoid domain errors
    if (cos_angle > 1.0) cos_angle = 1.0;
    if (cos_angle < -1.0) cos_angle = -1.0;

    double angle_rad = acos(cos_angle);
    return angle_rad / DEGREE_TO_RAD;  // convert back to degrees
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " datafile.txt\n";
        return 1;
    }

    const char* filename = argv[1];
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error opening file " << filename << "\n";
        return 1;
    }

    std::vector<std::pair<double, double>> points;
    double ra, dec;
    while (infile >> ra >> dec) {
        points.emplace_back(ra, dec);
    }
    infile.close();

    const int num_bins = 10;
    std::vector<int> histogram(num_bins, 0);
    double max_distance = 10.0; // degrees
    double bin_size = max_distance / num_bins;

    std::cout << "Calculating angular correlation function for " << points.size() << " points...\n";

    auto start = std::chrono::high_resolution_clock::now();

    // Calculate ACF
    for (size_t i = 0; i < points.size(); ++i) {
        for (size_t j = i + 1; j < points.size(); ++j) {
            double dist = angularDistance(points[i].first, points[i].second,
                                          points[j].first, points[j].second);
            if (dist < max_distance) {
                int bin_index = static_cast<int>(dist / bin_size);
                histogram[bin_index]++;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Output to console
    std::cout << "\nAngular Correlation Function Histogram:\n";
    for (int i = 0; i < num_bins; ++i) {
        std::cout << i * bin_size << "-" << (i + 1) * bin_size << " deg: "
                  << histogram[i] << "\n";
    }

    int counted_pairs = 0;
    for (int count : histogram) counted_pairs += count;

    std::cout << "\nTotal pairs counted: " << counted_pairs << "\n";
    std::cout << "Time taken (CPU): " << elapsed.count() << " seconds\n";

    // Append results to output file
    std::ofstream outfile("acf_results_cpu.txt", std::ios::app);
    outfile << "\n=== Results for file: " << filename << " ===\n";
    outfile << "Angular Correlation Function Histogram:\n";
    for (int i = 0; i < num_bins; ++i) {
        outfile << i * bin_size << "-" << (i + 1) * bin_size << " deg: "
                << histogram[i] << "\n";
    }
    outfile << "Total pairs counted: " << counted_pairs << "\n";
    outfile << "Expected total pairs (n(n-1)/2): " << (points.size() * (points.size() - 1)) / 2 << "\n";
    outfile << "Time taken (CPU): " << elapsed.count() << " seconds\n";

    outfile.close();

    return 0;
}
