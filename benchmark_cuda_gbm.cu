#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__   \
                      << " - " << cudaGetErrorString(err) << std::endl;    \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

__global__ void gbm_kernel(
    double *results,
    double S0,
    double drift,
    double volatility,
    int n_steps,
    int N_sim,
    unsigned long seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N_sim) {
        return;
    }

    curandState state;
    curand_init(seed, idx, 0, &state);

    double S = S0;

    for (int t = 0; t < n_steps; t++) {
        double Z = curand_normal_double(&state);

        double step = exp(
            (drift - 0.5 * volatility * volatility)
            + volatility * Z
        );

        S *= step;
    }

    results[idx] = S;
}

std::vector<double> read_close_prices(const std::string &filename) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filename << std::endl;
        exit(1);
    }

    std::vector<double> prices;
    std::string line;

    // Skip header
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> cols;

        while (std::getline(ss, cell, ',')) {
            cols.push_back(cell);
        }

        // Assumes CSV columns include close price as the 5th column:
        // timestamp, open, high, low, close, volume
        if (cols.size() >= 5) {
            prices.push_back(std::stod(cols[4]));
        }
    }

    return prices;
}

void estimate_gbm_parameters(
    const std::vector<double> &prices,
    double &drift,
    double &volatility
) {
    std::vector<double> log_returns;

    for (int i = 1; i < (int)prices.size(); i++) {
        double r = log(prices[i] / prices[i - 1]);
        log_returns.push_back(r);
    }

    double sum = 0.0;
    for (int i = 0; i < (int)log_returns.size(); i++) {
        sum += log_returns[i];
    }

    drift = sum / log_returns.size();

    double variance_sum = 0.0;
    for (int i = 0; i < (int)log_returns.size(); i++) {
        double diff = log_returns[i] - drift;
        variance_sum += diff * diff;
    }

    volatility = sqrt(variance_sum / (log_returns.size() - 1));
}

double percentile(std::vector<double> values, double pct) {
    std::sort(values.begin(), values.end());

    double index = (pct / 100.0) * (values.size() - 1);
    int lower = (int)floor(index);
    int upper = (int)ceil(index);

    if (lower == upper) {
        return values[lower];
    }

    double weight = index - lower;
    return values[lower] * (1.0 - weight) + values[upper] * weight;
}

void run_benchmark(const std::string &input_file, const std::string &output_csv) {
    std::vector<double> prices = read_close_prices(input_file);

    double drift = 0.0;
    double volatility = 0.0;
    estimate_gbm_parameters(prices, drift, volatility);

    double S0 = prices.back();

    int n_steps = 200;
    std::vector<int> workload_sizes = {1000, 10000, 100000, 1000000};

    std::ofstream out(output_csv);
    out << "execution_mode,input_file,paths,time_steps,execution_time_ms,"
        << "mean_final_price,median_final_price,5th_percentile,95th_percentile\n";

    for (int i = 0; i < (int)workload_sizes.size(); i++) {
        int N_sim = workload_sizes[i];

        std::cout << "Running CUDA GBM with " << N_sim
                  << " paths for " << input_file << "..." << std::endl;

        double *d_results = nullptr;
        CHECK_CUDA(cudaMalloc(&d_results, N_sim * sizeof(double)));

        int threads = 256;
        int blocks = (N_sim + threads - 1) / threads;

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));

        gbm_kernel<<<blocks, threads>>>(
            d_results,
            S0,
            drift,
            volatility,
            n_steps,
            N_sim,
            42
        );

        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        CHECK_CUDA(cudaGetLastError());

        float elapsed_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

        std::vector<double> final_prices(N_sim);
        CHECK_CUDA(cudaMemcpy(
            final_prices.data(),
            d_results,
            N_sim * sizeof(double),
            cudaMemcpyDeviceToHost
        ));

        double sum = 0.0;
        for (int j = 0; j < N_sim; j++) {
            sum += final_prices[j];
        }

        double mean = sum / N_sim;
        double median = percentile(final_prices, 50.0);
        double p5 = percentile(final_prices, 5.0);
        double p95 = percentile(final_prices, 95.0);

        out << "CUDA-GPU,"
            << input_file << ","
            << N_sim << ","
            << n_steps << ","
            << elapsed_ms << ","
            << mean << ","
            << median << ","
            << p5 << ","
            << p95 << "\n";

        CHECK_CUDA(cudaFree(d_results));
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    out.close();

    std::cout << "Saved results to " << output_csv << std::endl;
}

int main() {
    run_benchmark("usdjpy-m1-bid-2013.csv", "cuda_gbm_bid_benchmark.csv");
    run_benchmark("usdjpy-m1-ask-2013.csv", "cuda_gbm_ask_benchmark.csv");

    return 0;
}