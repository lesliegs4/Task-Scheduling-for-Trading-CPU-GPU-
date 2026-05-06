#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__   \
                      << " - " << cudaGetErrorString(err) << std::endl;    \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

struct JumpDiffusionParams {
    double s0{};
    double mu{};
    double sigma_diffusion{};
    double lambda{};
    double muJ{};
    double sigmaJ{};
    double threshold{};
};

static std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> out;
    std::string cell;
    std::stringstream ss(line);
    while (std::getline(ss, cell, ',')) {
        out.push_back(cell);
    }
    return out;
}

static bool parse_utc_timestamp_seconds(const std::string& ts, std::int64_t& out_epoch_seconds) {
    // Accepts common formats:
    // - "2013-02-01 00:00:00+00:00"
    // - "2013-02-01T00:00:00+00:00"
    // - "2013-02-01 00:00:00Z"
    // Keeps only "YYYY-MM-DD HH:MM:SS" and treats it as UTC.
    std::string s = ts;
    if (s.empty()) return false;
    for (char& c : s) {
        if (c == 'T') c = ' ';
    }
    // Strip timezone suffix.
    size_t plus = s.find('+');
    if (plus != std::string::npos) s = s.substr(0, plus);
    // Handle trailing 'Z'
    if (!s.empty() && (s.back() == 'Z' || s.back() == 'z')) s.pop_back();
    // Keep first 19 chars: YYYY-MM-DD HH:MM:SS
    if (s.size() < 19) return false;
    s = s.substr(0, 19);

    auto to_int = [&](int start, int len) -> int {
        return std::stoi(s.substr((size_t)start, (size_t)len));
    };

    try {
        int year = to_int(0, 4);
        int mon = to_int(5, 2);
        int day = to_int(8, 2);
        int hour = to_int(11, 2);
        int min = to_int(14, 2);
        int sec = to_int(17, 2);

        std::tm tm{};
        tm.tm_year = year - 1900;
        tm.tm_mon = mon - 1;
        tm.tm_mday = day;
        tm.tm_hour = hour;
        tm.tm_min = min;
        tm.tm_sec = sec;
        tm.tm_isdst = 0;

        // timegm is available on Linux (CARC).
        std::time_t t = timegm(&tm);
        if (t == (std::time_t)-1) return false;
        out_epoch_seconds = (std::int64_t)t;
        return true;
    } catch (...) {
        return false;
    }
}

static double median_step_seconds(const std::vector<std::string>& timestamps) {
    if (timestamps.size() < 2) return 0.0;
    std::vector<double> deltas;
    deltas.reserve(timestamps.size() - 1);

    std::int64_t prev = 0;
    if (!parse_utc_timestamp_seconds(timestamps[0], prev)) return 0.0;
    for (size_t i = 1; i < timestamps.size(); i++) {
        std::int64_t cur = 0;
        if (!parse_utc_timestamp_seconds(timestamps[i], cur)) continue;
        deltas.push_back((double)(cur - prev));
        prev = cur;
    }
    if (deltas.empty()) return 0.0;
    std::sort(deltas.begin(), deltas.end());
    size_t mid = deltas.size() / 2;
    if (deltas.size() % 2 == 1) return deltas[mid];
    return 0.5 * (deltas[mid - 1] + deltas[mid]);
}

// Load timestamps + close from CSV with header including at least timestamp and close.
static std::pair<std::vector<std::string>, std::vector<double>> read_timestamps_and_close(
    const std::string& filename,
    int max_rows
) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string header;
    if (!std::getline(file, header)) {
        throw std::runtime_error("Empty CSV: " + filename);
    }

    auto cols = split_csv_line(header);
    int ts_idx = -1;
    int close_idx = -1;
    for (int i = 0; i < (int)cols.size(); i++) {
        // Accept common names.
        if (cols[i] == "timestamp") ts_idx = i;
        if (cols[i] == "close") close_idx = i;
    }
    if (ts_idx < 0 || close_idx < 0) {
        throw std::runtime_error("CSV must contain columns: timestamp, close");
    }

    std::vector<std::string> timestamps;
    std::vector<double> close_prices;
    timestamps.reserve(max_rows > 0 ? (size_t)max_rows : 4096);
    close_prices.reserve(max_rows > 0 ? (size_t)max_rows : 4096);

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        auto row = split_csv_line(line);
        if ((int)row.size() <= std::max(ts_idx, close_idx)) continue;

        const std::string& ts = row[ts_idx];
        const std::string& close_s = row[close_idx];
        char* endptr = nullptr;
        errno = 0;
        double c = std::strtod(close_s.c_str(), &endptr);
        if (errno != 0 || endptr == close_s.c_str()) continue;

        timestamps.push_back(ts);
        close_prices.push_back(c);

        if (max_rows > 0 && (int)close_prices.size() >= max_rows) break;
    }

    if (close_prices.size() < 3) {
        throw std::runtime_error("Need at least 3 close prices to fit jump-diffusion params.");
    }
    return {timestamps, close_prices};
}

static std::vector<double> log_returns(const std::vector<double>& prices) {
    std::vector<double> r;
    r.reserve(prices.size() - 1);
    for (size_t i = 1; i < prices.size(); i++) {
        if (prices[i] <= 0.0 || prices[i - 1] <= 0.0) {
            throw std::runtime_error("Close prices must be positive.");
        }
        r.push_back(std::log(prices[i] / prices[i - 1]));
    }
    return r;
}

static double mean(const std::vector<double>& x) {
    if (x.empty()) return 0.0;
    double s = std::accumulate(x.begin(), x.end(), 0.0);
    return s / (double)x.size();
}

static double stddev_sample(const std::vector<double>& x) {
    if (x.size() < 2) return 0.0;
    double m = mean(x);
    double acc = 0.0;
    for (double v : x) {
        double d = v - m;
        acc += d * d;
    }
    return std::sqrt(acc / (double)(x.size() - 1));
}

static JumpDiffusionParams fit_jump_diffusion_params(
    const std::vector<double>& close_prices,
    double jump_threshold_mult
) {
    auto r = log_returns(close_prices);

    double s0 = close_prices.front();
    double mu = mean(r);
    double sigma = stddev_sample(r);
    if (sigma <= 0.0) sigma = 1e-6;

    double threshold = jump_threshold_mult * sigma;

    std::vector<double> jump_residuals;
    std::vector<double> non_jump_residuals;
    jump_residuals.reserve(r.size() / 10);
    non_jump_residuals.reserve(r.size());

    for (double ri : r) {
        double residual = ri - mu;
        if (std::abs(residual) > threshold) {
            jump_residuals.push_back(residual);
        } else {
            non_jump_residuals.push_back(residual);
        }
    }

    double lam = (double)jump_residuals.size() / std::max<size_t>(1, r.size());
    double muJ = jump_residuals.empty() ? 0.0 : mean(jump_residuals);
    double sigmaJ = 0.0;
    if (jump_residuals.size() > 1) {
        sigmaJ = stddev_sample(jump_residuals);
    } else {
        sigmaJ = std::max(1e-6, sigma * 0.5);
    }

    double sigma_diff = 0.0;
    if (non_jump_residuals.size() > 1) {
        sigma_diff = stddev_sample(non_jump_residuals);
    } else {
        sigma_diff = sigma;
    }

    JumpDiffusionParams p;
    p.s0 = s0;
    p.mu = mu;
    p.sigma_diffusion = sigma_diff;
    p.lambda = lam;
    p.muJ = muJ;
    p.sigmaJ = sigmaJ;
    p.threshold = threshold;
    return p;
}

__global__ void jump_diffusion_kernel_final_and_first_path(
    double* final_prices_out,   // N paths
    double* first_path_out,     // n_steps (only written by idx==0)
    double* sample_paths_out,   // sample_paths_count * n_steps (only written by idx < sample_paths_count)
    int sample_paths_count,
    JumpDiffusionParams params,
    int n_steps,
    int n_paths,
    double dt,
    unsigned long long seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_paths) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, (unsigned long long)idx, 0ULL, &state);

    double S = params.s0;
    if (idx == 0 && first_path_out != nullptr && n_steps > 0) {
        first_path_out[0] = S;
    }
    if (sample_paths_out != nullptr && idx < sample_paths_count && n_steps > 0) {
        sample_paths_out[(size_t)idx * (size_t)n_steps + 0] = S;
    }

    double sqrt_dt = sqrt(dt);
    double drift_term = (params.mu - 0.5 * params.sigma_diffusion * params.sigma_diffusion) * dt;
    double diffusion_scale = params.sigma_diffusion * sqrt_dt;
    double lambda_dt = params.lambda * dt;

    for (int t = 1; t < n_steps; t++) {
        // Jump count ~ Poisson(lambda*dt)
        unsigned int jump_count = curand_poisson(&state, lambda_dt);

        double jump_term = 0.0;
        // Sum of jump_count normal(muJ, sigmaJ) terms (matches Python intent).
        for (unsigned int j = 0; j < jump_count; j++) {
            double z = curand_normal_double(&state);
            jump_term += params.muJ + params.sigmaJ * z;
        }

        double z = curand_normal_double(&state);
        double diffusion = drift_term + diffusion_scale * z;

        double step = exp(diffusion + jump_term);
        S = fmax(1e-8, S * step);

        if (idx == 0 && first_path_out != nullptr) {
            first_path_out[t] = S;
        }
        if (sample_paths_out != nullptr && idx < sample_paths_count) {
            sample_paths_out[(size_t)idx * (size_t)n_steps + (size_t)t] = S;
        }
    }

    final_prices_out[idx] = S;
}

static double percentile_sorted(const std::vector<double>& sorted_values, double pct) {
    if (sorted_values.empty()) return 0.0;
    double index = (pct / 100.0) * (double)(sorted_values.size() - 1);
    size_t lower = (size_t)std::floor(index);
    size_t upper = (size_t)std::ceil(index);
    if (lower == upper) return sorted_values[lower];
    double w = index - (double)lower;
    return sorted_values[lower] * (1.0 - w) + sorted_values[upper] * w;
}

struct Percentiles {
    double p5{};
    double p50{};
    double p95{};
};

static Percentiles compute_percentiles_5_50_95(const std::vector<double>& values) {
    if (values.empty()) return {};
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    Percentiles out;
    out.p5 = percentile_sorted(sorted, 5.0);
    out.p50 = percentile_sorted(sorted, 50.0);
    out.p95 = percentile_sorted(sorted, 95.0);
    return out;
}

static void write_params_json(
    const std::string& out_path,
    const std::string& fit_from,
    double step_seconds,
    int n_paths,
    int n_steps,
    double elapsed_seconds,
    const JumpDiffusionParams& p,
    double mean_final,
    double std_final,
    double min_final,
    double max_final,
    const std::vector<std::string>& plot_files,
    const std::string& sample_paths_csv,
    const std::string& final_prices_csv,
    double load_csv_seconds,
    double fit_parameters_seconds,
    double jump_diffusion_simulation_seconds,
    double gpu_alloc_seconds,
    double gpu_kernel_seconds,
    double gpu_d2h_seconds,
    double host_postprocess_seconds,
    double save_files_seconds,
    double generate_plots_seconds
) {
    std::ofstream out(out_path);
    if (!out.is_open()) throw std::runtime_error("Could not write: " + out_path);

    out << std::setprecision(12);
    out << "{\n";
    out << "  \"fit_from\": " << "\"" << fit_from << "\",\n";
    out << "  \"step_seconds\": " << step_seconds << ",\n";
    out << "  \"n_paths\": " << n_paths << ",\n";
    out << "  \"n_steps\": " << n_steps << ",\n";
    out << "  \"elapsed_seconds\": " << elapsed_seconds << ",\n";
    out << "  \"timing_breakdown\": {\n";
    out << "    \"fit_parameters_seconds\": " << fit_parameters_seconds << ",\n";
    out << "    \"generate_plots_seconds\": " << generate_plots_seconds << ",\n";
    out << "    \"jump_diffusion_simulation_seconds\": " << jump_diffusion_simulation_seconds << ",\n";
    out << "    \"load_sv_seconds\": " << load_csv_seconds << ",\n";
    out << "    \"save_files_seconds\": " << save_files_seconds << "\n";
    out << "  },\n";
    out << "  \"params\": {\n";
    out << "    \"model\": \"jump_diffusion\",\n";
    out << "    \"s0\": " << p.s0 << ",\n";
    out << "    \"mu\": " << p.mu << ",\n";
    out << "    \"sigma\": " << p.sigma_diffusion << ",\n";
    out << "    \"extra\": {\n";
    out << "      \"lambda\": " << p.lambda << ",\n";
    out << "      \"muJ\": " << p.muJ << ",\n";
    out << "      \"sigmaJ\": " << p.sigmaJ << ",\n";
    out << "      \"threshold\": " << p.threshold << "\n";
    out << "    }\n";
    out << "  },\n";
    out << "  \"path_stats\": {\n";
    out << "    \"final_price_mean\": " << mean_final << ",\n";
    out << "    \"final_price_std\": " << std_final << ",\n";
    out << "    \"min_final_price\": " << min_final << ",\n";
    out << "    \"max_final_price\": " << max_final << "\n";
    out << "  }\n";
    if (!sample_paths_csv.empty() || !final_prices_csv.empty()) {
        out << "  ,\"artifacts\": {\n";
        bool first = true;
        if (!sample_paths_csv.empty()) {
            out << "    \"sample_paths_csv\": " << "\"" << sample_paths_csv << "\"";
            first = false;
        }
        if (!final_prices_csv.empty()) {
            if (!first) out << ",\n";
            out << "    \"final_prices_csv\": " << "\"" << final_prices_csv << "\"";
        }
        out << "\n  }\n";
    }
    out << "  ,\"plot_files\": [";
    for (size_t i = 0; i < plot_files.size(); i++) {
        if (i) out << ",";
        out << "\n    " << "\"" << plot_files[i] << "\"";
    }
    if (!plot_files.empty()) out << "\n  ";
    out << "]\n";
    out << "}\n";
}

static void write_synthetic_bars_csv(
    const std::string& out_path,
    const std::vector<std::string>& timestamps,
    const std::vector<double>& close_path,
    unsigned int seed
) {
    if (timestamps.size() != close_path.size()) {
        throw std::runtime_error("timestamps and close_path must have same length for OHLC output.");
    }
    if (timestamps.size() < 2) {
        throw std::runtime_error("Need at least 2 steps to write synthetic OHLC bars.");
    }

    // Simple wick model similar to Python: wick scales with move + noise.
    std::mt19937 rng(seed);
    std::normal_distribution<double> norm01(0.0, 1.0);

    constexpr double base_spread_fraction = 0.15;

    std::ofstream out(out_path);
    if (!out.is_open()) throw std::runtime_error("Could not write: " + out_path);

    out << "timestamp,open,high,low,close\n";

    // First row: flat bar at s0.
    {
        double c = close_path[0];
        out << timestamps[0] << "," << c << "," << c << "," << c << "," << c << "\n";
    }

    for (size_t i = 1; i < close_path.size(); i++) {
        double o = close_path[i - 1];
        double c = close_path[i];
        double move = std::abs(c - o);
        double noise = std::abs(norm01(rng));
        double wick = std::max(move * base_spread_fraction, 1e-8) * (1.0 + 0.75 * noise);
        double h = std::max(o, c) + wick;
        double l = std::max(std::min(o, c) - wick, 1e-8);

        out << timestamps[i] << "," << o << "," << h << "," << l << "," << c << "\n";
    }
}

static void write_final_prices_csv(const std::string& out_path, const std::vector<double>& final_prices) {
    std::ofstream out(out_path);
    if (!out.is_open()) throw std::runtime_error("Could not write: " + out_path);
    out << "final_price\n";
    out << std::setprecision(12);
    for (double v : final_prices) {
        out << v << "\n";
    }
}

static void write_sample_paths_csv(
    const std::string& out_path,
    const std::vector<std::string>& timestamps,
    const std::vector<double>& sample_paths,  // sample_paths_count * n_steps
    int sample_paths_count,
    int n_steps
) {
    if ((int)timestamps.size() != n_steps) {
        throw std::runtime_error("timestamps length must equal n_steps for sample paths output.");
    }
    if ((int)sample_paths.size() != sample_paths_count * n_steps) {
        throw std::runtime_error("sample_paths size mismatch.");
    }
    std::ofstream out(out_path);
    if (!out.is_open()) throw std::runtime_error("Could not write: " + out_path);

    out << "timestamp";
    for (int p = 0; p < sample_paths_count; p++) {
        out << ",path_" << p;
    }
    out << "\n";
    out << std::setprecision(12);

    for (int t = 0; t < n_steps; t++) {
        out << timestamps[(size_t)t];
        for (int p = 0; p < sample_paths_count; p++) {
            out << "," << sample_paths[(size_t)p * (size_t)n_steps + (size_t)t];
        }
        out << "\n";
    }
}

static int parse_int_arg(const std::string& v, const std::string& name) {
    char* endptr = nullptr;
    errno = 0;
    long x = std::strtol(v.c_str(), &endptr, 10);
    if (errno != 0 || endptr == v.c_str()) {
        throw std::runtime_error("Invalid int for " + name + ": " + v);
    }
    return (int)x;
}

static double parse_double_arg(const std::string& v, const std::string& name) {
    char* endptr = nullptr;
    errno = 0;
    double x = std::strtod(v.c_str(), &endptr);
    if (errno != 0 || endptr == v.c_str()) {
        throw std::runtime_error("Invalid double for " + name + ": " + v);
    }
    return x;
}

int main(int argc, char** argv) {
    // Defaults mirror the Python script’s intent but keep step count controllable for benchmarking.
    std::string input_csv;
    std::string output_bars = "reports/jump_diffusion/jump_diffusion_synthetic_bars_cuda.csv";
    std::string output_params = "reports/jump_diffusion/jump_diffusion_params_cuda.json";
    std::string output_final_prices = "";
    std::string output_sample_paths = "";
    int sample_paths = 0;
    int n_paths = 10000;
    int n_steps = 200;
    int seed = 42;
    double jump_threshold_mult = 2.5;
    double dt = 1.0;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto need_value = [&](const std::string& flag) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + flag);
            return std::string(argv[++i]);
        };

        if (a == "--input") input_csv = need_value(a);
        else if (a == "--output-bars") output_bars = need_value(a);
        else if (a == "--output-params") output_params = need_value(a);
        else if (a == "--output-final-prices") output_final_prices = need_value(a);
        else if (a == "--output-sample-paths") output_sample_paths = need_value(a);
        else if (a == "--sample-paths") sample_paths = parse_int_arg(need_value(a), a);
        else if (a == "--n-paths") n_paths = parse_int_arg(need_value(a), a);
        else if (a == "--n-steps") n_steps = parse_int_arg(need_value(a), a);
        else if (a == "--seed") seed = parse_int_arg(need_value(a), a);
        else if (a == "--jump-threshold-mult") jump_threshold_mult = parse_double_arg(need_value(a), a);
        else if (a == "--dt") dt = parse_double_arg(need_value(a), a);
        else if (a == "--help" || a == "-h") {
            std::cout
                << "Jump-Diffusion Monte Carlo (CUDA)\n\n"
                << "Required:\n"
                << "  --input <csv>              CSV with at least columns: timestamp, close\n\n"
                << "Optional:\n"
                << "  --n-paths <int>            Number of Monte Carlo paths (default 10000)\n"
                << "  --n-steps <int>            Steps to simulate (uses first N CSV rows; default 200)\n"
                << "  --seed <int>               RNG seed (default 42)\n"
                << "  --jump-threshold-mult <f>  Jump threshold multiplier (default 2.5)\n"
                << "  --dt <f>                   Time step (default 1.0)\n"
                << "  --output-bars <csv>        Output synthetic OHLC CSV (default reports/...)\n"
                << "  --output-params <json>     Output params + stats JSON (default reports/...)\n"
                << "  --sample-paths <int>       Also record the first K full paths (for plotting)\n"
                << "  --output-sample-paths <csv>  CSV of sample paths (requires --sample-paths)\n"
                << "  --output-final-prices <csv>  CSV of all final prices (optional; can be large)\n\n"
                << "Build example:\n"
                << "  nvcc -O3 -std=c++17 monte_carlo/GPU/jump_diffusion_synth.cu -o jump_diffusion_cuda -lcurand\n"
                << "Run example:\n"
                << "  ./jump_diffusion_cuda --input usdjpy-m1-bid-2013.csv --n-paths 100000 --n-steps 200\n";
            return 0;
        } else {
            throw std::runtime_error("Unknown arg: " + a);
        }
    }

    if (input_csv.empty()) {
        throw std::runtime_error("Missing required arg: --input <csv>");
    }
    if (n_paths <= 0) throw std::runtime_error("--n-paths must be > 0");
    if (n_steps <= 1) throw std::runtime_error("--n-steps must be >= 2");
    if (dt <= 0.0) throw std::runtime_error("--dt must be > 0");
    if (sample_paths < 0) throw std::runtime_error("--sample-paths must be >= 0");
    if (sample_paths > n_paths) sample_paths = n_paths;
    if (sample_paths > 0 && output_sample_paths.empty()) {
        output_sample_paths = "reports/jump_diffusion/jump_diffusion_sample_paths_cuda.csv";
    }
    if (!output_final_prices.empty() && output_final_prices == "1") {
        // Convenience: --output-final-prices 1 uses a default path.
        output_final_prices = "reports/jump_diffusion/jump_diffusion_final_prices_cuda.csv";
    }

    auto t_total_start = std::chrono::steady_clock::now();

    // Load only the first n_steps rows (so you can benchmark fixed steps like the GBM benchmark does).
    auto t_load_start = std::chrono::steady_clock::now();
    auto [timestamps, close_prices] = read_timestamps_and_close(input_csv, n_steps);
    if ((int)close_prices.size() < n_steps) {
        throw std::runtime_error("Input CSV had fewer usable rows than --n-steps.");
    }
    double step_seconds = median_step_seconds(timestamps);
    auto t_load_end = std::chrono::steady_clock::now();

    // Fit params on CPU to match Python behavior.
    auto t_fit_start = std::chrono::steady_clock::now();
    JumpDiffusionParams params = fit_jump_diffusion_params(close_prices, jump_threshold_mult);
    auto t_fit_end = std::chrono::steady_clock::now();

    // Best-effort: create output directories if they don't exist.
    try {
        std::filesystem::create_directories(std::filesystem::path(output_bars).parent_path());
        std::filesystem::create_directories(std::filesystem::path(output_params).parent_path());
        if (!output_sample_paths.empty()) {
            std::filesystem::create_directories(std::filesystem::path(output_sample_paths).parent_path());
        }
        if (!output_final_prices.empty()) {
            std::filesystem::create_directories(std::filesystem::path(output_final_prices).parent_path());
        }
    } catch (...) {
        // Ignore; writing will fail later if paths are invalid.
    }

    auto t_gpu_sim_start = std::chrono::steady_clock::now();

    // Allocate device buffers: final prices for all paths + first path series.
    auto t_gpu_alloc_start = std::chrono::steady_clock::now();
    double* d_final = nullptr;
    double* d_first_path = nullptr;
    double* d_sample_paths = nullptr;
    CHECK_CUDA(cudaMalloc(&d_final, (size_t)n_paths * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_first_path, (size_t)n_steps * sizeof(double)));
    if (sample_paths > 0) {
        CHECK_CUDA(cudaMalloc(&d_sample_paths, (size_t)sample_paths * (size_t)n_steps * sizeof(double)));
    }
    auto t_gpu_alloc_end = std::chrono::steady_clock::now();

    int threads = 256;
    int blocks = (n_paths + threads - 1) / threads;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    jump_diffusion_kernel_final_and_first_path<<<blocks, threads>>>(
        d_final,
        d_first_path,
        d_sample_paths,
        sample_paths,
        params,
        n_steps,
        n_paths,
        dt,
        (unsigned long long)seed
    );
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaGetLastError());

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // Copy results back.
    auto t_d2h_start = std::chrono::steady_clock::now();
    std::vector<double> final_prices((size_t)n_paths);
    std::vector<double> first_path((size_t)n_steps);
    std::vector<double> sample_paths_host;
    if (sample_paths > 0) {
        sample_paths_host.resize((size_t)sample_paths * (size_t)n_steps);
    }
    CHECK_CUDA(cudaMemcpy(final_prices.data(), d_final, (size_t)n_paths * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(first_path.data(), d_first_path, (size_t)n_steps * sizeof(double), cudaMemcpyDeviceToHost));
    if (sample_paths > 0) {
        CHECK_CUDA(cudaMemcpy(
            sample_paths_host.data(),
            d_sample_paths,
            (size_t)sample_paths * (size_t)n_steps * sizeof(double),
            cudaMemcpyDeviceToHost
        ));
    }
    auto t_d2h_end = std::chrono::steady_clock::now();

    // Compute simple path stats on CPU.
    auto t_post_start = std::chrono::steady_clock::now();
    double sum = 0.0;
    double minv = final_prices[0];
    double maxv = final_prices[0];
    for (double v : final_prices) {
        sum += v;
        minv = std::min(minv, v);
        maxv = std::max(maxv, v);
    }
    double mean_final = sum / (double)n_paths;
    double var_acc = 0.0;
    if (n_paths > 1) {
        for (double v : final_prices) {
            double d = v - mean_final;
            var_acc += d * d;
        }
        var_acc /= (double)(n_paths - 1);
    }
    double std_final = std::sqrt(var_acc);

    // Save output artifacts (kept separate from kernel timing).
    auto t_save_start = std::chrono::steady_clock::now();
    write_synthetic_bars_csv(output_bars, timestamps, first_path, (unsigned int)seed);
    if (!output_final_prices.empty()) {
        write_final_prices_csv(output_final_prices, final_prices);
    }
    if (sample_paths > 0 && !output_sample_paths.empty()) {
        write_sample_paths_csv(output_sample_paths, timestamps, sample_paths_host, sample_paths, n_steps);
    }
    auto t_save_end = std::chrono::steady_clock::now();

    // Write params/stats JSON. Use kernel time as the elapsed value (benchmark style).
    Percentiles pct = compute_percentiles_5_50_95(final_prices);
    auto t_post_end = std::chrono::steady_clock::now();

    auto t_gpu_sim_end = std::chrono::steady_clock::now();

    double elapsed_seconds_kernel = (double)elapsed_ms / 1000.0;
    double elapsed_seconds_total = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_total_start).count();

    double load_csv_seconds = std::chrono::duration<double>(t_load_end - t_load_start).count();
    double fit_parameters_seconds = std::chrono::duration<double>(t_fit_end - t_fit_start).count();
    double jump_diffusion_sim_seconds = std::chrono::duration<double>(t_gpu_sim_end - t_gpu_sim_start).count();
    double gpu_alloc_seconds = std::chrono::duration<double>(t_gpu_alloc_end - t_gpu_alloc_start).count();
    double gpu_d2h_seconds = std::chrono::duration<double>(t_d2h_end - t_d2h_start).count();
    double host_postprocess_seconds = std::chrono::duration<double>(t_post_end - t_post_start).count();
    double save_files_seconds = std::chrono::duration<double>(t_save_end - t_save_start).count();

    // Now that percentiles are computed, write JSON.
    std::vector<std::string> plot_files;  // Generated by a post-step (see scripts).
    double generate_plots_seconds = 0.0;

    write_params_json(
        output_params,
        input_csv,
        step_seconds,
        n_paths,
        n_steps,
        elapsed_seconds_total,
        params,
        mean_final,
        std_final,
        minv,
        maxv,
        plot_files,
        output_sample_paths,
        output_final_prices,
        load_csv_seconds,
        fit_parameters_seconds,
        jump_diffusion_sim_seconds,
        gpu_alloc_seconds,
        elapsed_seconds_kernel,
        gpu_d2h_seconds,
        host_postprocess_seconds,
        save_files_seconds,
        generate_plots_seconds
    );

    std::cout << "CUDA kernel time: " << elapsed_ms << " ms\n";
    std::cout << "Simulated " << n_paths << " paths in " << elapsed_seconds_kernel << " seconds\n";
    std::cout << "Total wall time (host+GPU): " << elapsed_seconds_total << " seconds\n";
    std::cout << "Saved synthetic bars to " << output_bars << "\n";
    std::cout << "Saved params/stats to " << output_params << "\n";
    if (!output_final_prices.empty()) {
        std::cout << "Saved final prices to " << output_final_prices << "\n";
    }
    if (sample_paths > 0 && !output_sample_paths.empty()) {
        std::cout << "Saved sample paths to " << output_sample_paths << "\n";
    }
    std::cout << "Final price mean=" << mean_final
              << " median=" << pct.p50
              << " p5=" << pct.p5
              << " p95=" << pct.p95
              << "\n";
    std::cout << "\n=== Timing Breakdown ===\n";
    std::cout << "  Load CSV:                  " << load_csv_seconds << "s\n";
    std::cout << "  Fit parameters:            " << fit_parameters_seconds << "s\n";
    std::cout << "  Jump-diffusion (GPU total):" << jump_diffusion_sim_seconds << "s\n";
    std::cout << "    GPU alloc:               " << gpu_alloc_seconds << "s\n";
    std::cout << "    GPU kernel:              " << elapsed_seconds_kernel << "s\n";
    std::cout << "    GPU D2H memcpy:          " << gpu_d2h_seconds << "s\n";
    std::cout << "  Host postprocess:          " << host_postprocess_seconds << "s\n";
    std::cout << "  Save files:                " << save_files_seconds << "s\n";
    std::cout << "  Generate plots:            " << generate_plots_seconds << "s\n";
    std::cout << "  Total wall:                " << elapsed_seconds_total << "s\n";

    CHECK_CUDA(cudaFree(d_final));
    CHECK_CUDA(cudaFree(d_first_path));
    if (d_sample_paths != nullptr) CHECK_CUDA(cudaFree(d_sample_paths));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}

