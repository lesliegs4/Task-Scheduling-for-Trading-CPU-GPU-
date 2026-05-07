/*
 * heston_gpu.cu  —  fully self-contained Heston Monte Carlo on GPU
 *
 * Does everything heston_synth.py does:
 *   1. Parse input CSV
 *   2. Fit Heston params (identical math to heston_synth.py)
 *   3. Simulate n_paths on GPU  (timed with cudaEvent — kernel only)
 *   4. Build synthetic bars from path 0
 *   5. Write synthetic_bars.csv, heston_params.json, timing.json
 *
 * Build:
 *   nvcc -O3 -arch=sm_80 -o heston_gpu heston_gpu.cu -lcurand -std=c++17
 *
 * Run:
 *   ./heston_gpu --input usdjpy-m1-bid-2013.csv \
 *               --n-paths 1000000              \
 *               --out-bars    heston_synthetic_bars.csv \
 *               --out-params  heston_params.json        \
 *               --out-timing  heston_timing.json        \
 *               --seed 42
 *
 * For benchmarking: --out-bars and --out-params can be /dev/null if you only
 * care about timing.  The kernel timer is always written to --out-timing.
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

/* ======================================================================
 *  CUDA error macro
 * ====================================================================== */
#define CUDA_CHECK(expr)                                                   \
    do {                                                                   \
        cudaError_t _e = (expr);                                           \
        if (_e != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(_e));           \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

/* ======================================================================
 *  Data structures
 * ====================================================================== */

struct Bar {
    char   timestamp[32];   /* stored as-is from CSV */
    double open, high, low, close;
};

/* Parameters that go to the kernel (kept in a flat struct to minimise
 * register pressure — same fields as Python BarModelParams.extra). */
struct HestonParams {
    double s0;
    double mu;
    double v0;
    double theta;
    double kappa;
    double xi;
    double rho;
    double dt;
    double sqrt_dt;
    double rho_bar;   /* sqrt(1 - rho^2), precomputed */
};

/* ======================================================================
 *  1.  CSV PARSER
 *  Expects header: timestamp,open,high,low,close  (any extra cols ignored)
 *  Handles both Unix (\n) and Windows (\r\n) line endings.
 * ====================================================================== */

static inline char* ltrim(char* s) {
    while (*s == ' ' || *s == '\t') ++s;
    return s;
}

static inline void rtrim(char* s) {
    size_t n = strlen(s);
    while (n > 0 && (s[n-1] == '\r' || s[n-1] == '\n' ||
                     s[n-1] == ' '  || s[n-1] == '\t'))
        s[--n] = '\0';
}

/* Returns index of column name in comma-split header, or -1. */
static int find_col(const std::vector<std::string>& hdr, const char* name) {
    for (int i = 0; i < (int)hdr.size(); ++i)
        if (hdr[i] == name) return i;
    return -1;
}

static std::vector<std::string> split_csv_line(const char* line) {
    std::vector<std::string> fields;
    const char* p = line;
    while (true) {
        const char* q = strchr(p, ',');
        if (!q) { fields.emplace_back(p); break; }
        fields.emplace_back(p, q - p);
        p = q + 1;
    }
    return fields;
}

static std::vector<Bar> load_csv(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) { perror(path); exit(1); }

    char line[512];
    /* Read header. */
    if (!fgets(line, sizeof(line), f)) {
        fprintf(stderr, "Empty file: %s\n", path); exit(1);
    }
    rtrim(line);
    auto hdr = split_csv_line(line);
    for (auto& h : hdr) { h = ltrim(h.data()); rtrim(h.data()); }

    int col_ts    = find_col(hdr, "timestamp");
    int col_open  = find_col(hdr, "open");
    int col_high  = find_col(hdr, "high");
    int col_low   = find_col(hdr, "low");
    int col_close = find_col(hdr, "close");

    if (col_ts < 0 || col_open < 0 || col_high < 0 ||
        col_low < 0 || col_close < 0) {
        fprintf(stderr,
            "CSV must have columns: timestamp,open,high,low,close\n"
            "Found header: %s\n", line);
        exit(1);
    }

    std::vector<Bar> bars;
    bars.reserve(600000);   /* ~1 year of 1-min bars */

    while (fgets(line, sizeof(line), f)) {
        rtrim(line);
        if (line[0] == '\0') continue;
        auto fields = split_csv_line(line);
        if ((int)fields.size() <= std::max({col_ts, col_open,
                                            col_high, col_low, col_close}))
            continue;

        Bar b;
        strncpy(b.timestamp, fields[col_ts].c_str(), sizeof(b.timestamp) - 1);
        b.timestamp[sizeof(b.timestamp) - 1] = '\0';
        b.open  = atof(fields[col_open].c_str());
        b.high  = atof(fields[col_high].c_str());
        b.low   = atof(fields[col_low].c_str());
        b.close = atof(fields[col_close].c_str());
        bars.push_back(b);
    }
    fclose(f);

    if (bars.empty()) {
        fprintf(stderr, "No data rows in %s\n", path); exit(1);
    }
    printf("[csv]   loaded %zu bars from %s\n", bars.size(), path);
    return bars;
}

/* ======================================================================
 *  2.  PARAM FITTING  (mirrors heston_synth.py fit_heston_params exactly)
 * ====================================================================== */

static HestonParams fit_heston_params(const std::vector<Bar>& bars,
                                      double* out_sigma) {
    int n = (int)bars.size();

    /* Log returns: r[i] = log(close[i+1] / close[i]) */
    std::vector<double> r(n - 1);
    for (int i = 0; i < n - 1; ++i)
        r[i] = std::log(bars[i+1].close / bars[i].close);

    int nr = (int)r.size();

    double mu = 0.0;
    for (double x : r) mu += x;
    mu /= nr;

    /* Variance of returns  →  v0, theta */
    double v0 = 0.0;
    for (double x : r) v0 += (x - mu) * (x - mu);
    v0 = (nr > 1) ? v0 / (nr - 1) : 1e-8;
    double theta = v0;

    /* Squared demeaned returns */
    std::vector<double> sq(nr);
    for (int i = 0; i < nr; ++i) sq[i] = (r[i] - mu) * (r[i] - mu);

    /* Lag-1 autocorrelation of sq  →  kappa */
    double corr = 0.0;
    if (nr > 2) {
        double mx = 0.0, my = 0.0;
        for (int i = 0; i < nr - 1; ++i) { mx += sq[i]; my += sq[i+1]; }
        mx /= (nr - 1); my /= (nr - 1);
        double num = 0.0, dx2 = 0.0, dy2 = 0.0;
        for (int i = 0; i < nr - 1; ++i) {
            double dx = sq[i] - mx, dy = sq[i+1] - my;
            num += dx * dy; dx2 += dx * dx; dy2 += dy * dy;
        }
        if (dx2 > 0.0 && dy2 > 0.0)
            corr = num / std::sqrt(dx2 * dy2);
        if (std::isnan(corr)) corr = 0.0;
        corr = std::max(-0.99, std::min(0.99, corr));
    }
    double abs_corr = std::abs(corr);
    double kappa = (abs_corr > 1e-6)
                   ? std::max(0.5, -std::log(std::max(1e-6, abs_corr)))
                   : 2.0;

    /* Std of squared returns  →  xi (vol-of-vol) */
    double sq_mean = 0.0;
    for (double x : sq) sq_mean += x;
    sq_mean /= nr;
    double sq_var = 0.0;
    for (double x : sq) sq_var += (x - sq_mean) * (x - sq_mean);
    double xi = (nr > 1) ? std::sqrt(sq_var / (nr - 1)) : 1e-4;
    xi = std::max(1e-6, xi);

    double rho  = -0.2;   /* conservative FX default, same as Python */
    double rho2 = rho * rho;

    if (out_sigma) *out_sigma = std::sqrt(v0);

    HestonParams p;
    p.s0     = bars[0].close;
    p.mu     = mu;
    p.v0     = v0;
    p.theta  = theta;
    p.kappa  = kappa;
    p.xi     = xi;
    p.rho    = rho;
    p.dt     = 1.0;
    p.sqrt_dt = 1.0;
    p.rho_bar = (rho2 < 1.0) ? std::sqrt(1.0 - rho2) : 0.0;

    printf("[fit]   s0=%.6g  mu=%.4g  v0=%.4g  theta=%.4g  "
           "kappa=%.4g  xi=%.4g  rho=%.3g\n",
           p.s0, p.mu, p.v0, p.theta, p.kappa, p.xi, p.rho);

    return p;
}

/* Infer step size in seconds from first two timestamps.
 * Timestamps can be ISO 8601 ("2013-01-02 00:01:00") or Unix epoch strings. */
static double infer_step_seconds(const std::vector<Bar>& bars) {
    if (bars.size() < 2) return 60.0;
    /* Try to parse as Unix epoch first. */
    char* end1; char* end2;
    double t1 = strtod(bars[0].timestamp, &end1);
    double t2 = strtod(bars[1].timestamp, &end2);
    if (*end1 == '\0' && *end2 == '\0' && t2 > t1)
        return t2 - t1;
    /* Fallback: assume 1-minute bars. */
    return 60.0;
}

/* ======================================================================
 *  3.  HESTON KERNEL  (one thread = one full path)
 * ====================================================================== */

__global__ void heston_kernel(
    const HestonParams  p,
    const int           n_paths,
    const int           n_steps,
    const uint64_t      base_seed,
    double* __restrict__ d_finals,
    double* __restrict__ d_path0   /* path index 0 only, length n_steps; may be NULL */
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_paths) return;

    curandStatePhilox4_32_10_t rng;
    curand_init(base_seed, (uint64_t)idx, 0, &rng);

    double s = p.s0;
    double v = p.v0;

    if (d_path0 != nullptr && idx == 0)
        d_path0[0] = s;

    for (int t = 1; t < n_steps; ++t) {
        double2 zz = curand_normal2_double(&rng);
        double w1 = zz.x;
        double w2 = p.rho * zz.x + p.rho_bar * zz.y;

        /* Variance — CIR Euler with reflection */
        double v_pos = fmax(v, 0.0);
        v = v + p.kappa * (p.theta - v) * p.dt
              + p.xi * sqrt(v_pos) * p.sqrt_dt * w2;
        if (v < 0.0) v = 0.0;

        /* Price — log-normal Euler */
        double v2 = fmax(v, 0.0);
        s = fmax(1e-8,
                 s * exp((p.mu - 0.5 * v2) * p.dt
                          + sqrt(v2) * p.sqrt_dt * w1));

        if (d_path0 != nullptr && idx == 0)
            d_path0[t] = s;
    }

    d_finals[idx] = s;
}

/* ======================================================================
 *  4.  OHLC CONSTRUCTION FROM CLOSE PATH
 *  Mirrors trading_model_utils.make_ohlc_from_close_path:
 *    open  = previous close
 *    high  = max(open, close) * (1 + |N(0, sigma_hl)|)
 *    low   = min(open, close) * (1 - |N(0, sigma_hl)|)
 *  where sigma_hl is estimated from the historical bars.
 * ====================================================================== */

/* Simple LCG for host-side noise — not for the kernel, just OHLC dressing. */
struct HostRng {
    uint64_t state;
    explicit HostRng(uint64_t seed) : state(seed) {}
    double normal() {
        /* Box-Muller */
        double u, v2, s2;
        do {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            u = (double)(state >> 11) / (double)(1ULL << 53) * 2.0 - 1.0;
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            v2 = (double)(state >> 11) / (double)(1ULL << 53) * 2.0 - 1.0;
            s2 = u * u + v2 * v2;
        } while (s2 >= 1.0 || s2 == 0.0);
        return u * std::sqrt(-2.0 * std::log(s2) / s2);
    }
};

static std::vector<Bar> make_ohlc(const std::vector<Bar>& real_bars,
                                   const double* close_path,
                                   int n_steps,
                                   uint64_t seed) {
    /* Estimate hl_sigma from real bars */
    double sum_hl = 0.0;
    int cnt = 0;
    for (auto& b : real_bars) {
        if (b.low > 0.0 && b.high > 0.0) {
            sum_hl += std::log(b.high / b.low);
            ++cnt;
        }
    }
    double sigma_hl = (cnt > 0) ? sum_hl / cnt * 0.5 : 1e-4;

    HostRng rng(seed);
    std::vector<Bar> out(n_steps);

    for (int i = 0; i < n_steps; ++i) {
        strncpy(out[i].timestamp, real_bars[i].timestamp,
                sizeof(out[i].timestamp) - 1);
        out[i].close = close_path[i];
        out[i].open  = (i == 0) ? close_path[0] : close_path[i - 1];

        double hi = std::max(out[i].open, out[i].close);
        double lo = std::min(out[i].open, out[i].close);
        double noise = std::abs(rng.normal()) * sigma_hl;
        out[i].high = hi * (1.0 + noise);
        out[i].low  = lo * (1.0 - noise);
    }
    return out;
}

/* ======================================================================
 *  5.  OUTPUT WRITERS
 * ====================================================================== */

static void write_bars_csv(const char* path,
                            const std::vector<Bar>& bars) {
    FILE* f = fopen(path, "w");
    if (!f) { perror(path); exit(1); }
    fprintf(f, "timestamp,open,high,low,close\n");
    for (auto& b : bars)
        fprintf(f, "%s,%.10g,%.10g,%.10g,%.10g\n",
                b.timestamp, b.open, b.high, b.low, b.close);
    fclose(f);
}

/* Minimal JSON writer — avoids pulling in a library. */
static void write_params_json(const char* path,
                               const char* input_csv,
                               double      step_seconds,
                               int         n_paths,
                               int         n_steps,
                               int         block_size,
                               double      kernel_ms,
                               double      total_secs,
                               const HestonParams& p,
                               double      sigma,
                               /* path stats */
                               double mean_fp, double std_fp,
                               double min_fp,  double max_fp,
                               /* timing breakdown */
                               double load_csv_secs,
                               double fit_params_secs,
                               double ohlc_secs,
                               double save_files_secs) {
    FILE* f = fopen(path, "w");
    if (!f) { perror(path); exit(1); }

    fprintf(f,
        "{\n"
        "  \"chunk_size\": %d,\n"
        "  \"elapsed_seconds\": %.15g,\n"
        "  \"fit_from\": \"%s\",\n"
        "  \"n_paths\": %d,\n"
        "  \"params\": {\n"
        "    \"extra\": {\n"
        "      \"kappa\": %.15g,\n"
        "      \"rho\": %.15g,\n"
        "      \"theta\": %.15e,\n"
        "      \"v0\": %.15e,\n"
        "      \"xi\": %.15e\n"
        "    },\n"
        "    \"model\": \"heston\",\n"
        "    \"mu\": %.15e,\n"
        "    \"s0\": %.10g,\n"
        "    \"sigma\": %.15g\n"
        "  },\n"
        "  \"path_stats\": {\n"
        "    \"final_price_mean\": %.15g,\n"
        "    \"final_price_std\": %.15g,\n"
        "    \"max_final_price\": %.15g,\n"
        "    \"min_final_price\": %.15g\n"
        "  },\n"
        "  \"step_seconds\": %.6g,\n"
        "  \"timing_breakdown\": {\n"
        "    \"fit_parameters_seconds\": %.15g,\n"
        "    \"heston_simulation_seconds\": %.15g,\n"
        "    \"load_csv_seconds\": %.15g,\n"
        "    \"ohlc_generation_seconds\": %.15g,\n"
        "    \"save_files_seconds\": %.15g\n"
        "  }\n"
        "}\n",
        block_size, total_secs,
        input_csv, n_paths,
        p.kappa, p.rho, p.theta, p.v0, p.xi,
        p.mu, p.s0, sigma,
        mean_fp, std_fp, max_fp, min_fp,
        step_seconds,
        fit_params_secs, kernel_ms / 1000.0, load_csv_secs, ohlc_secs, save_files_secs
    );
    fclose(f);
}

/* ======================================================================
 *  6.  CLI HELPERS
 * ====================================================================== */

static const char* get_arg(int argc, char** argv,
                            const char* flag, const char* def) {
    for (int i = 1; i < argc - 1; ++i)
        if (strcmp(argv[i], flag) == 0) return argv[i+1];
    return def;
}
static int    iarg(int argc, char** argv, const char* f, int    d) {
    const char* s = get_arg(argc, argv, f, nullptr);
    return s ? atoi(s) : d;
}
static double darg(int argc, char** argv, const char* f, double d) {
    const char* s = get_arg(argc, argv, f, nullptr);
    return s ? atof(s) : d;
}
static bool flag_set(int argc, char** argv, const char* f) {
    for (int i = 1; i < argc; ++i)
        if (strcmp(argv[i], f) == 0) return true;
    return false;
}

static void write_final_prices_csv(const char* path, const std::vector<double>& finals) {
    FILE* f = fopen(path, "w");
    if (!f) { perror(path); exit(1); }
    fprintf(f, "final_price\n");
    for (double x : finals)
        fprintf(f, "%.17g\n", x);
    fclose(f);
}

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s --input FILE [options]\n\n"
        "  --input        FILE   input OHLC CSV (timestamp,open,high,low,close)\n"
        "  --out-bars     FILE   synthetic OHLC CSV          [heston_synthetic_bars.csv]\n"
        "  --out-params   FILE   fitted params + timing JSON  [heston_params.json]\n"
        "  --output-final-prices FILE  per-path final prices CSV (header + one column)\n"
        "  --n-paths      N      Monte Carlo paths             [10000]\n"
        "  --block-size   N      CUDA threads/block            [256]\n"
        "  --seed         N      RNG seed                      [42]\n"
        "  --no-bars             skip writing synthetic bars\n"
        "\nBenchmark tip: use --no-bars when only kernel timing matters.\n",
        prog);
}

/* ======================================================================
 *  7.  MAIN
 * ====================================================================== */

int main(int argc, char** argv) {

    if (argc < 3 || flag_set(argc, argv, "--help") ||
        flag_set(argc, argv, "-h")) {
        print_usage(argv[0]); return 1;
    }

    const char* input_csv  = get_arg(argc, argv, "--input",      nullptr);
    const char* out_bars   = get_arg(argc, argv, "--out-bars",   "heston_synthetic_bars.csv");
    const char* out_params = get_arg(argc, argv, "--out-params", "heston_params.json");
    const char* out_final_prices = get_arg(argc, argv, "--output-final-prices", nullptr);
    int         n_paths    = iarg(argc, argv, "--n-paths",   10000);
    int         block_size = iarg(argc, argv, "--block-size", 256);
    int         seed       = iarg(argc, argv, "--seed",       42);
    bool        no_bars    = flag_set(argc, argv, "--no-bars");

    if (!input_csv) {
        fprintf(stderr, "Error: --input is required\n");
        print_usage(argv[0]); return 1;
    }

    /* ---- wall-clock start ---- */
    struct timespec wall0, wall1, t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &wall0);

    /* ---- 1. Load data ---- */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    auto bars    = load_csv(input_csv);
    int  n_steps = (int)bars.size();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double load_csv_secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;

    /* ---- 2. Fit params ---- */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    double sigma = 0.0;
    HestonParams p = fit_heston_params(bars, &sigma);
    double step_secs = infer_step_seconds(bars);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double fit_params_secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;

    /* ---- 3. GPU simulation ---- */
    printf("[gpu]   n_paths=%d  n_steps=%d  block_size=%d  seed=%d\n",
           n_paths, n_steps, block_size, seed);

    double* d_finals  = nullptr;
    double* d_path0   = nullptr;

    CUDA_CHECK(cudaMalloc(&d_finals, (size_t)n_paths * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_path0,  (size_t)n_steps * sizeof(double)));

    int grid = (n_paths + block_size - 1) / block_size;

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));

    heston_kernel<<<grid, block_size>>>(
        p, n_paths, n_steps, (uint64_t)seed,
        d_finals, d_path0
    );

    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaGetLastError());

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev0, ev1));
    printf("[gpu]   kernel time: %.3f ms\n", (double)kernel_ms);

    /* ---- Copy results ---- */
    auto h_finals = std::vector<double>(n_paths);
    CUDA_CHECK(cudaMemcpy(h_finals.data(), d_finals,
                          (size_t)n_paths * sizeof(double),
                          cudaMemcpyDeviceToHost));

    auto h_path0 = std::vector<double>(n_steps);
    CUDA_CHECK(cudaMemcpy(h_path0.data(), d_path0,
                          (size_t)n_steps * sizeof(double),
                          cudaMemcpyDeviceToHost));

    cudaFree(d_finals);
    cudaFree(d_path0);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);

    /* ---- 4. Path stats ---- */
    double sum = 0.0, sum2 = 0.0,
           mn  = h_finals[0], mx = h_finals[0];
    for (double x : h_finals) {
        sum += x; sum2 += x * x;
        if (x < mn) mn = x;
        if (x > mx) mx = x;
    }
    double mean_fp = sum / n_paths;
    double std_fp  = (n_paths > 1)
        ? std::sqrt((sum2 - (double)n_paths * mean_fp * mean_fp) / (n_paths - 1))
        : 0.0;
    printf("[stats] final price  mean=%.6g  std=%.6g  min=%.6g  max=%.6g\n",
           mean_fp, std_fp, mn, mx);

    if (out_final_prices) {
        write_final_prices_csv(out_final_prices, h_finals);
        printf("[out]   final prices   -> %s\n", out_final_prices);
    }

    /* ---- 5. Synthetic OHLC ---- */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    double ohlc_secs = 0.0;
    if (!no_bars) {
        auto synth = make_ohlc(bars, h_path0.data(), n_steps, (uint64_t)seed);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        ohlc_secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        
        clock_gettime(CLOCK_MONOTONIC, &t0);
        write_bars_csv(out_bars, synth);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double write_bars_secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        printf("[out]   synthetic bars -> %s\n", out_bars);
    } else {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        ohlc_secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    }

    /* ---- 6. Save JSON params/timing ---- */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    clock_gettime(CLOCK_MONOTONIC, &wall1);
    double total_secs = (wall1.tv_sec  - wall0.tv_sec) +
                        (wall1.tv_nsec - wall0.tv_nsec) * 1e-9;

    write_params_json(out_params, input_csv, step_secs,
                      n_paths, n_steps, block_size,
                      (double)kernel_ms, total_secs,
                      p, sigma,
                      mean_fp, std_fp, mn, mx,
                      load_csv_secs, fit_params_secs, ohlc_secs, 0.0);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double save_files_secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    printf("[out]   params/timing  -> %s\n", out_params);
    printf("[done]  total wall time: %.3f s\n", total_secs);

    return 0;
}
