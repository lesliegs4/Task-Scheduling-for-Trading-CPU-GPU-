import time
import pandas as pd
import numpy as np

#load historical price data from bid or ask CSV file
def load_prices(filename):
    df = pd.read_csv(filename, parse_dates=["timestamp"])
    prices = df["close"].dropna()
    return prices

#estimate GBM parameters; compute log returns and calculate
#drift (average return) & volatility (standard dev of returns)
def estimate_gbm_parameters(prices):
    log_returns = np.log(prices / prices.shift(1)).dropna()

    drift = log_returns.mean()
    volatility = log_returns.std()

    return drift, volatility

#run Monte Carlo simulation
def run_cpu_gbm(S0, drift, volatility, n_timeintervals, N_sim, dt=1):
    #matrix of random values
    Z = np.random.standard_normal((n_timeintervals, N_sim))

    #GBM step calculation
    step = np.exp(
        (drift - 0.5 * volatility**2) * dt
        + volatility * np.sqrt(dt) * Z
    )

    #create storage for all simulations & first row = starting price
    sim = np.zeros((n_timeintervals + 1, N_sim))
    sim[0] = S0

    #new price = prev price * growth factor
    for t in range(1, n_timeintervals + 1):
        sim[t] = sim[t - 1] * step[t - 1]

    return sim

#this ties everything together and collects results
def benchmark_cpu_gbm(filename, output_csv):
    #load prices, estimate drift and volatility, and get last price as starting val
    prices = load_prices(filename)

    drift, volatility = estimate_gbm_parameters(prices)
    S0 = prices.iloc[-1]

    #different simulation sizes 
    n_timeintervals = 200
    workload_sizes = [1_000, 10_000, 100_000, 1_000_000]

    results = []

    #loop over workloads
    for N_sim in workload_sizes:
        print(f"Running CPU GBM with {N_sim} paths...")

        #start timer
        start = time.perf_counter()

        #run simulation
        sim = run_cpu_gbm(
            S0=S0,
            drift=drift,
            volatility=volatility,
            n_timeintervals=n_timeintervals,
            N_sim=N_sim,
            dt=1,
        )

        #stop timer
        end = time.perf_counter()
        elapsed_time = end - start

        final = sim[-1]

        #get all results and extract final values
        results.append({
            "execution_mode": "CPU-only",
            "input_file": filename,
            "paths": N_sim,
            "time_steps": n_timeintervals,
            "execution_time_seconds": elapsed_time,
            "mean_final_price": final.mean(),
            "median_final_price": np.median(final),
            "5th_percentile": np.percentile(final, 5),
            "95th_percentile": np.percentile(final, 95),
        })

    #save results into csv
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)

    print("\nCPU GBM Benchmark Results:")
    print(results_df)

    print(f"\nSaved results to {output_csv}")


if __name__ == "__main__":
    np.random.seed(42)

    benchmark_cpu_gbm(
        filename="usdjpy-m1-bid-2013.csv",
        output_csv="cpu_gbm_bid_benchmark.csv",
    )

    benchmark_cpu_gbm(
        filename="usdjpy-m1-ask-2013.csv",
        output_csv="cpu_gbm_ask_benchmark.csv",
    )