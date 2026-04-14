import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Script simulates price paths of USDJPY ask prices with a Geometric Brownian Motion equation
parameters are estimated from historical log returns of historical ask price data (usdjpy-m1-ask-2013.csv)

"""


# Load data
df = pd.read_csv("usdjpy-m1-ask-2013.csv", parse_dates=["timestamp"]) # one month of data 

# print(df["timestamp"].head())

prices = df["close"].dropna()

# estimate volatility and drift parameters from log returns 
log_returns = np.log(prices / prices.shift(1)).dropna()

volatility = log_returns.std()
drift = log_returns.mean()
dt = 1 # 1 time step, data is one min apart


S0 = prices.iloc[-1] # last known price
n_timeintervals = 200 
N_sim = 10000
# np.random.seed(42) # uncomment to reproduce same paths

Z = np.random.standard_normal((n_timeintervals, N_sim))
step = np.exp((drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * Z)

sim = np.zeros((n_timeintervals + 1, N_sim))
sim[0] = S0
for t in range(1, n_timeintervals + 1):
    sim[t] = sim[t - 1] * step[t - 1]

fig, axes = plt.subplots(2, 1, figsize=(12, 8), squeeze=False)

# Historical prices
axes[0][0].plot(prices.values, color="steelblue", linewidth=1)
axes[0][0].set_title("Historical Close Prices")
axes[0][0].set_xlabel("Time step (minutes)")
axes[0][0].set_ylabel("Price")
axes[0][0].grid(alpha=0.3)

# Simulated paths
axes[1][0].plot(sim, alpha=0.3, linewidth=0.8)
axes[1][0].axhline(S0, color="black", linestyle="--", linewidth=1, label=f"S₀ = {S0:.3f}")
axes[1][0].set_title(f"GBM Simulation — {N_sim} paths, {n_timeintervals} steps")
axes[1][0].set_xlabel("Steps ahead")
axes[1][0].set_ylabel("Price")
axes[1][0].legend()
axes[1][0].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("gbm_ask_sim.png", dpi=150)
plt.show()

# ── 5. Summary stats at final step ──────────────────────────────────────────
final = sim[-1]
print(f"\nFinal-step statistics across {N_sim} paths:")
print(f"  Mean:   {final.mean():.4f}")
print(f"  Median: {np.median(final):.4f}")
print(f"  5th pct:{np.percentile(final, 5):.4f}")
print(f"  95th pct:{np.percentile(final, 95):.4f}")