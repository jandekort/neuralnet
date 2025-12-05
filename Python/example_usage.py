"""
Simple example showing how to use the neural network evaluator.

Before running this:
1. Run MainTrain to train the network and create best_net.txt
2. Copy the normalization parameters from the MainTrain output
3. Update the values below
"""

import numpy as np
import matplotlib.pyplot as plt
from neural_net_evaluator import NeuralNetworkEvaluator

# =============================================================================
# STEP 1: Set normalization parameters from MainTrain output
# =============================================================================
# When MainTrain runs, it prints these values. Copy them here:

INPUT_MEAN = np.array([-3.85356, 0.959457, 0.534587, 0.0499923], dtype=np.float32)
INPUT_STD = np.array([2.65276, 0.25382, 0.271758, 0.0288707], dtype=np.float32)
OUTPUT_MEAN = np.array([0.547926], dtype=np.float32)
OUTPUT_STD = np.array([0.263503], dtype=np.float32)

# =============================================================================
# STEP 2: Load the trained network
# =============================================================================

evaluator = NeuralNetworkEvaluator(
    input_mean=INPUT_MEAN,
    input_std=INPUT_STD,
    output_mean=OUTPUT_MEAN,
    output_std=OUTPUT_STD
)

evaluator.load_network("../cmake-build-release/best_net.txt")

# =============================================================================
# STEP 3: Predict implied volatility for an option
# =============================================================================

# Example: ATM European call option
F = 100.0      # Forward price
K = 100.0      # Strike price
r = 0.05       # Interest rate (5%)
tau = 0.5      # Time to maturity (6 months)
V = 5.637      # Observed market price of the option

# Predict the implied volatility
sigma_pred = evaluator.predict_sigma(F, K, r, tau, V)

# Compute the option price using the predicted sigma
V_pred = evaluator.black_call(F, K, r, tau, sigma_pred)

# Calculate pricing error
price_error = abs(V_pred - V)
relative_price_error = price_error / V * 100

print(f"Option parameters:")
print(f"  Forward price (F):   {F:.2f}")
print(f"  Strike price (K):    {K:.2f}")
print(f"  Interest rate (r):   {r:.4f}")
print(f"  Time to maturity:    {tau:.4f} years")
print(f"  Market price (V):    {V:.4f}")
print(f"\nPredicted implied volatility: {sigma_pred:.4f}")
print(f"\nPrice validation:")
print(f"  Price from predicted σ: {V_pred:.6f}")
print(f"  Original market price:  {V:.6f}")
print(f"  Absolute price error:   {price_error:.6f}")
print(f"  Relative price error:   {relative_price_error:.4f}%")

# =============================================================================
# STEP 4: Batch prediction for multiple options
# =============================================================================

print("\n" + "="*70)
print("Batch prediction example:")
print("="*70)

# Multiple options with different strikes (volatility smile)
# Extended range from deep ITM to deep OTM
strikes = np.array([70.0, 80.0, 90.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 110.0, 120.0, 130.0])
forwards = np.full(len(strikes), 100.0)
rates = np.full(len(strikes), 0.05)
maturities = np.full(len(strikes), 0.5)

# Generate realistic prices using a true sigma for each strike
# Using a volatility smile: higher vol for OTM options (typical equity smile)
true_sigmas = np.array([0.28, 0.25, 0.23, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.23, 0.23, 0.23, 0.24, 0.27, 0.30])
prices = np.array([
    evaluator.black_call(forwards[i], strikes[i], rates[i], maturities[i], true_sigmas[i])
    for i in range(len(strikes))
])

print("\nComputing implied volatilities using:")
print("  1. Neural Network (fast)")
print("  2. Newton-Raphson solver (classical method)")
print()

# Method 1: Neural Network prediction
import time
start_nn = time.time()
sigmas_nn = evaluator.predict_sigma_batch(forwards, strikes, rates, maturities, prices)
time_nn = time.time() - start_nn

# Method 2: Numerical solver (Newton-Raphson)
start_solver = time.time()
sigmas_solver = evaluator.implied_volatility_batch(forwards, strikes, rates, maturities, prices)
time_solver = time.time() - start_solver

print(f"Neural Network time: {time_nn*1000:.2f} ms")
print(f"Numerical solver time: {time_solver*1000:.2f} ms")
print(f"Speedup: {time_solver/time_nn:.1f}x")

# Compute prices from predicted sigmas (NN)
prices_pred_nn = np.array([
    evaluator.black_call(forwards[i], strikes[i], rates[i], maturities[i], sigmas_nn[i])
    if not np.isnan(sigmas_nn[i]) else np.nan
    for i in range(len(strikes))
])

# Compute prices from solved sigmas (Newton-Raphson)
prices_pred_solver = np.array([
    evaluator.black_call(forwards[i], strikes[i], rates[i], maturities[i], sigmas_solver[i])
    if not np.isnan(sigmas_solver[i]) else np.nan
    for i in range(len(strikes))
])

# Calculate errors for both methods
sigma_errors_nn = np.abs(sigmas_nn - true_sigmas)
sigma_errors_solver = np.abs(sigmas_solver - true_sigmas)
price_errors_nn = np.abs(prices_pred_nn - prices)
price_errors_solver = np.abs(prices_pred_solver - prices)

print("\nLegend:")
print("  K = Strike price")
print("  M = Moneyness (F/K)")
print("  σ_true = True volatility used to generate price")
print("  V_mkt = Market price")
print("  σ_NN = Neural network prediction")
print("  σ_Solv = Newton-Raphson solution (should equal σ_true within ~1e-10)")
print("  σ_err_NN = |σ_NN - σ_true|")
print("  σ_err_Solv = |σ_Solv - σ_true| (should be ~0, proving solver correctness)")
print("  V_err = Absolute error when recomputing price with predicted σ")
print()

print(f"{'K':<6} {'M':<6} {'σ_true':<10} {'V_mkt':<10} {'σ_NN':<10} {'σ_Solv':<10} {'σ_err_NN':<12} {'σ_err_Solv':<14} {'V_err_NN':<12} {'V_err_Solv':<14}")
print("-" * 120)
for i in range(len(strikes)):
    moneyness = forwards[i] / strikes[i]
    print(f"{strikes[i]:<6.0f} {moneyness:<6.3f} "
          f"{true_sigmas[i]:<10.6f} {prices[i]:<10.6f} "
          f"{sigmas_nn[i]:<10.6f} {sigmas_solver[i]:<10.6f} "
          f"{sigma_errors_nn[i]:<12.6f} {sigma_errors_solver[i]:<14.6e} "
          f"{price_errors_nn[i]:<12.8f} {price_errors_solver[i]:<14.6e}")

# Summary statistics
valid_idx_nn = ~np.isnan(sigma_errors_nn)
valid_idx_solver = ~np.isnan(sigma_errors_solver)

print("\n" + "="*110)
print("COMPARISON SUMMARY: Neural Network vs Newton-Raphson Solver")
print("="*110)

if np.any(valid_idx_nn) and np.any(valid_idx_solver):
    print(f"\n{'Method':<20} {'Mean σ err':<18} {'Max σ err':<18} {'Mean V err':<18} {'Max V err':<18}")
    print("-" * 90)
    print(f"{'Neural Network':<20} {np.mean(sigma_errors_nn[valid_idx_nn]):<18.8f} "
          f"{np.max(sigma_errors_nn[valid_idx_nn]):<18.8f} "
          f"{np.mean(price_errors_nn[valid_idx_nn]):<18.10e} "
          f"{np.max(price_errors_nn[valid_idx_nn]):<18.10e}")
    print(f"{'Newton-Raphson':<20} {np.mean(sigma_errors_solver[valid_idx_solver]):<18.12e} "
          f"{np.max(sigma_errors_solver[valid_idx_solver]):<18.12e} "
          f"{np.mean(price_errors_solver[valid_idx_solver]):<18.12e} "
          f"{np.max(price_errors_solver[valid_idx_solver]):<18.12e}")

    # Relative performance
    sigma_ratio = np.mean(sigma_errors_nn[valid_idx_nn]) / np.mean(sigma_errors_solver[valid_idx_solver])
    price_ratio = np.mean(price_errors_nn[valid_idx_nn]) / np.mean(price_errors_solver[valid_idx_solver])

    print(f"\nRelative Performance (NN / Solver):")
    print(f"  Sigma error ratio:  {sigma_ratio:.2f}x")
    print(f"  Price error ratio:  {price_ratio:.2f}x")
    print(f"  Speed advantage:    {time_solver/time_nn:.1f}x faster")

    if sigma_ratio < 1.5 and price_ratio < 2.0:
        print(f"\n✓ Neural network achieves comparable accuracy to numerical solver!")
    elif sigma_ratio < 3.0:
        print(f"\n⚠ Neural network is less accurate but much faster.")
    else:
        print(f"\n✗ Neural network needs more training to match solver accuracy.")

# =============================================================================
# STEP 5: Visualization - Plot predictions vs actual values
# =============================================================================

print("\n" + "="*70)
print("Generating volatility smile plot...")
print("="*70)

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Compute moneyness for x-axis
moneyness = forwards / strikes

# Plot 1: Sigma vs Strike
ax1.plot(strikes, true_sigmas, 'o', markersize=8, label='True σ (used to generate prices)',
         color='black', markerfacecolor='none', markeredgewidth=2)
ax1.plot(strikes, sigmas_solver, 'o', markersize=6, label='Solver σ (Newton-Raphson)',
         color='green', alpha=0.7)
ax1.plot(strikes, sigmas_nn, 'x', markersize=8, label='NN σ (Neural Network)',
         color='red', markeredgewidth=2)

ax1.set_xlabel('Strike Price (K)', fontsize=12)
ax1.set_ylabel('Implied Volatility (σ)', fontsize=12)
ax1.set_title('Implied Volatility vs Strike\n(Volatility Smile)', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='best', fontsize=10)
ax1.axvline(x=100, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='ATM (F=100)')

# Plot 2: Sigma vs Moneyness
ax2.plot(moneyness, true_sigmas, 'o', markersize=8, label='True σ (used to generate prices)',
         color='black', markerfacecolor='none', markeredgewidth=2)
ax2.plot(moneyness, sigmas_solver, 'o', markersize=6, label='Solver σ (Newton-Raphson)',
         color='green', alpha=0.7)
ax2.plot(moneyness, sigmas_nn, 'x', markersize=8, label='NN σ (Neural Network)',
         color='red', markeredgewidth=2)

ax2.set_xlabel('Moneyness (F/K)', fontsize=12)
ax2.set_ylabel('Implied Volatility (σ)', fontsize=12)
ax2.set_title('Implied Volatility vs Moneyness\n(Volatility Smile)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='best', fontsize=10)
ax2.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='ATM (M=1)')

# Invert x-axis for moneyness plot (standard convention: ITM on left, OTM on right for calls)
ax2.invert_xaxis()

plt.tight_layout()
plt.savefig('volatility_smile_comparison.png', dpi=150, bbox_inches='tight')
print("Plot saved as 'volatility_smile_comparison.png'")
plt.show()

print("\nPlot interpretation:")
print("  • Black circles (o): True volatilities used to generate the option prices")
print("  • Green circles (o): Newton-Raphson solver results (should overlap with black circles)")
print("  • Red crosses (x): Neural network predictions")
print("  • The closer the red crosses are to the black/green circles, the better the NN performs")
