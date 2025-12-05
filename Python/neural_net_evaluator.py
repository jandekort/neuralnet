"""
Neural Network Evaluator for Black-Scholes Implied Volatility

This module loads a trained neural network from C++ output and performs
feedforward evaluation to predict implied volatility from option parameters.

Usage:
1. Train the network using MainTrain (C++)
2. Copy the normalization parameters from the training output
3. Update the hardcoded values in main() or pass them to __init__
4. Run this script to evaluate the network on new data

Required file:
- best_net.txt: Network weights and biases (created by MainTrain)
"""

import numpy as np
from typing import Tuple, Optional
from scipy.stats import norm


class NeuralNetworkEvaluator:
    """Evaluates a trained neural network for implied volatility prediction."""

    def __init__(self,
                 input_mean=None,
                 input_std=None,
                 output_mean=None,
                 output_std=None):
        self.weights = []
        self.biases = []
        self.layer_sizes = []

        # Normalization parameters (hardcoded or provided)
        # These values are printed by MainTrain during the normalizer.fit() step
        # Copy them from the training output and paste them here
        self.input_mean = input_mean
        self.input_std = input_std
        self.output_mean = output_mean
        self.output_std = output_std
        self.normalized = (input_mean is not None and input_std is not None and
                          output_mean is not None and output_std is not None)

    def load_network(self, network_path: str = "../best_net.txt"):
        """
        Load network architecture and parameters from file.

        File format:
        - Line 1: number of layers
        - Line 2: layer sizes (space-separated)
        - For each weight matrix: rows cols, then matrix values
        - For each bias vector: size, then bias values
        """
        with open(network_path, 'r') as f:
            # Read number of layers
            n_layers = int(f.readline().strip())

            # Read layer sizes
            self.layer_sizes = list(map(int, f.readline().strip().split()))

            if len(self.layer_sizes) != n_layers:
                raise ValueError(f"Expected {n_layers} layer sizes, got {len(self.layer_sizes)}")

            # Read weights for each layer (skip input layer)
            self.weights = []
            for i in range(1, n_layers):
                line = f.readline().strip().split()
                rows, cols = int(line[0]), int(line[1])

                # Read weight matrix
                W = np.zeros((rows, cols), dtype=np.float32)
                for r in range(rows):
                    W[r, :] = list(map(float, f.readline().strip().split()))

                self.weights.append(W)

            # Read biases for each layer (skip input layer)
            self.biases = []
            for i in range(1, n_layers):
                size = int(f.readline().strip())
                b = np.array(list(map(float, f.readline().strip().split())), dtype=np.float32)

                if len(b) != size:
                    raise ValueError(f"Expected {size} bias values, got {len(b)}")

                self.biases.append(b)

        print(f"Loaded network with architecture: {self.layer_sizes}")
        print(f"Number of weight matrices: {len(self.weights)}")
        print(f"Number of bias vectors: {len(self.biases)}")

    def set_normalization_params(self,
                                  input_mean: np.ndarray,
                                  input_std: np.ndarray,
                                  output_mean: np.ndarray,
                                  output_std: np.ndarray):
        """
        Set normalization parameters (alternative to passing them in __init__).

        Args:
            input_mean: Mean of input features (4-element array)
            input_std: Std of input features (4-element array)
            output_mean: Mean of output (1-element array)
            output_std: Std of output (1-element array)
        """
        self.input_mean = np.array(input_mean, dtype=np.float32)
        self.input_std = np.array(input_std, dtype=np.float32)
        self.output_mean = np.array(output_mean, dtype=np.float32)
        self.output_std = np.array(output_std, dtype=np.float32)
        self.normalized = True

        print(f"Set normalization parameters:")
        print(f"  Input mean: {self.input_mean}")
        print(f"  Input std:  {self.input_std}")
        print(f"  Output mean: {self.output_mean}")
        print(f"  Output std:  {self.output_std}")

    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, z)

    @staticmethod
    def linear(z: np.ndarray) -> np.ndarray:
        """Linear activation function (identity)."""
        return z

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        """
        Perform feedforward pass through the network.

        Args:
            x: Input vector (already transformed and normalized)

        Returns:
            Output vector (normalized)
        """
        a = x.astype(np.float32)

        # Forward pass through all layers
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = W @ a + b

            # Apply activation function
            if i == len(self.weights) - 1:
                # Output layer: linear activation
                a = self.linear(z)
            else:
                # Hidden layers: ReLU activation
                a = self.relu(z)

        return a

    def normalize_input(self, x: np.ndarray) -> np.ndarray:
        """Normalize input using stored mean and std."""
        if not self.normalized:
            raise ValueError("Normalization parameters not loaded. Call load_normalization_params() first.")
        return (x - self.input_mean) / self.input_std

    def denormalize_output(self, y_norm: np.ndarray) -> np.ndarray:
        """Denormalize output using stored mean and std."""
        if not self.normalized:
            raise ValueError("Normalization parameters not loaded. Call load_normalization_params() first.")
        return y_norm * self.output_std + self.output_mean

    @staticmethod
    def black_call(F: float, K: float, r: float, tau: float, sigma: float) -> float:
        """
        Calculate Black-Scholes call option price.

        Args:
            F: Forward price
            K: Strike price
            r: Interest rate
            tau: Time to maturity (years)
            sigma: Volatility

        Returns:
            Call option price
        """
        if sigma <= 0 or tau <= 0:
            return np.exp(-r * tau) * max(F - K, 0.0)

        st = sigma * np.sqrt(tau)
        d1 = (np.log(F / K) + 0.5 * sigma**2 * tau) / st
        d2 = d1 - st

        return np.exp(-r * tau) * (F * norm.cdf(d1) - K * norm.cdf(d2))

    @staticmethod
    def black_vega(F: float, K: float, r: float, tau: float, sigma: float) -> float:
        """
        Calculate vega (derivative of call price with respect to sigma).

        Args:
            F: Forward price
            K: Strike price
            r: Interest rate
            tau: Time to maturity (years)
            sigma: Volatility

        Returns:
            Vega (∂C/∂σ)
        """
        if sigma <= 0 or tau <= 0:
            return 0.0

        st = sigma * np.sqrt(tau)
        d1 = (np.log(F / K) + 0.5 * sigma**2 * tau) / st

        # Vega = F * exp(-r*tau) * phi(d1) * sqrt(tau)
        # where phi is the standard normal PDF
        vega = F * np.exp(-r * tau) * norm.pdf(d1) * np.sqrt(tau)
        return vega

    def implied_volatility_newton(self, F: float, K: float, r: float, tau: float,
                                   V_market: float, initial_guess: float = 0.3,
                                   max_iterations: int = 100, tolerance: float = 1e-8) -> float:
        """
        Calculate implied volatility using Newton-Raphson method.

        This is the classical numerical approach to solve for sigma such that:
        Black-Scholes(F, K, r, tau, sigma) = V_market

        Args:
            F: Forward price
            K: Strike price
            r: Interest rate
            tau: Time to maturity (years)
            V_market: Observed market price
            initial_guess: Starting point for sigma
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance

        Returns:
            Implied volatility (sigma)
        """
        sigma = initial_guess

        for i in range(max_iterations):
            # Calculate price and vega at current sigma
            price = self.black_call(F, K, r, tau, sigma)
            vega = self.black_vega(F, K, r, tau, sigma)

            # Price difference
            diff = price - V_market

            # Check convergence
            if abs(diff) < tolerance:
                return sigma

            # Check for zero vega (shouldn't happen with valid inputs)
            if abs(vega) < 1e-10:
                # Fall back to bisection for this step
                if diff > 0:
                    sigma *= 0.9  # Price too high, reduce sigma
                else:
                    sigma *= 1.1  # Price too low, increase sigma
            else:
                # Newton-Raphson update: sigma_new = sigma_old - f(sigma)/f'(sigma)
                sigma -= diff / vega

            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 0.001, 5.0)

        # If we didn't converge, return the best estimate
        return sigma

    def implied_volatility_batch(self, F: np.ndarray, K: np.ndarray, r: np.ndarray,
                                 tau: np.ndarray, V_market: np.ndarray) -> np.ndarray:
        """
        Calculate implied volatility for a batch of options using Newton-Raphson.

        Args:
            F: Array of forward prices
            K: Array of strike prices
            r: Array of interest rates
            tau: Array of times to maturity
            V_market: Array of observed market prices

        Returns:
            Array of implied volatilities
        """
        n = len(F)
        sigmas = np.zeros(n)

        for i in range(n):
            try:
                sigmas[i] = self.implied_volatility_newton(
                    F[i], K[i], r[i], tau[i], V_market[i]
                )
            except (ValueError, RuntimeError) as e:
                print(f"Warning: Could not solve for IV at index {i}: {e}")
                sigmas[i] = np.nan

        return sigmas

    def transform_inputs(self, F: float, K: float, r: float, tau: float, V: float) -> np.ndarray:
        """
        Transform original option parameters to network input format.

        The network expects inputs in the form:
        x[0] = log(Vtilde / K) where Vtilde = V - intrinsic_value
        x[1] = F / K (moneyness)
        x[2] = tau (time to maturity)
        x[3] = r (interest rate)

        Args:
            F: Forward price
            K: Strike price
            r: Interest rate
            tau: Time to maturity (years)
            V: Observed option price

        Returns:
            Transformed input vector
        """
        # Calculate intrinsic value (discounted)
        intrinsic_disc = np.exp(-r * tau) * max(F - K, 0.0)

        # Calculate time value
        Vtilde = V - intrinsic_disc

        if Vtilde <= 1e-7:
            raise ValueError(f"Time value Vtilde={Vtilde:.2e} is too small. Option may be deep OTM or invalid.")

        # Create transformed input vector
        x = np.array([
            np.log(Vtilde / K),  # log time value normalized by strike
            F / K,               # moneyness
            tau,                 # time to maturity
            r                    # interest rate
        ], dtype=np.float32)

        return x

    def predict_sigma(self, F: float, K: float, r: float, tau: float, V: float) -> float:
        """
        Predict implied volatility from option parameters.

        This is the main entry point that handles the complete pipeline:
        1. Transform original inputs to feature space
        2. Normalize inputs
        3. Feedforward through network
        4. Denormalize output

        Args:
            F: Forward price
            K: Strike price
            r: Interest rate
            tau: Time to maturity (years)
            V: Observed option price

        Returns:
            Predicted implied volatility (sigma)
        """
        # Step 1: Transform inputs
        x_transformed = self.transform_inputs(F, K, r, tau, V)

        # Step 2: Normalize
        x_normalized = self.normalize_input(x_transformed)

        # Step 3: Feedforward
        y_normalized = self.feedforward(x_normalized)

        # Step 4: Denormalize
        sigma = self.denormalize_output(y_normalized)

        return sigma[0]

    def predict_sigma_batch(self, F: np.ndarray, K: np.ndarray, r: np.ndarray,
                           tau: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Predict implied volatility for a batch of options.

        Args:
            F: Array of forward prices
            K: Array of strike prices
            r: Array of interest rates
            tau: Array of times to maturity
            V: Array of observed option prices

        Returns:
            Array of predicted implied volatilities
        """
        n = len(F)
        sigmas = np.zeros(n)

        for i in range(n):
            try:
                sigmas[i] = self.predict_sigma(F[i], K[i], r[i], tau[i], V[i])
            except ValueError as e:
                print(f"Warning: Could not predict sigma for example {i}: {e}")
                sigmas[i] = np.nan

        return sigmas


def main():
    """Example usage of the neural network evaluator."""

    # STEP 1: Copy normalization parameters from MainTrain output
    # When you run MainTrain, it prints these values during training.
    # Look for lines like:
    #   Input mean:  -1.234 0.987 0.456 0.032
    #   Input std:   0.567 0.123 0.234 0.012
    #   Output mean: 0.345
    #   Output std:  0.156
    #
    # Replace the example values below with your actual values:

    input_mean = np.array([-3.85356, 0.959457, 0.534587, 0.0499923], dtype=np.float32)
    input_std = np.array([2.65276, 0.25382, 0.271758, 0.0288707], dtype=np.float32)
    output_mean = np.array([0.547926], dtype=np.float32)
    output_std = np.array([0.263503], dtype=np.float32)

    # STEP 2: Initialize evaluator with normalization parameters
    evaluator = NeuralNetworkEvaluator(
        input_mean=input_mean,
        input_std=input_std,
        output_mean=output_mean,
        output_std=output_std
    )

    # STEP 3: Load the trained network
    try:
        evaluator.load_network("../best_net.txt")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure best_net.txt exists in the parent directory (created by running MainTrain)")
        return

    if not evaluator.normalized:
        print("WARNING: Normalization parameters not set!")
        print("Please update the hardcoded values in main() with your actual values from training.")
        return

    # Example: Predict implied volatility
    print("\n" + "="*60)
    print("Example Prediction")
    print("="*60)

    # Sample parameters (ATM option)
    F = 100.0      # Forward price
    K = 100.0      # Strike price
    r = 0.05       # 5% interest rate
    tau = 0.5      # 6 months to maturity

    # First, generate an option price using a known sigma
    sigma_true = 0.25  # 25% volatility
    V = evaluator.black_call(F, K, r, tau, sigma_true)
    print(f"\nInput parameters:")
    print(f"  F (Forward):     {F:.2f}")
    print(f"  K (Strike):      {K:.2f}")
    print(f"  r (Rate):        {r:.4f}")
    print(f"  tau (Maturity):  {tau:.4f}")
    print(f"  V (Option price): {V:.6f}")
    print(f"  True sigma:      {sigma_true:.4f}")

    # Predict sigma
    sigma_pred = evaluator.predict_sigma(F, K, r, tau, V)

    # Compute price from predicted sigma
    V_pred = evaluator.black_call(F, K, r, tau, sigma_pred)

    print(f"\nPredicted sigma:   {sigma_pred:.4f}")
    print(f"Sigma error:       {abs(sigma_pred - sigma_true):.6f}")

    print(f"\nPrice validation:")
    print(f"  Original price (V):     {V:.6f}")
    print(f"  Price from pred σ:      {V_pred:.6f}")
    print(f"  Absolute price error:   {abs(V_pred - V):.6f}")
    print(f"  Relative price error:   {abs(V_pred - V) / V * 100:.4f}%")

    # Example: Batch prediction
    print("\n" + "="*60)
    print("Batch Prediction Example")
    print("="*60)

    # Generate test data
    n_test = 10
    np.random.seed(42)

    F_test = np.random.uniform(80, 120, n_test)
    K_test = np.full(n_test, 100.0)
    r_test = np.random.uniform(0.01, 0.08, n_test)
    tau_test = np.random.uniform(0.1, 1.0, n_test)
    sigma_true_test = np.random.uniform(0.1, 0.6, n_test)

    # Calculate option prices
    V_test = np.array([
        evaluator.black_call(F_test[i], K_test[i], r_test[i], tau_test[i], sigma_true_test[i])
        for i in range(n_test)
    ])

    # Predict sigmas
    sigma_pred_test = evaluator.predict_sigma_batch(F_test, K_test, r_test, tau_test, V_test)

    # Compute prices from predicted sigmas
    V_pred_test = np.array([
        evaluator.black_call(F_test[i], K_test[i], r_test[i], tau_test[i], sigma_pred_test[i])
        if not np.isnan(sigma_pred_test[i]) else np.nan
        for i in range(n_test)
    ])

    print(f"\n{'i':<3} {'F/K':<8} {'tau':<8} {'σ_true':<10} {'σ_pred':<10} {'σ_err':<10} {'V_err':<12} {'V_err %':<10}")
    print("-" * 90)
    for i in range(n_test):
        if not np.isnan(sigma_pred_test[i]):
            sigma_error = abs(sigma_pred_test[i] - sigma_true_test[i])
            price_error = abs(V_pred_test[i] - V_test[i])
            rel_price_error = price_error / V_test[i] * 100
            print(f"{i:<3} {F_test[i]/K_test[i]:<8.3f} {tau_test[i]:<8.3f} "
                  f"{sigma_true_test[i]:<10.4f} {sigma_pred_test[i]:<10.4f} {sigma_error:<10.6f} "
                  f"{price_error:<12.6f} {rel_price_error:<10.4f}")

    # Statistics
    valid_idx = ~np.isnan(sigma_pred_test)
    if np.any(valid_idx):
        sigma_errors = np.abs(sigma_pred_test[valid_idx] - sigma_true_test[valid_idx])
        price_errors = np.abs(V_pred_test[valid_idx] - V_test[valid_idx])
        rel_price_errors = price_errors / V_test[valid_idx] * 100

        print(f"\nSigma Error Statistics:")
        print(f"  Mean Absolute Error: {np.mean(sigma_errors):.6f}")
        print(f"  Max Absolute Error:  {np.max(sigma_errors):.6f}")

        print(f"\nPrice Validation Statistics:")
        print(f"  Mean absolute price error:    {np.mean(price_errors):.6f}")
        print(f"  Max absolute price error:     {np.max(price_errors):.6f}")
        print(f"  Mean relative price error:    {np.mean(rel_price_errors):.4f}%")
        print(f"  Max relative price error:     {np.max(rel_price_errors):.4f}%")


if __name__ == "__main__":
    main()
