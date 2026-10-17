# NeuralNet: a neural network from scratch in C++, applied to implied volatility

This project is a neural network written from scratch in modern C++ (C++20, using the Eigen
linear algebra library). No machine-learning framework such as PyTorch or TensorFlow is used:
every part of the learning process is implemented by hand. The network is then trained to solve
a classic problem in quantitative finance: finding the **implied volatility** of an option.

Python tools are included for quick investigation of the results produced by the C++ code, such
as checking accuracy and plotting volatility smiles.

## What is a neural network, in short?

A neural network is a flexible function that learns from examples. You show it many inputs
together with the correct answers, and it gradually adjusts its internal parameters (its
"weights") until its own answers get close to the correct ones. After training, it can give
answers for inputs it has never seen before, and it does so very quickly.

This project implements all of the building blocks of that process:

- **Layers of neurons**: the network used here has 4 inputs, four hidden layers of 400 neurons
  each, and 1 output.
- **Activation functions** (ReLU, Sigmoid, Linear) that let the network represent non-linear
  relationships.
- **Backpropagation**: the algorithm that works out how each weight should change to reduce
  the error.
- **Optimisers**: stochastic gradient descent and Adam, with learning-rate decay.
- **Data normalisation**, so that inputs on very different scales are treated fairly.
- **Parallel training** with OpenMP, to use all CPU cores.
- **Unit tests** with GoogleTest.

## The application: implied volatility

An option is a contract that gives the right to buy (or sell) an asset at a fixed price in the
future. Its market price depends on a handful of quantities: the current price of the asset,
the agreed strike price, the time to expiry, the interest rate, and the **volatility**, which
measures how much the asset price is expected to move.

All of these are directly observable except volatility. The Black-Scholes (Black-76) formula
turns a volatility into an option price. In practice the question is the reverse one: *given
the option price we see in the market, what volatility does it imply?* That number is the
implied volatility. It is the standard language in which options are quoted and compared, and
it is computed millions of times a day across the industry.

There is no closed-form formula for this inverse, so it has to be solved numerically, usually
with an iterative root-finder such as Newton's method. The model is well understood and
the answer is known exactly, yet computing it still requires real numerical work. That makes it
an ideal test case for a neural network: if the network can learn this inverse accurately, the
same approach can be used for harder problems, such as calibrating more complex pricing models,
where speed matters much more.

### How the network is trained

1. **Generate data** (`MainGenerate`): random option scenarios are drawn (moneyness 0.5-1.4,
   maturity 0.05-1 year, interest rate 0-10%, volatility 5-100%). For each, the exact option
   price is computed with the Black formula. This gives 1.5 million training examples and
   50,000 test examples where the correct volatility is known.
2. **Train** (`MainTrain`): the network learns to map the option price and its parameters back
   to the volatility. Following Liu, Oosterlee and Bothe (2019), the price is first transformed
   (the intrinsic value is removed and the logarithm is taken) which makes the problem much
   easier to learn.
3. **Evaluate** (Python): the trained network is loaded in Python and compared with a classic
   Newton root-finder, for example by plotting volatility smiles side by side.

On the test set the network reaches a mean absolute error of about 0.0094 in volatility, i.e.
just under one volatility point.

## Project structure

| Part | Files |
|------|-------|
| Neural network core | `Network.*`, `Activation.h`, `ReLU.h`, `Sigmoid.*`, `Linear.h`, `Cost.*` |
| Data handling | `DataGenerator.*`, `DataNormalizer.*`, `VectorXdPairReader.*`, `BinaryFileWriter.h`, `MnistReader.*` |
| Programs | `MainGenerate.cpp`, `MainTrain.cpp`, `MainMnist.cpp` |
| Tests | `TestSuite*.cpp` |
| Python analysis | `Python/neural_net_evaluator.py`, `Python/example_usage.py` |

`MainMnist` trains the same network on the MNIST handwritten-digit dataset. This classic
image-recognition benchmark was used to verify that the network implementation works correctly
before applying it to finance.

## Building and running

Requirements: CMake 3.29+, a C++20 compiler, Eigen 3 and (optionally) OpenMP.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/MainGenerate   # creates GeneratedData.bin
./build/MainTrain      # trains the network, saves best_net.txt
```

Then use the scripts in `Python/` to evaluate the trained network.

## References

- S. Liu, C. W. Oosterlee, S. M. Bohte, *Pricing options and computing implied volatilities
  using neural networks*, Risks 7(1), 2019.

## Dataset

Dataset: MNIST handwritten digits
Source: http://yann.lecun.com/exdb/mnist/
Created by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges.
