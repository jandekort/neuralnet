#include <gtest/gtest.h>

#include <cstdio>     // for std::remove
#include <string>

#include "BinaryFileWriter.h"
#include "VectorXdPairReader.h"

using Eigen::VectorXd;
using Example = std::pair<VectorXd, VectorXd>;


TEST(MnistTest, FeedforwardMatchesManualComputation)
{
    auto act = std::make_shared<Sigmoid>();

    // 2-2-1 network: 2 inputs -> 2 hidden -> 1 output
    Network net({2, 2, 1}, act, /*use_scaled_weight_init=*/false);

    // Manually set weights and biases to deterministic values.
    auto& biases  = net.biases();
    auto& weights = net.weights();

    biases.clear();
    weights.clear();

    biases.push_back(Eigen::VectorXd(2));
    biases[0] << 0.1, -0.2;           // hidden layer biases

    biases.push_back(Eigen::VectorXd(1));
    biases[1] << 0.05;                // output layer bias

    weights.push_back(Eigen::MatrixXd(2, 2));
    weights[0] << 0.5, -0.3, 0.8,  0.2;          // input -> hidden

    weights.push_back(Eigen::MatrixXd(1, 2));
    weights[1] << -0.7, 0.9;          // hidden -> output

    // Input vector
    Eigen::VectorXd x(2);
    x << 0.4, 0.6;

    // Manual computation: a1 = σ(W1 x + b1), a2 = σ(W2 a1 + b2)
    Eigen::VectorXd z1 = weights[0] * x + biases[0];
    Eigen::VectorXd a1 = act->f(z1);
    Eigen::VectorXd z2 = weights[1] * a1 + biases[1];
    Eigen::VectorXd a2 = act->f(z2);

    Eigen::VectorXd out = net.feedforward(x);

    ASSERT_EQ(out.size(), 1);
    EXPECT_NEAR(out(0), a2(0), 1e-10);
}


TEST(MnistTest, BackpropGradientMatchesFiniteDifference)
{
    auto act = std::make_shared<Sigmoid>();

    Network net({2, 2, 1}, act, /*use_scaled_weight_init=*/false);

    auto& biases  = net.biases();
    auto& weights = net.weights();

    // Deterministic parameters.
    biases.clear();
    weights.clear();

    biases.push_back(Eigen::VectorXd(2));
    biases[0] << 0.1, -0.2;

    biases.push_back(Eigen::VectorXd(1));
    biases[1] << 0.05;

    weights.push_back(Eigen::MatrixXd(2, 2));
    weights[0] << 0.5, -0.3,
            0.8,  0.2;

    weights.push_back(Eigen::MatrixXd(1, 2));
    weights[1] << -0.7, 0.9;

    // A single training example (x, y) — like a tiny "MNIST" with 2 pixels and 1 output.
    Eigen::VectorXd x(2);
    x << 0.4, 0.6;

    Eigen::VectorXd y(1);
    y << 1.0;  // target "label"

    // Analytical gradient via backprop.
    std::vector<Eigen::VectorXd> nabla_b;
    std::vector<Eigen::MatrixXd> nabla_w;
    std::tie(nabla_b, nabla_w) = net.backprop(x, y);

    // We'll check the gradient w.r.t. a single weight, e.g. weights[1](0,0).
    const int layer = 1;    // hidden -> output layer
    const int row   = 0;
    const int col   = 0;

    double base_weight = weights[layer](row, col);
    const double eps = 1e-5;

    auto cost = [&](const Network& net)
    {
        Eigen::VectorXd a = net.feedforward(x);
        return 0.5 * (a - y).squaredNorm();
    };

    // Numerical gradient using central difference:
    // (C(w+eps) - C(w-eps)) / (2*eps)
    weights[layer](row, col) = base_weight + eps;
    double cost_up = cost(net);

    weights[layer](row, col) = base_weight - eps;
    double cost_down = cost(net);

    // Restore parameter
    weights[layer](row, col) = base_weight;

    double numerical_grad = (cost_up - cost_down) / (2.0 * eps);
    double backprop_grad  = nabla_w[layer](row, col);

    // Reasonable tolerance – backprop and finite differences should be very close.
    EXPECT_NEAR(backprop_grad, numerical_grad, 1e-4);
}