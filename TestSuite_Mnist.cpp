#include <gtest/gtest.h>

#include <cstdio>     // for std::remove
#include <string>

#include "BinaryFileWriter.h"
#include "VectorXdPairReader.h"

using Eigen::VectorXf;
using Eigen::MatrixXf;
using Example = std::pair<VectorXf, VectorXf>;


TEST(MnistTest, FeedforwardMatchesManualComputation)
{
    auto act = std::make_shared<Sigmoid>();

    // 2-2-1 network: 2 inputs -> 2 hidden -> 1 output
    Network net({2, 2, 1}, act);

    // Manually set weights and biases to deterministic values.
    auto& biases  = net.biases();
    auto& weights = net.weights();

    biases.clear();
    weights.clear();

    biases.push_back(Eigen::VectorXf(2));
    biases[0] << 0.1f, -0.2f;           // hidden layer biases

    biases.push_back(Eigen::VectorXf(1));
    biases[1] << 0.05f;                // output layer bias

    weights.push_back(Eigen::MatrixXf(2, 2));
    weights[0] << 0.5f, -0.3f, 0.8f,  0.2f;          // input -> hidden

    weights.push_back(Eigen::MatrixXf(1, 2));
    weights[1] << -0.7f, 0.9f;          // hidden -> output

    // Input vector
    Eigen::VectorXf x(2);
    x << 0.4f, 0.6f;

    // Manual computation: a1 = σ(W1 x + b1), a2 = σ(W2 a1 + b2)
    Eigen::VectorXf z1 = weights[0] * x + biases[0];
    Eigen::VectorXf a1 = act->f(z1);
    Eigen::VectorXf z2 = weights[1] * a1 + biases[1];
    Eigen::VectorXf a2 = act->f(z2);

    Eigen::VectorXf out = net.feedforward(x);

    ASSERT_EQ(out.size(), 1);
    EXPECT_NEAR(out(0), a2(0), 1e-10);
}


TEST(MnistTest, BackpropGradientMatchesFiniteDifference)
{
    auto act = std::make_shared<Sigmoid>();

    Network net({2, 2, 1}, act);

    auto& biases  = net.biases();
    auto& weights = net.weights();

    // Deterministic parameters.
    biases.clear();
    weights.clear();

    biases.push_back(Eigen::VectorXf(2));
    biases[0] << 0.1f, -0.2f;

    biases.push_back(Eigen::VectorXf(1));
    biases[1] << 0.05f;

    weights.push_back(Eigen::MatrixXf(2, 2));
    weights[0] << 0.5f, -0.3f,
            0.8f,  0.2f;

    weights.push_back(Eigen::MatrixXf(1, 2));
    weights[1] << -0.7f, 0.9f;

    // A single training example (x, y) — like a tiny "MNIST" with 2 pixels and 1 output.
    Eigen::VectorXf x(2);
    x << 0.4f, 0.6f;

    Eigen::VectorXf y(1);
    y << 1.0f;  // target "label"

    // Analytical gradient via backprop.
    std::vector<Eigen::VectorXf> nabla_b;
    std::vector<Eigen::MatrixXf> nabla_w;
    std::tie(nabla_b, nabla_w) = net.backprop(x, y);

    // We'll check the gradient w.r.t. a single weight, e.g. weights[1](0,0).
    const int layer = 1;    // hidden -> output layer
    const int row   = 0;
    const int col   = 0;

    float base_weight = weights[layer](row, col);
    const float eps = 1e-3f;  // Use larger epsilon for float precision (vs 1e-5 for double)

    auto cost = [&](const Network& net)
    {
        Eigen::VectorXf a = net.feedforward(x);
        return 0.5f * (a - y).squaredNorm();
    };

    // Numerical gradient using central difference:
    // (C(w+eps) - C(w-eps)) / (2*eps)
    weights[layer](row, col) = base_weight + eps;
    float cost_up = cost(net);

    weights[layer](row, col) = base_weight - eps;
    float cost_down = cost(net);

    // Restore parameter
    weights[layer](row, col) = base_weight;

    float numerical_grad = (cost_up - cost_down) / (2.0f * eps);
    float backprop_grad  = nabla_w[layer](row, col);

    // Tolerance adjusted for float precision
    EXPECT_NEAR(backprop_grad, numerical_grad, 1e-4f);
}