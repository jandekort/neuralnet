
#pragma once

#include <Eigen/Dense>
#include <random>
#include <iostream>
#include "Activation.h"
#include "Sigmoid.h"
#include "Cost.h"

#define DEBUG 1 // 0: debug off, 1: debug on

using Eigen::MatrixXd;
using Eigen::VectorXd;

inline void check_finite(const Eigen::Ref<const Eigen::VectorXd>& x, const char* msg)
{
    if (!x.array().isFinite().all())
    {
        std::ostringstream oss;
        oss << "NaN/Inf detected in " << msg << "\n" << x;
        throw std::runtime_error(oss.str());
    }
}

struct RNG
{
    std::mt19937 gen{std::random_device{}()};
    std::normal_distribution<double> normal{0.0, 1.0};
    template<typename It>
    void shuffle(It first, It last)
    {
        std::shuffle(first, last, gen);
    }
    double gauss()
    {
        return normal(gen);
    }
};

static int argmax(const VectorXd& v)
{
    Eigen::Index idx;
    v.maxCoeff(&idx); // index of maximum coefficient in vec
    return static_cast<int>(idx);
}

static VectorXd std_basis_vec(int index, int size)
{
    VectorXd v = VectorXd::Zero(size);
    v(index) = 1.0; // standard basis vector e_i
    return v;
}

enum class EvalMode
{
    Classification,
    Regression
};

struct EvaluationResult
{
    EvalMode mode = EvalMode::Classification;

    // Classification
    int correct = 0;
    int total   = 0;

    // Regression
    double mse  = 0.0;  // mean squared error
    double mae  = 0.0;  // mean absolute error
};

class Network
{
public:
    using Example = std::pair<VectorXd, VectorXd>;

    Network(const std::vector<int>& sizes,
            std::shared_ptr<const Activation> hidden_act = std::make_shared<Sigmoid>(),
            std::shared_ptr<const Activation> output_act = nullptr,
            const Cost* cost = &MSE_COST);

    VectorXd feedforward(const VectorXd& a0) const;

    void SGD(std::vector<Example>& training_data, int epochs, int mini_batch_size, double eta, const std::vector<Example>* test_data = nullptr,
             EvalMode eval_mode = EvalMode::Classification, double decay = 1.0);

    EvaluationResult evaluate(const std::vector<Example>& test_data, EvalMode mode = EvalMode::Classification) const;

    std::pair<std::vector<VectorXd>, std::vector<MatrixXd>> backprop(const VectorXd& x, const VectorXd& y) const;

    void print() const;

    std::vector<VectorXd>& biases();
    std::vector<MatrixXd>& weights();
    const std::vector<VectorXd>& biases() const;
    const std::vector<MatrixXd>& weights() const;

private:
    void update_mini_batch(std::vector<Example>::iterator begin, std::vector<Example>::iterator end, double eta);

private:
    int num_layers_;
    std::vector<int> sizes_;
    std::vector<VectorXd> biases_;
    std::vector<MatrixXd> weights_;
    std::shared_ptr<const Activation> hidden_act_;  // for hidden layers
    std::shared_ptr<const Activation> output_act_;  // for last layer
    const Cost* cost_;  // Cost function for training
    RNG rng_;  // Random number generator (non-global, thread-safe)
};