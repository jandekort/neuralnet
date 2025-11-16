
#pragma once

#include <Eigen/Dense>
#include <random>
#include <iostream>
#include "Activation.h"
#include "Sigmoid.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

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

extern RNG RNG_GLOBAL;

// Utility functions
static int argmax(const VectorXd& v)
{
    Eigen::Index idx;
    v.maxCoeff(&idx);
    return static_cast<int>(idx);
}

// standard basis vector e_i
static VectorXd std_basis_vec(int index, int size)
{
    VectorXd v = VectorXd::Zero(size);
    v(index) = 1.0;
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
            bool use_scaled_weight_init = true );

    VectorXd feedforward(const VectorXd& a0) const;

    void SGD(std::vector<Example>& training_data, int epochs, int mini_batch_size, double eta, const std::vector<Example>* test_data = nullptr,
             EvalMode eval_mode = EvalMode::Classification);

    EvaluationResult evaluate(const std::vector<Example>& test_data, EvalMode mode = EvalMode::Classification) const;

    std::pair<std::vector<VectorXd>, std::vector<MatrixXd>> backprop(const VectorXd& x, const VectorXd& y) const;

    void print() const;

    std::vector<VectorXd>& biases();
    std::vector<MatrixXd>& weights();
    const std::vector<VectorXd>& biases() const;
    const std::vector<MatrixXd>& weights() const;

private:
    template<class It>
    void update_mini_batch(It begin, It end, double eta)
    {
        std::vector<VectorXd> nabla_b(biases_.size());
        std::vector<MatrixXd> nabla_w(weights_.size());
        for (size_t l = 0; l < nabla_b.size(); ++l)
        {
            nabla_b[l] = VectorXd::Zero(biases_[l].size());
        }
        for (size_t l = 0; l < nabla_w.size(); ++l)
        {
            nabla_w[l] = MatrixXd::Zero(weights_[l].rows(), weights_[l].cols());
        }

        for (auto it = begin; it != end; ++it)
        {
            const VectorXd& x = it->first;
            const VectorXd& y = it->second;

            std::vector<VectorXd> dnb;
            std::vector<MatrixXd> dnw;

            std::tie(dnb, dnw) = backprop(x, y); // compute per-example gradients using backpropagation

            for (size_t l = 0; l < nabla_b.size(); ++l)
            {
                nabla_b[l] += dnb[l]; // accumulate gradients in mini batch
            }
            for (size_t l = 0; l < nabla_w.size(); ++l)
            {
                nabla_w[l] += dnw[l]; // accumulate gradients in mini batch
            }
        }

        double lr = eta / static_cast<double>(std::distance(begin, end));
        for (size_t l = 0; l < weights_.size(); ++l)
        {
            weights_[l] -= lr * nabla_w[l];
            biases_[l]  -= lr * nabla_b[l];
        }
    }

private:
    int num_layers_;
    std::vector<int> sizes_;
    std::vector<VectorXd> biases_;
    std::vector<MatrixXd> weights_;
    std::shared_ptr<const Activation> hidden_act_;  // for hidden layers
    std::shared_ptr<const Activation> output_act_;  // for last layer
};