#pragma once

#include <Eigen/Dense>
#include <memory>

class Activation;
class Sigmoid;

using Eigen::VectorXf;

class Cost
{
public:
    virtual ~Cost() = default;

    // Compute the cost for a single example
    // a: network output (prediction)
    // y: true label/value
    virtual float fn(const VectorXf& a, const VectorXf& y) const = 0;

    // Compute the output layer error delta for backpropagation
    // z: weighted input to output layer
    // a: activation of output layer (network prediction)
    // y: true label/value
    // output_act: activation function used in output layer
    virtual VectorXf delta(const VectorXf& z, const VectorXf& a, const VectorXf& y,
                           const Activation& output_act) const = 0;
};

// Mean Squared Error (compatible with ANY differentiable activation)
class MSECost : public Cost
{
public:
    // Cost: C = 0.5 * ||a - y||^2
    float fn(const VectorXf& a, const VectorXf& y) const override;

    // Delta: δ^L = (a - y) * σ'(z)    // elementwise multiplication
    VectorXf delta(const VectorXf& z, const VectorXf& a, const VectorXf& y,
                   const Activation& output_act) const override;
};

// Cross-entropy WITH sigmoid output
class SigmoidCrossEntropyCost : public Cost
{
public:
    // C = - Σ [ y_i log(a_i) + (1 - y_i) log(1 - a_i) ]
    float fn(const VectorXf& a, const VectorXf& y) const override;

    // δ^L = a - y  (valid if output activation is Sigmoid)
    VectorXf delta(const VectorXf& z, const VectorXf& a, const VectorXf& y,
                   const Activation& output_act) const override;
};

inline const std::shared_ptr<const Cost> MSE_COST = std::make_shared<MSECost>();
inline const std::shared_ptr<const Cost> SIGMOID_CE_COST = std::make_shared<SigmoidCrossEntropyCost>();
