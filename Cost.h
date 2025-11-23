#pragma once

#include <Eigen/Dense>

class Activation;
class Sigmoid;

using Eigen::VectorXd;

class Cost
{
public:
    virtual ~Cost() = default;

    // Compute the cost for a single example
    // a: network output (prediction)
    // y: true label/value
    virtual double fn(const VectorXd& a, const VectorXd& y) const = 0;

    // Compute the output layer error delta for backpropagation
    // z: weighted input to output layer
    // a: activation of output layer (network prediction)
    // y: true label/value
    // output_act: activation function used in output layer
    virtual VectorXd delta(const VectorXd& z, const VectorXd& a, const VectorXd& y,
                           const Activation* output_act) const = 0;
};

// Mean Squared Error (compatible with ANY differentiable activation)
class MSECost : public Cost
{
public:
    // Cost: C = 0.5 * ||a - y||^2
    double fn(const VectorXd& a, const VectorXd& y) const override;

    // Delta: δ^L = (a - y) * σ'(z)    // elementwise multiplication
    VectorXd delta(const VectorXd& z, const VectorXd& a, const VectorXd& y,
                   const Activation* output_act) const override;
};

// Cross-entropy WITH sigmoid output
class SigmoidCrossEntropyCost : public Cost
{
public:
    // C = - Σ [ y_i log(a_i) + (1 - y_i) log(1 - a_i) ]
    double fn(const VectorXd& a, const VectorXd& y) const override;

    // δ^L = a - y  (valid if output activation is Sigmoid)
    VectorXd delta(const VectorXd& z, const VectorXd& a, const VectorXd& y,
                   const Activation* output_act) const override;
};

// Global cost function instances
extern const MSECost                 MSE_COST;
extern const SigmoidCrossEntropyCost SIGMOID_CE_COST;
