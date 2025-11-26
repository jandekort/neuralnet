
#pragma once

#include <Eigen/Dense>
#include "RNG.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class Activation
{
public:
    virtual ~Activation() = default;
    virtual VectorXd f(const VectorXd& z) const = 0;   // activation function
    virtual VectorXd df(const VectorXd& z) const = 0;  // derivative wrt z

    virtual double initWeight(RNG& rng, int num_in) const = 0; // random weight initialisation
};
