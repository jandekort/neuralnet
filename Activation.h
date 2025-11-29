
#pragma once

#include <Eigen/Dense>
#include "RNG.h"

using Eigen::MatrixXf;
using Eigen::VectorXf;

class Activation
{
public:
    virtual ~Activation() = default;
    virtual VectorXf f(const VectorXf& z) const = 0;   // activation function
    virtual VectorXf df(const VectorXf& z) const = 0;  // derivative wrt z

    virtual float initWeight(RNG& rng, int num_in) const = 0; // random weight initialisation
};
