
#pragma once

#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class Activation
{
public:
    virtual ~Activation() = default;
    virtual VectorXd f(const VectorXd& z) const = 0;   // activation function
    virtual VectorXd df(const VectorXd& z) const = 0;  // derivative wrt z
};
