// Linear.h
#pragma once

#include "Activation.h"

class Linear : public Activation
{
public:
    VectorXd f(const VectorXd& z) const override
    {
        return z;  // identity
    }

    VectorXd df(const VectorXd& z) const override
    {
        // derivative is 1 everywhere
        return VectorXd::Ones(z.size());
    }

    double scaledWeight(int num_in) const override
    {
        return 1.0 / std::sqrt(static_cast<double>(num_in));
    }
};
