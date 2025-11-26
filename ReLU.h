#pragma once

#include "Activation.h"
#include <algorithm>
#include <cmath>

class ReLU : public Activation
{
public:
    VectorXd f(const VectorXd& z) const override
    {
        return z.unaryExpr([](double v){ return std::max(0.0, v); });
    }

    VectorXd df(const VectorXd& z) const override
    {
        return z.unaryExpr([](double v){ return v > 0.0 ? 1.0 : 0.0; });
    }

    double initWeight(RNG& rng, int num_in) const override
    {
        // He Uniform initialization: U(-limit, limit) where limit = sqrt(6 / num_in)
        const double limit = std::sqrt(6.0 / static_cast<double>(num_in));
        return rng.uniform(-limit, limit);
    }
};
