#pragma once

#include "Activation.h"
#include <algorithm>

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

    double scaledWeight(int num_in) const override
    {
        return std::sqrt(2.0 / static_cast<double>(num_in));
    }
};
