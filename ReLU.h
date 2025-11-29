#pragma once

#include "Activation.h"
#include <algorithm>
#include <cmath>

class ReLU : public Activation
{
public:
    VectorXf f(const VectorXf& z) const override
    {
        return z.unaryExpr([](float v){ return std::max(0.0f, v); });
    }

    VectorXf df(const VectorXf& z) const override
    {
        return z.unaryExpr([](float v){ return v > 0.0f ? 1.0f : 0.0f; });
    }

    float initWeight(RNG& rng, int num_in) const override
    {
        // He Uniform initialization: U(-limit, limit) where limit = sqrt(6 / num_in)
        const float limit = std::sqrt(6.0f / static_cast<float>(num_in));
        return static_cast<float>(rng.uniform(-limit, limit));
    }
};
