
#pragma once

#include "Activation.h"
#include <cmath>

class Sigmoid : public Activation
{
    static inline float s(float z)
    {
        if (z >= 0)
        {
            float e = std::exp(-z); return 1.0f / (1.0f + e);
        }
        else
        {
            float e = std::exp(z);  return e / (1.0f + e);
        }
    }
public:
    VectorXf f(const VectorXf& z) const override
    {
        return z.unaryExpr([](float v){ return s(v); });
    }
    VectorXf df(const VectorXf& z) const override
    {
        return z.unaryExpr([](float v){ float a = s(v); return a*(1.0f - a); });
    }

    float initWeight(RNG& rng, int num_in) const override
    {
        // Xavier/Glorot Normal initialization: N(0, 1/sqrt(num_in))
        const float scale = 1.0f / std::sqrt(static_cast<float>(num_in));
        return static_cast<float>(rng.gauss()) * scale;
    }
};



