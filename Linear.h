// Linear.h
#pragma once

#include "Activation.h"
#include <cmath>

class Linear : public Activation
{
public:
    VectorXf f(const VectorXf& z) const override
    {
        return z;  // identity
    }

    VectorXf df(const VectorXf& z) const override
    {
        // derivative is 1 everywhere
        return VectorXf::Ones(z.size());
    }

    float initWeight(RNG& rng, int num_in) const override
    {
        // Xavier/Glorot Normal initialization: N(0, 1/sqrt(num_in))
        const float scale = 1.0f / std::sqrt(static_cast<float>(num_in));
        return static_cast<float>(rng.gauss()) * scale;
    }
};
