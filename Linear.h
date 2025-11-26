// Linear.h
#pragma once

#include "Activation.h"
#include <cmath>

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

    double initWeight(RNG& rng, int num_in) const override
    {
        // Xavier/Glorot Normal initialization: N(0, 1/sqrt(num_in))
        const double scale = 1.0 / std::sqrt(static_cast<double>(num_in));
        return rng.gauss() * scale;
    }
};
