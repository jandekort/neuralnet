
#pragma once

#include "Activation.h"

class Sigmoid : public Activation
{
    static inline double s(double z)
    {
        if (z >= 0)
        {
            double e = std::exp(-z); return 1.0 / (1.0 + e);
        }
        else
        {
            double e = std::exp(z);  return e / (1.0 + e);
        }
    }
public:
    VectorXd f(const VectorXd& z) const override
    {
        return z.unaryExpr([](double v){ return s(v); });
    }
    VectorXd df(const VectorXd& z) const override
    {
        return z.unaryExpr([](double v){ double a = s(v); return a*(1.0 - a); });
    }
};



