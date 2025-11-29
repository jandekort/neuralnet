#include "Cost.h"
#include "Activation.h"
#include "Sigmoid.h"
//#include "ReLU.h"
//#include "Linear.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <sstream>

using Eigen::VectorXf;



float MSECost::fn(const VectorXf& a, const VectorXf& y) const
{
    VectorXf diff = a - y;
    return 0.5f * diff.squaredNorm();
}

VectorXf MSECost::delta(const VectorXf& z, const VectorXf& a, const VectorXf& y,
                        const Activation& output_act) const
{
    return (a - y).cwiseProduct(output_act.df(z));
}

float SigmoidCrossEntropyCost::fn(const VectorXf& a, const VectorXf& y) const
{
    const float eps = 1e-10f;
    float cost = 0.0f;

    for (int i = 0; i < a.size(); ++i)
    {
        float ai = std::clamp(a(i), eps, 1.0f - eps);
        float yi = y(i);
        cost -= yi * std::log(ai) + (1.0f - yi) * std::log(1.0f - ai);
    }
    return cost;
}

VectorXf SigmoidCrossEntropyCost::delta(const VectorXf& z, const VectorXf& a, const VectorXf& y,
                                        const Activation& output_act) const
{
    const Sigmoid* sigmoid_act = dynamic_cast<const Sigmoid*>(&output_act);
    if (!sigmoid_act)
    {
        std::ostringstream oss;
        oss << "SigmoidCrossEntropyCost::delta requires output activation of type Sigmoid, but got ";
        oss << typeid(output_act).name();
        throw std::logic_error(oss.str());
    }

    return a - y;
}
