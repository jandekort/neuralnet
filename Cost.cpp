#include "Cost.h"
#include "Activation.h"
#include "Sigmoid.h"
//#include "ReLU.h"
//#include "Linear.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <sstream>

using Eigen::VectorXd;



double MSECost::fn(const VectorXd& a, const VectorXd& y) const
{
    VectorXd diff = a - y;
    return 0.5 * diff.squaredNorm();
}

VectorXd MSECost::delta(const VectorXd& z, const VectorXd& a, const VectorXd& y,
                        const Activation& output_act) const
{
    return (a - y).cwiseProduct(output_act.df(z));
}

double SigmoidCrossEntropyCost::fn(const VectorXd& a, const VectorXd& y) const
{
    const double eps = 1e-10;
    double cost = 0.0;

    for (int i = 0; i < a.size(); ++i)
    {
        double ai = std::clamp(a(i), eps, 1.0 - eps);
        double yi = y(i);
        cost -= yi * std::log(ai) + (1.0 - yi) * std::log(1.0 - ai);
    }
    return cost;
}

VectorXd SigmoidCrossEntropyCost::delta(const VectorXd& z, const VectorXd& a, const VectorXd& y,
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
