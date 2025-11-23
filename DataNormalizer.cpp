// DataNormalizer.cpp
#include "DataNormalizer.h"

using Eigen::VectorXd;

void StandardScalerNormalizer::check_fitted() const
{
    if (!fitted_)
    {
        throw std::runtime_error("StandardScalerNormalizer: fit() must be called before transform.");
    }
}

void StandardScalerNormalizer::fit(const std::vector<Example>& training_data)
{
    if (training_data.empty())
    {
        throw std::runtime_error("StandardScalerNormalizer::fit: training_data is empty.");
    }

    const std::size_t n = training_data.size();
    const int in_dim  = static_cast<int>(training_data[0].first.size());
    const int out_dim = static_cast<int>(training_data[0].second.size());

    input_mean_  = VectorXd::Zero(in_dim);
    input_std_   = VectorXd::Zero(in_dim);
    output_mean_ = VectorXd::Zero(out_dim);
    output_std_  = VectorXd::Zero(out_dim);

    // First pass: compute means
    for (const auto& ex : training_data)
    {
        const VectorXd& x = ex.first;
        const VectorXd& y = ex.second;

        if (x.size() != in_dim || y.size() != out_dim)
        {
            throw std::runtime_error("StandardScalerNormalizer::fit: inconsistent dimensions in training_data.");
        }

        input_mean_  += x;
        output_mean_ += y;
    }

    input_mean_  /= static_cast<double>(n);
    output_mean_ /= static_cast<double>(n);

    // Second pass: compute variances
    VectorXd input_var  = VectorXd::Zero(in_dim);
    VectorXd output_var = VectorXd::Zero(out_dim);

    for (const auto& ex : training_data)
    {
        const VectorXd& x = ex.first;
        const VectorXd& y = ex.second;

        VectorXd dx = x - input_mean_;
        VectorXd dy = y - output_mean_;

        input_var  += dx.cwiseProduct(dx);
        output_var += dy.cwiseProduct(dy);
    }

    input_var  /= static_cast<double>(n);
    output_var /= static_cast<double>(n);

    // Compute std dev; avoid division by zero by clamping very small stds to 1.0
    const double eps = 1e-12;
    input_std_.resize(in_dim);
    output_std_.resize(out_dim);

    for (int i = 0; i < in_dim; ++i)
    {
        double v = input_var(i);
        if (v < eps)
        {
            v = 1.0; // no scaling
        }
        input_std_(i) = std::sqrt(v);
    }

    for (int i = 0; i < out_dim; ++i)
    {
        double v = output_var(i);
        if (v < eps)
        {
            v = 1.0; // no scaling
        }
        output_std_(i) = std::sqrt(v);
    }

    fitted_ = true;

    std::cout << "  Input mean:  " << input_mean_.transpose() << "\n";
    std::cout << "  Input std:   " << input_std_.transpose()  << "\n";
    std::cout << "  Output mean: " << output_mean_.transpose() << "\n";
    std::cout << "  Output std:  " << output_std_.transpose()  << "\n\n";
}

void StandardScalerNormalizer::transform_inplace(Example& ex) const
{
    check_fitted();

    VectorXd& x = ex.first;
    VectorXd& y = ex.second;

    if (x.size() != input_mean_.size() || y.size() != output_mean_.size())
    {
        throw std::runtime_error("StandardScalerNormalizer::transform_inplace: dimension mismatch.");
    }

    // Normalize inputs
    x = (x - input_mean_).cwiseQuotient(input_std_);

    // Normalize outputs (targets)
    y = (y - output_mean_).cwiseQuotient(output_std_);
}

void StandardScalerNormalizer::transform_dataset_inplace(std::vector<Example>& data) const
{
    check_fitted();
    for (auto& ex : data)
    {
        transform_inplace(ex);
    }
}

VectorXd StandardScalerNormalizer::inverse_transform_output(const VectorXd& y_norm) const
{
    check_fitted();

    if (y_norm.size() != output_mean_.size())
    {
        throw std::runtime_error("StandardScalerNormalizer::inverse_transform_output: dimension mismatch.");
    }

    // y = y_norm * std + mean
    VectorXd y = y_norm.cwiseProduct(output_std_) + output_mean_;
    return y;
}
