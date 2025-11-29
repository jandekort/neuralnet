// DataNormalizer.cpp
#include "DataNormalizer.h"

using Eigen::VectorXf;

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

    input_mean_  = VectorXf::Zero(in_dim);
    input_std_   = VectorXf::Zero(in_dim);
    output_mean_ = VectorXf::Zero(out_dim);
    output_std_  = VectorXf::Zero(out_dim);

    // First pass: compute means
    for (const auto& ex : training_data)
    {
        const VectorXf& x = ex.first;
        const VectorXf& y = ex.second;

        if (x.size() != in_dim || y.size() != out_dim)
        {
            throw std::runtime_error("StandardScalerNormalizer::fit: inconsistent dimensions in training_data.");
        }

        input_mean_  += x;
        output_mean_ += y;
    }

    input_mean_  /= static_cast<float>(n);
    output_mean_ /= static_cast<float>(n);

    // Second pass: compute variances
    VectorXf input_var  = VectorXf::Zero(in_dim);
    VectorXf output_var = VectorXf::Zero(out_dim);

    for (const auto& ex : training_data)
    {
        const VectorXf& x = ex.first;
        const VectorXf& y = ex.second;

        VectorXf dx = x - input_mean_;
        VectorXf dy = y - output_mean_;

        input_var  += dx.cwiseProduct(dx);
        output_var += dy.cwiseProduct(dy);
    }

    input_var  /= static_cast<float>(n);
    output_var /= static_cast<float>(n);

    // Compute std dev; avoid division by zero by clamping very small stds to 1.0
    const float eps = 1e-12f;
    input_std_.resize(in_dim);
    output_std_.resize(out_dim);

    for (int i = 0; i < in_dim; ++i)
    {
        float v = input_var(i);
        if (v < eps)
        {
            v = 1.0f; // no scaling
        }
        input_std_(i) = std::sqrt(v);
    }

    for (int i = 0; i < out_dim; ++i)
    {
        float v = output_var(i);
        if (v < eps)
        {
            v = 1.0f; // no scaling
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

    VectorXf& x = ex.first;
    VectorXf& y = ex.second;

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

VectorXf StandardScalerNormalizer::inverse_transform_output(const VectorXf& y_norm) const
{
    check_fitted();

    if (y_norm.size() != output_mean_.size())
    {
        throw std::runtime_error("StandardScalerNormalizer::inverse_transform_output: dimension mismatch.");
    }

    // y = y_norm * std + mean
    VectorXf y = y_norm.cwiseProduct(output_std_) + output_mean_;
    return y;
}
