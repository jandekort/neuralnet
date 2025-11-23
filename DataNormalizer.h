// DataNormalizer.h
#pragma once

#include <vector>
#include <memory>
#include <stdexcept>
#include <string>
#include <Eigen/Dense>

#include "Network.h"

class IDataNormalizer
{
public:
    using Example = Network::Example;

    virtual ~IDataNormalizer() = default;

    // Learn normalization parameters from the training set
    virtual void fit(const std::vector<Example>& training_data) = 0;

    // Normalize a single (x, y) example in-place
    virtual void transform_inplace(Example& ex) const = 0;

    // Normalize an entire dataset in-place
    virtual void transform_dataset_inplace(std::vector<Example>& data) const = 0;

    // Undo normalization for an output vector (sigma prediction)
    virtual Eigen::VectorXd inverse_transform_output(const Eigen::VectorXd& y_norm) const = 0;
};


class StandardScalerNormalizer : public IDataNormalizer
{
public:
    using Example = Network::Example;

    StandardScalerNormalizer() = default;

    void fit(const std::vector<Example>& training_data) override;
    void transform_inplace(Example& ex) const override;
    void transform_dataset_inplace(std::vector<Example>& data) const override;
    Eigen::VectorXd inverse_transform_output(const Eigen::VectorXd& y_norm) const override;

private:
    bool fitted_ = false;
    Eigen::VectorXd input_mean_;
    Eigen::VectorXd input_std_;
    Eigen::VectorXd output_mean_;
    Eigen::VectorXd output_std_;

    void check_fitted() const;
};
