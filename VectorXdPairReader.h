#pragma once
#include "IDataReader.h"
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <Eigen/Dense>

using Eigen::VectorXf;


class VectorXdPairReader : public IDataReader
{
public:
    explicit VectorXdPairReader(std::string dataset_file)
            : path_(std::move(dataset_file)) {}

    void load() override ;

    const std::vector<Example>& training() const override { return training_; }
    const std::vector<Example>& test() const override { return test_; }

    void print(const VectorXf& img, int /*label*/, int rows = 1, int cols = 1) ;

private:
    std::string path_;
    std::vector<Example> training_, test_;
};
