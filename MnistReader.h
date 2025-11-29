#pragma once

#include "IDataReader.h"


class MnistReader : public IDataReader
{
public:
    MnistReader(std::string train_images, std::string train_labels, std::string test_images, std::string test_labels);

    void load() override;
    
    const std::vector<Example>& training() const override { return training_; }
    const std::vector<Example>& test() const override { return test_; }

    void print(const VectorXf& img, int label, int rows = 28, int cols = 28);

private:
    static std::vector<VectorXf> load_images(const std::string& path);
    static std::vector<int> load_labels(const std::string& path);

private:
    std::string train_images_, train_labels_, test_images_, test_labels_;
    std::vector<Example> training_, test_;
};

