#pragma once

#include "IDataReader.h"

static uint32_t be_u32(std::ifstream& f)
{
    uint8_t b[4]; f.read(reinterpret_cast<char*>(b),4);
    return (uint32_t(b[0])<<24) | (uint32_t(b[1])<<16) | (uint32_t(b[2])<<8) | uint32_t(b[3]);
}

class MnistReader : public IDataReader
{
public:
    MnistReader(std::string train_images, std::string train_labels, std::string test_images, std::string test_labels);

    void load() override;
    
    const std::vector<Example>& training() const override { return training_; }
    const std::vector<std::pair<VectorXd,int>>& test() const override { return test_; }

    void print(const VectorXd& img, int label, int rows = 28, int cols = 28);

private:
    static std::vector<VectorXd> load_images(const std::string& path);
    static std::vector<int> load_labels(const std::string& path);

private:
    std::string train_images_, train_labels_, test_images_, test_labels_;
    std::vector<Example> training_;
    std::vector<std::pair<VectorXd,int>> test_;
};

