
#pragma once

#include "Network.h"
#include <fstream>

// ============================================================================
// Data reading interfaces
// ============================================================================
class IDataReader
{
public:
    using Example = Network::Example;
    virtual ~IDataReader() = default;
    virtual void load() = 0;
    virtual const std::vector<Example>& training() const = 0;
    virtual const std::vector<std::pair<VectorXd,int>>& test() const = 0;
};

// ============================================================================
// MNIST reader (IDX format)
// ============================================================================
static uint32_t be_u32(std::ifstream& f)
{
    uint8_t b[4]; f.read(reinterpret_cast<char*>(b),4);
    return (uint32_t(b[0])<<24) | (uint32_t(b[1])<<16) | (uint32_t(b[2])<<8) | uint32_t(b[3]);
}

class MnistReader : public IDataReader
{
public:
    MnistReader(std::string train_images, std::string train_labels,
                std::string test_images,  std::string test_labels)
            : train_images_(std::move(train_images)), train_labels_(std::move(train_labels)),
              test_images_(std::move(test_images)),   test_labels_(std::move(test_labels)) {}

    void load() override {
        auto trX = load_images(train_images_);
        auto trY = load_labels(train_labels_);
        auto teX = load_images(test_images_);
        auto teY = load_labels(test_labels_);

        if (trX.size() != trY.size()) throw std::runtime_error("train images/labels mismatch");
        if (teX.size() != teY.size()) throw std::runtime_error("test images/labels mismatch");

        training_.clear(); test_.clear();
        training_.reserve(trX.size());
        for (size_t i = 0; i < trX.size(); ++i)
            training_.emplace_back(std::move(trX[i]), std_basis_vec(trY[i], 10));

        test_.reserve(teX.size());
        for (size_t i = 0; i < teX.size(); ++i)
            test_.emplace_back(std::move(teX[i]), teY[i]);
    }

    const std::vector<Example>& training() const override { return training_; }
    const std::vector<std::pair<VectorXd,int>>& test() const override { return test_; }

private:
    static std::vector<VectorXd> load_images(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("open images failed: " + path);
        if (be_u32(f) != 2051) throw std::runtime_error("bad images magic");
        uint32_t num = be_u32(f), rows = be_u32(f), cols = be_u32(f);
        uint32_t dim = rows*cols;

        std::vector<VectorXd> out; out.reserve(num);
        std::vector<uint8_t> buf(dim);
        for (uint32_t i=0;i<num;++i) {
            f.read(reinterpret_cast<char*>(buf.data()), dim);
            VectorXd v(dim);
            for (uint32_t j=0;j<dim;++j) v(j) = double(buf[j]) / 255.0;
            out.push_back(std::move(v));
        }
        return out;
    }

    static std::vector<int> load_labels(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("open labels failed: " + path);
        if (be_u32(f) != 2049) throw std::runtime_error("bad labels magic");
        uint32_t num = be_u32(f);
        std::vector<int> out(num);
        for (uint32_t i=0;i<num;++i) { unsigned char c; f.read(reinterpret_cast<char*>(&c),1); out[i]=int(c); }
        return out;
    }

public:
    void print_ascii_digit(const VectorXd& img, int label, int rows = 28, int cols = 28) {
        static const char* shades = " .:-=+*#%@";   // 10 gradations of intensity

        std::cout << "Label: " << label << "\n";
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                double v = img(r * cols + c);       // value in [0,1]
                int idx = std::clamp(int(v * 9.999), 0, 9);
                std::cout << shades[idx];
            }
            std::cout << "\n";
        }
        std::cout << std::flush;
    }

private:
    std::string train_images_, train_labels_, test_images_, test_labels_;
    std::vector<Example> training_;
    std::vector<std::pair<VectorXd,int>> test_;
};