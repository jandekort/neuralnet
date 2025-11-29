#include "VectorXdPairReader.h"

void VectorXdPairReader::load() {
    training_.clear();
    test_.clear();

    std::ifstream f(path_, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Cannot open dataset file: " + path_);
    }

    // Magic header check
    char magic[4];
    f.read(magic, 4);
    if (f.gcount() != 4 || std::string(magic, 4) != "EXDS")
        throw std::runtime_error("Invalid file header");

    uint32_t n_sections = be_u32(f);
    if (n_sections != 2)
        throw std::runtime_error("Expected 2 sections");

    for (uint32_t s = 0; s < n_sections; ++s)
    {
        char tag[4];
        f.read(tag, 4);
        uint32_t count = be_u32(f);
        uint32_t in_dim = be_u32(f);
        uint32_t out_dim = be_u32(f);

        auto read_double = [&]() {
            double x{};
            f.read(reinterpret_cast<char *>(&x), sizeof(double));
            return x;
        };

        std::vector<Example> bucket;
        bucket.reserve(count);

        for (uint32_t i = 0; i < count; ++i) {
            VectorXf x(in_dim);
            VectorXf y(out_dim);
            for (uint32_t k = 0; k < in_dim; ++k) x[k] = static_cast<float>(read_double());
            for (uint32_t k = 0; k < out_dim; ++k) y[k] = static_cast<float>(read_double());
            bucket.emplace_back(std::move(x), std::move(y));
        }

        std::string stag(tag, 4);
        if (stag == "TRAI") {
            training_ = std::move(bucket);
        } else if (stag == "TEST") {
            test_ = std::move(bucket);
        } else {
            throw std::runtime_error("Unknown section tag: " + stag);
        }
    }
}

void VectorXdPairReader::print(const VectorXf &img, int, int rows, int cols)
{
    if (rows == 1)
    {
        int n = std::min<int>(cols, img.size());
        for (int c = 0; c < n; ++c)
        {
            std::cout << img[c];
            if (c + 1 < n) std::cout << " ";
        }
        std::cout << "\n";
        return;
    }
    else
    {
        throw std::range_error("VectorXdPairReader::print: Not supported");
    }
}
