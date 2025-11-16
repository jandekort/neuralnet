#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include "BinaryFileWriter.h"

using Eigen::VectorXd;
using Example = std::pair<VectorXd, VectorXd>;

class DataGenerator {
public:
    struct Ranges {
        double m_min = 0.5,   m_max = 1.4;   // moneyness F/K
        double tau_min = 0.05, tau_max = 10.0; // maturity (years)
        double r_min = 0.0,    r_max = 0.10;  // interest rate
        double sig_min = 0.01, sig_max = 0.80;// volatility
    };

public:
    DataGenerator(uint64_t seed, Ranges ranges )
            : rng_(seed), ranges_(ranges) {}

    DataGenerator(uint64_t seed)
            : rng_(seed), ranges_(Ranges{}) {}

    // Generate and write a binary dataset containing validation + test sets
    void generate_file(const std::string& out_path, size_t n_validation, size_t n_test, uint32_t in_dim = 3, uint32_t out_dim = 1);

    // Produce a vector of Example objects
    std::vector<Example> make_examples(size_t n, uint32_t in_dim, uint32_t out_dim);

    static double black_call(double F, double K, double r, double tau, double sigma);

private:
    static double norm_cdf(double x);

    // Write one dataset section
    static void write_section(BinaryFileWriter& w, const char tag[4], const std::vector<Example>& data, uint32_t in_dim, uint32_t out_dim);

private:
    std::mt19937_64 rng_;
    Ranges ranges_;
};
