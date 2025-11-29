#include <gtest/gtest.h>

#include <cstdio>     // for std::remove
#include <string>

#include "BinaryFileWriter.h"
#include "VectorXdPairReader.h"
#include "DataGenerator.h"

using Eigen::VectorXf;
using Example = std::pair<VectorXf, VectorXf>;



TEST(BlackTest, MakeExamplesTest)
{
    // Fixed ranges chosen to eliminate randomness
    DataGenerator::Ranges r;
    r.m_min = r.m_max = 1.2;
    r.tau_min = r.tau_max = 0.5;
    r.r_min = r.r_max = 0.02;
    r.sig_min = r.sig_max = 0.3;

    DataGenerator gen(1234, r);

    const uint32_t in_dim = 4;
    const uint32_t out_dim = 1;

    auto examples = gen.make_examples(1, in_dim, out_dim);
    ASSERT_EQ(examples.size(), 1);

    const VectorXf& x = examples[0].first;
    const VectorXf& y = examples[0].second;

    // Basic dimension checks
    EXPECT_EQ(x.size(), in_dim);
    EXPECT_EQ(y.size(), out_dim);

    // Parameters
    const double K = 1.0;
    const double m     = r.m_min;     // 1.2
    const double tau   = r.tau_min;   // 0.5
    const double rate  = r.r_min;     // 0.02
    const double sigma = r.sig_min;   // 0.3

    const double F = m * K;

    const double V = DataGenerator::black_call(F, K, rate, tau, sigma);
    const double intrinsic_disc = std::exp(-rate * tau) * std::max(F - K, 0.0);
    const double Vtilde = V - intrinsic_disc;

    // Check output y (use float-appropriate tolerance)
    EXPECT_NEAR(y[0], sigma, 1e-6);

    // Check input x = [log(Vtilde/K), F/K, tau, r] (use float-appropriate tolerance)
    EXPECT_NEAR(x[0], std::log(Vtilde / K), 1e-6);
    EXPECT_NEAR(x[1], F / K, 1e-6);
    EXPECT_NEAR(x[2], tau, 1e-6);
    EXPECT_NEAR(x[3], rate, 1e-6);
}