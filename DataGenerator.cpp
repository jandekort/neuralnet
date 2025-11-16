#include "DataGenerator.h"

void DataGenerator::generate_file(const std::string &out_path, size_t n_validation, size_t n_test, uint32_t in_dim, uint32_t out_dim)
{
    BinaryFileWriter w(out_path);

    // File header
    w.write_tag("EXDS");       // magic
    w.write_be_u32(2u);        // 2 sections (VALD + TEST)

    auto val = make_examples(n_validation, in_dim, out_dim);
    auto tst = make_examples(n_test,       in_dim, out_dim);

    write_section(w, "VALD", val, in_dim, out_dim);
    write_section(w, "TEST", tst, in_dim, out_dim);
}

double DataGenerator::norm_cdf(double x)
{
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

double DataGenerator::black_call(double F, double K, double r, double tau, double sigma)
{
    if (sigma <= 0.0 || tau <= 0.0)
    {
        return std::exp(-r * tau) * std::max(F - K, 0.0);
    }
    double st = sigma * std::sqrt(tau);
    double d1 = (std::log(F/K) + 0.5*sigma*sigma*tau) / st;
    double d2 = d1 - st;
    return std::exp(-r*tau) * (F * norm_cdf(d1) - K * norm_cdf(d2));
}

std::vector<Example> DataGenerator::make_examples(size_t n, uint32_t in_dim, uint32_t out_dim)
{
    std::uniform_real_distribution<double> U_m(ranges_.m_min, ranges_.m_max);
    std::uniform_real_distribution<double> U_tau(ranges_.tau_min, ranges_.tau_max);
    std::uniform_real_distribution<double> U_r(ranges_.r_min, ranges_.r_max);
    std::uniform_real_distribution<double> U_sig(ranges_.sig_min, ranges_.sig_max);

    const double K = 1.0;
    const double eps_tv = 1e-12;

    std::vector<Example> out;
    out.reserve(n);

    while (out.size() < n)
    {
        double m     = U_m(rng_);
        double tau   = U_tau(rng_);
        double r     = U_r(rng_);
        double sigma = U_sig(rng_);

        double F = m * K;
        double V = black_call(F, K, r, tau, sigma);

        double intrinsic_disc = std::exp(-r*tau) * std::max(F - K, 0.0);
        double Vtilde = V - intrinsic_disc;
        if (Vtilde <= eps_tv)
        {
            continue;
        }

        VectorXd x(in_dim);
        x[0] = std::log(Vtilde / K);
        x[1] = F / K;
        x[2] = tau;

        VectorXd y(out_dim);
        y[0] = sigma;

        out.emplace_back(std::move(x), std::move(y));
    }
    return out;
}

void DataGenerator::write_section(BinaryFileWriter &w, const char *tag, const std::vector<Example> &data, uint32_t in_dim, uint32_t out_dim)
{
    w.write_tag(tag);
    w.write_be_u32(static_cast<uint32_t>(data.size()));
    w.write_be_u32(in_dim);
    w.write_be_u32(out_dim);

    for (const auto& ex : data)
    {
        const VectorXd& x = ex.first;
        const VectorXd& y = ex.second;
        for (int i = 0; i < x.size(); ++i)
        {
            w.write_double(x[i]);
        }
        for (int i = 0; i < y.size(); ++i)
        {
            w.write_double(y[i]);
        }
    }
}
