#include "Network.h"

Network::Network(const std::vector<int> &sizes,
                 std::shared_ptr<const Activation> hidden_act,
                 std::shared_ptr<const Activation> output_act )
        : num_layers_(static_cast<int>(sizes.size())),
          sizes_(sizes),
          hidden_act_(std::move(hidden_act)),
          output_act_(std::move(output_act))
{
    if (num_layers_ < 2) throw std::invalid_argument("Need at least 2 layers");

    if (!output_act_)
    {
        output_act_ = hidden_act_; // by default, use same activation for output as hidden
    }

    // biases (no bias for input layer)
    for (int l = 1; l < num_layers_; ++l)
    {
        VectorXd b(sizes_[l]);
        for (int i = 0; i < b.size(); ++i)
        {
            b(i) = rng_.gauss();
        }
        biases_.push_back(std::move(b));
    }
    // weights
    for (int l = 1; l < num_layers_; ++l)
    {
        const int num_in = sizes_[l - 1];

        double scale = 1.0;
        if(l == num_layers_ - 1)
        {
            scale = output_act_->scaledWeight(num_in);
        }
        else
        {
            scale = hidden_act_->scaledWeight(num_in);
        }

        MatrixXd w(sizes_[l], num_in);
        for (int r = 0; r < w.rows(); ++r)
        {
            for (int c = 0; c < w.cols(); ++c)
            {
                w(r, c) = rng_.gauss() * scale;
            }
        }
        weights_.push_back(std::move(w));
    }
}


VectorXd Network::feedforward(const VectorXd& a0) const
{
    VectorXd a = a0;
    const size_t L = weights_.size();

    for (size_t l = 0; l < L; ++l)
    {
        VectorXd z = weights_[l] * a + biases_[l];

#ifdef DEBUG
        check_finite(z, "feedforward() z pre-activation");
#endif

        if (l + 1 == L)
        {
            a = output_act_->f(z); // output layer
        }
        else
        {
            a = hidden_act_->f(z); // hidden layer
        }

#ifdef DEBUG
        check_finite(a, "feedforward() activation output");
#endif
    }
    return a;
}

void Network::SGD(std::vector<Example>& training_data, int epochs, int mini_batch_size, double eta, const std::vector<Example>* test_data,
                  EvalMode mode, double decay)
{
    const int n = static_cast<int>(training_data.size());

    double current_eta = eta;

    for (int ep = 0; ep < epochs; ++ep) // every epoch loops through training data
    {
        rng_.shuffle(training_data.begin(), training_data.end());

        for (int k = 0; k < n; k += mini_batch_size) // update weights and biases on every mini-batch
        {
            int end = std::min(n, k + mini_batch_size);
            update_mini_batch(training_data.begin() + k, training_data.begin() + end, current_eta);
        }

        if (!test_data)
        {
            std::cout << "No test data for epoch " << ep << "\n";
            continue;
        }

        EvaluationResult er = evaluate(*test_data, mode);
        switch (mode)
        {
            case EvalMode::Classification:
                std::cout << "Epoch " << ep << ": " << er.correct << " / " << er.total << "\n";
                break;
            case EvalMode::Regression:
                std::cout << "Epoch " << ep << ": " << "MSE=" << er.mse << ", MAE=" << er.mae << "\n";
                break;
            default:
                throw std::runtime_error("Unknown EvalMode in SGD");
        }

        current_eta *= decay; // decay learning rate after finishing the epoch
    }
}

EvaluationResult Network::evaluate(const std::vector<Example>& test_data, EvalMode mode) const
{
    const std::size_t count = test_data.size();
    const double dcount = static_cast<double>(count);

    EvaluationResult res;
    res.mode  = mode;
    res.total = static_cast<int>(count);

    if (test_data.empty())
    {
        return res; // everything stays at 0
    }

    if (mode == EvalMode::Classification)
    {
        int correct = 0;
        for (const auto& [x, y_label] : test_data)
        {
            if (argmax(feedforward(x)) == argmax(y_label))
            {
                ++correct;
            }
        }
        res.correct = correct;
    }
    else if(mode == EvalMode::Regression)
    {
        double sum_sq = 0.0;
        double sum_abs = 0.0;

        for (const auto& [x, y_true] : test_data)
        {
            VectorXd y_pred = feedforward(x);
            VectorXd diff   = y_pred - y_true;

            sum_sq  += diff.squaredNorm(); // total L2 norm over output vec
            sum_abs += diff.cwiseAbs().mean(); // average abs per output vec
        }

        res.mse = sum_sq / dcount;
        res.mae = sum_abs / dcount;
    }
    else
    {
        throw std::runtime_error("Unknown EvalMode");
    }

    return res;
}

std::pair<std::vector<VectorXd>, std::vector<MatrixXd>> Network::backprop(const VectorXd& x, const VectorXd& y) const
{
    std::vector<VectorXd> nabla_b(biases_.size());
    std::vector<MatrixXd> nabla_w(weights_.size());
    for (size_t l = 0; l < nabla_b.size(); ++l)
    {
        nabla_b[l] = VectorXd::Zero(biases_[l].size());
    }
    for (size_t l = 0; l < nabla_w.size(); ++l)
    {
        nabla_w[l] = MatrixXd::Zero(weights_[l].rows(), weights_[l].cols());
    }

    std::vector<VectorXd> activations; activations.reserve(num_layers_);
    std::vector<VectorXd> zs;          zs.reserve(num_layers_-1);

    VectorXd a = x;
    activations.push_back(a);
    const size_t L = weights_.size();
    for (size_t l = 0; l < L; ++l)
    {
        VectorXd z = weights_[l] * a + biases_[l];
        zs.push_back(z);

#ifdef DEBUG
        check_finite(z, "backprop z");
#endif

        if (l + 1 == L)
        {
            a = output_act_->f(z);
        }
        else
        {
            a = hidden_act_->f(z);
        }

#ifdef DEBUG
        check_finite(a, "backprop activation");
#endif

        activations.push_back(a);
    }

    // Output layer delta
    VectorXd delta = (activations.back() - y).cwiseProduct(output_act_->df(zs.back()));
    nabla_b.back() = delta;
    nabla_w.back() = delta * activations[activations.size()-2].transpose();

    // Hidden layers (L-2 down to 0)
    for (int l = static_cast<int>(L) - 2; l >= 0; --l)
    {
        VectorXd sp = hidden_act_->df(zs[l]);
        delta = (weights_[l+1].transpose() * delta).cwiseProduct(sp);
        nabla_b[l] = delta;
        nabla_w[l] = delta * activations[l].transpose();
    }

    return {std::move(nabla_b), std::move(nabla_w)};
}

void Network::print() const
{
    std::cout << "\n=== Network Parameters ===\n";
    for (size_t l = 0; l < weights_.size(); ++l)
    {
        std::cout << "Layer " << l + 1 << ":\n";

        std::cout << "  Weights (" << weights_[l].rows() << " x "
                  << weights_[l].cols() << "):\n";
        std::cout << weights_[l] << "\n";

        std::cout << "  Biases (" << biases_[l].size() << "):\n";
        std::cout << biases_[l].transpose() << "\n\n";
    }
    std::cout << "==========================\n";
}

std::vector<VectorXd>& Network::biases()
{
    return biases_;
}

std::vector<MatrixXd>& Network::weights()
{
    return weights_;
}

const std::vector<VectorXd>& Network::biases() const
{
    return biases_;
}

const std::vector<MatrixXd>& Network::weights() const
{
    return weights_;
}