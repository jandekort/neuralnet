#include "Network.h"

RNG RNG_GLOBAL;

Network::Network(const std::vector<int> &sizes, std::shared_ptr<const Activation> act, bool use_scaled_weight_init)
    : num_layers_(static_cast<int>(sizes.size())),
      sizes_(sizes),
      act_(std::move(act)),
      use_scaled_weight_init_(use_scaled_weight_init)
{
    if (num_layers_ < 2) throw std::invalid_argument("Need at least 2 layers");

    // biases (no bias for input layer)
    for (int l = 1; l < num_layers_; ++l)
    {
        VectorXd b(sizes_[l]);
        for (int i = 0; i < b.size(); ++i)
        {
            b(i) = RNG_GLOBAL.gauss();
        }
        biases_.push_back(std::move(b));
    }
    // weights
    for (int l = 1; l < num_layers_; ++l)
    {
        const int num_in = sizes_[l - 1];
        const double scale = use_scaled_weight_init_ ? (1.0 / std::sqrt(static_cast<double>(num_in))) : 1.0;

        MatrixXd w(sizes_[l], num_in);
        for (int r = 0; r < w.rows(); ++r)
        {
            for (int c = 0; c < w.cols(); ++c)
            {
                w(r, c) = RNG_GLOBAL.gauss() * scale;
            }
        }
        weights_.push_back(std::move(w));
    }
}


VectorXd Network::feedforward(const VectorXd& a0) const
{
    VectorXd a = a0;
    for (size_t l = 0; l < weights_.size(); ++l)
    {
        VectorXd z = weights_[l] * a + biases_[l]; // feedforward one layer
        a = act_->f(z); // update all activations for layer l
    }
    return a;
}

void Network::SGD(std::vector<Example>& training_data,
         int epochs, int mini_batch_size, double eta,
         const std::vector<std::pair<VectorXd,int>>* test_data)
{
    const int n = static_cast<int>(training_data.size());
    for (int ep = 0; ep < epochs; ++ep)
    {
        RNG_GLOBAL.shuffle(training_data.begin(), training_data.end());
        for (int k = 0; k < n; k += mini_batch_size)
        {
            int end = std::min(n, k + mini_batch_size);
            update_mini_batch(training_data.begin()+k, training_data.begin()+end, eta);
        }
        if (test_data)
        {
            int correct = evaluate(*test_data);
            std::cout << "Epoch " << ep << ": " << correct << " / " << test_data->size() << "\n";
        }
        else
        {
            std::cout << "Epoch " << ep << " complete\n";
        }
    }
}

int Network::evaluate(const std::vector<std::pair<VectorXd,int>>& test_data) const
{
    int correct = 0;
    for (const auto& [x, y_label] : test_data)
    {
        if (argmax(feedforward(x)) == y_label)
        {
            ++correct;
        }
    }
    return correct;
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
    for (size_t l = 0; l < weights_.size(); ++l)
    {
        VectorXd z = weights_[l] * a + biases_[l];
        zs.push_back(z);
        a = act_->f(z);
        activations.push_back(a);
    }

    VectorXd delta = (activations.back() - y).cwiseProduct(act_->df(zs.back()));
    nabla_b.back() = delta;
    nabla_w.back() = delta * activations[activations.size()-2].transpose();

    for (int l = static_cast<int>(weights_.size()) - 2; l >= 0; --l)
    {
        VectorXd sp = act_->df(zs[l]);
        delta = (weights_[l+1].transpose() * delta).cwiseProduct(sp);
        nabla_b[l] = delta;
        nabla_w[l] = delta * activations[l].transpose();
    }

    return {std::move(nabla_b), std::move(nabla_w)};
}