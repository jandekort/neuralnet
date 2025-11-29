#include "Network.h"
#include <fstream>

Network::Network(const std::vector<int> &sizes,
                 std::shared_ptr<const Activation> hidden_act,
                 std::shared_ptr<const Activation> output_act,
                 std::shared_ptr<const Cost> cost)
        : num_layers_(static_cast<int>(sizes.size())),
          sizes_(sizes),
          hidden_act_(std::move(hidden_act)),
          output_act_(std::move(output_act)),
          cost_(std::move(cost))
{
    if (num_layers_ < 2)
    {
        throw std::invalid_argument("Need at least 2 layers");
    }

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

        const Activation* act = (l == num_layers_ - 1) ? output_act_.get() : hidden_act_.get();

        MatrixXd w(sizes_[l], num_in);
        for (int r = 0; r < w.rows(); ++r)
        {
            for (int c = 0; c < w.cols(); ++c)
            {
                w(r, c) = act->initWeight(rng_, num_in);
            }
        }
        weights_.push_back(std::move(w));
    }

    // Initialize Adam moment vectors to zero with same shapes as weights/biases
    m_w_.resize(weights_.size());
    v_w_.resize(weights_.size());
    m_b_.resize(biases_.size());
    v_b_.resize(biases_.size());

    for (size_t l = 0; l < weights_.size(); ++l)
    {
        m_w_[l] = MatrixXd::Zero(weights_[l].rows(), weights_[l].cols());
        v_w_[l] = MatrixXd::Zero(weights_[l].rows(), weights_[l].cols());
        m_b_[l] = VectorXd::Zero(biases_[l].size());
        v_b_[l] = VectorXd::Zero(biases_[l].size());
    }

    adam_t_ = 0;
}


VectorXd Network::feedforward(const VectorXd& a0) const
{
#ifdef DEBUG
    // Validate input dimension (only in debug builds for performance)
    const int expected_input_dim = sizes_.front();
    if (a0.size() != expected_input_dim)
    {
        throw std::invalid_argument(
            "feedforward() input dimension mismatch: network expects " +
            std::to_string(expected_input_dim) + " inputs but got " +
            std::to_string(a0.size())
        );
    }
#endif

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

void Network::validate_dataset_dimensions(const std::vector<Example>& data, const std::string& dataset_name) const
{
    if (data.empty())
    {
        return; // Nothing to validate
    }

    const int expected_input_dim = sizes_.front();
    const int expected_output_dim = sizes_.back();

    for (size_t i = 0; i < data.size(); ++i)
    {
        const VectorXd& x = data[i].first;
        const VectorXd& y = data[i].second;

        if (x.size() != expected_input_dim)
        {
            throw std::invalid_argument(
                dataset_name + " input dimension mismatch at example " + std::to_string(i) +
                ": network expects " + std::to_string(expected_input_dim) +
                " inputs but got " + std::to_string(x.size())
            );
        }

        if (y.size() != expected_output_dim)
        {
            throw std::invalid_argument(
                dataset_name + " output dimension mismatch at example " + std::to_string(i) +
                ": network expects " + std::to_string(expected_output_dim) +
                " outputs but got " + std::to_string(y.size())
            );
        }
    }
}

void Network::SGD(std::vector<Example>& training_data, int epochs, int mini_batch_size, double eta, const std::vector<Example>* test_data,
                  EvalMode mode, double decay, Optimizer optimizer)
{
    validate_dataset_dimensions(training_data, "Training data");
    if (test_data)
    {
        validate_dataset_dimensions(*test_data, "Test data");
    }

    const int n = static_cast<int>(training_data.size());

    double current_eta = eta;

    // Checkpointing: track best evaluation and best network
    EvaluationResult best_eval;
    best_eval.mode = mode;
    Network best_net = *this; // copy current network as initial best

    for (int ep = 0; ep < epochs; ++ep) // every epoch loops through training data
    {
        rng_.shuffle(training_data.begin(), training_data.end());

        for (int k = 0; k < n; k += mini_batch_size) // update weights and biases on every mini-batch
        {
            int end = std::min(n, k + mini_batch_size);

            if (optimizer == Optimizer::SGD)
            {
                update_mini_batch(training_data.begin() + k,
                                  training_data.begin() + end,
                                  current_eta);
            }
            else // Optimizer::Adam
            {
                update_mini_batch_adam(training_data.begin() + k,
                                       training_data.begin() + end,
                                       current_eta);
            }
        }

        if (!test_data)
        {
            std::cout << "No test data for epoch " << ep << "\n";
            continue;
        }

        EvaluationResult er = evaluate(*test_data, mode);
        er.epoch = ep;

        switch (mode)
        {
            case EvalMode::Classification:
                std::cout << "Epoch " << ep << ": " << er.correct << " / " << er.total;
                break;
            case EvalMode::Regression:
                std::cout << "Epoch " << ep << ": " << "MSE=" << er.mse << ", MAE=" << er.mae;
                break;
            default:
                throw std::runtime_error("Unknown EvalMode in SGD");
        }

        // Check if current evaluation is better than best
        if (er < best_eval)
        {
            best_eval = er;
            best_net = *this; // save current network as best
            std::cout << " , New best";
        }

        std::cout << "\n";

        current_eta *= decay; // decay learning rate after finishing the epoch
    }

    // Restore best network
    *this = best_net;

    std::cout << "\nTraining complete. Best model from epoch " << best_eval.epoch << ":\n";
    switch (mode)
    {
        case EvalMode::Classification:
            std::cout << "  Accuracy: " << best_eval.correct << " / " << best_eval.total << "\n";
            break;
        case EvalMode::Regression:
            std::cout << "  MSE=" << best_eval.mse << ", MAE=" << best_eval.mae << "\n";
            break;
    }
}

void Network::update_mini_batch(std::vector<Example>::iterator begin, std::vector<Example>::iterator end, double eta)
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

    for (auto it = begin; it != end; ++it)
    {
        const VectorXd& x = it->first;
        const VectorXd& y = it->second;

        std::vector<VectorXd> dnb;
        std::vector<MatrixXd> dnw;

        std::tie(dnb, dnw) = backprop(x, y); // compute per-example gradients using backpropagation

        for (size_t l = 0; l < nabla_b.size(); ++l)
        {
            nabla_b[l] += dnb[l]; // accumulate gradients in mini batch
        }
        for (size_t l = 0; l < nabla_w.size(); ++l)
        {
            nabla_w[l] += dnw[l]; // accumulate gradients in mini batch
        }
    }

    double lr = eta / static_cast<double>(std::distance(begin, end));

    for (size_t l = 0; l < weights_.size(); ++l)
    {
        weights_[l] -= lr * nabla_w[l];
        biases_[l]  -= lr * nabla_b[l];

#ifdef DEBUG
        check_finite(weights_[l], "weights after update");
#endif
    }
}

void Network::update_mini_batch_adam(std::vector<Example>::iterator begin,
                                     std::vector<Example>::iterator end,
                                     double eta)
{
    // 1) Accumulate gradients over the mini-batch (exactly like update_mini_batch)
    std::vector<VectorXd> nabla_b(biases_.size());
    std::vector<MatrixXd> nabla_w(weights_.size());

    for (size_t l = 0; l < biases_.size(); ++l)
    {
        nabla_b[l] = VectorXd::Zero(biases_[l].size());
    }
    for (size_t l = 0; l < weights_.size(); ++l)
    {
        nabla_w[l] = MatrixXd::Zero(weights_[l].rows(), weights_[l].cols());
    }

    for (auto it = begin; it != end; ++it)
    {
        const VectorXd& x = it->first;
        const VectorXd& y = it->second;

        std::vector<VectorXd> dnb;
        std::vector<MatrixXd> dnw;
        std::tie(dnb, dnw) = backprop(x, y);

        for (size_t l = 0; l < nabla_b.size(); ++l)
        {
            nabla_b[l] += dnb[l];
        }

        for (size_t l = 0; l < nabla_w.size(); ++l)
        {
            nabla_w[l] += dnw[l];
        }
    }

    // 2) Compute average gradient for the batch
    const double batch_size = static_cast<double>(std::distance(begin, end));

    std::vector<VectorXd> g_b(nabla_b.size());
    std::vector<MatrixXd> g_w(nabla_w.size());

    for (size_t l = 0; l < nabla_b.size(); ++l)
    {
        g_b[l] = nabla_b[l] / batch_size;
    }

    for (size_t l = 0; l < nabla_w.size(); ++l)
    {
        g_w[l] = nabla_w[l] / batch_size;
    }

    // 3) Adam hyperparameters
    const double beta1 = 0.9;
    const double beta2 = 0.999;
    const double eps   = 1e-8;

    // 4) Increment timestep (global across training)
    ++adam_t_;
    const double beta1_t = std::pow(beta1, adam_t_);
    const double beta2_t = std::pow(beta2, adam_t_);

    // 5) Update moments and apply Adam step
    for (size_t l = 0; l < weights_.size(); ++l)
    {
        // weights moments
        m_w_[l] = beta1 * m_w_[l] + (1.0 - beta1) * g_w[l];
        v_w_[l] = beta2 * v_w_[l] + (1.0 - beta2) * g_w[l].cwiseProduct(g_w[l]);

        MatrixXd m_hat_w = m_w_[l] / (1.0 - beta1_t);
        MatrixXd v_hat_w = v_w_[l] / (1.0 - beta2_t);

        weights_[l].array() -= eta * m_hat_w.array() / (v_hat_w.array().sqrt() + eps);

        // biases moments
        m_b_[l] = beta1 * m_b_[l] + (1.0 - beta1) * g_b[l];
        v_b_[l] = beta2 * v_b_[l] + (1.0 - beta2) * g_b[l].cwiseProduct(g_b[l]);

        VectorXd m_hat_b = m_b_[l] / (1.0 - beta1_t);
        VectorXd v_hat_b = v_b_[l] / (1.0 - beta2_t);

        biases_[l].array()  -= eta * m_hat_b.array() / (v_hat_b.array().sqrt() + eps);
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

        const double out_dim = static_cast<double>(sizes_.back());
        res.mse = sum_sq / (dcount * out_dim); // overall mean of the sum of squared errors per element (per output)
        res.mae = sum_abs / dcount; // overall mean absolute error per element (per output)
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

    // Output layer delta (uses cost function's delta method)
    VectorXd delta = cost_->delta(zs.back(), activations.back(), y, *output_act_);
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

void Network::load(const std::string& path)
{
    std::ifstream in(path);
    if (!in)
    {
        throw std::runtime_error("Cannot open " + path);
    }

    int n_layers;
    in >> n_layers;
    std::vector<int> sizes(n_layers);
    for (int i = 0; i < n_layers; ++i)
    {
        in >> sizes[i];
    }

    weights_.clear();
    biases_.clear();

    // weights
    for (auto& W : weights_)
    {
        int rows, cols;
        in >> rows >> cols;
        if (rows != W.rows() || cols != W.cols())
        {
            throw std::runtime_error("Dimension mismatch in weight matrix");
        }
        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < cols; ++c)
            {
                in >> W(r,c);
            }
        }

    }

    // biases
    for (auto& b : biases_)
    {
        int n;
        in >> n;
        if (n != b.size())
        {
            throw std::runtime_error("Dimension mismatch in bias vector");
        }
        for (int i = 0; i < n; ++i)
        {
            in >> b(i);
        }
    }
}

void Network::save(const std::string& path)
{
    std::ofstream out(path);
    if (!out)
    {
        throw std::runtime_error("Cannot open " + path);
    }

    out << std::setprecision(17);

    // 1) architecture
    out << sizes_.size() << "\n";
    for (int s : sizes_)
    {
        out << s << " ";
    }
    out << "\n";

    // 2) weights
    for (const auto& W : weights_)
    {
        out << W.rows() << " " << W.cols() << "\n";
        for (int r = 0; r < W.rows(); ++r)
        {
            for (int c = 0; c < W.cols(); ++c)
            {
                out << W(r,c) << " ";
            }
            out << "\n";
        }
    }

    // 3) biases
    for (const auto& b : biases_)
    {
        out << b.size() << "\n";
        for (int i = 0; i < b.size(); ++i)
        {
            out << b(i) << " ";
        }
        out << "\n";
    }
}