#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "VectorXdPairReader.h"
#include "Sigmoid.h"
#include "ReLU.h"
#include "Linear.h"
#include "Network.h"
#include "DataNormalizer.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

void main_black_train()
{
    const std::string dataset_path = "GeneratedData.bin";

    // Hyperparameters
    int    epochs           = 150;
    int    mini_batch_size  = 1024;
    double eta              = 0.002; // chosen as the largest learning rate that remains numerically stable (no NaNs) and yields consistently decreasing loss
    double decay            = 0.999;

    // Setup activation function
    //auto sigmoid = std::make_shared<Sigmoid>();
    auto relu = std::make_shared<ReLU>();
    auto linear = std::make_shared<Linear>();

    // Load the dataset
    VectorXdPairReader reader(dataset_path);
    reader.load();

    const auto& train_data = reader.training();
    const auto& test_data  = reader.test();

    std::cout << "Loaded dataset from " << dataset_path << "\n";
    std::cout << "  Training size: " << train_data.size() << "\n";
    std::cout << "  Test size:     " << test_data.size() << "\n";

    if (train_data.empty() || test_data.empty())
    {
        throw std::runtime_error("Training or test set is empty – run MainGenerate first.");
    }

    // Show a preview example
    const auto& ex0 = train_data[0];
    const VectorXd& x0 = ex0.first;
    const VectorXd& y0 = ex0.second;

    std::cout << "\nExample training input x (log(Vtilde/K), F/K, tau):\n";
    reader.print(x0, 0, /*rows*/ 1, /*cols*/ x0.size());

    std::cout << "Example training target y (sigma):\n";
    reader.print(y0, 0, /*rows*/ 1, /*cols*/ y0.size());

    // Copy training data into a modifiable vector for SGD (SGD method makes in-place changes)
    std::vector<Network::Example> train(train_data.begin(), train_data.end());
    std::vector<Network::Example> test(test_data.begin(), test_data.end());

    StandardScalerNormalizer normalizer;
    normalizer.fit(train); // compute means/stds from training set
    normalizer.transform_dataset_inplace(train); // normalize training data
    normalizer.transform_dataset_inplace(test);  // normalize test data

    // Initialise network
    Network net({3, 300, 300, 300, 1}, relu, linear);

    // Train and evaluate in regression mode on NORMALIZED data
    std::cout << "\nStarting training...\n";
    net.SGD(train, epochs, mini_batch_size, eta, &test, EvalMode::Regression, decay, Optimizer::Adam);

    // Print network parameters
    //net.print();

    // Save network parameters
    net.save("net.txt");

    // Evaluate a few test examples (denormalize predictions + targets)
    std::cout << "\nSample predictions on test data:\n";
    for (size_t i = 0; i < std::min<size_t>(100, test.size()); ++i)
    {
        const auto& ex_norm = test[i];
        const VectorXd& x_norm     = ex_norm.first;
        const VectorXd& y_true_norm = ex_norm.second;

        // Predict in normalized space
        VectorXd y_pred_norm = net.feedforward(x_norm);

        // Denormalize both prediction and target for reporting
        VectorXd y_pred_raw = normalizer.inverse_transform_output(y_pred_norm);
        VectorXd y_true_raw = normalizer.inverse_transform_output(y_true_norm);

        std::cout << "Test example " << i;
        std::cout << ": true sigma = " << y_true_raw[0] ;
        std::cout << ", predicted sigma = " << y_pred_raw[0] ;
        std::cout << ", abs error = " << std::abs(y_pred_raw[0] - y_true_raw[0]) << "\n";
    }
}

int main()
{
    try
    {
        main_black_train();
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
