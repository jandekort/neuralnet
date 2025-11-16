#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "VectorXdPairReader.h"
#include "Sigmoid.h"
#include "ReLU.h"
#include "Network.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

void main_black_train()
{
    const std::string dataset_path = "GeneratedData.bin";

    // Hyperparameters
    int    epochs          = 30;
    int    mini_batch_size = 10;
    double eta             = 1.0;

    // Setup activation function
    auto act = std::make_shared<Sigmoid>();
    //auto act = std::make_shared<ReLU>();

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

    // Initialise network
    Network net({3, 40, 40, 1}, act);

    // Copy training data into a modifiable vector for SGD (SGD method makes in-place changes)
    std::vector<Network::Example> train(train_data.begin(), train_data.end());

    // Train and evaluate in regression mode
    std::cout << "\nStarting training...\n";
    net.SGD(train, epochs, mini_batch_size, eta, &test_data, EvalMode::Regression);

    // Print network parameters
    net.print();

    // Evaluate a few test examples
    std::cout << "\nSample predictions on test data:\n";
    for (size_t i = 0; i < std::min<size_t>(5, test_data.size()); ++i)
    {
        const auto& ex = test_data[i];
        const VectorXd& x = ex.first;
        const VectorXd& y_true = ex.second;

        VectorXd y_pred = net.feedforward(x);

        std::cout << "Test example " << i << ":\n";
        std::cout << "  true sigma       = " << y_true[0] << "\n";
        std::cout << "  predicted sigma  = " << y_pred[0] << "\n";
        std::cout << "  abs error        = " << std::abs(y_pred[0] - y_true[0]) << "\n\n";
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
