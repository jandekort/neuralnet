
#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <filesystem>

#include "MnistReader.h"
#include "Sigmoid.h"
#include "Network.h"

using Eigen::MatrixXf;
using Eigen::VectorXf;


void main_mnist()
{
    std::filesystem::path cwd = std::filesystem::current_path().parent_path();
    std::string train_images = (cwd / "train-images-idx3-ubyte").string();
    std::string train_labels = (cwd / "train-labels-idx1-ubyte").string();
    std::string test_images  = (cwd / "t10k-images-idx3-ubyte").string();
    std::string test_labels  = (cwd / "t10k-labels-idx1-ubyte").string();

    int epochs = 30;
    int mini_batch_size = 10;
    double eta = 3.0;

    // Setup
    auto act = std::make_shared<Sigmoid>();
    MnistReader reader(train_images, train_labels, test_images, test_labels);
    reader.load();

    // Show examples
    const auto& trX = reader.training()[4]; // 5th training example
    int label = 0;
    trX.second.maxCoeff(&label);
    reader.print(trX.first, label);

    const auto& teX = reader.test()[90]; // 91st test example
    teX.second.maxCoeff(&label);
    reader.print(teX.first, label);

    // Initialise network
    Network net({784, 30, 10}, act);
    auto train = reader.training();

    // Train and evaluate
    net.SGD(train, epochs, mini_batch_size, eta, &reader.test());

    // Apply trained network
    std::cout << argmax(net.feedforward(trX.first)) << std::endl;
    std::cout << argmax(net.feedforward(teX.first)) << std::endl;
}


int main()
{
    main_mnist();
    return 0;
}
