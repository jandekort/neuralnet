
#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <filesystem>

#include "Sigmoid.h"
#include "Network.h"
//#include "IDataReader.h"
#include "MnistReader.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;


int main()
{
    try
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

        // Show the 5th training example as a sanity check
        const auto& ex = reader.training()[4];
        int label = 0;
        ex.second.maxCoeff(&label);
        reader.print(ex.first, label);

        // Show the 91st test example as a sanity check
        auto [img, lbl] = reader.test()[90];
        reader.print(img, lbl);

        Network net({784, 30, 10}, act);
        auto train = reader.training();

        // Train and evaluate
        net.SGD(train, epochs, mini_batch_size, eta, &reader.test());


        std::cout << argmax(net.feedforward(ex.first)) << std::endl;
        std::cout << argmax(net.feedforward(img)) << std::endl;

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }
}
