
#include <Eigen/Dense>
#include <iostream>
#include <stdexcept>
#include <string>
#include "DataGenerator.h"
#include "VectorXdPairReader.h"


using Eigen::MatrixXd;
using Eigen::VectorXd;


void main_generate()
{
    const std::string out_path = "GeneratedData.bin";
    const size_t n_train = 9e5;
    const size_t n_test  = 1e5;
    const uint64_t seed = 1;

    // Create training and test data using the generator
    DataGenerator gen(seed);
    gen.generate_file(out_path, n_train, n_test);

    std::cout << "Wrote " << out_path
              << " with " << n_train << " training examples and "
              << n_test << " test examples.\n";

    // Sanity check: read back using VectorXdPairReader
    try
    {
        VectorXdPairReader reader(out_path);
        reader.load();

        const auto& vald = reader.training();
        std::cout << "Loaded " << vald.size() << " validation examples.\n";

        if (!vald.empty()) {
            const auto& ex = vald.front();
            const auto& x = ex.first;
            const auto& y = ex.second;

            std::cout << "\n=== First Validation Example ===\n";

            std::cout << "Input vector (x):\n";
            reader.print(x, /*label (ignored)*/ 0, /*rows*/ 1, /*cols*/ x.size());

            std::cout << "\nOutput vector (y):\n";
            reader.print(y, /*label (ignored)*/ 0, /*rows*/ 1, /*cols*/ y.size());
        }

    }
    catch (const std::exception& ex)
    {
        std::cerr << "Error while reading back " << out_path << ": " << ex.what() << "\n";
    }
}


int main()
{
    main_generate();
    return 0;
}
