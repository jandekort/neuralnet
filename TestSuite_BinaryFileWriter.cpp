#include <gtest/gtest.h>

#include <cstdio>     // for std::remove
#include <string>

#include "BinaryFileWriter.h"
#include "VectorXdPairReader.h"

using Eigen::VectorXf;
using Example = std::pair<VectorXf, VectorXf>;


TEST(BinaryFileWriterReaderTest, WriteThenReadBack)
{
    std::string filename = "test_temp_dataset.bin";

    {
        BinaryFileWriter w(filename);

        // File header (just like the generator would write)
        w.write_tag("EXDS");
        w.write_be_u32(2u);   // two sections: train and test

        // --- Section 1: VALD ---
        w.write_tag("TRAI");
        w.write_be_u32(1u); // 1 example
        w.write_be_u32(3u); // input dimension
        w.write_be_u32(1u); // output dimension

        // Example: x = [1.0, 2.0, 3.0], y = [0.5]
        w.write_double(1.0);
        w.write_double(2.0);
        w.write_double(3.0);
        w.write_double(0.5);

        // --- Section 2: TEST ---
        w.write_tag("TEST");
        w.write_be_u32(1u); // 1 example
        w.write_be_u32(3u); // input dimension
        w.write_be_u32(1u); // output dimension

        // Example: x = [9.0, 8.0, 7.0], y = [0.1]
        w.write_double(9.0);
        w.write_double(8.0);
        w.write_double(7.0);
        w.write_double(0.1);
    }

    // ----- 3) Read file using VectorXdPairReader -----
    VectorXdPairReader reader(filename);
    reader.load();

    const auto& vald = reader.training();
    const auto& test = reader.test();

    // ----- 4) Assertions -----

    ASSERT_EQ(vald.size(), 1);
    ASSERT_EQ(test.size(), 1);

    // Validate VALD example
    {
        const Example& ex = vald[0];
        const VectorXf& x = ex.first;
        const VectorXf& y = ex.second;

        ASSERT_EQ(x.size(), 3);
        ASSERT_EQ(y.size(), 1);

        EXPECT_FLOAT_EQ(x[0], 1.0f);
        EXPECT_FLOAT_EQ(x[1], 2.0f);
        EXPECT_FLOAT_EQ(x[2], 3.0f);
        EXPECT_FLOAT_EQ(y[0], 0.5f);
    }

    // Validate TEST example
    {
        const Example& ex = test[0];
        const VectorXf& x = ex.first;
        const VectorXf& y = ex.second;

        ASSERT_EQ(x.size(), 3);
        ASSERT_EQ(y.size(), 1);

        EXPECT_FLOAT_EQ(x[0], 9.0f);
        EXPECT_FLOAT_EQ(x[1], 8.0f);
        EXPECT_FLOAT_EQ(x[2], 7.0f);
        EXPECT_FLOAT_EQ(y[0], 0.1f);
    }

    // ----- 5) Cleanup -----
    std::remove(filename.c_str());
}

