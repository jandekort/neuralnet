cmake_minimum_required(VERSION 3.29)
project(NeuralNet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Eigen (Homebrew install assumed)
set(CMAKE_PREFIX_PATH "/opt/homebrew/share/eigen3/cmake")
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# OpenMP for Eigen parallelization
# On macOS with Homebrew, libomp is keg-only, so we need to set paths explicitly
if(APPLE)
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found - enabling Eigen parallelization")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(WARNING "OpenMP not found - Eigen will run single-threaded")
endif()

# Compiler optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -ffast-math")

# --------------------- Library ---------------------
add_library(NeuralNet_Lib
        Network.cpp
        Network.h
        Activation.h
        Cost.h
        Cost.cpp
        IDataReader.h
        Sigmoid.h
        MnistReader.cpp
        MnistReader.h
        BinaryFileWriter.h
        DataGenerator.h
        VectorXdPairReader.h
        VectorXdPairReader.cpp
        DataGenerator.cpp
        ReLU.h
        Linear.h
        DataNormalizer.cpp
        DataNormalizer.h
        RNG.h
)

target_link_libraries(NeuralNet_Lib
        PUBLIC Eigen3::Eigen
)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(NeuralNet_Lib PUBLIC OpenMP::OpenMP_CXX)
endif()

# --------------------- Main Executables ---------------------
# Each of these has its own main() in its respective .cpp file.

add_executable(MainMnist
        MainMnist.cpp
)

target_link_libraries(MainMnist
        PRIVATE NeuralNet_Lib
)

add_executable(MainGenerate
        MainGenerate.cpp
)

target_link_libraries(MainGenerate
        PRIVATE NeuralNet_Lib
)

add_executable(MainTrain
        MainTrain.cpp
)

target_link_libraries(MainTrain
        PRIVATE NeuralNet_Lib
)

# --------------------- GoogleTest ---------------------
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()

# --------------------- Tests ---------------------
add_executable(NeuralNet_Test
        TestSuite_BinaryFileWriter.cpp
        TestSuite_Mnist.cpp
        TestSuite_Black.cpp
        TestSuite.cpp
)

target_link_libraries(NeuralNet_Test
        PRIVATE
        NeuralNet_Lib
        gtest_main
)

include(GoogleTest)
gtest_discover_tests(NeuralNet_Test)
