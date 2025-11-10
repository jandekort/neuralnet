cmake_minimum_required(VERSION 3.29)
project(NeuralNet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Eigen (Homebrew install assumed)
set(CMAKE_PREFIX_PATH "/opt/homebrew/share/eigen3/cmake")
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# --------------------- Library ---------------------
add_library(NeuralNet_Lib
        Network.cpp
        Network.h
        Activation.h
        IDataReader.h
        #Network.cpp
        #Activation.cpp
        #IDataReader.cpp
)

target_link_libraries(NeuralNet_Lib
        PUBLIC Eigen3::Eigen
)

# --------------------- Main Executable ---------------------
add_executable(NeuralNet
        main.cpp
        Sigmoid.cpp
        Sigmoid.h
        MnistReader.cpp
        MnistReader.h
)
target_link_libraries(NeuralNet
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
        TestSuite.cpp
)

target_link_libraries(NeuralNet_Test
        PRIVATE
        NeuralNet_Lib
        gtest_main
)

include(GoogleTest)
gtest_discover_tests(NeuralNet_Test)
