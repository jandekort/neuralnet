cmake_minimum_required(VERSION 3.29)
project(NeuralNetwork LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_PREFIX_PATH "/opt/homebrew/share/eigen3/cmake")

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(NeuralNetwork main.cpp
        Network.h
        Activation.h
        IDataReader.h)

target_link_libraries(NeuralNetwork PRIVATE Eigen3::Eigen)