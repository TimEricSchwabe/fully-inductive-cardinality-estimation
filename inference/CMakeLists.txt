cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(fice_inference CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find libtorch — set Torch_DIR or CMAKE_PREFIX_PATH to your libtorch install.
# Example: cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
find_package(Torch REQUIRED)

add_executable(fice_inference example_inference.cpp)
target_link_libraries(fice_inference "${TORCH_LIBRARIES}")

