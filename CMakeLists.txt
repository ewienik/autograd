cmake_minimum_required(VERSION 3.16)

project(autograd)

add_executable(autograd main.cpp tests.cpp)

set_target_properties(autograd PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_CLANG_TIDY clang-tidy
)
