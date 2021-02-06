cmake_minimum_required(VERSION 3.16)

project(autograd)

add_library(ag INTERFACE)

target_include_directories(ag
    INTERFACE
    ${PROJECT_SOURCE_DIR}
    ext
)

add_executable(ag_tests main_tests.cpp ag_tests.cpp reqs_tests.cpp)
target_link_libraries(ag_tests ag)
set_target_properties(ag_tests PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_CLANG_TIDY clang-tidy
)

add_executable(mlc mlc.cpp)
target_link_libraries(mlc ag)
set_target_properties(mlc PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_CLANG_TIDY clang-tidy
)
