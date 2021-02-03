#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <array>
#include <cassert>
#include "types.hpp"

template <int N, int M>
struct Matrix {
    constexpr Matrix(std::initializer_list<Float> initial) : value_(initial) { assert(initial.size() == N * M); }

    auto at(int n, int m) const { return value_.at(n, m); }

private:
    std::array<Float, N * M> value_;
};

#endif
