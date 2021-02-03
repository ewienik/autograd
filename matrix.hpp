#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <array>
#include <cassert>
#include "types.hpp"

template <int N, int M>
struct Matrix {
    constexpr Matrix() = default;
    template <typename... Args>
    constexpr Matrix(Args&&... args) : value_{std::forward<Args>(args)...} {
        static_assert(sizeof...(args) == N * M);
    }

    auto at(int n, int m) -> Float& { return value_.at(n * N + m); }
    [[nodiscard]] auto at(int n, int m) const -> Float const& { return value_.at(n * N + m); }

private:
    std::array<Float, N * M> value_{0.F};
};

#endif
