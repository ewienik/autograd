#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <array>
#include <cassert>
#include "types.hpp"

template <int N, int M>
struct Value {
    static_assert(N > 0 && M > 0);

    constexpr Value() = default;
    template <typename... Args>
    constexpr Value(Args&&... args) : data_{std::forward<Args>(args)...} {
        static_assert(sizeof...(args) == N * M);
    }

    [[nodiscard]] auto item() -> Float& {
        static_assert(N == 1 && M == 1, "item() method is only for scalars");
        return data_[0];
    }
    [[nodiscard]] auto item() const -> Float const& {
        static_assert(N == 1 && M == 1, "item() method is only for scalars");
        return data_[0];
    }

    [[nodiscard]] auto item(int n) -> Float& {
        static_assert(M == 1, "item() method is only for vectors");
        assert(n >= 0 && n < N);
        return data_[n];
    }
    [[nodiscard]] auto item(int n) const -> Float const& {
        static_assert(M == 1, "item() method is only for vectors");
        assert(n >= 0 && n < N);
        return data_[n];
    }

    auto item(int n, int m) -> Float& {
        assert(n >= 0 && n < N && m >= 0 && m < M);
        return data_[m * N + n];
    }
    [[nodiscard]] auto item(int n, int m) const -> Float const& {
        assert(n >= 0 && n < N && m >= 0 && m < M);
        return data_[m * N + n];
    }

private:
    std::array<Float, N * M> data_{0.F};
};

#endif
