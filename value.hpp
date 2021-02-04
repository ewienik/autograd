#ifndef VALUE_HPP
#define VALUE_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include "types.hpp"

template <typename Operation>
concept ValueTransformOperation = requires(Operation op) {
    { op(1, 1) }
    ->std::convertible_to<Float>;
};

template <int N, int M>
struct Value {
    static_assert(N > 0 && M > 0);

    using ValueT = Value<N, M>;

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

    auto fill(Float value) { data_.fill(value); }

    template <ValueTransformOperation Operation>
    auto transform(Operation&& operation) {
        std::transform(
            std::begin(data_), std::end(data_), std::begin(data_),
            [start = data_.data(), operation = std::forward<Operation>(operation)](auto& item) {
                auto distance = &item - start;
                return operation(distance % N, distance / N);
            });
    }

    static auto one() {
        auto value = ValueT{};
        value.fill(1.F);
        return value;
    }

private:
    std::array<Float, N * M> data_{0.F};
};

#endif
