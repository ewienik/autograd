#ifndef AG_VALUE_HPP
#define AG_VALUE_HPP

#include <ag/types.hpp>
#include <algorithm>
#include <array>
#include <ext/gsl/gsl>

namespace ag {

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
    constexpr Value(Args&&... args) : data_{std::forward<Args>(args)...} {}

    [[nodiscard]] auto item() -> Float& {
        static_assert(N == 1 && M == 1, "item() method is only for scalars");
        return data_[0];
    }
    [[nodiscard]] auto item() const -> Float const& {
        static_assert(N == 1 && M == 1, "item() method is only for scalars");
        return data_[0];
    }

    [[nodiscard]] auto item(int n) -> Float& {
        static_assert(N == 1 || M == 1, "item() method is only for vectors");
        Expects(n >= 0 && n < std::max(N, M));
        return data_[n];
    }
    [[nodiscard]] auto item(int n) const -> Float const& {
        static_assert(N == 1 || M == 1, "item() method is only for vectors");
        Expects(n >= 0 && n < std::max(N, M));
        return data_[n];
    }

    auto item(int n, int m) -> Float& {
        Expects(n >= 0 && n < N && m >= 0 && m < M);
        return data_[n * M + m];
    }
    [[nodiscard]] auto item(int n, int m) const -> Float const& {
        Expects(n >= 0 && n < N && m >= 0 && m < M);
        return data_[n * M + m];
    }

    auto fill(Float value) { data_.fill(value); }

    template <ValueTransformOperation Operation>
    auto transform(Operation&& operation) {
        std::transform(
            std::begin(data_), std::end(data_), std::begin(data_),
            [start = data_.data(), operation = std::forward<Operation>(operation)](auto& item) {
                auto distance = &item - start;
                return operation(distance / M, distance % M);
            });
    }

    static auto ones() {
        auto value = ValueT{};
        value.fill(1.F);
        return value;
    }

    static auto zeros() { return ValueT{}; }

    auto begin() { return data_.begin(); }
    auto begin() const { return data_.begin(); }
    auto end() { return data_.end(); }
    auto end() const { return data_.end(); }

private:
    std::array<Float, N * M> data_{0.F};
};

template <int N, int M>
auto transpose(ag::Value<N, M> const& src) {
    ag::Value<M, N> dst;
    dst.transform([&src](auto n, auto m) { return src.item(m, n); });
    return dst;
}

template <int N, int M, int O>
auto matmul(ag::Value<N, M> const& left, ag::Value<M, O> const& right) {
    ag::Value<N, O> dst;
    dst.transform([&left, &right](auto n, auto o) {
        auto sum = 0.F;
        for (auto m = 0; m < M; ++m) { sum += left.item(n, m) * right.item(m, o); }
        return sum;
    });
    return dst;
}

}  // namespace ag

template <int N, int M>
auto operator+(ag::Value<N, M> const& left, ag::Float right) {
    ag::Value<N, M> dst;
    dst.transform([&left, right](auto n, auto m) { return left.item(n, m) + right; });
    return dst;
}

template <int N, int M>
auto operator+(ag::Float left, ag::Value<N, M> const& right) {
    return operator+(right, left);
}

template <int N, int M>
auto operator+(ag::Value<N, M> const& left, ag::Value<N, M> const& right) {
    ag::Value<N, M> dst;
    dst.transform([&left, &right](auto n, auto m) { return left.item(n, m) + right.item(n, m); });
    return dst;
}

template <int N, int M>
auto operator-(ag::Value<N, M> const& left, ag::Float right) {
    ag::Value<N, M> dst;
    dst.transform([&left, right](auto n, auto m) { return left.item(n, m) - right; });
    return dst;
}

template <int N, int M>
auto operator-(ag::Float left, ag::Value<N, M> const& right) {
    return operator-(right, left);
}

template <int N, int M>
auto operator-(ag::Value<N, M> const& left, ag::Value<N, M> const& right) {
    ag::Value<N, M> dst;
    dst.transform([&left, &right](auto n, auto m) { return left.item(n, m) - right.item(n, m); });
    return dst;
}

template <int N, int M>
auto operator*(ag::Value<N, M> const& left, ag::Float right) {
    ag::Value<N, M> dst;
    dst.transform([&left, right](auto n, auto m) { return left.item(n, m) * right; });
    return dst;
}

template <int N, int M>
auto operator*(ag::Float left, ag::Value<N, M> const& right) {
    return operator*(right, left);
}

template <int N, int M>
auto operator*(ag::Value<N, M> const& left, ag::Value<N, M> const& right) {
    ag::Value<N, M> dst;
    dst.transform([&left, &right](auto n, auto m) { return left.item(n, m) * right.item(n, m); });
    return dst;
}

#endif
