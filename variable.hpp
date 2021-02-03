#ifndef VARIABLE_HPP
#define VARIABLE_HPP

#include "types.hpp"
#include "value.hpp"

template <int N, int M>
struct Variable {
    static_assert(N > 0 && M > 0);

    using ValueT = Value<N, M>;

    template <typename... Args>
    constexpr Variable(Args&&... args) : value_{std::forward<Args>(args)...} {
        static_assert(sizeof...(args) == N * M);
    }

    [[nodiscard]] auto value() -> ValueT& { return value_; }
    [[nodiscard]] auto value() const -> ValueT const& { return value_; }
    [[nodiscard]] auto grad() const -> ValueT const& { return grad_; }

private:
    Value<N, M> value_{};
    Value<N, M> grad_{};
};

using Scalar = Variable<1, 1>;
template <int N>
using Vector = Variable<N, 1>;
template <int N, int M>
using Matrix = Variable<N, M>;

#endif
