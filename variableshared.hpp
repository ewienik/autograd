#ifndef VARIABLESHARED_HPP
#define VARIABLESHARED_HPP

#include "value.hpp"

template <int N, int M>
struct VariableShared {
    static_assert(N > 0 && M > 0);

    using ValueT = Value<N, M>;

    template <typename... Args>
    constexpr VariableShared(Args&&... args) : value_{std::forward<Args>(args)...} {
        static_assert(sizeof...(args) == N * M);
    }

    [[nodiscard]] auto value() -> ValueT& { return value_; }
    [[nodiscard]] auto value() const -> ValueT const& { return value_; }
    [[nodiscard]] auto grad() const -> ValueT const& { return grad_; }

private:
    Value<N, M> value_{};
    Value<N, M> grad_{};
};

#endif

