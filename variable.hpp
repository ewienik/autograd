#ifndef VARIABLE_HPP
#define VARIABLE_HPP

#include <memory>
#include "variableshared.hpp"

template <int N, int M>
struct Variable {
    static_assert(N > 0 && M > 0);

    using VariableSharedT = VariableShared<N, M>;
    using ValueT = typename VariableSharedT::ValueT;

    template <typename... Args>
    constexpr Variable(Args&&... args) : variable_{std::make_shared<VariableSharedT>(std::forward<Args>(args)...)} {
        static_assert(sizeof...(args) == N * M);
    }

    [[nodiscard]] auto value() -> ValueT& { return variable_->value(); }
    [[nodiscard]] auto value() const -> ValueT const& { return variable_->value(); }
    [[nodiscard]] auto grad() const -> ValueT const& { return variable_->grad(); }

private:
    std::shared_ptr<VariableSharedT> variable_{};
};

using Scalar = Variable<1, 1>;
template <int N>
using Vector = Variable<N, 1>;
template <int N, int M>
using Matrix = Variable<N, M>;

#endif
