#ifndef VARIABLE_HPP
#define VARIABLE_HPP

#include <cassert>
#include <memory>
#include "operationleaf.hpp"
#include "types.hpp"
#include "variableshared.hpp"

template <int N, int M, VariableOperation<N, M> Operation = OperationLeaf>
struct Variable final {
    static_assert(N > 0 && M > 0);

    using OperationT = Operation;
    using VariableSharedT = VariableShared<N, M, Operation>;
    using ValueT = typename VariableSharedT::ValueT;

    template <typename... Args>
    constexpr Variable(Args&&... args) : variable_{std::make_shared<VariableSharedT>(std::forward<Args>(args)...)} {}

    [[nodiscard]] auto shared() { return variable_; }
    [[nodiscard]] auto shared() const { return variable_; }

    [[nodiscard]] auto value(bool force = {}) -> ValueT& {
        if (force) { variable_->reset(); }
        return variable_->value();
    }
    [[nodiscard]] auto value() const -> ValueT const& { return variable_->value(); }

    [[nodiscard]] auto grad() const -> ValueT const& { return variable_->grad(); }

    auto backprop(ValueT const& chain = ValueT::one()) { variable_->backprop(chain); }
    auto zerograd() { variable_->zerograd(); }

private:
    std::shared_ptr<VariableSharedT> variable_{std::make_shared<VariableSharedT>()};
};

using Scalar = Variable<1, 1>;
template <int N>
using Vector = Variable<N, 1>;
template <int N, int M>
using Matrix = Variable<N, M>;

#endif
