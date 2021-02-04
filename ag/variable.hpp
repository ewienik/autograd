#ifndef AG_VARIABLE_HPP
#define AG_VARIABLE_HPP

#include <ag/operation/leaf.hpp>
#include <ag/types.hpp>
#include <ag/value.hpp>
#include <cassert>
#include <memory>

namespace ag {

template <typename Operation, int N, int M>
concept VariableOperation = requires(Operation op, Value<N, M>& valuerw, Value<N, M> const& valuero) {
    op.fwdprop(valuerw);
    op.backprop(valuero);
    { op.grad(valuero) }
    ->std::same_as<Value<N, M>>;
    op.reset();
    op.zerograd();
};

template <int N, int M, VariableOperation<N, M> Operation>
struct VariableShared {
    static_assert(N > 0 && M > 0);

    using ValueT = Value<N, M>;

    template <typename... Args>
    constexpr VariableShared(Args&&... args) : value_{std::forward<Args>(args)...} {}

    VariableShared(Operation operation) : operation_{std::move(operation)} {}

    [[nodiscard]] auto value() -> ValueT& {
        if (!calculated_) {
            calculated_ = true;
            operation_.fwdprop(value_);
        }
        return value_;
    }
    [[nodiscard]] auto value() const -> ValueT const& { return value_; }

    auto reset() {
        calculated_ = false;
        operation_.reset();
    }

    [[nodiscard]] auto grad() const -> ValueT const& { return grad_; }

    auto backprop(ValueT const& chain) {
        auto grad = chain * operation_.grad(value_);
        grad_ = grad_ + grad;
        operation_.backprop(grad);
    }
    auto zerograd() {
        grad_.fill(0.F);
        operation_.zerograd();
    }

private:
    Value<N, M> value_{};
    Value<N, M> grad_{};
    Operation operation_{};
    bool calculated_{};
};

template <int N, int M, VariableOperation<N, M> Operation = operation::Leaf>
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

}  // namespace ag

#endif
