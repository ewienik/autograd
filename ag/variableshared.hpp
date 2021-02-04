#ifndef AG_VARIABLESHARED_HPP
#define AG_VARIABLESHARED_HPP

#include <ag/value.hpp>

namespace ag {

template <typename Operation, int N, int M>
concept VariableOperation = requires(Operation op, Value<N, M>& valuerw, Value<N, M> const& valuero) {
    op.fwdprop(valuerw);
    op.backprop(valuerw);
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
        auto grad = transformMul(chain, operation_.grad(value_));
        grad_ = transformAdd(grad_, grad);
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

}  // namespace ag

#endif
