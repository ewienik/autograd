#ifndef AG_OPERATION_MINUS_HPP
#define AG_OPERATION_MINUS_HPP

#include <ag/value.hpp>
#include <ag/variable.hpp>
#include <memory>

namespace ag::operation {

template <int N, int M, typename OL, typename OR>
struct Minus {
    using LeftT = std::shared_ptr<VariableShared<N, M, OL>>;
    using RightT = std::shared_ptr<VariableShared<N, M, OR>>;
    using ValueT = Value<N, M>;

    Minus(LeftT left, RightT right) : left_(std::move(left)), right_(std::move(right)) {}

    auto fwdprop(ValueT& value) { value = left_->value() - right_->value(); }
    auto backprop(ValueT const& chain, [[maybe_unused]] ValueT const& value) {
        left_->backprop(chain);
        right_->backprop(-1.F * chain);
    }
    auto reset() {
        left_->reset();
        right_->reset();
    }
    auto zerograd() {
        left_->zerograd();
        right_->zerograd();
    }

private:
    LeftT left_{};
    RightT right_{};
};

}  // namespace ag::operation

#endif
