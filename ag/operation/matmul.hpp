#ifndef AG_OPERATION_MATMUL_HPP
#define AG_OPERATION_MATMUL_HPP

#include <ag/value.hpp>
#include <ag/variable.hpp>
#include <memory>

namespace ag::operation {

template <int N, int M, int O, typename OL, typename OR>
struct MatMul {
    using LeftT = std::shared_ptr<VariableShared<N, M, OL>>;
    using RightT = std::shared_ptr<VariableShared<M, O, OR>>;
    using ValueT = Value<N, O>;

    MatMul(LeftT left, RightT right) : left_(std::move(left)), right_(std::move(right)) {}

    auto fwdprop(ValueT& value) { value = matmul(left_->value(), right_->value()); }
    auto backprop(ValueT const& chain, [[maybe_unused]] ValueT const& value) {
        left_->backprop(matmul(chain, transpose(right_->value())));
        right_->backprop(matmul(transpose(left_->value()), chain));
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
