#ifndef AG_OPERATION_PLUS_HPP
#define AG_OPERATION_PLUS_HPP

#include <ag/value.hpp>
#include <ag/variable.hpp>
#include <memory>

namespace ag::operation {

template <int N, int M, typename OL, typename OR>
struct Plus {
    using LeftT = std::shared_ptr<VariableShared<N, M, OL>>;
    using RightT = std::shared_ptr<VariableShared<N, M, OR>>;
    using ValueT = Value<N, M>;

    Plus(LeftT left, RightT right) : left_(std::move(left)), right_(std::move(right)) {}

    auto fwdprop(ValueT& value) { value = right_->value() + left_->value(); }
    auto backprop(ValueT const& chain, [[maybe_unused]] ValueT const& value) {
        left_->backprop(chain);
        right_->backprop(chain);
    }
    static auto grad() { return ValueT::one(); }
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
