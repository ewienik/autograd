#ifndef AG_OPERATION_PLUS_HPP
#define AG_OPERATION_PLUS_HPP

#include <ag/transform.hpp>
#include <ag/value.hpp>
#include <ag/variableshared.hpp>
#include <memory>

namespace ag::operation {

template <int N, int M, int O, int P, typename OL, typename OR>
struct Plus {};

template <int N, int M, typename OL, typename OR>
struct Plus<1, 1, N, M, OL, OR> {
    using LeftT = std::shared_ptr<VariableShared<1, 1, OL>>;
    using RightT = std::shared_ptr<VariableShared<N, M, OR>>;
    using ValueT = Value<N, M>;

    Plus(LeftT left, RightT right) : left_(std::move(left)), right_(std::move(right)) {}

    auto fwdprop(ValueT& value) { transformSum(value, right_->value(), left_->value().item()); }
    static auto backprop([[maybe_unused]] ValueT const& value) { return ValueT::one(); }
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

template <int N, int M, typename OL, typename OR>
struct Plus<N, M, 1, 1, OL, OR> {
    using LeftT = std::shared_ptr<VariableShared<N, M, OL>>;
    using RightT = std::shared_ptr<VariableShared<1, 1, OR>>;
    using ValueT = Value<N, M>;

    Plus(LeftT left, RightT right) : left_(std::move(left)), right_(std::move(right)) {}

    auto fwdprop(ValueT& value) { transformSum(value, left_->value(), right_->value().item()); }
    static auto backprop([[maybe_unused]] ValueT const& value) { return ValueT::one(); }
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

template <typename OL, typename OR>
struct Plus<1, 1, 1, 1, OL, OR> {
    using LeftT = std::shared_ptr<VariableShared<1, 1, OL>>;
    using RightT = std::shared_ptr<VariableShared<1, 1, OR>>;
    using ValueT = Value<1, 1>;

    Plus(LeftT left, RightT right) : left_(std::move(left)), right_(std::move(right)) {}

    auto fwdprop(ValueT& value) { value.item() = right_->value().item() + left_->value().item(); }
    static auto backprop([[maybe_unused]] ValueT const& value) { return ValueT::one(); }
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
