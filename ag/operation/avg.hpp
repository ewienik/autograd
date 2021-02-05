#ifndef AG_OPERATION_AVG_HPP
#define AG_OPERATION_AVG_HPP

#include <ag/value.hpp>
#include <ag/variable.hpp>
#include <memory>
#include <numeric>

namespace ag::operation {

template <int N, int M, typename O>
struct Avg {
    using VariableT = std::shared_ptr<VariableShared<N, M, O>>;
    using ValueT = Value<1, 1>;

    Avg(VariableT variable) : variable_(std::move(variable)) {}

    auto fwdprop(ValueT& value) {
        value.item() =
            std::accumulate(std::begin(variable_->value()), std::end(variable_->value()), 0.F) / (1.F * N * M);
    }
    auto backprop(ValueT const& chain, [[maybe_unused]] ValueT const& value) {
        using VariableValueT = typename VariableShared<N, M, O>::ValueT;
        variable_->backprop((chain.item() / (1.F * N * M)) * VariableValueT::ones());
    }
    auto reset() { variable_->reset(); }
    auto zerograd() { variable_->zerograd(); }

private:
    VariableT variable_{};
};

}  // namespace ag::operation

#endif
