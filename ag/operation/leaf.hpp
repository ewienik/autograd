#ifndef AG_OPERATION_LEAF_HPP
#define AG_OPERATION_LEAF_HPP

namespace ag::operation {

struct Leaf {
    template <typename Value>
    static auto fwdprop(Value& value) {}
    template <typename Value>
    static auto backprop([[maybe_unused]] Value const& chain, [[maybe_unused]] Value const& value) {}
    static auto reset() {}
    static auto zerograd() {}
};

}  // namespace ag::operation

#endif

