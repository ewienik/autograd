#ifndef OPERATIONLEAF_HPP
#define OPERATIONLEAF_HPP

struct OperationLeaf {
    template <typename Value>
    static auto fwdprop(Value& value) {}
    template <typename Value>
    static auto backprop([[maybe_unused]] Value const& value) {
        return Value::one();
    }
    static auto reset() {}
    static auto zerograd() {}
};

#endif

