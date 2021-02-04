#ifndef AG_TRANSFORM_HPP
#define AG_TRANSFORM_HPP

#include <ag/types.hpp>
#include <ag/value.hpp>

namespace ag {
template <int N, int M>
auto transformSum(Value<N, M> const& src, Float bias) {
    Value<N, M> dst;
    dst.transform([&src, bias](auto n, auto m) { return src.item(n, m) + bias; });
    return dst;
}

template <int N, int M>
auto transformSum(Value<N, M> const& left, Value<N, M> const& right) {
    Value<N, M> dst;
    dst.transform([&left, &right](auto n, auto m) { return left.item(n, m) + right.item(n, m); });
    return dst;
}

template <int N, int M>
auto transformMul(Value<N, M> const& src, Float bias) {
    Value<N, M> dst;
    dst.transform([&src, bias](auto n, auto m) { return src.item(n, m) * bias; });
    return dst;
}

template <int N, int M>
auto transformMul(Value<N, M> const& left, Value<N, M> const& right) {
    Value<N, M> dst;
    dst.transform([&left, &right](auto n, auto m) { return left.item(n, m) * right.item(n, m); });
    return dst;
}

template <int N, int M>
auto transformAdd(Value<N, M> const& left, Value<N, M> const& right) {
    Value<N, M> dst;
    dst.transform([&left, &right](auto n, auto m) { return left.item(n, m) + right.item(n, m); });
    return dst;
}

}  // namespace ag

#endif

