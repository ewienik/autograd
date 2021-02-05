#ifndef AG_HPP
#define AG_HPP

#include <ag/operation/avg.hpp>
#include <ag/operation/matmul.hpp>
#include <ag/operation/minus.hpp>
#include <ag/operation/mul.hpp>
#include <ag/operation/plus.hpp>
#include <ag/variable.hpp>

template <int N, int M, typename O>
auto avg(ag::Variable<N, M, O>& variable) {
    using Operation = ag::operation::Avg<N, M, O>;
    return ag::Variable<1, 1, Operation>{Operation{variable.shared()}};
}

template <int N, int M, int O, typename OL, typename OR>
auto matmul(ag::Variable<N, M, OL>& left, ag::Variable<M, O, OR>& right) {
    using Operation = ag::operation::MatMul<N, M, O, OL, OR>;
    return ag::Variable<N, O, Operation>{Operation{left.shared(), right.shared()}};
}

template <int N, int M, typename OL, typename OR>
auto operator+(ag::Variable<N, M, OL>& left, ag::Variable<N, M, OR>& right) {
    using Operation = ag::operation::Plus<N, M, OL, OR>;
    return ag::Variable<N, M, Operation>{Operation{left.shared(), right.shared()}};
}

template <int N, int M, typename OL, typename OR>
auto operator-(ag::Variable<N, M, OL>& left, ag::Variable<N, M, OR>& right) {
    using Operation = ag::operation::Minus<N, M, OL, OR>;
    return ag::Variable<N, M, Operation>{Operation{left.shared(), right.shared()}};
}

template <int N, int M, typename OL, typename OR>
auto operator*(ag::Variable<N, M, OL>& left, ag::Variable<N, M, OR>& right) {
    using Operation = ag::operation::Mul<N, M, OL, OR>;
    return ag::Variable<N, M, Operation>{Operation{left.shared(), right.shared()}};
}

#endif

