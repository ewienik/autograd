#ifndef VARIABLE_HPP
#define VARIABLE_HPP

#include "matrix.hpp"
#include "types.hpp"

template <int N, int M>
struct Variable {
    using MatrixT = Matrix<N, M>;

    constexpr Variable(std::initializer_list<Float> initial) : value_(initial) {}

    auto value() -> MatrixT& { return value_; }
    auto value() const -> MatrixT& { return value_; }
    auto grad() const -> MatrixT& { return grad_; }

private:
    Matrix<N, M> value_{};
    Matrix<N, M> grad_{};
};

#endif
