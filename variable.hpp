#ifndef VARIABLE_HPP
#define VARIABLE_HPP

#include "matrix.hpp"
#include "types.hpp"

template <int N, int M>
struct Variable {
    using MatrixT = Matrix<N, M>;

    template <typename... Args>
    constexpr Variable(Args&&... args) : value_{std::forward<Args>(args)...} {
        static_assert(sizeof...(args) == N * M);
    }

    [[nodiscard]] auto value() -> MatrixT& { return value_; }
    [[nodiscard]] auto value() const -> MatrixT const& { return value_; }
    [[nodiscard]] auto grad() const -> MatrixT const& { return grad_; }

private:
    Matrix<N, M> value_{};
    Matrix<N, M> grad_{};
};

#endif
