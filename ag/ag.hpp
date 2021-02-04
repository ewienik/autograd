#ifndef AG_HPP
#define AG_HPP

#include <ag/operation/mul.hpp>
#include <ag/operation/plus.hpp>
#include <ag/variable.hpp>

template <int NL, int ML, typename OL, int NR, int MR, typename OR>
auto operator+(ag::Variable<NL, ML, OL>& left, ag::Variable<NR, MR, OR>& right) {
    static_assert(NL == NR && ML == MR || NL == 1 && ML == 1 || NR == 1 && MR == 1);
    using Operation = ag::operation::Plus<NL, NR, NR, MR, OL, OR>;
    return ag::Variable<NR, MR, Operation>{Operation{left.shared(), right.shared()}};
}

template <int NL, int ML, typename OL, int NR, int MR, typename OR>
auto operator*(ag::Variable<NL, ML, OL>& left, ag::Variable<NR, MR, OR>& right) {
    static_assert(NL == NR && ML == MR || NL == 1 && ML == 1 || NR == 1 && MR == 1);
    using Operation = ag::operation::Mul<NL, NR, NR, MR, OL, OR>;
    return ag::Variable<NR, MR, Operation>{Operation{left.shared(), right.shared()}};
}

#endif

