#ifndef OPERATION_HPP
#define OPERATION_HPP

#include "operationplus.hpp"
#include "variable.hpp"

template <int NL, int ML, typename OL, int NR, int MR, typename OR>
auto operator+(Variable<NL, ML, OL>& left, Variable<NR, MR, OR>& right) {
    static_assert(NL == NR && ML == MR || NL == 1 && ML == 1 || NR == 1 && MR == 1);
    using Operation = OperationPlus<NL, NR, NR, MR, OL, OR>;
    return Variable<NR, MR, Operation>{Operation{left.shared(), right.shared()}};
}

#endif
