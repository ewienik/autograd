#include <ag/ag.hpp>
#include <ext/catch2/catch.hpp>

using namespace ag;

TEST_CASE("FromReqs1") {
    auto a = Scalar{3.F};
    auto b = Scalar{4.F};
    auto c = a * b;
    REQUIRE(c.value().item() == 12.F);
    c.backprop();
    REQUIRE(a.grad().item() == 4.F);
    c.zerograd();
    REQUIRE(a.grad().item() == 0.F);
}

TEST_CASE("FromReqs2") {
    auto a = Scalar{3.F};
    auto b = Scalar{4.F};
    auto c = Scalar{5.F};
    auto d = (a * b + c) - a;

    REQUIRE(d.value().item() == 14.F);
    d.backprop();
    REQUIRE(a.grad().item() == 3.F);
    REQUIRE(b.grad().item() == 3.F);
    REQUIRE(c.grad().item() == 1.F);
}

TEST_CASE("FromReqs3") {
    auto a = Scalar{3.F};
    auto b = Scalar{4.F};
    auto c = a + b;
    auto d = a * b;
    auto e = c - d;

    REQUIRE(e.value().item() == -5.F);
    e.backprop();
    REQUIRE(a.grad().item() == -3.F);
    REQUIRE(b.grad().item() == -2.F);
}

TEST_CASE("FromReqsFinal") {
    auto weights = Matrix<2, 2>{1.F, 2.F, 3.F, 4.F};
    auto bias = VectorRow<2>{5.F, 6.F};
    auto x = VectorRow<2>{7.F, 8.F};
    auto y = VectorRow<2>{9.F, 10.F};
    auto y_pred = matmul(x, weights) + bias;
    auto loss = avg((y - y_pred) * (y - y_pred));

    REQUIRE(loss.value().item() == 1246.5);
    loss.backprop();
    REQUIRE(weights.grad().item(0, 0) == 189.F);
    REQUIRE(weights.grad().item(0, 1) == 294.F);
    REQUIRE(weights.grad().item(1, 0) == 216.F);
    REQUIRE(weights.grad().item(1, 1) == 336.F);
    REQUIRE(bias.grad().item(0) == 27.F);
    REQUIRE(bias.grad().item(1) == 42.F);
}

