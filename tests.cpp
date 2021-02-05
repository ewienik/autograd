#include <ag/ag.hpp>
#include <catch.hpp>

using namespace ag;

TEST_CASE("UseCases") {  // NOLINT

    SECTION("FromReqs1") {
        auto a = Scalar{3.F};
        auto b = Scalar{4.F};
        auto c = a * b;
        REQUIRE(c.value().item() == 12.F);
        c.backprop();
        REQUIRE(a.grad().item() == 4.F);
        c.zerograd();
        REQUIRE(a.grad().item() == 0.F);
    }

    SECTION("FromReqs2") {
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

    SECTION("FromReqs3") {
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

    SECTION("FromReqsFinal") {
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
}

TEST_CASE("Framework") {  // NOLINT

    SECTION("Scalar") {
        auto scalar = Scalar{4.F};
        REQUIRE(scalar.value().item() == 4.F);
        REQUIRE(scalar.grad().item() == 0.F);
    }

    SECTION("VectorRow2") {
        auto vector = VectorRow<2>{1.F, 2.F};
        REQUIRE(vector.value().item(0) == 1.F);
        REQUIRE(vector.value().item(1) == 2.F);
        REQUIRE(vector.grad().item(0) == 0.F);
        REQUIRE(vector.grad().item(1) == 0.F);
    }

    SECTION("VectorCol2") {
        auto vector = VectorCol<2>{1.F, 2.F};
        REQUIRE(vector.value().item(0) == 1.F);
        REQUIRE(vector.value().item(1) == 2.F);
        REQUIRE(vector.grad().item(0) == 0.F);
        REQUIRE(vector.grad().item(1) == 0.F);
    }

    SECTION("Matrix2x2") {
        auto matrix = Matrix<2, 2>{1.F, 2.F, 3.F, 4.F};
        REQUIRE(matrix.value().item(0, 0) == 1.F);
        REQUIRE(matrix.value().item(0, 1) == 2.F);
        REQUIRE(matrix.value().item(1, 0) == 3.F);
        REQUIRE(matrix.value().item(1, 1) == 4.F);
        REQUIRE(matrix.grad().item(0, 0) == 0.F);
        REQUIRE(matrix.grad().item(0, 1) == 0.F);
        REQUIRE(matrix.grad().item(1, 0) == 0.F);
        REQUIRE(matrix.grad().item(1, 1) == 0.F);
    }

    SECTION("PlusScalars") {
        auto a = Scalar{3.F};
        auto b = Scalar{4.F};
        auto c = a + b;
        REQUIRE(c.value().item() == 7.F);
        REQUIRE(c.value(true).item() == 7.F);

        REQUIRE(a.grad().item() == 0.F);
        REQUIRE(b.grad().item() == 0.F);
        c.backprop();
        REQUIRE(a.grad().item() == 1.F);
        REQUIRE(b.grad().item() == 1.F);
        c.zerograd();
        REQUIRE(a.grad().item() == 0.F);
        REQUIRE(b.grad().item() == 0.F);

        a.value().fill(4.F);
        b.value().fill(5.F);
        REQUIRE(c.value().item() == 7.F);
        REQUIRE(c.value(true).item() == 9.F);

        REQUIRE(a.grad().item() == 0.F);
        REQUIRE(b.grad().item() == 0.F);
        c.backprop();
        REQUIRE(a.grad().item() == 1.F);
        REQUIRE(b.grad().item() == 1.F);
        c.zerograd();
        REQUIRE(a.grad().item() == 0.F);
        REQUIRE(b.grad().item() == 0.F);
    }

    SECTION("PlusVectors") {
        auto a = VectorRow<2>{2.F, 3.F};
        auto b = VectorRow<2>{4.F, 5.F};
        auto c = a + b;
        REQUIRE(c.value().item(0) == 6.F);
        REQUIRE(c.value().item(1) == 8.F);

        c.backprop();
        REQUIRE(a.grad().item(0) == 1.F);
        REQUIRE(a.grad().item(1) == 1.F);
        REQUIRE(b.grad().item(0) == 1.F);
        REQUIRE(b.grad().item(1) == 1.F);

        a.value() = {3.F, 4.F};
        b.value() = {5.F, 6.F};
        REQUIRE(c.value().item(0) == 6.F);
        REQUIRE(c.value().item(1) == 8.F);
        REQUIRE(c.value(true).item(0) == 8.F);
        REQUIRE(c.value().item(1) == 10.F);

        c.backprop();
        REQUIRE(a.grad().item(0) == 2.F);
        REQUIRE(a.grad().item(1) == 2.F);
        REQUIRE(b.grad().item(0) == 2.F);
        REQUIRE(b.grad().item(1) == 2.F);

        c.zerograd();
        c.backprop();
        REQUIRE(a.grad().item(0) == 1.F);
        REQUIRE(a.grad().item(1) == 1.F);
        REQUIRE(b.grad().item(0) == 1.F);
        REQUIRE(b.grad().item(1) == 1.F);
    }

    SECTION("MinusScalars") {
        auto a = Scalar{3.F};
        auto b = Scalar{4.F};
        auto c = a - b;
        REQUIRE(c.value().item() == -1.F);

        REQUIRE(a.grad().item() == 0.F);
        REQUIRE(b.grad().item() == 0.F);
        c.backprop();
        REQUIRE(a.grad().item() == 1.F);
        REQUIRE(b.grad().item() == -1.F);
        c.zerograd();
        REQUIRE(a.grad().item() == 0.F);
        REQUIRE(b.grad().item() == 0.F);

        a.value() = {6.F};
        b.value() = {5.F};
        REQUIRE(c.value().item() == -1.F);
        REQUIRE(c.value(true).item() == 1.F);

        REQUIRE(a.grad().item() == 0.F);
        REQUIRE(b.grad().item() == 0.F);
        c.backprop();
        REQUIRE(a.grad().item() == 1.F);
        REQUIRE(b.grad().item() == -1.F);
        c.zerograd();
        REQUIRE(a.grad().item() == 0.F);
        REQUIRE(b.grad().item() == 0.F);
    }

    SECTION("MinusVectors") {
        auto a = VectorCol<2>{2.F, 3.F};
        auto b = VectorCol<2>{4.F, 5.F};
        auto c = a - b;
        REQUIRE(c.value().item(0) == -2.F);
        REQUIRE(c.value().item(1) == -2.F);

        c.backprop();
        REQUIRE(a.grad().item(0) == 1.F);
        REQUIRE(a.grad().item(1) == 1.F);
        REQUIRE(b.grad().item(0) == -1.F);
        REQUIRE(b.grad().item(1) == -1.F);

        a.value() = {7.F, 9.F};
        b.value() = {5.F, 6.F};
        REQUIRE(c.value().item(0) == -2.F);
        REQUIRE(c.value().item(1) == -2.F);
        REQUIRE(c.value(true).item(0) == 2.F);
        REQUIRE(c.value().item(1) == 3.F);

        c.zerograd();
        c.backprop();
        REQUIRE(a.grad().item(0) == 1.F);
        REQUIRE(a.grad().item(1) == 1.F);
        REQUIRE(b.grad().item(0) == -1.F);
        REQUIRE(b.grad().item(1) == -1.F);
    }

    SECTION("MulScalars") {
        auto a = Scalar{3.F};
        auto b = Scalar{4.F};
        auto c = a * b;
        REQUIRE(c.value().item() == 12.F);
        REQUIRE(c.value(true).item() == 12.F);

        REQUIRE(a.grad().item() == 0.F);
        REQUIRE(b.grad().item() == 0.F);
        c.backprop();
        REQUIRE(a.grad().item() == 4.F);
        REQUIRE(b.grad().item() == 3.F);
        c.zerograd();
        REQUIRE(a.grad().item() == 0.F);
        REQUIRE(b.grad().item() == 0.F);

        a.value().fill(4.F);
        b.value().fill(5.F);
        REQUIRE(c.value().item() == 12.F);
        REQUIRE(c.value(true).item() == 20.F);

        REQUIRE(a.grad().item() == 0.F);
        REQUIRE(b.grad().item() == 0.F);
        c.backprop();
        REQUIRE(a.grad().item() == 5.F);
        REQUIRE(b.grad().item() == 4.F);
        c.zerograd();
        REQUIRE(a.grad().item() == 0.F);
        REQUIRE(b.grad().item() == 0.F);
    }

    SECTION("MulVectors") {
        auto a = VectorRow<2>{2.F, 3.F};
        auto b = VectorRow<2>{4.F, 5.F};
        auto c = a * b;
        REQUIRE(c.value().item(0) == 8.F);
        REQUIRE(c.value().item(1) == 15.F);

        c.backprop();
        REQUIRE(a.grad().item(0) == 4.F);
        REQUIRE(a.grad().item(1) == 5.F);
        REQUIRE(b.grad().item(0) == 2.F);
        REQUIRE(b.grad().item(1) == 3.F);

        a.value() = {3.F, 4.F};
        b.value() = {5.F, 6.F};
        REQUIRE(c.value().item(0) == 8.F);
        REQUIRE(c.value().item(1) == 15.F);
        REQUIRE(c.value(true).item(0) == 15.F);
        REQUIRE(c.value().item(1) == 24.F);

        c.zerograd();
        c.backprop();
        REQUIRE(a.grad().item(0) == 5.F);
        REQUIRE(a.grad().item(1) == 6.F);
        REQUIRE(b.grad().item(0) == 3.F);
        REQUIRE(b.grad().item(1) == 4.F);
    }

    SECTION("MatMulScalars") {
        auto a = Scalar{3.F};
        auto b = Scalar{4.F};
        auto c = matmul(a, b);
        REQUIRE(c.value().item() == 12.F);
        REQUIRE(c.value(true).item() == 12.F);

        REQUIRE(a.grad().item() == 0.F);
        REQUIRE(b.grad().item() == 0.F);
        c.backprop();
        REQUIRE(a.grad().item() == 4.F);
        REQUIRE(b.grad().item() == 3.F);
        c.zerograd();
        REQUIRE(a.grad().item() == 0.F);
        REQUIRE(b.grad().item() == 0.F);

        a.value().fill(4.F);
        b.value().fill(5.F);
        REQUIRE(c.value().item() == 12.F);
        REQUIRE(c.value(true).item() == 20.F);

        REQUIRE(a.grad().item() == 0.F);
        REQUIRE(b.grad().item() == 0.F);
        c.backprop();
        REQUIRE(a.grad().item() == 5.F);
        REQUIRE(b.grad().item() == 4.F);
        c.zerograd();
        REQUIRE(a.grad().item() == 0.F);
        REQUIRE(b.grad().item() == 0.F);
    }

    SECTION("MatMulVectorsToScalar") {
        auto a = VectorRow<2>{2.F, 3.F};
        auto b = VectorCol<2>{4.F, 5.F};
        auto c = matmul(a, b);
        REQUIRE(c.value().item() == 23.F);

        c.backprop();
        REQUIRE(a.grad().item(0) == 4.F);
        REQUIRE(a.grad().item(1) == 5.F);
        REQUIRE(b.grad().item(0) == 2.F);
        REQUIRE(b.grad().item(1) == 3.F);

        a.value() = {3.F, 4.F};
        b.value() = {5.F, 6.F};
        REQUIRE(c.value().item() == 23.F);
        REQUIRE(c.value(true).item() == 39.F);

        c.zerograd();
        c.backprop();
        REQUIRE(a.grad().item(0) == 5.F);
        REQUIRE(a.grad().item(1) == 6.F);
        REQUIRE(b.grad().item(0) == 3.F);
        REQUIRE(b.grad().item(1) == 4.F);
    }

    SECTION("MatMulVectorsToMatrix") {
        auto a = VectorCol<2>{2.F, 3.F};
        auto b = VectorRow<2>{4.F, 5.F};
        auto c = matmul(a, b);
        REQUIRE(c.value().item(0, 0) == 8.F);
        REQUIRE(c.value().item(0, 1) == 10.F);
        REQUIRE(c.value().item(1, 0) == 12.F);
        REQUIRE(c.value().item(1, 1) == 15.F);

        c.backprop();
        REQUIRE(a.grad().item(0) == 9.F);
        REQUIRE(a.grad().item(1) == 9.F);
        REQUIRE(b.grad().item(0) == 5.F);
        REQUIRE(b.grad().item(1) == 5.F);

        a.value() = {3.F, 4.F};
        b.value() = {5.F, 6.F};
        REQUIRE(c.value(true).item(0, 0) == 15.F);
        REQUIRE(c.value().item(0, 1) == 18.F);
        REQUIRE(c.value().item(1, 0) == 20.F);
        REQUIRE(c.value().item(1, 1) == 24.F);

        c.zerograd();
        c.backprop();
        REQUIRE(a.grad().item(0) == 11.F);
        REQUIRE(a.grad().item(1) == 11.F);
        REQUIRE(b.grad().item(0) == 7.F);
        REQUIRE(b.grad().item(1) == 7.F);
    }

    SECTION("MatMulMatrixes") {
        auto a = Matrix<2, 2>{2.F, 3.F, 4.F, 5.F};
        auto b = Matrix<2, 2>{6.F, 7.F, 8.F, 9.F};
        auto c = matmul(a, b);
        REQUIRE(c.value().item(0, 0) == 36.F);
        REQUIRE(c.value().item(0, 1) == 41.F);
        REQUIRE(c.value().item(1, 0) == 64.F);
        REQUIRE(c.value().item(1, 1) == 73.F);

        c.backprop();
        REQUIRE(a.grad().item(0, 0) == 13.F);
        REQUIRE(a.grad().item(0, 1) == 17.F);
        REQUIRE(a.grad().item(1, 0) == 13.F);
        REQUIRE(a.grad().item(1, 1) == 17.F);
        REQUIRE(b.grad().item(0, 0) == 6.F);
        REQUIRE(b.grad().item(0, 1) == 6.F);
        REQUIRE(b.grad().item(1, 0) == 8.F);
        REQUIRE(b.grad().item(1, 1) == 8.F);
    }

    SECTION("VarMatrix") {
        auto a = Matrix<2, 2>{2.F, 3.F, 4.F, 5.F};
        auto b = avg(a);
        REQUIRE(b.value().item() == 3.5F);

        b.backprop();
        REQUIRE(a.grad().item(0, 0) == 0.25F);
        REQUIRE(a.grad().item(0, 1) == 0.25F);
        REQUIRE(a.grad().item(1, 0) == 0.25F);
        REQUIRE(a.grad().item(1, 1) == 0.25F);
    }
}
