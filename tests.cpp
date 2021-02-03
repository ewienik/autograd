#include "catch.hpp"
#include "variable.hpp"

TEST_CASE("Scalar") {
    auto scalar = Scalar{4.F};
    REQUIRE(scalar.value().item() == 4.F);
    REQUIRE(scalar.grad().item() == 0.F);
}

TEST_CASE("Vector2") {
    auto vector = Vector<2>{1.F, 2.F};
    REQUIRE(vector.value().item(0) == 1.F);
    REQUIRE(vector.value().item(1) == 2.F);
}

TEST_CASE("Matrix2x2") {
    auto matrix = Matrix<2, 2>{1.F, 2.F, 3.F, 4.F};
    REQUIRE(matrix.value().item(0, 0) == 1.F);
    REQUIRE(matrix.value().item(1, 0) == 2.F);
    REQUIRE(matrix.value().item(0, 1) == 3.F);
    REQUIRE(matrix.value().item(1, 1) == 4.F);
    REQUIRE(matrix.grad().item(0, 0) == 0.F);
    REQUIRE(matrix.grad().item(1, 0) == 0.F);
    REQUIRE(matrix.grad().item(0, 1) == 0.F);
    REQUIRE(matrix.grad().item(1, 1) == 0.F);
}

