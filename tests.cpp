#include "catch.hpp"
#include "variable.hpp"

TEST_CASE("Scalar") {
    auto scalar = Variable<1, 1>{4.F};
    REQUIRE(scalar.value().at(0, 0) == 4.F);
    REQUIRE(scalar.grad().at(0, 0) == 0.F);
}

TEST_CASE("2x2") {
    auto scalar = Variable<2, 2>{1.F, 2.F, 3.F, 4.F};
    REQUIRE(scalar.value().at(0, 0) == 1.F);
    REQUIRE(scalar.value().at(0, 1) == 2.F);
    REQUIRE(scalar.value().at(1, 0) == 3.F);
    REQUIRE(scalar.value().at(1, 1) == 4.F);
    REQUIRE(scalar.grad().at(0, 0) == 0.F);
    REQUIRE(scalar.grad().at(0, 1) == 0.F);
    REQUIRE(scalar.grad().at(1, 0) == 0.F);
    REQUIRE(scalar.grad().at(1, 1) == 0.F);
}

