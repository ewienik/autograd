#include "catch.hpp"
#include "variable.hpp"

TEST_CASE("Scalar") {
    auto scalar = Variable<1, 1>{4.f};
    REQUIRE(scalar.value().at(0, 0) == 4.);
    REQUIRE(scalar.grad().at(0, 0) == 0.);
}

TEST_CASE("2x2") {
    auto scalar = Variable<2, 2>{1.f, 2.f, 3.f, 4.f};
    REQUIRE(scalar.value().at(0, 0) == 4.);
    REQUIRE(scalar.grad().at(0, 0) == 0.);
}

