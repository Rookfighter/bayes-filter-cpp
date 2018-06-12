/*
 * test_unscented_transform.cpp
 *
 *  Created on: 12 Jun 2018
 *      Author: Fabian Meyer
 */

#include <catch.hpp>
#include "bayes_filter/unscented_transform.h"
#include "eigen_assert.h"

using namespace bf;

TEST_CASE("sigma points")
{
    const double eps = 1e-6;
    UnscentedTransform trans;

    SECTION("with simple params")
    {
        trans.setAlpha(1.0);
        trans.setBeta(1.0);
        trans.setKappa(1.0);

        Eigen::VectorXd state(3);
        state << 1, 1, 1;
        Eigen::MatrixXd cov(3,3);
        cov << 1, 0, 0,
               0, 1, 0,
               0, 0, 1;

        Eigen::MatrixXd wexp(2, 7);
        wexp << 0.25, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125,
                1.25, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125;

        Eigen::MatrixXd sexp(3, 7);
        sexp << 1, 3, 1, 1, -1,  1,  1,
                1, 1, 3, 1,  1, -1,  1,
                1, 1, 1, 3,  1,  1, -1;

        auto result = trans.calcSigmaPoints(state, cov);

        REQUIRE(trans.calcLambda(state.size()) == Approx(1.0).epsilon(eps));
        REQUIRE_MAT(wexp, result.weights, eps);
        REQUIRE_MAT(sexp, result.points, eps);
    }

    SECTION("with different params")
    {
        trans.setAlpha(1.0);
        trans.setBeta(2.0);
        trans.setKappa(2.0);

        Eigen::VectorXd state(2);
        state << 1,1;
        Eigen::MatrixXd cov(2,2);
        cov << 1, 0,
               0, 1;

        Eigen::MatrixXd wexp(2, 5);
        wexp << 0.5, 0.125, 0.125, 0.125, 0.125,
                2.5, 0.125, 0.125, 0.125, 0.125;

        Eigen::MatrixXd sexp(2, 5);
        sexp << 1, 3, 1, -1,  1,
                1, 1, 3,  1, -1;

        auto result = trans.calcSigmaPoints(state, cov);

        REQUIRE(trans.calcLambda(state.size()) == Approx(2.0).epsilon(eps));
        REQUIRE_MAT(wexp, result.weights, eps);
        REQUIRE_MAT(sexp, result.points, eps);
    }
}
