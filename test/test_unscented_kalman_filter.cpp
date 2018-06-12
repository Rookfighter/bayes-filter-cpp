/*
 * test_unscented_kalman_filter.cpp
 *
 *  Created on: 12 Jun 2018
 *      Author: Fabian Meyer
 */

#include <catch.hpp>
#include "bayes_filter/unscented_kalman_filter.h"
#include "eigen_assert.h"
#include "invitro_models.h"

using namespace bf;

TEST_CASE("Unscented Kalman Filter")
{
    SECTION("initialize")
    {
        const double eps = 1e-6;
        UnscentedKalmanFilter ukf(
            new IdentityMotionModel(),
            new IdentitySensorModel());

        Eigen::VectorXd state(3);
        state << 1, 2, 3;
        Eigen::MatrixXd cov(3,3);
        cov << 4, 0 ,0,
               0, 5, 0,
               0, 0, 6;

        ukf.init(state, cov);
        auto result = ukf.getEstimate();

        REQUIRE_MAT(state, result.first, eps);
        REQUIRE_MAT(cov, result.second, eps);
    }

    SECTION("prediction step")
    {
        const double eps = 1e-6;
        UnscentedKalmanFilter ukf(
            new ConstVelMotionModel(),
            new IdentitySensorModel());
        Eigen::MatrixXd noise(4,4);
        noise << 0.1,   0,   0,   0,
                   0, 0.1,   0,   0,
                   0,   0, 0.1,   0,
                   0,   0,   0, 0.1;

        SECTION("with regular input")
        {
            Eigen::VectorXd state(4);
            state << 0, 0, 0.5, 0.5;
            Eigen::MatrixXd cov(4, 4);
            cov << 2, 0 ,0, 0,
                   0, 2, 0, 0,
                   0, 0, 1, 0,
                   0, 0, 0, 1;
            Eigen::VectorXd controls(1);
            controls << 1;

            ukf.init(state, cov);
            ukf.predict(controls, Eigen::MatrixXd(), noise);

            auto result = ukf.getEstimate();

            state << 0.5, 0.5, 0.5, 0.5;
            cov << 3.1,   0, 1.0,   0,
                     0, 3.1,   0, 1.0,
                   1.0,   0, 1.1,   0,
                     0, 1.0,   0, 1.1;

            REQUIRE_MAT(state, result.first, eps);
            REQUIRE_MAT(cov, result.second, eps);
        }
    }

    SECTION("correction step")
    {

    }
}
