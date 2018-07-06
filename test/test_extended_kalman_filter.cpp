/*
 * test_extended_kalman_filter.cpp
 *
 *  Created on: 06 Jul 2018
 *      Author: Fabian Meyer
 */

#include <catch.hpp>
#include "bayes_filter/extended_kalman_filter.h"
#include "eigen_assert.h"
#include "invitro_models.h"

using namespace bf;

TEST_CASE("Extended Kalman Filter")
{
    SECTION("initialize")
    {
        const double eps = 1e-6;
        ExtendedKalmanFilter ekf(
            new IdentityMotionModel(),
            new IdentitySensorModel());

        Eigen::VectorXd state(3);
        state << 1, 2, 3;
        Eigen::MatrixXd cov(3,3);
        cov << 4, 0 ,0,
               0, 5, 0,
               0, 0, 6;

        ekf.init(state, cov);
        auto result = ekf.getEstimate();

        REQUIRE_MAT(state, result.state, eps);
        REQUIRE_MAT(cov, result.cov, eps);
    }

    SECTION("prediction step")
    {
        const double eps = 1e-6;
        ExtendedKalmanFilter ekf(
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

            ekf.init(state, cov);
            ekf.predict(controls, Eigen::MatrixXd(), noise);

            auto result = ekf.getEstimate();

            state << 0.5, 0.5, 0.5, 0.5;
            cov << 3.01,   0, 1.0,   0,
                     0, 3.01,   0, 1.0,
                   1.0,   0, 1.01,   0,
                     0, 1.0,   0, 1.01;

            REQUIRE_MAT(state, result.state, eps);
            REQUIRE_MAT(cov, result.cov, eps);
        }

    }

    SECTION("correction step")
    {
        const double eps = 1e-6;
        ExtendedKalmanFilter ekf(
            new ConstVelMotionModel(),
            new IdentitySensorModel());
        Eigen::MatrixXd noise(2,2);
        noise << 0.1,   0,
               0, 0.1;

       SECTION("with regular input")
       {

            Eigen::VectorXd state(4);
            state << 0, 0, 0.5, 0.5;
            Eigen::MatrixXd cov(4, 4);
            cov << 2, 0 ,0, 0,
                   0, 2, 0, 0,
                   0, 0, 1, 0,
                   0, 0, 0, 1;
            Eigen::MatrixXd obs(2,4);
            obs << 1, 2, 3, 4,
                        4, 3, 2, 1;

            ekf.init(state, cov);
            ekf.correct(obs, noise);

            auto result = ekf.getEstimate();

            state << 0.0, 0.0, 0.5, 0.5;
            cov << 0.00995025, 0, 0, 0,
                     0, 0.00995025, 0, 0,
                     0, 0, 0.00990099, 0,
                     0, 0, 0, 0.00990099;

            REQUIRE_MAT(state, result.state, eps);
            REQUIRE_MAT(cov, result.cov, eps);
        }

        SECTION("with empty observations")
        {
            Eigen::VectorXd state(4);
            state << 0, 0, 0.5, 0.5;
            Eigen::MatrixXd cov(4, 4);
            cov << 2, 0 ,0, 0,
                   0, 2, 0, 0,
                   0, 0, 1, 0,
                   0, 0, 0, 1;

            ekf.init(state, cov);
            ekf.correct(Eigen::MatrixXd(2,0), noise);

            auto result = ekf.getEstimate();

            state << 0, 0, 0.5, 0.5;
            cov << 2, 0, 0, 0,
                   0, 2, 0, 0,
                   0, 0, 1, 0,
                   0, 0, 0, 1;

            REQUIRE_MAT(state, result.state, eps);
            REQUIRE_MAT(cov, result.cov, eps);
        }
    }
}
