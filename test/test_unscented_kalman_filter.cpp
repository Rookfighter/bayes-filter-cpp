/*
 * test_unscented_kalman_filter.cpp
 *
 *  Created on: 12 Jun 2018
 *      Author: Fabian Meyer
 */

#include "bayes_filter/unscented_kalman_filter.h"
#include "eigen_assert.h"
#include "invitro_models.h"
#include <catch.hpp>

using namespace bf;

TEST_CASE("Unscented Kalman Filter")
{
    SECTION("initialize")
    {
        const double eps = 1e-6;
        UnscentedKalmanFilter ukf(
            new IdentityMotionModel(), new IdentitySensorModel());

        Eigen::VectorXd state(3);
        state << 1, 2, 3;
        Eigen::MatrixXd cov(3, 3);
        cov << 4, 0, 0, 0, 5, 0, 0, 0, 6;

        ukf.init(state, cov);
        auto result = ukf.getEstimate();

        REQUIRE_MAT(state, result.state, eps);
        REQUIRE_MAT(cov, result.cov, eps);
    }

    SECTION("prediction step")
    {
        const double eps = 1e-6;
        UnscentedKalmanFilter ukf(
            new ConstVelMotionModel(), new IdentitySensorModel());
        Eigen::MatrixXd noise(4, 4);
        noise << 0.1, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0.1;

        SECTION("with regular input")
        {
            Eigen::VectorXd state(4);
            state << 0, 0, 0.5, 0.5;
            Eigen::MatrixXd cov(4, 4);
            cov << 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
            Eigen::VectorXd controls(1);
            controls << 1;

            ukf.init(state, cov);
            ukf.predict(controls, Eigen::MatrixXd(), noise);

            auto result = ukf.getEstimate();

            state << 0.5, 0.5, 0.5, 0.5;
            cov << 3.01, 0, 1.0, 0, 0, 3.01, 0, 1.0, 1.0, 0, 1.01, 0, 0, 1.0, 0,
                1.01;

            REQUIRE_MAT(state, result.state, eps);
            REQUIRE_MAT(cov, result.cov, eps);
        }
    }

    SECTION("correction step")
    {
        const double eps = 1e-6;
        UnscentedKalmanFilter ukf(
            new ConstVelMotionModel(), new IdentitySensorModel());
        Eigen::MatrixXd noise(2, 2);
        noise << 0.1, 0, 0, 0.1;

        SECTION("with regular input")
        {
            Eigen::VectorXd state(4);
            state << 0, 0, 0.5, 0.5;
            Eigen::MatrixXd cov(4, 4);
            cov << 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
            Eigen::MatrixXd obs(2, 4);
            obs << 1, 2, 3, 4, 4, 3, 2, 1;

            ukf.init(state, cov);
            ukf.correct(obs, noise);

            auto result = ukf.getEstimate();

            state << 0, 0, 0.5, 0.5;
            cov << 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;

            REQUIRE_MAT(state, result.state, eps);
            REQUIRE_MAT(cov, result.cov, eps);
        }

        SECTION("with empty observations")
        {
            Eigen::VectorXd state(4);
            state << 0, 0, 0.5, 0.5;
            Eigen::MatrixXd cov(4, 4);
            cov << 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;

            ukf.init(state, cov);
            ukf.correct(Eigen::MatrixXd(2, 0), noise);

            auto result = ukf.getEstimate();

            state << 0, 0, 0.5, 0.5;
            cov << 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;

            REQUIRE_MAT(state, result.state, eps);
            REQUIRE_MAT(cov, result.cov, eps);
        }
    }
}
