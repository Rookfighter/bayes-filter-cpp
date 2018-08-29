/*
 * test_particle_filter.cpp
 *
 *  Created on: 12 Jun 2018
 *      Author: Fabian Meyer
 */

#include "bayes_filter/particle_filter.h"
#include "eigen_assert.h"
#include "invitro_models.h"
#include <catch.hpp>
#include <iostream>

using namespace bf;

TEST_CASE("Particle Filter")
{
    SECTION("initialize")
    {
        ParticleFilter pf(new IdentityMotionModel(), new IdentitySensorModel());
        pf.setParticleCount(50000);
        pf.setSeed(0);

        Eigen::VectorXd state(3);
        state << 1, 2, 3;
        Eigen::MatrixXd cov(3, 3);
        cov << 4, 0, 0, 0, 5, 0, 0, 0, 6;

        pf.init(state, cov);
        auto result = pf.getEstimate();

        REQUIRE_MAT(state, result.state, 0.1);
        REQUIRE_MAT(cov, result.cov, 0.1);
    }

    SECTION("prediction step")
    {
        ParticleFilter pf(new ConstVelMotionModel(), new IdentitySensorModel());
        pf.setParticleCount(10000);
        pf.setSeed(0);

        Eigen::MatrixXd noise(4, 4);
        noise << 0.01, 0, 0, 0,
                 0, 0.01, 0, 0,
                 0, 0, 0.01, 0,
                 0, 0, 0, 0.01;

        SECTION("with regular input")
        {
            Eigen::VectorXd state(4);
            state << 0, 0, 0.5, 0.5;
            Eigen::MatrixXd cov(4, 4);
            cov << 2, 0, 0, 0,
                   0, 2, 0, 0,
                   0, 0, 1, 0,
                   0, 0, 0, 1;
            // move by one second
            Eigen::VectorXd controls(1);
            controls << 1;

            pf.init(state, cov);
            pf.predict(controls, Eigen::MatrixXd(), noise);

            auto result = pf.getEstimate();

            state << 0.5, 0.5, 0.5, 0.5;
            cov << 3, 0, 1, 0,
                   0, 3, 0, 1,
                   1, 0, 1, 0,
                   0, 1, 0, 1;

            REQUIRE_MAT(state, result.state, 0.1);
            REQUIRE_MAT(cov, result.cov, 0.1);
        }
    }

    SECTION("correction step")
    {
        ParticleFilter pf(new ConstVelMotionModel(), new IdentitySensorModel());
        pf.setParticleCount(10000);
        pf.setSeed(0);

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

            pf.init(state, cov);
            pf.correct(obs, noise);

            auto result = pf.getEstimate();

            REQUIRE_MAT(state, result.state, 0.1);
            REQUIRE_MAT(cov, result.cov, 0.1);
        }

        SECTION("with empty observations")
        {
            Eigen::VectorXd state(4);
            state << 0, 0, 0.5, 0.5;
            Eigen::MatrixXd cov(4, 4);
            cov << 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;

            pf.init(state, cov);
            pf.correct(Eigen::MatrixXd(2, 0), noise);

            auto result = pf.getEstimate();

            REQUIRE_MAT(state, result.state, 0.1);
            REQUIRE_MAT(cov, result.cov, 0.1);
        }
    }
}
