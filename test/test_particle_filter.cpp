/*
 * test_particle_filter.cpp
 *
 *  Created on: 12 Jun 2018
 *      Author: Fabian Meyer
 */

#include <iostream>
#include <catch.hpp>
#include "bayes_filter/particle_filter.h"
#include "eigen_assert.h"
#include "invitro_models.h"

using namespace bf;

TEST_CASE("Particle Filter")
{
    SECTION("initialize")
    {
        const double eps = 1e-5;
        ParticleFilter pf(
            new IdentityMotionModel(),
            new IdentitySensorModel());
        pf.setParticleCount(1000);
        pf.setSeed(0);

        Eigen::VectorXd state(3);
        state << 1, 2, 3;
        Eigen::MatrixXd cov(3,3);
        cov << 4, 0 ,0,
               0, 5, 0,
               0, 0, 6;

        pf.init(state, cov);
        auto result = pf.getEstimate();

        state << 1.05751, 1.98381, 2.86014;
        cov << 3.96552, 3.89074, -0.516142,
              -0.109259, 7.84079, 1.78106,
               0.0406131, 2.23055, 9.93785;

        REQUIRE_MAT(state, result.state, eps);
        REQUIRE_MAT(cov, result.cov, eps);
    }

    SECTION("prediction step")
    {
        const double eps = 1e-6;
        ParticleFilter pf(
            new ConstVelMotionModel(),
            new IdentitySensorModel());
        pf.setParticleCount(1000);
        pf.setSeed(0);

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

            pf.init(state, cov);
            pf.predict(controls, Eigen::MatrixXd(), noise);

            auto result = pf.getEstimate();

            state << 0.44533, 0.495151, 0.466665, 0.509483;
            cov << 2.93074, -4.57248, -2.65221, 1.09949,
                -0.0307314, 7.01612, 3.86597, 4.98291,
                0.950251, 0.00365655, 6.2079, 4.31703,
                -0.0665163, 1.38553, 7.64476, -1.35022;

            REQUIRE_MAT(state, result.state, eps);
            REQUIRE_MAT(cov, result.cov, eps);
        }

    }

    SECTION("correction step")
    {
        const double eps = 1e-6;
        ParticleFilter pf(
            new ConstVelMotionModel(),
            new IdentitySensorModel());
        pf.setParticleCount(1000);
        pf.setSeed(0);

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

            pf.init(state, cov);
            pf.correct(obs, noise);

            auto result = pf.getEstimate();

            state <<  -0.0178023, -0.00721643, 0.468539, 0.508503;
            cov << 1.98429, 0.0359382, -0.0185409, -0.031198,
                0.0349382, 2.07037, 0.00561825, -0.0103627,
                -0.0195409, 0.00561825, 0.966072, -0.0394861,
                -0.032198, -0.0103627, -0.0394861, 1.01497;

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

            pf.init(state, cov);
            pf.correct(Eigen::MatrixXd(2,0), noise);

            auto result = pf.getEstimate();

            state <<  -0.0178023, -0.00721643, 0.468539, 0.508503;
            cov << 1.98429, 0.0349382, 0.480459, 0.467802,
                0.0349382, 3.48358, 1.00462, 0.988637,
                0.0504525, -0.196151, -0.33417, 0.0114127,
                -0.064396, -0.0217254, -0.0799722, 2.02894;

            REQUIRE_MAT(state, result.state, eps);
            REQUIRE_MAT(cov, result.cov, eps);
        }
    }
}
