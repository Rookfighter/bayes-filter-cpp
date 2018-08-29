/*
 * invitro_models.h
 *
 *  Created on: 12 Jun 2018
 *      Author: Fabian Meyer
 */

#ifndef BFCPP_TEST_INVITRO_MODELS_H_
#define BFCPP_TEST_INVITRO_MODELS_H_

#include "bayes_filter/models.h"

class IdentityMotionModel : public bf::MotionModel
{
public:
    IdentityMotionModel()
    {}
    ~IdentityMotionModel()
    {}

    void _estimateState(const Eigen::VectorXd &state,
        const Eigen::VectorXd &,
        const Eigen::MatrixXd &,
        Eigen::VectorXd &outValue,
        Eigen::MatrixXd &outJacobian) const override
    {
        outValue = state;
        outJacobian = Eigen::MatrixXd::Identity(state.size(), state.size());
    }
};

class IdentitySensorModel : public bf::SensorModel
{
public:
    IdentitySensorModel()
    {}
    ~IdentitySensorModel()
    {}

    void _estimateObservations(const Eigen::VectorXd &state,
        const Eigen::MatrixXd &observations,
        Eigen::MatrixXd &outValue,
        Eigen::MatrixXd &outJacobian) const override
    {
        outValue = observations;
        outJacobian =
            Eigen::MatrixXd::Identity(observations.size(), state.size());
    }

    double likelihood(const Eigen::VectorXd &,
        const Eigen::MatrixXd &,
        const Eigen::MatrixXd &) const override
    {
        return 1.0;
    }
};

class ConstVelMotionModel : public bf::MotionModel
{
public:
    ConstVelMotionModel()
    {}
    ~ConstVelMotionModel()
    {}

    void _estimateState(const Eigen::VectorXd &state,
        const Eigen::VectorXd &controls,
        const Eigen::MatrixXd &,
        Eigen::VectorXd &outValue,
        Eigen::MatrixXd &outJacobian) const override
    {
        assert(state.size() == 4);
        assert(controls.size() == 1);

        double dt = controls(0);
        Eigen::Vector2d pos(state(0), state(1));
        Eigen::Vector2d vel(state(2), state(3));

        outValue.resize(state.size());
        outValue << pos(0) + vel(0) * dt, pos(1) + vel(1) * dt, vel(0),
            vel(1);
        outJacobian.resize(state.size(), state.size());
        outJacobian << 1, 0, dt, 0, 0, 1, 0, dt, 0, 0, 1, 0, 0, 0, 0, 1;
    }
};

#endif
