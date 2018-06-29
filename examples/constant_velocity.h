/*
 * constant_velocity.h
 *
 *  Created on: 29 Jun 2018
 *      Author: Fabian Meyer
 */

#ifndef CONSTANT_VELOCITY_H_
#define CONSTANT_VELOCITY_H_

#include <bayes_filter/models.h>
#include <cassert>

class ConstantVelocity : public bf::MotionModel
{
public:
    ConstantVelocity() {}
    ~ConstantVelocity() {}

    Result estimateState(const Eigen::VectorXd &state,
        const Eigen::VectorXd &controls,
        const Eigen::MatrixXd &) const override
    {
        // state is of form [x, y, vx, vy]^T
        assert(state.size() == 4);
        assert(controls.size() == 1);

        Result result;
        result.val.resize(state.size());
        result.jac.resize(result.val.size(), state.size());

        double dt = controls(0);
        double x  = state(0);
        double y  = state(1);
        double vx  = state(2);
        double vy  = state(3);

        result.val << x + vx * dt, y + vy * dt, vx, vy;
        result.jac <<
            1.0, 0.0,  dt, 0.0,
            0.0, 1.0, 0.0,  dt,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0;

        return result;
    }
};

#endif
