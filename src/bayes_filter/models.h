/*
 * models.h
 *
 *  Created on: 17 Apr 2018
 *      Author: Fabian Meyer
 */

#ifndef BFCPP_MODELS_H_
#define BFCPP_MODELS_H_

#include <vector>
#include <sstream>
#include <iomanip>
#include <Eigen/Dense>

namespace bf
{
    class MotionModel
    {
    public:
        /** The first component of a MotionModel::Result is the new state
          * estimate. The second component is the Jacobian at the given
          * point. */
        struct Result
        {
            Eigen::VectorXd val;
            Eigen::MatrixXd jac;
        };

        MotionModel() {}
        virtual ~MotionModel() {}

        /** Estimates a new state given the current state, controls,
         *  observations and time since last state.
         *  @param state current state vector
         *  @param controls current control vector
         *  @param observations matrix of observations
         *  @return new state vector */
        virtual Result estimateState(const Eigen::VectorXd &state,
                                     const Eigen::VectorXd &controls,
                                     const Eigen::MatrixXd &observations) const = 0;
    };

    class SensorModel
    {
    public:
        SensorModel() {}
        virtual ~SensorModel() {}

        struct Result
        {
            Eigen::MatrixXd val;
            Eigen::MatrixXd jac;
        };

        /** Estimates observations given the current state,
         *  observations and absolute time of the system.
         *  @param state current state vector
         *  @param observations matrix of observations
         *  @return observation estimate. */
        virtual Result estimateObservations(const Eigen::VectorXd &state,
                                            const Eigen::MatrixXd &observations) const = 0;

        /** Calculates the likelihood of p(z|x).
         *  @param pose current pose estimate
         *  @param observations matrix of observations
         *  @param noise uncertainty of observations (sqrt of covariance)
         *  @return probability of these observations given the state
         */
        virtual double likelihood(const Eigen::VectorXd &state,
                                  const Eigen::MatrixXd &observations,
                                  const Eigen::MatrixXd &noise) const = 0;
    };
}

#endif
