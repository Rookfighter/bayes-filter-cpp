/*
 * models.h
 *
 *  Created on: 17 Apr 2018
 *      Author: Fabian Meyer
 */

#ifndef BFCPP_MODELS_H_
#define BFCPP_MODELS_H_

#include <Eigen/Geometry>

namespace bf
{
    /** Base class for motion models. It estimates the jacobian if no jacobian
      * is given. */
    class MotionModel
    {
    private:
        bool calcJacobian_;

        void computeFiniteDifferences(const Eigen::VectorXd &state,
            const Eigen::VectorXd &controls,
            const Eigen::MatrixXd &observations,
            const Eigen::VectorXd &value,
            Eigen::MatrixXd &outJacobian,
            const double diff) const;

    public:
        MotionModel();
        virtual ~MotionModel();

        void setCalculateJacobian(const bool calcJac);
        bool calculateJacobian() const;

        /** Estimates a new state given the current state, controls,
         *  observations and time since last state.
         *  If the jacobian is not calculated it will be estimated by a black
         *  box approach.
         *  @param state current state vector
         *  @param controls current control vector
         *  @param observations matrix of observations
         *  @param outValue the estimated state
         *  @param outJacobian the jacobian of the motion model */
        virtual void _estimateState(const Eigen::VectorXd &state,
            const Eigen::VectorXd &controls,
            const Eigen::MatrixXd &observations,
            Eigen::VectorXd &outValue,
            Eigen::MatrixXd &outJacobian) const = 0;

        virtual void estimateState(const Eigen::VectorXd &state,
            const Eigen::VectorXd &controls,
            const Eigen::MatrixXd &observations,
            Eigen::VectorXd &outValue,
            Eigen::MatrixXd &outJacobian) const;
    };

    class SensorModel
    {
    private:
        bool calcJacobian_;

        void computeFiniteDifferences(const Eigen::VectorXd &state,
            const Eigen::MatrixXd &observations,
            const Eigen::MatrixXd &value,
            Eigen::MatrixXd &outJacobian,
            const double diff) const;
    public:
        SensorModel();
        virtual ~SensorModel();

        void setCalculateJacobian(const bool calcJac);
        bool calculateJacobian() const;

        /** Estimates observations given the current state,
         *  observations and absolute time of the system.
         *  If the jacobian is not calculated it will be estimated by a black
         *  box approach.
         *  @param state current state vector
         *  @param observations matrix of observations
         *  @param outValue the expected observations
         *  @param outJacobian the jacobian of the sensor model (vectorized)*/
        virtual void _estimateObservations(const Eigen::VectorXd &state,
            const Eigen::MatrixXd &observations,
            Eigen::MatrixXd &outValue,
            Eigen::MatrixXd &outJacobian) const = 0;

        virtual void estimateObservations(const Eigen::VectorXd &state,
            const Eigen::MatrixXd &observations,
            Eigen::MatrixXd &outValue,
            Eigen::MatrixXd &outJacobian) const;

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
