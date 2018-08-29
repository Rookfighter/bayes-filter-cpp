/*
 * models.h
 *
 *  Created on: 17 Apr 2018
 *      Author: Fabian Meyer
 */

#ifndef BFCPP_MODELS_H_
#define BFCPP_MODELS_H_

#include <limits>
#include <Eigen/Geometry>
#include "bayes_filter/math.h"

namespace bf
{
    /** Base class for motion models. It estimates the jacobian if no jacobian
      * is given. */
    class MotionModel
    {
    private:
        void computeFiniteDifferences(const Eigen::VectorXd &state,
            const Eigen::VectorXd &controls,
            const Eigen::MatrixXd &observations,
            const Eigen::VectorXd &value,
            Eigen::MatrixXd &outJacobian,
            const double diff) const
        {
            Eigen::VectorXd stateTmp;
            Eigen::VectorXd valueTmp;
            Eigen::MatrixXd jacobianTmp;

            outJacobian.resize(value.size(), state.size());

            for(unsigned int i = 0; i < state.size(); ++i)
            {
                stateTmp = state;
                stateTmp(i) += diff;

                _estimateState(stateTmp, controls, observations, valueTmp,
                    jacobianTmp);
                assert(valueTmp.size() == value.size());

                outJacobian.col(i) = (valueTmp - value) / diff;
            }
        }

    public:
        MotionModel()
        {}
        virtual ~MotionModel()
        {}

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
            Eigen::MatrixXd &outJacobian) const
        {
            static const double diff = std::sqrt(
                std::numeric_limits<double>::epsilon());

            outJacobian.resize(0, 0);

            _estimateState(state, controls, observations, outValue,
                outJacobian);
            if(outJacobian.size() == 0)
            {
                computeFiniteDifferences(state, controls, observations,
                    outValue, outJacobian, diff);
            }
        }
    };

    class SensorModel
    {
    private:
        void computeFiniteDifferences(const Eigen::VectorXd &state,
            const Eigen::MatrixXd &observations,
            const Eigen::VectorXd &value,
            Eigen::MatrixXd &outJacobian,
            const double diff) const
        {
            Eigen::VectorXd stateTmp;
            Eigen::MatrixXd valueTmp;
            Eigen::MatrixXd jacobianTmp;

            outJacobian.resize(value.size(), state.size());

            for(unsigned int i = 0; i < state.size(); ++i)
            {
                stateTmp = state;
                stateTmp(i) += diff;

                _estimateObservations(stateTmp, observations, valueTmp, jacobianTmp);
                assert(valueTmp.size() == value.size());

                outJacobian.col(i) = mat2vec(valueTmp - value) / diff;
            }
        }
    public:
        SensorModel()
        {}
        virtual ~SensorModel()
        {}

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
            Eigen::MatrixXd &outJacobian) const
        {
            static const double diff = std::sqrt(
                std::numeric_limits<double>::epsilon());

            outJacobian.resize(0, 0);

            _estimateObservations(state, observations, outValue, outJacobian);
            if(outJacobian.size() == 0)
            {
                computeFiniteDifferences(state, observations, outValue,
                    outJacobian, diff);
            }
        }

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
