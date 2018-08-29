/*
 * models.cpp
 *
 *  Created on: 17 Apr 2018
 *      Author: Fabian Meyer
 */

#include <limits>
#include "bayes_filter/models.h"
#include "bayes_filter/math.h"

namespace bf
{
    MotionModel::MotionModel()
    {}
    MotionModel::~MotionModel()
    {}

    void MotionModel::computeFiniteDifferences(const Eigen::VectorXd &state,
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

    void MotionModel::estimateState(const Eigen::VectorXd &state,
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

    SensorModel::SensorModel()
    {}
    SensorModel::~SensorModel()
    {}

    void SensorModel::computeFiniteDifferences(const Eigen::VectorXd &state,
        const Eigen::MatrixXd &observations,
        const Eigen::MatrixXd &value,
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

    void SensorModel::estimateObservations(const Eigen::VectorXd &state,
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
}
