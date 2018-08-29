/*
 * extended_kalman_filter.cpp
 *
 *  Created on: 03 Jul 2018
 *      Author: Fabian Meyer
 */

#include "bayes_filter/extended_kalman_filter.h"

namespace bf
{
    ExtendedKalmanFilter::ExtendedKalmanFilter()
        : BayesFilter(), state_(), cov_()
    {}
    ExtendedKalmanFilter::ExtendedKalmanFilter(MotionModel *mm, SensorModel *sm)
        : BayesFilter(mm, sm), state_(), cov_()
    {}
    ExtendedKalmanFilter::~ExtendedKalmanFilter()
    {}

    StateEstimate ExtendedKalmanFilter::getEstimate() const
    {
        return {state_, cov_};
    }

    void ExtendedKalmanFilter::init(const Eigen::VectorXd &state,
        const Eigen::MatrixXd &cov)
    {
        assert(state.size() == cov.rows());
        assert(state.size() == cov.cols());
        
        state_ = state;
        cov_ = cov;
    }

    void ExtendedKalmanFilter::predict(const Eigen::VectorXd &controls,
        const Eigen::MatrixXd &observations,
        const Eigen::MatrixXd &noise)
    {
        assert(noise.cols() == cov_.cols());
        assert(noise.rows() == cov_.rows());

        Eigen::VectorXd value;
        Eigen::MatrixXd jacobian;

        motionModel().estimateState(state_, controls, observations, value,
            jacobian);

        assert(value.size() == state_.size());
        assert(jacobian.rows() == state_.size());
        assert(jacobian.cols() == state_.size());

        state_ = value;
        cov_ = jacobian * cov_ * jacobian.transpose() +
            noise * noise.transpose();
    }

    void ExtendedKalmanFilter::correct(const Eigen::MatrixXd &observations,
        const Eigen::MatrixXd &noise)
    {
        assert(noise.rows() == observations.rows());
        assert(noise.cols() == observations.rows());

        // if there are no observations, simply return
        if(observations.cols() == 0)
            return;

        Eigen::MatrixXd value;
        Eigen::MatrixXd jacobian;

        sensorModel().estimateObservations(state_, observations, value,
            jacobian);

        assert(value.rows() == observations.rows());
        assert(value.cols() == observations.cols());
        assert(jacobian.rows() == observations.size());
        assert(jacobian.cols() == state_.size());

        Eigen::MatrixXd tmp = jacobian * cov_ * jacobian.transpose();
        for(unsigned int i = 0; i < tmp.rows(); ++i)
        {
            unsigned int j = i % noise.rows();
            tmp(i, i) += noise(j, j) * noise(j, j);
        }
        tmp = tmp.inverse();

        // calculate kalman gain
        Eigen::MatrixXd kalGain = cov_ * jacobian.transpose() * tmp;
        Eigen::VectorXd diff = mat2vec(observations - result.val);

        state_ += kalGain * diff;
        cov_ -= kalGain * result.jac * cov_;
    }
}
