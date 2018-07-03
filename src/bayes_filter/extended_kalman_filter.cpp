/*
 * extended_kalman_filter.cpp
 *
 *  Created on: 03 Jul 2018
 *      Author: Fabian Meyer
 */


#include "bayes_filter/extended_kalman_filter.h"
#include "bayes_filter/math.h"

namespace bf
{
    ExtendedKalmanFilter::ExtendedKalmanFilter()
        : BayesFilter(), state_(), cov_()
    {

    }

    ExtendedKalmanFilter::ExtendedKalmanFilter(MotionModel *mm, SensorModel *sm)
        : BayesFilter(mm, sm), state_(), cov_()
    {

    }

    ExtendedKalmanFilter::~ExtendedKalmanFilter()
    {

    }

    StateEstimate ExtendedKalmanFilter::getEstimate() const
    {
        return {state_, cov_};
    }

    void ExtendedKalmanFilter::init(const Eigen::VectorXd &state,
        const Eigen::MatrixXd &cov)
    {
        state_ = state;
        cov_ = cov;
    }

    void ExtendedKalmanFilter::predict(const Eigen::VectorXd &controls,
        const Eigen::MatrixXd &observations,
        const Eigen::MatrixXd &motionCov)
     {
         auto result = motionModel().estimateState(state_, controls,
             observations);

         state_ = result.val;
         cov_ = result.jac * cov_ * result.jac.transpose() + motionCov;
     }
    void ExtendedKalmanFilter::correct(const Eigen::MatrixXd &observations,
        const Eigen::MatrixXd &sensorCov)
    {
        auto result = sensorModel().estimateObservations(state_, observations);

        Eigen::MatrixXd sensorCovScal = diagMat(sensorCov, observations.cols());

        Eigen::MatrixXd jacT = result.jac.transpose();
        Eigen::MatrixXd kalGain = cov_ * jacT * (result.jac * cov_ * jacT + sensorCovScal).inverse();
        Eigen::MatrixXd diff = observations - result.val;
        normalizeObservations(diff);
        Eigen::VectorXd diffV = mat2vec(diff);

        state_ += kalGain * diffV;
        cov_ -= kalGain * result.jac * cov_;
    }
}
