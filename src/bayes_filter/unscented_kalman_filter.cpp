/*
 * unscented_kalman_filter.cpp
 *
 *  Created on: 18 May 2018
 *      Author: Fabian Meyer
 */

#include "bayes_filter/unscented_kalman_filter.h"
#include "bayes_filter/math.h"

namespace bf
{
    UnscentedKalmanFilter::UnscentedKalmanFilter()
        : BayesFilter(), unscentTrans_()
    {}

    UnscentedKalmanFilter::UnscentedKalmanFilter(
        MotionModel *mm, SensorModel *sm)
        : BayesFilter(mm, sm), unscentTrans_()
    {}

    UnscentedKalmanFilter::~UnscentedKalmanFilter()
    {}

    StateEstimate UnscentedKalmanFilter::getEstimate() const
    {
        return {state_, cov_};
    }

    void UnscentedKalmanFilter::init(const Eigen::VectorXd &state,
        const Eigen::MatrixXd &cov)
    {
        assert(state.size() == cov.rows());
        assert(state.size() == cov.cols());

        state_ = state;
        cov_ = cov;
    }

    void UnscentedKalmanFilter::predict(const Eigen::VectorXd &controls,
        const Eigen::MatrixXd &observations,
        const Eigen::MatrixXd &noise)
    {
        assert(state_.size() == noise.rows());
        assert(state_.size() == noise.cols());

        // calculate sigma points
        SigmaPoints sigma;
        unscentTrans_.calcSigmaPoints(state_, cov_, sigma);

        // transform sigma points through motion model
        Eigen::VectorXd value;
        Eigen::MatrixXd jacobian;
        for(unsigned int i = 0; i < sigma.points.cols(); ++i)
        {
            // estimate new state for sigma point
            motionModel().estimateState(sigma.points.col(i), controls,
                observations, value, jacobian);
            // normalize resulting state
            sigma.points.col(i) = value;
        }

        Eigen::VectorXd mean;
        Eigen::MatrixXd cov;
        unscentTrans_.recoverMean(sigma, mean);
        unscentTrans_.recoverCovariance(sigma, mean, cov);

        // update current state estimate
        state_ = mean;
        cov_ = cov + noise * noise.transpose();
    }

    void UnscentedKalmanFilter::correct(const Eigen::MatrixXd &observations,
        const Eigen::MatrixXd &noise)
    {
        assert(noise.rows() == noise.cols());
        assert(noise.rows() == observations.rows());

        if(observations.cols() == 0)
            return;

        // transform observation matrix into vector
        Eigen::VectorXd obs = mat2vec(observations);

        // calculate sigma points
        SigmaPoints sigmaA;
        unscentTrans_.calcSigmaPoints(state_, cov_, sigmaA);

        SigmaPoints sigmaB;
        sigmaB.points.resize(0, sigmaA.points.cols());
        sigmaB.weights = sigmaA.weights;

        // transform sigma points through sensor model
        Eigen::MatrixXd value;
        Eigen::MatrixXd jacobian;
        for(unsigned int i = 0; i < sigmaA.points.cols(); ++i)
        {
            // estimate observation for this sigma point
            sensorModel().estimateObservations(sigmaA.points.col(i),
                observations, value, jacobian);
            // if sigmaB was not initialized init it now
            if(sigmaB.points.rows() < value.size())
                sigmaB.points.resize(value.size(), sigmaA.points.cols());
            // reshape resulting observations
            sigmaB.points.col(i) = mat2vec(value);
        }

        Eigen::VectorXd mean;
        Eigen::MatrixXd cov;
        Eigen::MatrixXd crossCov;

        unscentTrans_.recoverMean(sigmaB, mean);
        unscentTrans_.recoverCovariance(sigmaB, mean, cov);
        unscentTrans_.recoverCrossCorrelation(sigmaA, state_, sigmaB, mean,
            crossCov);

        assert(mean.size() == obs.size());
        assert(cov.rows() == obs.size());
        assert(cov.cols() == obs.size());
        assert(crossCov.rows() == state_.size());
        assert(crossCov.cols() == mean.size());

        // add noise to covariance
        for(unsigned int i = 0; i < obs.size(); ++i)
        {
            unsigned int j = i % noise.rows();
            cov(i, i) += noise(j, j) * noise(j, j);
        }

        // calculate kalman gain
        Eigen::MatrixXd kalGain = crossCov * cov.inverse();

        // correct current state estimate
        Eigen::MatrixXd diff = obs - mean;
        state_ += kalGain * diff;
        cov_ -= kalGain * cov * kalGain.transpose();
    }
}
