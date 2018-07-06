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
    {

    }

    UnscentedKalmanFilter::UnscentedKalmanFilter(MotionModel *mm,
            SensorModel *sm)
        : BayesFilter(mm, sm), unscentTrans_()
    {

    }

    UnscentedKalmanFilter::~UnscentedKalmanFilter()
    {

    }

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
        auto sigma = unscentTrans_.calcSigmaPoints(state_, cov_, normState_);
        // transform sigma points through motion model
        for(unsigned int i = 0; i < sigma.points.cols(); ++i)
        {
            // estimate new state for sigma point
            auto mmResult = motionModel().estimateState(
                                sigma.points.col(i), controls, observations);
            // normalize resulting state
            normalizeState(mmResult.val);
            sigma.points.col(i) = mmResult.val;
        }

        auto mu = unscentTrans_.recoverMean(sigma, meanState_);
        auto cov = unscentTrans_.recoverCovariance(sigma, mu, normState_);

        // update current state estimate
        state_ = mu;
        cov_ = cov + noise * noise.transpose();
    }

    void UnscentedKalmanFilter::correct(const Eigen::MatrixXd &observations,
                                        const Eigen::MatrixXd &noise)
    {
        // transform observation matrix into vector
        Eigen::VectorXd obs = mat2vec(observations);

        assert(noise.rows() == noise.cols());
        assert(noise.rows() == observations.rows());

        // reshape noise (=single measurement stddev on main diagonal) into
        // covariance matrix for all received observations
        Eigen::MatrixXd obsCov;
        obsCov.setZero(obs.size(), obs.size());
        for(unsigned int i = 0; i <  obs.size(); ++i)
        {
            unsigned int j = i % noise.rows();
            obsCov(i, i) = noise(j, j) * noise(j, j);
        }

        // calculate sigma points
        auto sigmaA = unscentTrans_.calcSigmaPoints(state_, cov_, normState_);
        SigmaPoints sigmaB;
        sigmaB.points.resize(0, sigmaA.points.cols());
        sigmaB.weights = sigmaA.weights;

        // transform sigma points through sensor model
        for(unsigned int i = 0; i < sigmaA.points.cols(); ++i)
        {
            // estimate observation for this sigma point
            auto smResult = sensorModel().estimateObservations(
                sigmaA.points.col(i), observations);
            // if sigmaB was not initialized init it now
            if(sigmaB.points.rows() < smResult.val.size())
            {
                sigmaB.points.resize(smResult.val.size(),
                                     sigmaA.points.cols());
            }
            // reshape resulting observations
            sigmaB.points.col(i) = mat2vec(smResult.val);

        }

        auto mu = unscentTrans_.recoverMean(sigmaB, meanObs_);
        auto cov = unscentTrans_.recoverCovariance(sigmaB, mu, normObs_);
        auto crossCov = unscentTrans_.recoverCrossCorrelation(
                            sigmaA, state_, normState_,
                            sigmaB, mu, normObs_);

        assert(mu.size() == obs.size());
        assert(cov.rows() == noise.rows());
        assert(cov.cols() == noise.cols());
        assert(crossCov.rows() == state_.size());
        assert(crossCov.cols() == mu.size());

        cov += obsCov;
        // calculate kalman gain
        Eigen::MatrixXd kalGain = crossCov * cov.inverse();

        // correct current state estimate
        Eigen::MatrixXd diff = obs - mu;
        normalizeObservations(diff);
        state_ += kalGain * diff;
        cov_ -= kalGain * cov * kalGain.transpose();
    }
}
