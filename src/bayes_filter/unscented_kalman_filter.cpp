/*
 * unscented_kalman_filter.cpp
 *
 *  Created on: 18 May 2018
 *      Author: Fabian Meyer
 */

#include "bayes_filter/unscented_kalman_filter.h"
#include "bayes_filter/math.h"

using namespace std::placeholders;

namespace bf
{
    static Eigen::VectorXd noNormalize(const Eigen::VectorXd &v)
    {
        return v;
    }

    UnscentedKalmanFilter::UnscentedKalmanFilter()
        : BayesFilter(), unscentTrans_(),
        normState_(std::bind(noNormalize, _1)),
        normObs_(std::bind(noNormalize, _1))
    {

    }

    UnscentedKalmanFilter::UnscentedKalmanFilter(MotionModel *mm, SensorModel *sm)
        : BayesFilter(mm, sm),
        normState_(std::bind(noNormalize, _1)),
        normObs_(std::bind(noNormalize, _1))
    {

    }

    UnscentedKalmanFilter::~UnscentedKalmanFilter()
    {

    }

    void UnscentedKalmanFilter::setNormalizeState(const NormalizeFunc &normalize)
    {
        normState_ = normalize;
    }

    void UnscentedKalmanFilter::setNormalizeObservation(const NormalizeFunc &normalize)
    {
        normObs_ = normalize;
    }

    std::pair<Eigen::VectorXd, Eigen::MatrixXd> UnscentedKalmanFilter::getEstimate()
    const
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
                                        const Eigen::MatrixXd &motionCov)
    {
        assert(state_.size() == motionCov.rows());
        assert(state_.size() == motionCov.cols());

        auto sigma = unscentTrans_.calcSigmaPoints(state_, cov_, normState_);
        for(unsigned int i = 0; i < sigma.points.cols(); ++i)
        {
            // transform points through motion model
            auto mmResult = motionModel().estimateState(
                sigma.points.col(i), controls, observations);
            // normalize resulting state
            sigma.points.col(i) = normState_(mmResult.val);
        }

        auto mu = unscentTrans_.recoverMean(sigma, normState_);
        auto cov = unscentTrans_.recoverCovariance(sigma, mu, normState_);

        // update current state estimate
        state_ = mu;
        cov_ = cov + motionCov;
    }

    void UnscentedKalmanFilter::correct(const Eigen::MatrixXd &observations,
                                        const Eigen::MatrixXd &sensorCov)
    {
        // transform observation matrix into vector
        Eigen::VectorXd obs = mat2vec(observations);

        assert(sensorCov.rows() == sensorCov.cols());
        Eigen::MatrixXd obsCov = Eigen::MatrixXd::Zero(obs.size(), obs.size());
        for(unsigned int i = 0; i <  obs.size(); ++i)
        {
            unsigned int j = i % sensorCov.rows();
            obsCov(i, i) = sensorCov(j, j);
        }

        auto sigmaA = unscentTrans_.calcSigmaPoints(state_, cov_, normState_);
        SigmaPoints sigmaB;
        for(unsigned int i = 1; i < sigmaA.points.cols(); ++i)
        {
            // transform points through sensor model
            auto smResult = sensorModel().estimateObservations(
                sigmaA.points.col(i), observations);
            // if sigmaB was not initialized init it now
            if(sigmaB.points.rows() < smResult.val.size())
                sigmaB.points.resize(smResult.val.size(), sigmaA.points.cols());
            // normalize resulting observations
            sigmaB.points.col(i) = normObs_(mat2vec(smResult.val));
        }

        auto mu = unscentTrans_.recoverMean(sigmaB, normObs_);
        auto cov = unscentTrans_.recoverCovariance(sigmaB, mu, normObs_);
        auto crossCov = unscentTrans_.recoverCrossCovariance(
            sigmaA, state_, normState_,
            sigmaB, mu, normObs_);

        assert(mu.size() == obs.size());
        assert(cov.rows() == obsCov.rows());
        assert(cov.cols() == obsCov.cols());
        assert(crossCov.rows() == state_.size());
        assert(crossCov.cols() == mu.size());

        cov += obsCov;
        // calculate kalman gain
        Eigen::MatrixXd kalGain = crossCov * cov.inverse();

        // correct current state estimate
        state_ += kalGain * normObs_(obs - mu);
        cov_ -= kalGain * cov * kalGain.transpose();
    }
}
