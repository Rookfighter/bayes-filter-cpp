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
    UnscentedKalmanFilter::UnscentedKalmanFilter()
        : BayesFilter()
    {

    }

    UnscentedKalmanFilter::UnscentedKalmanFilter(MotionModel *mm, SensorModel *sm)
        : BayesFilter(mm, sm)
    {

    }

    UnscentedKalmanFilter::~UnscentedKalmanFilter()
    {

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

    Eigen::VectorXd UnscentedKalmanFilter::estimateState(
        const Eigen::VectorXd &state,
        const Eigen::VectorXd &controls,
        const Eigen::MatrixXd &observations) const
    {
        return motionModel().estimateState(state, controls, observations).val;
    }

    void UnscentedKalmanFilter::predict(const Eigen::VectorXd &controls,
                                        const Eigen::MatrixXd &observations,
                                        const Eigen::MatrixXd &motionCov)
    {
        assert(state_.size() == motionCov.rows());
        assert(state_.size() == motionCov.cols());

        // bind transform function for UT
        UnscentedTransform::TransformFunc func =
            std::bind(&UnscentedKalmanFilter::estimateState, this,
                      std::placeholders::_1,
                      std::cref(controls),
                      std::cref(observations));
        // perform unscented transform on current state estimate
        auto result = unscentTrans_.transform(state_, cov_, func);

        // update current state estimate
        state_ = result.state;
        cov_ = result.cov + motionCov;
    }

    Eigen::VectorXd UnscentedKalmanFilter::estimateObservations(
        const Eigen::VectorXd &state,
        const Eigen::MatrixXd &observations) const
    {
        Eigen::MatrixXd estObs = sensorModel().estimateObservations(state,
                                 observations).val;
        return mat2vec(estObs);
    }

    void UnscentedKalmanFilter::correct(const Eigen::MatrixXd &observations,
                                        const Eigen::MatrixXd &sensorCov)
    {
        Eigen::VectorXd obs = mat2vec(observations);

        assert(sensorCov.rows() == sensorCov.cols());
        Eigen::MatrixXd obsCov = Eigen::MatrixXd::Zero(obs.size(), obs.size());
        for(unsigned int i = 0; i <  obs.size(); ++i)
        {
            unsigned int j = i % sensorCov.rows();
            obsCov(i, i) = sensorCov(j, j);
        }

        // bind transform function for UT
        UnscentedTransform::TransformFunc func =
            std::bind(&UnscentedKalmanFilter::estimateObservations, this,
                      std::placeholders::_1,
                      std::cref(observations));

        // perform unscented transform
        auto result = unscentTrans_.transform(state_, cov_, func, true);

        assert(result.state.size() == obs.size());
        assert(result.cov.rows() == obsCov.rows());
        assert(result.cov.cols() == obsCov.cols());

        result.cov += obsCov;
        for(unsigned int i = 0; i < result.cov.cols(); ++ i)
        {
            if(iszero(result.cov(i, i), 1e-12))
                result.cov(i, i) = 1;
        }

        // calculate kalman gain
        Eigen::MatrixXd kalGain = result.crossCov * result.cov.inverse();

        // correct current state estimate
        state_ += kalGain * (obs - result.state);
        cov_ -= kalGain * result.cov * kalGain.transpose();
    }
}
