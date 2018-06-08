/*
 * unscented_transform.cpp
 *
 *  Created on: 18 May 2018
 *      Author: Fabian Meyer
 */

#include "bfcpp/unscented_transform.h"

namespace bf
{
    UnscentedTransform::UnscentedTransform()
        : kappa_(1.0), alpha_(0.9), beta_(2.0)
    {

    }

    UnscentedTransform::~UnscentedTransform()
    {

    }

    void UnscentedTransform::setKappa(const double kappa)
    {
        kappa_ = kappa;
    }

    void UnscentedTransform::setAlpha(const double alpha)
    {
        alpha_ = alpha;
    }

    void UnscentedTransform::setBeta(const double beta)
    {
        beta_ = beta;
    }

    SigmaPoints UnscentedTransform::calcSigmaPoints(const Eigen::VectorXd &state,
            const Eigen::MatrixXd &cov) const
    {
        assert(state.size() == cov.rows());
        assert(state.size() == cov.cols());

        unsigned int n = state.size();
        double nd = static_cast<double>(n);
        double lambda = alpha_ * alpha_ * (nd + kappa_) - nd;

        SigmaPoints result;
        result.points.resize(n, 2 * n + 1);
        result.weights.resize(2, 2 * n + 1);

        Eigen::MatrixXd sigmaOffset = ((nd + lambda) * cov).llt().matrixL();

        result.points.col(0) = state;
        result.weights(0, 0) = lambda / (nd + lambda);
        result.weights(1, 0) = result.weights(0, 0) + (1 - alpha_ * alpha_ + beta_);
        double constWeight = 1.0 / (2.0 * (nd + lambda));

        for(unsigned int i = 0; i < n; ++i)
        {
            result.points.col(i + 1) = state + sigmaOffset.col(i);
            result.weights(0, i + 1) = constWeight;
            result.weights(1, i + 1) = constWeight;

            result.points.col(i + 1 + n) = state - sigmaOffset.col(i);
            result.weights(0, i + 1 + n) = constWeight;
            result.weights(1, i + 1 + n) = constWeight;
        }

        return result;
    }

    std::pair<Eigen::VectorXd, Eigen::MatrixXd> UnscentedTransform::recoverDistrib(
        const SigmaPoints &sigmaPoints) const
    {
        assert(sigmaPoints.points.cols() == sigmaPoints.weights.cols());

        Eigen::VectorXd mu = Eigen::VectorXd::Zero(sigmaPoints.points.rows());
        Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(mu.size(), mu.size());

        // calculate new mean
        for(unsigned int i = 0; i < sigmaPoints.points.cols(); ++i)
            mu += sigmaPoints.weights(0, i) * sigmaPoints.points.col(i);

        for(unsigned int i = 0; i < sigmaPoints.points.cols(); ++i)
        {
            Eigen::VectorXd diff = sigmaPoints.points.col(i) - mu;
            cov += sigmaPoints.weights(1, i) * diff * diff.transpose();
        }

        return {mu, cov};
    }

    Eigen::MatrixXd UnscentedTransform::calcCrossCov(const Eigen::VectorXd
            &stateOld,
            const SigmaPoints &sigmaOld,
            const Eigen::VectorXd &stateNew,
            const SigmaPoints &sigmaNew) const
    {
        assert(sigmaOld.points.rows() == stateOld.size());
        assert(sigmaNew.points.rows() == stateNew.size());
        assert(sigmaOld.points.cols() == sigmaNew.points.cols());

        Eigen::MatrixXd result = Eigen::MatrixXd::Zero(stateOld.size(),
                                 stateNew.size());

        for(unsigned int i = 0; i < sigmaOld.points.cols(); ++i)
        {
            Eigen::VectorXd diffA = sigmaOld.points.col(i) - stateOld;
            Eigen::VectorXd diffB = sigmaNew.points.col(i) - stateNew;
            result += sigmaOld.weights(1, i) * diffA * diffB.transpose();
        }

        return result;
    }

    UnscentedTransform::Result UnscentedTransform::transform(
        const Eigen::VectorXd &state,
        const Eigen::MatrixXd &cov,
        const TransformFunc &func,
        const bool cross) const
    {
        Result result;

        // calc sigmapoints with given state and covariance
        SigmaPoints sigmaOld = calcSigmaPoints(state, cov);

        // setup result sigma points set
        SigmaPoints sigmaNew;
        sigmaNew.weights = sigmaOld.weights;

        Eigen::VectorXd val = func(sigmaOld.points.col(0));
        sigmaNew.points.resize(val.size(), sigmaOld.points.cols());
        sigmaNew.points.col(0) = val;

        // transform sigma points through given function
        for(unsigned int i = 1; i < sigmaOld.points.cols(); ++i)
            sigmaNew.points.col(i) = func(sigmaOld.points.col(i));

        // recover distribution from transformed points
        auto newDistrib =  recoverDistrib(sigmaNew);
        result.state = newDistrib.first;
        result.cov = newDistrib.second;

        if(cross)
            result.crossCov = calcCrossCov(state, sigmaOld, result.state, sigmaNew);

        return result;
    }

}
