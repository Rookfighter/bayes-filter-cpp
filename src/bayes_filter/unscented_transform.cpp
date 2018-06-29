/*
 * unscented_transform.cpp
 *
 *  Created on: 18 May 2018
 *      Author: Fabian Meyer
 */

#include "bayes_filter/unscented_transform.h"

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
        assert(kappa >= 0);
        kappa_ = kappa;
    }

    void UnscentedTransform::setAlpha(const double alpha)
    {
        assert(alpha > 0);
        assert(alpha <= 1);
        alpha_ = alpha;
    }

    void UnscentedTransform::setBeta(const double beta)
    {
        beta_ = beta;
    }

    double UnscentedTransform::calcLambda(const size_t n) const
    {
        double nd = static_cast<double>(n);
        return alpha_ * alpha_ * (nd + kappa_) - nd;
    }

    SigmaPoints UnscentedTransform::calcSigmaPoints(
        const Eigen::VectorXd &state,
        const Eigen::MatrixXd &cov,
        const NormalizeFunc& normalize) const
    {
        assert(state.size() == cov.rows());
        assert(state.size() == cov.cols());

        // keep track of dimension of a sigma point
        size_t n = state.size();
        double nd = static_cast<double>(n);
        double lambda = calcLambda(n);

        SigmaPoints result;
        result.points.resize(n, 2 * n + 1);
        result.weights.resize(2, 2 * n + 1);

        // calculate sigma point distances to mean
        // each column is a distance vector
        Eigen::MatrixXd sigmaOffset = ((nd + lambda) * cov).llt().matrixL();

        // first sigma point is always just the mean
        result.points.col(0) = state;
        // calc weight of first sigma point
        result.weights(0, 0) = lambda / (nd + lambda);
        result.weights(1, 0) = result.weights(0, 0) + (1 - alpha_ * alpha_ + beta_);

        // all remeaining sigma point have the same constant weight
        double constWeight = 1.0 / (2.0 * (nd + lambda));

        for(unsigned int i = 0; i < n; ++i)
        {
            // calculate sigma point in positive direction
            result.points.col(i + 1) = normalize(state + sigmaOffset.col(i));
            result.weights(0, i + 1) = constWeight;
            result.weights(1, i + 1) = constWeight;

            // calculate sigma point in negative direction
            result.points.col(i + 1 + n) = normalize(state - sigmaOffset.col(i));
            result.weights(0, i + 1 + n) = constWeight;
            result.weights(1, i + 1 + n) = constWeight;
        }

        return result;
    }

    Eigen::VectorXd UnscentedTransform::recoverMean(
        const SigmaPoints &sigma,
        const NormalizeFunc& normalize) const
    {
        assert(sigma.points.cols() == sigma.weights.cols());

        Eigen::VectorXd result;
        result.setZero(sigma.points.rows());

        for(unsigned int i = 0; i < sigma.points.cols(); ++i)
            result += sigma.weights(0, i) * sigma.points.col(i);

        return normalize(result);
    }

    Eigen::MatrixXd UnscentedTransform::recoverCovariance(
        const SigmaPoints &sigma,
        const Eigen::VectorXd &mean,
        const NormalizeFunc& normalize) const
    {
        assert(sigma.points.cols() == sigma.weights.cols());
        assert(sigma.points.rows() == mean.size());

        Eigen::MatrixXd result;
        result.setZero(mean.size(), mean.size());

        for(unsigned int i = 0; i < sigma.points.cols(); ++i)
        {
            Eigen::VectorXd diff = normalize(sigma.points.col(i) - mean);
            result += sigma.weights(1, i) * diff * diff.transpose();
        }

        return result;
    }

    Eigen::MatrixXd UnscentedTransform::recoverCrossCorrelation(
        const SigmaPoints &sigmaA,
        const Eigen::VectorXd &meanA,
        const NormalizeFunc& normalizeA,
        const SigmaPoints &sigmaB,
        const Eigen::VectorXd &meanB,
        const NormalizeFunc& normalizeB) const
    {
        assert(sigmaA.points.rows() == meanA.size());
        assert(sigmaB.points.rows() == meanB.size());
        assert(sigmaA.points.cols() == sigmaB.points.cols());

        Eigen::MatrixXd result;
        result.setZero(meanA.size(), meanB.size());

        for(unsigned int i = 0; i < sigmaA.points.cols(); ++i)
        {
            // calculate normalized diff of A
            Eigen::VectorXd diffA = normalizeA(sigmaA.points.col(i) - meanA);
            // calculate normalized diff of B
            Eigen::VectorXd diffB = normalizeB(sigmaB.points.col(i) - meanB);
            result += sigmaA.weights(1, i) * diffA * diffB.transpose();
        }

        return result;
    }

}
