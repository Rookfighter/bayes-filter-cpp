/*
 * unscented_transform.cpp
 *
 *  Created on: 18 May 2018
 *      Author: Fabian Meyer
 */

#include "bayes_filter/unscented_transform.h"
#include "bayes_filter/math.h"

namespace bf
{
    UnscentedTransform::UnscentedTransform()
        : kappa_(1.0), alpha_(0.9), beta_(2.0)
    {}

    UnscentedTransform::~UnscentedTransform()
    {}

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

    void UnscentedTransform::calcSigmaPoints(
        const Eigen::VectorXd &state,
        const Eigen::MatrixXd &cov,
        SigmaPoints &outSigma) const
    {
        assert(state.size() == cov.rows());
        assert(state.size() == cov.cols());

        // keep track of dimension of a sigma point
        size_t n = state.size();
        double nd = static_cast<double>(n);
        double lambda = calcLambda(n);

        outSigma.points.resize(n, 2 * n + 1);
        outSigma.weights.resize(2, 2 * n + 1);

        // calculate sigma point distances to mean
        // each column is a distance vector
        Eigen::MatrixXd sigmaOffset = ((nd + lambda) * cov).llt().matrixL();

        // first sigma point is always just the mean
        outSigma.points.col(0) = state;
        // calc weight of first sigma point
        outSigma.weights(0, 0) = lambda / (nd + lambda);
        outSigma.weights(1, 0) =
            outSigma.weights(0, 0) + (1 - alpha_ * alpha_ + beta_);

        // all remeaining sigma point have the same constant weight
        double constWeight = 1.0 / (2.0 * (nd + lambda));
        Eigen::VectorXd tmp;
        for(unsigned int i = 0; i < n; ++i)
        {
            // calculate sigma point in positive direction
            tmp = state + sigmaOffset.col(i);
            outSigma.points.col(i + 1) = tmp;
            outSigma.weights(0, i + 1) = constWeight;
            outSigma.weights(1, i + 1) = constWeight;

            // calculate sigma point in negative direction
            tmp = state - sigmaOffset.col(i);
            outSigma.points.col(i + 1 + n) = tmp;
            outSigma.weights(0, i + 1 + n) = constWeight;
            outSigma.weights(1, i + 1 + n) = constWeight;
        }
    }

    void UnscentedTransform::recoverMean(const SigmaPoints &sigma,
        Eigen::VectorXd &outMean) const
    {
        assert(sigma.weights.rows() == 2);
        assert(sigma.points.cols() == sigma.weights.cols());

        computeWeightedMean(sigma.points, sigma.weights.row(0), outMean);
    }

    void UnscentedTransform::recoverCovariance(
        const SigmaPoints &sigma,
        const Eigen::VectorXd &mean,
        Eigen::MatrixXd &outCovariance) const
    {
        assert(sigma.points.cols() == sigma.weights.cols());
        assert(sigma.points.rows() == mean.size());

        computeWeightedCovariance(sigma.points, sigma.weights.row(1), mean,
            outCovariance);
    }

    void UnscentedTransform::recoverCrossCorrelation(
        const SigmaPoints &sigmaA,
        const Eigen::VectorXd &meanA,
        const SigmaPoints &sigmaB,
        const Eigen::VectorXd &meanB,
        Eigen::MatrixXd &outCrossCorrelation) const
    {
        assert(sigmaA.points.rows() == meanA.size());
        assert(sigmaB.points.rows() == meanB.size());
        assert(sigmaA.points.cols() == sigmaB.points.cols());

        outCrossCorrelation.setZero(meanA.size(), meanB.size());

        for(unsigned int i = 0; i < sigmaA.points.cols(); ++i)
        {
            Eigen::VectorXd diffA = sigmaA.points.col(i) - meanA;
            Eigen::VectorXd diffB = sigmaB.points.col(i) - meanB;
            outCrossCorrelation +=
                sigmaA.weights(1, i) * diffA * diffB.transpose();
        }
    }
}
