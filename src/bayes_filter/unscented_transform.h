/*
 * unscented_transform.h
 *
 *  Created on: 18 May 2018
 *      Author: Fabian Meyer
 */

#ifndef BFCPP_UNSCENTED_TRANSFORM_H_
#define BFCPP_UNSCENTED_TRANSFORM_H_

#include <functional>
#include <Eigen/Dense>

namespace bf
{
    struct SigmaPoints
    {
        Eigen::MatrixXd points;
        Eigen::MatrixXd weights;
    };

    class UnscentedTransform
    {
    private:
        double kappa_;
        double alpha_;
        double beta_;
    public:
        typedef std::function<Eigen::VectorXd(const Eigen::VectorXd &)> NormalizeFunc;

        struct Result
        {
            Eigen::VectorXd state;
            Eigen::MatrixXd cov;
            Eigen::MatrixXd crossCov;
        };

        UnscentedTransform();
        ~UnscentedTransform();

        void setKappa(const double kappa);
        void setAlpha(const double alpha);
        void setBeta(const double beta);

        double calcLambda(const size_t n) const;

        /** Calculates sigma points from the given state and covariance.
         *  @param state state vector
         *  @param cov covariance matrix, (state.size(), state.size()) dimensions
         *  @return set of sigma points and corresponding weights */
        SigmaPoints calcSigmaPoints(const Eigen::VectorXd &state,
            const Eigen::MatrixXd &cov,
            const NormalizeFunc &normalize) const;

        /** Recovers the mean value from sigma points.
         *  @param sigma sigma points
         *  @param normalize normalization function for a sigma point
         *  @return mean of the sigma points, normalized */
        Eigen::VectorXd recoverMean(const SigmaPoints &sigma,
            const NormalizeFunc &normalize) const;

        Eigen::MatrixXd recoverCovariance(
            const SigmaPoints &sigma,
            const Eigen::VectorXd &mean,
            const NormalizeFunc& normalize) const;

        Eigen::MatrixXd recoverCrossCovariance(
            const SigmaPoints &sigmaA,
            const Eigen::VectorXd &meanA,
            const NormalizeFunc& normalizeA,
            const SigmaPoints &sigmaB,
            const Eigen::VectorXd &meanB,
            const NormalizeFunc& normalizeB) const;
    };
}

#endif
