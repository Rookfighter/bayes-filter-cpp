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
        typedef std::function<Eigen::VectorXd(const Eigen::VectorXd &)> TransformFunc;

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

        double calcLambda(const unsigned int n) const;

        /** Calculates sigma points from the given state and covariance.
         *  @param state state vector
         *  @param cov covariance matrix, has (state.size(), state.size()) dimensions
         *  @return set of sigma points and corresponding weights
         */
        SigmaPoints calcSigmaPoints(const Eigen::VectorXd &state,
                                    const Eigen::MatrixXd &cov) const;

        /** Recovers a gaussian distribution from the given sigma points.
         *  @param sigmaPoints set of sigma points and corresponding weights
         *  @return pair of recovered mean and covariance
         */
        std::pair<Eigen::VectorXd, Eigen::MatrixXd> recoverDistrib(
            const SigmaPoints &sigmaPoints) const;

        Eigen::MatrixXd calcCrossCov(const Eigen::VectorXd &stateOld,
                                     const SigmaPoints &sigmaOld,
                                     const Eigen::VectorXd &stateNew,
                                     const SigmaPoints &sigmaNew) const;

        /** Performs a unscented transform on the given state and covariance
         *  with the given function.
         *  @param state state vector
         *  @param cov covariance matrix, has (state.size(), state.size()) dimensions
         *  @param func function to be used on sigma points
         *  @param cross flag to enable cross correlation calculation
         *  @return struct containing transformed state, covariance and cross correlation.
         */
        Result transform(
            const Eigen::VectorXd &state,
            const Eigen::MatrixXd &cov,
            const TransformFunc &func,
            const bool cross = false) const;

    };
}

#endif
