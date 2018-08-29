/*
 * unscented_transform.h
 *
 *  Created on: 18 May 2018
 *      Author: Fabian Meyer
 */

#ifndef BFCPP_UNSCENTED_TRANSFORM_H_
#define BFCPP_UNSCENTED_TRANSFORM_H_

#include <Eigen/Dense>
#include <functional>

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
        UnscentedTransform();
        ~UnscentedTransform();

        /** Set the kappa factor for the unscented transform.
         *  Influences how far sigma points are away from mean value. The
         *  greater the value the farther apart.
         *  @param kappa kappa factor, has to be >= 0 */
        void setKappa(const double kappa);

        /** Set the alpha factor for the unscented transform.
         *  Influences how far sigma points are away from mean value. The
         *  greater the value the farther apart.
         *  @param alpha alpha factor, has to be in (0,1] */
        void setAlpha(const double alpha);

        /** Set the beta factor for the unscented transform.
         *  Optimal choice for gaussians: 2.
         *  @param beta beta factor */
        void setBeta(const double beta);

        /** Calculates the lamda factor for the unscented transform.
         *  Caclulated as: alpha^2 * (n + kappa) - n
         *  @param n dimension of a single sigma point
         *  @return lambda factor*/
        double calcLambda(const size_t n) const;

        /** Calculates sigma points from the given state and covariance.
         *  @param state state vector
         *  @param cov covariance matrix, (state.size(), state.size())
         * dimensions
         *  @return set of sigma points and corresponding weights */
        void calcSigmaPoints(const Eigen::VectorXd &state,
            const Eigen::MatrixXd &cov,
            SigmaPoints &outSigma) const;

        /** Recovers the mean value from sigma points.
         *  @param sigma sigma points
         *  @return mean of the sigma points, normalized */
        void recoverMean(const SigmaPoints &sigma,
            Eigen::VectorXd &outMean) const;

        /** Recovers the covariance from sigma points.
         *  @param sigma sigma points
         *  @param mean mean value of the sigma points
         *  @param normalize normalization function for a sigma point
         *  @return covariance of the sigma points */
        void recoverCovariance(const SigmaPoints &sigma,
            const Eigen::VectorXd &mean,
            Eigen::MatrixXd &covariance) const;

        /** Calculates the cross correlation between two sets of sigma points.
         *  The resulting matrix is of dimension nA x nB.
         *  The sigma point sets must have the same amount of sigma points (
         *  same amount of columns).
         *  @param sigmaA sigma points A
         *  @param meanA mean value of sigmaA
         *  @param normalizeA normalization function for sigmaB
         *  @param sigmaB sigma points B
         *  @param meanB mean value of sigmaB
         *  @param normalizeB normalization function for sigmaB
         *  @return cross correlation of the two sets (nA x nB) */
        void recoverCrossCorrelation(const SigmaPoints &sigmaA,
            const Eigen::VectorXd &meanA,
            const SigmaPoints &sigmaB,
            const Eigen::VectorXd &meanB,
            Eigen::MatrixXd &crossCorrelation) const;
    };
}

#endif
