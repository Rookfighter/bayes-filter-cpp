/*
 * extended_kalman_filter.h
 *
 *  Created on: 03 Jul 2018
 *      Author: Fabian Meyer
 */

#ifndef BFCPP_EXTENDED_KALMAN_FILTER_H_
#define BFCPP_EXTENDED_KALMAN_FILTER_H_

#include "bayes_filter/bayes_filter.h"

namespace bf
{
    /** Implementation of a Extended Kalman Filter.*/
    class ExtendedKalmanFilter : public BayesFilter
    {
    private:
        Eigen::VectorXd state_;
        Eigen::MatrixXd cov_;

    public:
        ExtendedKalmanFilter();
        ExtendedKalmanFilter(MotionModel *mm, SensorModel *sm);
        ~ExtendedKalmanFilter();

        StateEstimate getEstimate() const override;

        void init(const Eigen::VectorXd &state,
            const Eigen::MatrixXd &cov) override;

        void predict(const Eigen::VectorXd &controls,
            const Eigen::MatrixXd &observations,
            const Eigen::MatrixXd &noise) override;

        void correct(const Eigen::MatrixXd &observations,
            const Eigen::MatrixXd &noise) override;
    };
}

#endif
