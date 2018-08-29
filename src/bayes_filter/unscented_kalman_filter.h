/*
 * unscented_kalman_filter.h
 *
 *  Created on: 18 May 2018
 *      Author: Fabian Meyer
 */

#ifndef BFCPP_UNSCENTED_KALMAN_FILTER_H_
#define BFCPP_UNSCENTED_KALMAN_FILTER_H_

#include "bayes_filter/bayes_filter.h"
#include "bayes_filter/unscented_transform.h"

namespace bf
{
    /** Implementation of a Unscented Kalman Filter.*/
    class UnscentedKalmanFilter : public BayesFilter
    {
    private:
        UnscentedTransform unscentTrans_;

        Eigen::VectorXd state_;
        Eigen::MatrixXd cov_;
    public:
        UnscentedKalmanFilter();
        UnscentedKalmanFilter(MotionModel *mm, SensorModel *sm);
        ~UnscentedKalmanFilter();

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
