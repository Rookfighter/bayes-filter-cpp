/*
 * bayes_filter.h
 *
 *  Created on: 02 May 2018
 *      Author: Fabian Meyer
 */

#ifndef BFCPP_BAYES_FILTER_H_
#define BFCPP_BAYES_FILTER_H_

#include <cassert>
#include "bfcpp/models.h"

namespace bf
{
    class BayesFilter
    {
    private:
        MotionModel *motionModel_;
        SensorModel *sensorModel_;

    protected:
        MotionModel &motionModel()
        {
            return *motionModel_;
        }

        const MotionModel &motionModel() const
        {
            return *motionModel_;
        }

        SensorModel &sensorModel()
        {
            return *sensorModel_;
        }

        const SensorModel &sensorModel() const
        {
            return *sensorModel_;
        }

    public:

        BayesFilter(MotionModel *mm, SensorModel *sm)
            : motionModel_(mm), sensorModel_(sm)
        {
            assert(motionModel_ != nullptr);
            assert(sensorModel_ != nullptr);
        }

        virtual ~BayesFilter()
        {
            assert(motionModel_ != nullptr);
            assert(sensorModel_ != nullptr);

            delete motionModel_;
            delete sensorModel_;
        }

        virtual std::pair<Eigen::VectorXd, Eigen::MatrixXd> getEstimate() const = 0;

        virtual void init(const Eigen::VectorXd &state,
                          const Eigen::MatrixXd &cov) = 0;
        virtual void predict(const Eigen::VectorXd &controls,
                             const Eigen::MatrixXd &observations,
                             const Eigen::MatrixXd &motionCov) = 0;
        virtual void correct(const Eigen::MatrixXd &observations,
                             const Eigen::MatrixXd &sensorCov) = 0;
        void update(const Eigen::VectorXd &controls,
                    const Eigen::MatrixXd &observations,
                    const Eigen::MatrixXd &motionCov,
                    const Eigen::MatrixXd &sensorCov)
        {
            predict(controls, observations, motionCov);
            correct(observations, sensorCov);
        }
    };


}

#endif
