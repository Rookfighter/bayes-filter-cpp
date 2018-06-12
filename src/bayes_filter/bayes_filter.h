/*
 * bayes_filter.h
 *
 *  Created on: 02 May 2018
 *      Author: Fabian Meyer
 */

#ifndef BFCPP_BAYES_FILTER_H_
#define BFCPP_BAYES_FILTER_H_

#include "bayes_filter/models.h"

namespace bf
{
    /** base class for bayes filter implementations. */
    class BayesFilter
    {
    private:
        MotionModel *motionModel_;
        SensorModel *sensorModel_;

    public:

        BayesFilter()
            : motionModel_(nullptr), sensorModel_(nullptr)
        {

        }

        BayesFilter(MotionModel *mm, SensorModel *sm)
            : motionModel_(mm), sensorModel_(sm)
        {
        }

        virtual ~BayesFilter()
        {
            if(sensorModel_ != nullptr)
                delete sensorModel_;
            if(motionModel_ != nullptr)
                delete motionModel_;
        }

        void setMotionModel(MotionModel *mm)
        {
            motionModel_ = mm;
        }

        void setSensorModel(SensorModel *sm)
        {
            sensorModel_ = sm;
        }

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

        /** Return the current estimated state vector of the filter and its
         *  covariance.
         *  @return pair of state vector and covariance */
        virtual std::pair<Eigen::VectorXd, Eigen::MatrixXd> getEstimate() const = 0;

        /** Initialize the filter with the given state vector and covariance.
         *  @param state initial state vector of size Nx1
         *  @param cov initial covariance of size NxN*/
        virtual void init(const Eigen::VectorXd &state,
                          const Eigen::MatrixXd &cov) = 0;

        /** Execute prediction step of the bayes filter with the motion model.
         *  @param controls control vector
         *  @param observations observation matrix, each column is one observation
         *  @param noise noise matrix of the motion model of size NxN */
        virtual void predict(const Eigen::VectorXd &controls,
                             const Eigen::MatrixXd &observations,
                             const Eigen::MatrixXd &noise) = 0;

        /** Execute correction step of the bayes filter with the sensor model.
         *  @param observations observation matrix, each column is one observation of the length M
         *  @param noise noise matrix of the sensor model of size MxM */
        virtual void correct(const Eigen::MatrixXd &observations,
                             const Eigen::MatrixXd &noise) = 0;

        /** Update the filter by one discrete timestep, which runs prediction
         *  and correction step.
         *  @param controls control vector
         *  @param observations observation matrix, each column is one observation of the length M
         *  @param motionCov noise matrix of the motion model of size NxN
         *  @param sensorCov noise matrix of the sensor model of size MxM */
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
