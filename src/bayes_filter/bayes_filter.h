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
    /** Struct representing a state estimate with mean and covariance */
    struct StateEstimate
    {
        Eigen::VectorXd state;
        Eigen::MatrixXd cov;
    };

    /** Interface for bayes filter implementations. */
    class BayesFilter
    {
    protected:
        MotionModel *motionModel_;
        SensorModel *sensorModel_;
    public:
        BayesFilter();
        BayesFilter(MotionModel *mm, SensorModel *sm);
        virtual ~BayesFilter();

        void setMotionModel(MotionModel *mm);
        void setSensorModel(SensorModel *sm);

        MotionModel &motionModel();
        const MotionModel &motionModel() const;

        SensorModel &sensorModel();
        const SensorModel &sensorModel() const;

        /** Return the current estimated state vector of the filter and its
         *  covariance.
         *  @return state vector and covariance */
        virtual StateEstimate getEstimate() const = 0;

        /** Initialize the filter with the given state vector and covariance.
         *  @param state initial state vector of size Nx1
         *  @param cov initial covariance of size NxN*/
        virtual void init(const Eigen::VectorXd &state,
            const Eigen::MatrixXd &cov) = 0;

        /** Execute prediction step of the bayes filter with the motion model.
         *  @param controls control vector
         *  @param observations observation matrix, each column is one
         *         observation
         *  @param noise noise matrix of the motion model of size NxN */
        virtual void predict(const Eigen::VectorXd &controls,
            const Eigen::MatrixXd &observations,
            const Eigen::MatrixXd &noise) = 0;

        /** Execute correction step of the bayes filter with the sensor model.
         *  @param observations observation matrix, each column is one
         *         observation of the length M
         *  @param noise noise matrix of the sensor model of size MxM */
        virtual void correct(const Eigen::MatrixXd &observations,
            const Eigen::MatrixXd &noise) = 0;

        /** Update the filter by one discrete timestep, which runs prediction
         *  and correction step.
         *  @param controls control vector
         *  @param observations observation matrix, each column is one
         *         observation of the length M
         *  @param motionNoise noise matrix (sqrt of covariance) of the motion
         *         model of size NxN
         *  @param sensorNoise noise matrix (sqrt of the covariance) of the
         *         sensor model of size MxM */
        void update(const Eigen::VectorXd &controls,
            const Eigen::MatrixXd &observations,
            const Eigen::MatrixXd &motionNoise,
            const Eigen::MatrixXd &sensorNoise);
    };
}

#endif
