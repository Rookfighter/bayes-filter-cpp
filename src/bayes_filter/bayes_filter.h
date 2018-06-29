/*
 * bayes_filter.h
 *
 *  Created on: 02 May 2018
 *      Author: Fabian Meyer
 */

#ifndef BFCPP_BAYES_FILTER_H_
#define BFCPP_BAYES_FILTER_H_

#include <functional>
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
    public:
        /** Function that normalizes the input vector and returns normalized
         *  version */
        typedef std::function<Eigen::VectorXd(const Eigen::VectorXd &)>
            NormalizeFunc;
        /** Function that calculates the rowwise mean of the given matrix. Each
         *  column represents one sample / measurement / state. */
        typedef std::function<Eigen::VectorXd(const Eigen::MatrixXd &,
            const Eigen::VectorXd &)> WeightedMeanFunc;

    protected:
        MotionModel *motionModel_;
        SensorModel *sensorModel_;

        NormalizeFunc normState_;
        NormalizeFunc normObs_;

        WeightedMeanFunc meanState_;
        WeightedMeanFunc meanObs_;

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

        /** Set the normalization function for state vectors.
         *  @param normalize normalization function */
        void setNormalizeState(const NormalizeFunc &func);

        /** Set the normalization function for observation matrices.
         *  @param normalize normalization function */
        void setNormalizeObservation(const NormalizeFunc &func);

        Eigen::VectorXd normalizeState(const Eigen::VectorXd &state) const;
        Eigen::VectorXd normalizeObservations(
            const Eigen::MatrixXd &observations) const;

        void setMeanState(const WeightedMeanFunc &func);
        void setMeanObservation(const WeightedMeanFunc &func);

        Eigen::VectorXd meanOfStates(
            const Eigen::MatrixXd &states,
            const Eigen::VectorXd &weights) const;
        Eigen::VectorXd meanOfObservations(
            const Eigen::MatrixXd &observations,
            const Eigen::VectorXd &weights) const;


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
         *  @param motionCov noise matrix of the motion model of size NxN
         *  @param sensorCov noise matrix of the sensor model of size MxM */
        void update(const Eigen::VectorXd &controls,
                    const Eigen::MatrixXd &observations,
                    const Eigen::MatrixXd &motionCov,
                    const Eigen::MatrixXd &sensorCov);
    };


}

#endif
