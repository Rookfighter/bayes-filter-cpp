/*
 * particle_filter.h
 *
 *  Created on: 02 May 2018
 *      Author: Fabian Meyer
 */

#ifndef BFCPP_PARTICLE_FILTER_H_
#define BFCPP_PARTICLE_FILTER_H_

#include <random>
#include "bfcpp/bayes_filter.h"

namespace bf
{
    /** Particle representing a single state estimate with a weight. */
    struct Particle
    {
        Eigen::VectorXd state;
        double weight;
    };

    /** Implementation of a Particle Filter.*/
    class ParticleFilter: public BayesFilter
    {
    private:
        std::vector<Particle> particles_;
        std::default_random_engine rndgen_;

        Eigen::VectorXd randomizeState(const Eigen::VectorXd &state,
                                       const Eigen::MatrixXd &cov);
        void normalizeWeight();

        /** Calculates the number of effective particles as 1 / sum(weight^2).
         *  @return number of effective particles */
        double effectiveParticles() const;
        void resample();

    public:
        ParticleFilter();
        ParticleFilter(MotionModel *mm, SensorModel *sm);
        ~ParticleFilter();

        void setParticleCount(const unsigned int cnt);
        const std::vector<Particle> &particles() const;

        std::pair<Eigen::VectorXd, Eigen::MatrixXd> getEstimate() const override;

        /** Returns the most likely (highest weight) state estimate.
         *  @return most likely state estimate */
        Eigen::VectorXd getMostLikely() const;

        void init(const Eigen::VectorXd &state,
                  const Eigen::MatrixXd &cov) override;
        void predict(const Eigen::VectorXd &controls,
                     const Eigen::MatrixXd &observations,
                     const Eigen::MatrixXd &motionCov) override;
        void correct(const Eigen::MatrixXd &observations,
                     const Eigen::MatrixXd &sensorCov) override;
    };
}

#endif
