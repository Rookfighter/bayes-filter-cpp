/*
 * particle_filter.h
 *
 *  Created on: 02 May 2018
 *      Author: Fabian Meyer
 */

#ifndef BFCPP_PARTICLE_FILTER_H_
#define BFCPP_PARTICLE_FILTER_H_

#include "bayes_filter/bayes_filter.h"
#include <random>

namespace bf
{
    /** Particle representing a single state estimate with a weight. */
    struct Particle
    {
        Eigen::VectorXd state;
        double weight;
    };

    typedef std::vector<Particle> ParticleSet;

    /** Implementation of a Particle Filter.*/
    class ParticleFilter : public BayesFilter
    {
    private:
        ParticleSet particles_;
        std::default_random_engine rndgen_;

        void randomizeState(Eigen::VectorXd &state,
            const Eigen::MatrixXd &noise);
        void normalizeWeight();

        /** Calculates the number of effective particles as: 1 / sum(weight^2).
         *  @return number of effective particles */
        double effectiveParticles() const;
        void resample();

    public:
        ParticleFilter();
        ParticleFilter(MotionModel *mm, SensorModel *sm);
        ~ParticleFilter();

        void setSeed(const size_t seed);
        void setParticleCount(const unsigned int cnt);
        const ParticleSet &particles() const;

        StateEstimate getEstimate() const override;

        /** Returns the most likely (highest weight) state estimate.
         *  @return most likely state estimate */
        Eigen::VectorXd getMostLikely() const;

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
