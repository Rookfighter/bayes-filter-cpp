/*
 * particle_filter.cpp
 *
 *  Created on: 17 Apr 2018
 *      Author: Fabian Meyer
 */

#include "bayes_filter/particle_filter.h"
#include "bayes_filter/math.h"

namespace bf
{
    ParticleFilter::ParticleFilter()
        : BayesFilter()
    {

    }

    ParticleFilter::ParticleFilter(MotionModel *mm, SensorModel *sm)
        : BayesFilter(mm, sm)
    {

    }

    ParticleFilter::~ParticleFilter()
    {
    }

    void ParticleFilter::setSeed(const size_t seed)
    {
        rndgen_.seed(seed);
    }

    StateEstimate ParticleFilter::getEstimate() const
    {
        assert(particles_.size() > 0);

        Eigen::MatrixXd states(particles_.front().state.size(),
            particles_.size());
        Eigen::VectorXd weights(particles_.size());

        for(unsigned int i = 0; i < states.cols(); ++i)
        {
            states.col(i) = particles_[i].state;
            weights(i) = particles_[i].weight;
        }

        Eigen::VectorXd state = meanOfStates(states, weights);
        Eigen::MatrixXd cov(state.size(), state.size());
        Eigen::VectorXd diff;
        for(unsigned int i = 0; i < particles_.size(); ++i)
        {
            diff = particles_[i].state - state;
            normalizeState(diff);
            cov += particles_[i].weight * diff * diff.transpose();
        }

        return {state, cov};
    }

    Eigen::VectorXd ParticleFilter::getMostLikely() const
    {
        unsigned int idx = 0;
        for(unsigned int i = 1; i < particles_.size(); ++i)
        {
            if(particles_[i].weight > particles_[idx].weight)
                idx = i;
        }

        return particles_[idx].state;
    }

    void ParticleFilter::setParticleCount(const unsigned int cnt)
    {
        particles_.resize(cnt);
    }

    const ParticleSet &ParticleFilter::particles() const
    {
        return particles_;
    }

    Eigen::VectorXd ParticleFilter::randomizeState(
        const Eigen::VectorXd &state,
        const Eigen::MatrixXd &noise)
    {
        // noise matrix main diagonal holds stddev not variance!
        assert(state.size() == noise.cols());
        assert(state.size() == noise.rows());

        Eigen::VectorXd result(state.size());

        std::vector<std::normal_distribution<double>> distribs(state.size());
        for(unsigned int i = 0; i < state.size(); ++i)
        {
            distribs[i] = std::normal_distribution<double>(state(i),
                          noise(i, i));
        }

        for(unsigned int i = 0; i < state.size(); ++i)
            result(i) = distribs[i](rndgen_);
        normalizeState(result);

        return result;
    }

    void ParticleFilter::normalizeWeight()
    {
        double sum = 0.0;
        for(const Particle &p : particles_)
            sum += p.weight;

        if(iszero(sum, 1e-12) || sum < 0)
        {
            double weight = 1.0 / static_cast<double>(particles_.size());
            for(Particle &p : particles_)
                p.weight = weight;
        }
        else
        {
            for(Particle &p : particles_)
                p.weight /= sum;
        }
    }

    double ParticleFilter::effectiveParticles() const
    {
        double sum = 0.0;
        for(const Particle &p : particles_)
            sum += p.weight * p.weight;

        if(sum <= 0)
            return 0;

        return 1.0 / sum;
    }

    void ParticleFilter::resample()
    {
        ParticleSet result(particles_.size());

        // calc step between each sample
        // assuming normalized weights (sum is 1.0)
        double step =  1.0 / static_cast<double>(particles_.size());
        // assume uniform weight as default weight
        double defaultWeight = step;

        // calc a random starting point for resampling
        std::uniform_real_distribution<double> distrib(0.0, 1.0);
        double position = distrib(rndgen_);

        // init accumulated weights with weight of first particle
        double accumWeight = particles_[0].weight;
        unsigned int k = 0;

        // do low variance resampling (stochastic universal resampling)
        for(unsigned int i = 0; i < particles_.size(); ++i)
        {
            // check if accumulated weight is less than current sample
            // increment particle number k until not satisfied anymore
            while(accumWeight < position)
            {
                k = (k + 1) % particles_.size();
                accumWeight += particles_[k].weight;
            }

            result[i] = particles_[k];
            result[i].weight = defaultWeight;
            position += step;
        }

        particles_ = result;
    }

    void ParticleFilter::init(const Eigen::VectorXd &state,
                              const Eigen::MatrixXd &cov)
    {
        assert(state.size() == cov.cols());
        assert(state.size() == cov.rows());

        std::vector<std::normal_distribution<double>> distribs(state.size());
        for(unsigned int i = 0; i < state.size(); ++i)
        {
            distribs[i] = std::normal_distribution<double>(state(i),
                          cov(i, i));
        }

        for(Particle &p : particles_)
        {
            p.weight = 0;
            p.state.resize(state.size());
            for(unsigned int i = 0; i < state.size(); ++i)
                p.state(i) = distribs[i](rndgen_);
            normalizeState(p.state);
        }
    }

    void ParticleFilter::predict(const Eigen::VectorXd &controls,
                                 const Eigen::MatrixXd &observations,
                                 const Eigen::MatrixXd &noise)
    {
        assert(particles_.front().state.size() == noise.rows());
        assert(particles_.front().state.size() == noise.cols());

        for(Particle &p : particles_)
        {
            p.state = motionModel().estimateState(
                p.state, controls, observations).val;
            p.state = randomizeState(p.state, noise);
        }
    }

    void ParticleFilter::correct(const Eigen::MatrixXd &observations,
                                 const Eigen::MatrixXd &sensorCov)
    {
        assert(sensorCov.size() > 0);

        for(Particle &p : particles_)
        {
            p.weight = sensorModel().likelihood(p.state, observations,
                                                sensorCov);
        }

        normalizeWeight();
        if(effectiveParticles() <= particles_.size() / 2)
            resample();
    }
}
