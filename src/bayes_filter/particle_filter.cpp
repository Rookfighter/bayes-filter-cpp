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

    StateEstimate ParticleFilter::getEstimate() const
    {
        Eigen::VectorXd state = Eigen::VectorXd::Zero(particles_.front().state.size());
        for(const Particle &p : particles_)
            state += p.weight * p.state;

        Eigen::Vector2d ang(0, 0);
        for(const Particle &p : particles_)
        {
            ang << ang(0) + p.weight *std::cos(p.state(3)),
                ang(1) + p.weight *std::sin(p.state(3));
        }
        state(3) = std::atan2(ang(1), ang(0));

        Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(state.size(), state.size());
        for(unsigned int i = 0; i < particles_.size(); ++i)
        {
            Eigen::VectorXd diff = particles_[i].state - state;
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
        const Eigen::MatrixXd &cov)
    {
        assert(state.size() == cov.cols());
        assert(state.size() == cov.rows());

        Eigen::VectorXd result(state.size());

        std::vector<std::normal_distribution<double>> distribs(state.size());
        for(unsigned int i = 0; i < state.size(); ++i)
            distribs[i] = std::normal_distribution<double>(state(i), cov(i, i));

        for(unsigned int i = 0; i < state.size(); ++i)
            result(i) = distribs[i](rndgen_);

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
            distribs[i] = std::normal_distribution<double>(state(i), cov(i, i));

        for(Particle &p : particles_)
        {
            p.weight = 0;
            p.state.resize(state.size());
            for(unsigned int i = 0; i < state.size(); ++i)
                p.state(i) = distribs[i](rndgen_);
        }
    }

    void ParticleFilter::predict(const Eigen::VectorXd &controls,
                                 const Eigen::MatrixXd &observations,
                                 const Eigen::MatrixXd &motionCov)
    {
        assert(particles_.front().state.size() == motionCov.rows());
        assert(particles_.front().state.size() == motionCov.cols());

        for(Particle &p : particles_)
        {
            p.state = motionModel().estimateState(p.state,
                                                  controls,
                                                  observations).val;
            p.state = randomizeState(p.state, motionCov);
        }
    }

    void ParticleFilter::correct(const Eigen::MatrixXd &observations,
                                 const Eigen::MatrixXd &sensorCov)
    {
        assert(sensorCov.size() > 0);

        for(Particle &p : particles_)
            p.weight = sensorModel().likelihood(p.state, observations, sensorCov);

        normalizeWeight();
        if(effectiveParticles() <= particles_.size() / 2)
            resample();
    }
}
