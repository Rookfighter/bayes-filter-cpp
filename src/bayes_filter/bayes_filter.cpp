/*
 * bayes_filter.cpp
 *
 *  Created on: 02 May 2018
 *      Author: Fabian Meyer
 */


#include "bayes_filter/bayes_filter.h"

namespace ph = std::placeholders;

namespace bf
{
    static void noNormalize(Eigen::VectorXd &)
    {

    }

    static Eigen::VectorXd rowwiseMean(const Eigen::MatrixXd &m,
        const Eigen::VectorXd &w)
    {
        assert(m.cols() == w.size());

        Eigen::VectorXd result;
        result.setZero(m.rows());
        for(unsigned int i = 0; i < m.cols(); ++i)
            result += w(i) * m.col(i);

        return result;
    }

    BayesFilter::BayesFilter()
        : BayesFilter(nullptr, nullptr)
    {

    }

    BayesFilter::BayesFilter(MotionModel *mm, SensorModel *sm)
        : motionModel_(mm), sensorModel_(sm),
        normState_(std::bind(noNormalize, ph::_1)),
        normObs_(std::bind(noNormalize, ph::_1)),
        meanState_(std::bind(rowwiseMean, ph::_1, ph::_2)),
        meanObs_(std::bind(rowwiseMean, ph::_1, ph::_2))
    {
    }

    BayesFilter::~BayesFilter()
    {
        if(sensorModel_ != nullptr)
            delete sensorModel_;
        if(motionModel_ != nullptr)
            delete motionModel_;
    }

    void BayesFilter::setMotionModel(MotionModel *mm)
    {
        motionModel_ = mm;
    }

    void BayesFilter::setSensorModel(SensorModel *sm)
    {
        sensorModel_ = sm;
    }

    MotionModel &BayesFilter::motionModel()
    {
        return *motionModel_;
    }

    const MotionModel &BayesFilter::motionModel() const
    {
        return *motionModel_;
    }

    SensorModel &BayesFilter::sensorModel()
    {
        return *sensorModel_;
    }

    const SensorModel &BayesFilter::sensorModel() const
    {
        return *sensorModel_;
    }

    void BayesFilter::setNormalizeState(const NormalizeFunc &func)
    {
        normState_ = func;
    }

    void BayesFilter::setNormalizeObservation(const NormalizeFunc &func)
    {
        normObs_ = func;
    }

    void BayesFilter::normalizeState(Eigen::VectorXd &state) const
    {
        normState_(state);
    }

    void BayesFilter::normalizeObservations(Eigen::MatrixXd &obs) const
    {
        Eigen::VectorXd tmp(obs.rows());
        for(unsigned int i = 0; i < obs.cols(); ++i)
        {
            tmp = obs.col(i);
            normObs_(tmp);
            obs.col(i) = tmp;
        }
    }

    void BayesFilter::setMeanState(const WeightedMeanFunc &func)
    {
        meanState_ = func;
    }

    void BayesFilter::setMeanObservation(const WeightedMeanFunc &func)
    {
        meanObs_ = func;
    }

    Eigen::VectorXd BayesFilter::meanOfStates(
        const Eigen::MatrixXd &states,
        const Eigen::VectorXd &weights) const
    {
        assert(states.cols() == weights.size());
        return meanState_(states, weights);
    }

    Eigen::VectorXd BayesFilter::meanOfObservations(
        const Eigen::MatrixXd &observations,
        const Eigen::VectorXd &weights) const
    {
        assert(observations.cols() == weights.size());
        return meanObs_(observations, weights);
    }

    void BayesFilter::update(const Eigen::VectorXd &controls,
                const Eigen::MatrixXd &observations,
                const Eigen::MatrixXd &motionNoise,
                const Eigen::MatrixXd &sensorNoise)
    {
        predict(controls, observations, motionNoise);
        correct(observations, sensorNoise);
    }
}
