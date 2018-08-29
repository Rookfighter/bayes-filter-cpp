/*
 * bayes_filter.cpp
 *
 *  Created on: 02 May 2018
 *      Author: Fabian Meyer
 */

#include "bayes_filter/bayes_filter.h"

namespace bf
{
    BayesFilter::BayesFilter()
        : BayesFilter(nullptr, nullptr)
    {}
    BayesFilter::BayesFilter(MotionModel *mm, SensorModel *sm)
        : motionModel_(mm), sensorModel_(sm)
    {}
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

    void BayesFilter::update(const Eigen::VectorXd &controls,
        const Eigen::MatrixXd &observations,
        const Eigen::MatrixXd &motionNoise,
        const Eigen::MatrixXd &sensorNoise)
    {
        predict(controls, observations, motionNoise);
        correct(observations, sensorNoise);
    }
}
