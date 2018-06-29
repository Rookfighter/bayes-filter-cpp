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
    static Eigen::VectorXd noNormalize(const Eigen::VectorXd &v)
    {
        return v;
    }

    static Eigen::VectorXd rowwiseMean(const Eigen::MatrixXd &m)
    {
        return m.rowwise().mean();
    }

    BayesFilter::BayesFilter()
        : BayesFilter(nullptr, nullptr)
    {

    }

    BayesFilter::BayesFilter(MotionModel *mm, SensorModel *sm)
        : motionModel_(mm), sensorModel_(sm),
        normState_(std::bind(noNormalize, ph::_1)),
        normObs_(std::bind(noNormalize, ph::_1)),
        meanState_(std::bind(rowwiseMean, ph::_1)),
        meanObs_(std::bind(rowwiseMean, ph::_1))
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

    Eigen::VectorXd BayesFilter::normalizeState(
        const Eigen::VectorXd &state) const
    {
        return normState_(state);
    }

    Eigen::VectorXd BayesFilter::normalizeObservations(
        const Eigen::MatrixXd &observations) const
    {
        Eigen::MatrixXd result(observations.rows(), observations.cols());
        for(unsigned int i = 0; i < result.cols(); ++i)
            result.col(i) = normObs_(observations.col(i));
        return result;
    }

    void BayesFilter::setMeanState(const MeanFunc &func)
    {
        meanState_ = func;
    }

    void BayesFilter::setMeanObservation(const MeanFunc &func)
    {
        meanObs_ = func;
    }

    Eigen::VectorXd BayesFilter::meanOfStates(
        const Eigen::MatrixXd &states) const
    {
        return meanState_(states);
    }

    Eigen::VectorXd BayesFilter::meanOfObservations(
        const Eigen::MatrixXd &observations) const
    {
        return meanObs_(observations);
    }

    void BayesFilter::update(const Eigen::VectorXd &controls,
                const Eigen::MatrixXd &observations,
                const Eigen::MatrixXd &motionCov,
                const Eigen::MatrixXd &sensorCov)
    {
        predict(controls, observations, motionCov);
        correct(observations, sensorCov);
    }
}
