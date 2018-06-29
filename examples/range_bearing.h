/*
 * range_bearing.h
 *
 *  Created on: 29 Jun 2018
 *      Author: Fabian Meyer
 */

 #ifndef RANGE_BEARING_H_
 #define RANGE_BEARING_H_

#include <bayes_filter/models.h>
#include <cassert>
#include <cmath>

class RangeBearing : public bf::SensorModel
{
private:
    // each column is a 2d position of a landmark
    Eigen::MatrixXd map;
public:
    RangeBearing(const Eigen::MatrixXd &map)
        :map(map)
    {

    }

    ~RangeBearing()
    {

    }

    Result estimateObservations(const Eigen::VectorXd &state,
        const Eigen::MatrixXd &observations) const override
    {
        // state is of form [x, y, vx, vy]^T
        assert(state.size() == 4);
        // observation is of form [range, bearing, landmarkId]^T
        assert(observations.rows() == 3);

        Result result;
        result.val.resize(observations.rows(), observations.cols());
        result.jac.setZero(result.val.size(), state.size());

        Eigen::Vector2d pos(state(0), state(1));
        for(unsigned int i = 0; i < observations.cols(); ++i)
        {
            size_t lm = static_cast<size_t>(observations(2, i));
            Eigen::Vector2d diff = map.col(lm) - pos;
            double range = diff.norm();
            double bearing = std::atan2(diff(1), diff(0));
            result.val.col(i) << range, bearing, lm;

            // calc jacobian ...
        }

        return result;
    }

    double likelihood(const Eigen::VectorXd &state,
        const Eigen::MatrixXd &observations,
        const Eigen::MatrixXd &sensorCov) const override
    {
        Eigen::MatrixXd obsExpected = estimateObservations(state,
            observations).val;

        double result = 1.0;
        for(unsigned int i = 0; i < observations.cols(); ++i)
        {
            // double normalpdf(x, mean, covariance)
            result *= normalpdf(observations.col(i),
                obsExpected.col(i), sensorCov);
        }

        return result;
    }
};

#endif
