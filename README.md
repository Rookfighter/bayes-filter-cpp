# Bayes Filter Cpp

Bayes Filter Cpp is a basic C++ library for bayes filters. It provides bayes
filter implementations, such as

* Particle Filter
* Unscented Kalman Filter
* ~Extended Kalman Filter~ TBD

## Install

First download the dependencies locally as git submodules.

```bash
cd <path-to-repo>
git submodule update --init --recursive
```

Then build the library with CMake by running

```bash
cd <path-to-repo>
mkdir build
cd build
cmake ..
make
```

Or you can simply copy the source into your project and build it with the build
system of your choice. Keep in mind that this requires Eigen3 as dependency.

## Usage

There are three steps to use this library:

* implement your motion model
* implement your sensor model
* pick the filter of your choice

For example an implementation of a 2d constant velocity motion model might look
like this:

```cpp
#include <bayes_filter/models.h>
#include <cassert>

class ConstantVelocity : public bf::MotionModel
{
public:
    ConstantVelocity() {}
    ~ConstantVelocity() {}

    Result estimateState(const Eigen::VectorXd &state,
        const Eigen::VectorXd &controls,
        const Eigen::MatrixXd &) const override
    {
        // state is of form [x, y, vx, vy]^T
        assert(state.size() == 4);
        assert(controls.size() == 1);

        Result result;
        result.val.resize(state.size());
        result.jac.resize(result.val.size(), state.size());

        double dt = controls(0);
        double x  = state(0);
        double y  = state(1);
        double vx  = state(2);
        double vy  = state(3);

        result.val << x + vx * dt, y + vy * dt, vx, vy;
        result.jac <<
            1.0, 0.0,  dt, 0.0,
            0.0, 1.0, 0.0,  dt,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0;

        return result;
    }
}
```

A 2d range bearing sensor model might look like this:

```cpp
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
        :map(map) {}
    ~RangeBearing() {}

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

        Eigen::Vector2 pos(state(0), state(1));
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
            observations);

        double result = 1.0;
        for(unsigned int i = 0; i < observations.cols(); ++i)
        {
            // double normalpdf(x, mean, covariance)
            result *= normalpdf(observations.col(i),
                obsExpected.col(i), sensorCov);
        }

        return result;
    }
}
```
