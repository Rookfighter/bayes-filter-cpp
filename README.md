# Bayes Filter Cpp

Bayes Filter Cpp is a basic C++ library for bayes filters. It provides bayes
filter implementations, such as

* Particle Filter
* Unscented Kalman Filter
* Extended Kalman Filter

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

An example for a 2d constant velocity motion model can be found
in ```examples/constant_velocity.h```.

An example for a 2d range bearing sensor model can be found
in ```examples/range_bearing.h```.
