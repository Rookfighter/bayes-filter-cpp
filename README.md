# bayes-filter-cpp

![Cpp11](https://img.shields.io/badge/C%2B%2B-11-blue.svg)
![License](https://img.shields.io/packagist/l/doctrine/orm.svg)
![Travis Status](https://travis-ci.org/Rookfighter/bayes-filter-cpp.svg?branch=master)
![Appveyer Status](https://ci.appveyor.com/api/projects/status/kb1lydcuywyurj5r?svg=true)

bayes-filter-cpp is a C++ library implementing recursive bayes filters for state estimation. It provides the following filters:

* Particle Filter
* Unscented Kalman Filter
* Extended Kalman Filter

## Install

The library has the following dependencies:

* [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page)

Make sure these are installed on your system and/or can be found by your build
system. For ease of use the header-only dependencies are included as
submodules.

On Debian based systems you can install Eigen3 via apt:

```bash
apt-get install libeigen3-dev
```

Then you can simply copy the source files into your project or install
the library using the CMake build system.

```bash
cd path/to/repo
git submodule update --init
mkdir build
cd build
cmake ..
make install
```

## Usage

There are three steps to use this library:

* implement your motion model
* implement your sensor model
* pick the filter of your choice

An example for a 2d constant velocity motion model can be found
in ```examples/constant_velocity.h```.

An example for a 2d range bearing sensor model can be found
in ```examples/range_bearing.h```.
