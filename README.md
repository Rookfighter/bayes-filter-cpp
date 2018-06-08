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

Or you can simply copy the source into your project and build it with the build system of your choice.

## Usage
