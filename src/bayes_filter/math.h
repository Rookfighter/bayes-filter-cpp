/*
 * math.h
 *
 *  Created on: 02 May 2018
 *      Author: Fabian Meyer
 */

#ifndef BFCPP_MATH_H_
#define BFCPP_MATH_H_

#include <Eigen/Dense>
#include <cassert>
#include <cmath>

namespace bf
{
    inline double pi()
    {
        return 3.1415926535897932384626433832795;
    }

    inline bool equals(double a, double b, double eps)
    {
        return std::abs(a - b) <= eps;
    }

    inline bool iszero(double a, double eps)
    {
        return equals(a, 0.0, eps);
    }

    inline Eigen::VectorXd mat2vec(const Eigen::MatrixXd &mat)
    {
        Eigen::VectorXd result(mat.size());

        for(unsigned int col = 0; col < mat.cols(); ++col)
        {
            for(unsigned int row = 0; row < mat.rows(); ++row)
            {
                unsigned int idx = col * mat.rows() + row;
                result(idx) = mat(row, col);
            }
        }

        return result;
    }

    inline Eigen::MatrixXd vec2mat(const Eigen::VectorXd &vec,
        const unsigned int rows,
        const unsigned int cols)
    {
        assert(rows * cols == vec.size());
        Eigen::MatrixXd result(rows, cols);

        for(unsigned int col = 0; col < cols; ++col)
        {
            for(unsigned int row = 0; row < rows; ++row)
            {
                unsigned int idx = col * rows + row;
                result(row, col) = vec(idx);
            }
        }

        return result;
    }

    inline Eigen::MatrixXd diagMat(
        const Eigen::MatrixXd &mat, const unsigned int times)
    {
        unsigned int rows = times * mat.rows();
        unsigned int cols = times * mat.cols();

        Eigen::MatrixXd result;
        result.setZero(rows, cols);

        for(unsigned int m = 0; m * mat.cols() < cols && m * mat.rows() < rows;
            ++m)
        {
            for(unsigned int col = 0; col < mat.cols(); ++col)
            {
                unsigned int c = m * mat.cols() + col;
                if(c >= cols)
                    break;
                for(unsigned int row = 0; row < mat.rows(); ++row)
                {
                    unsigned int r = m * mat.rows() + row;
                    if(r >= rows)
                        break;
                    result(r, c) = mat(row, col);
                }
            }
        }

        return result;
    }

    inline void computeWeightedMean(const Eigen::MatrixXd &data,
        const Eigen::VectorXd &weights,
        Eigen::VectorXd &outMean)
    {
        assert(data.cols() == weights.size());

        outMean.setZero(data.rows());

        for(unsigned int i = 0; i < data.cols(); ++i)
            outMean += weights(i) * data.col(i);
    }

    inline void computeWeightedCovariance(const Eigen::MatrixXd &data,
        const Eigen::VectorXd &weights,
        const Eigen::VectorXd &mean,
        Eigen::MatrixXd &outCov)
    {
        assert(data.cols() == weights.size());
        assert(data.rows() == mean.size());

        outCov.setZero(data.rows(), data.rows());

        Eigen::VectorXd diff;
        for(unsigned int i = 0; i < data.cols(); ++i)
        {
            diff = data.col(i) - mean;
            outCov += weights(i) * diff * diff.transpose();
        }
    }
}

#endif
