/*
 * math.h
 *
 *  Created on: 02 May 2018
 *      Author: Fabian Meyer
 */

#ifndef BFCPP_MATH_H_
#define BFCPP_MATH_H_

#include <cmath>
#include <cassert>
#include <Eigen/Dense>

namespace bf
{
    constexpr double pi()
    {
        return std::atan(1.0) * 4.0;
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
}

#endif