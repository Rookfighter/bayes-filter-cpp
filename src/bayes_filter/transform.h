/*
 * transform.h
 *
 *  Created on: 20 Aug 2018
 *      Author: Fabian Meyer
 */

#ifndef BFCPP_TRANSFORM_H_
#define BFCPP_TRANSFORM_H_

#include <Eigen/Geometry>

namespace bf
{
    /** A 3D Transform representing translation and rotation. */
    typedef Eigen::Isometry3d Transform;
    typedef Eigen::Matrix<double, 12, 1> TransformVector;
    typedef Eigen::Matrix<double, 6, 1> TransformVectorQuat;

    inline Transform transformFromVector(const TransformVector &vec)
    {
        Transform result = Transform::Identity();
        result.translation() << vec(0), vec(1), vec(2);

        for(unsigned int r = 0; r < 3; ++r)
            for(unsigned int c = 0; c < 3; ++c)
                result.linear()(r, c) = vec(r*3+c+3);
        return result;
    }

    inline TransformVector transformToVector(const Transform &trans)
    {
        TransformVector result;

        for(unsigned int i = 0; i < 3; ++i)
            result(i) = trans.translation()(i);
        for(unsigned int r = 0; r < 3; ++r)
            for(unsigned int c = 0; c < 3; ++c)
                result(r*3+c+3) = trans.linear()(r, c);
        return result;
    }

    inline Transform transformFromVectorQuat(const TransformVectorQuat &vec)
    {
        Transform result;

        result.translation() = vec.block<3, 1>(0, 0);

        double w = 1 - vec.block<3, 1>(3, 0).squaredNorm();
        if(w < 0)
          result.linear() = Eigen::Matrix3d::Identity();
        else
        {
            result.linear() = Eigen::Quaterniond(
                std::sqrt(w), vec(3), vec(4), vec(5)).toRotationMatrix();
        }

        return result;
    }

    inline TransformVectorQuat transformToVectorQuat(const Transform &trans)
    {
        TransformVectorQuat result;
        Eigen::Quaterniond q(trans.linear());

        // normalize quaternion
        q.normalize();
        if(q.w() < 0)
            q.coeffs() *= -1;

        result.block<3,1>(0,0) = trans.translation();
        result.block<3,1>(3,0) = q.coeffs().head<3>();

        return result;
    }
}

#endif
