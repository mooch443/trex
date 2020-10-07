#pragma once
#include <gui/types/Basic.h>
#include <misc/defines.h>

////////////////////////////////////////////////////////////
//
// SFML - Simple and Fast Multimedia Library
// Copyright (C) 2007-2017 Laurent Gomila (laurent@sfml-dev.org)
//
// This software is provided 'as-is', without any express or implied warranty.
// In no event will the authors be held liable for any damages arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it freely,
// subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented;
//    you must not claim that you wrote the original software.
//    If you use this software in a product, an acknowledgment
//    in the product documentation would be appreciated but is not required.
//
// 2. Altered source versions must be plainly marked as such,
//    and must not be misrepresented as being the original software.
//
// 3. This notice may not be removed or altered from any source distribution.
//
////////////////////////////////////////////////////////////

/**
 * This is an altered version of SFMLs Transform implementation.
 * For more on SFMLs implementation, see https://github.com/SFML/SFML
 */

namespace gui {
class Transform {
    float m_matrix[16];
    
public:
    Transform();
    Transform(float a00, float a01, float a02,
              float a10, float a11, float a12,
              float a20, float a21, float a22);

    const float* getMatrix() const;
    
    Transform getInverse() const;
    Vec2 transformPoint(float x, float y) const;
    Vec2 transformPoint(const Vec2& pt) const;
    
    Bounds transformRect(const Bounds& bounds) const;
    
    Transform& combine(const Transform& other);
    
    Transform& translate(const Vec2&);
    Transform& translate(float x, float y);
    Transform& rotate(float angle);
    Transform& scale(float x, float y);
    Transform& scale(const Vec2& factors);
    
    cv::Mat toCV() const {
        cv::Mat t(2,3,CV_64F,cv::Scalar(0.0));
        
        t.at<double>(0, 0) = getMatrix()[0]; // 0
        t.at<double>(1, 0) = getMatrix()[1]; // 1
        
        t.at<double>(0, 1) = getMatrix()[4]; // 4
        t.at<double>(1, 1) = getMatrix()[5]; // 5
        
        t.at<double>(0, 2) = getMatrix()[12]; // 12
        t.at<double>(1, 2) = getMatrix()[13]; // 13
        
        return t;
    }
};
    
    Transform operator *(const Transform& left, const Transform& right);
    Transform& operator *=(Transform& left, const Transform& right);
    Vec2 operator *(const Transform& left, const Vec2& right);
}
