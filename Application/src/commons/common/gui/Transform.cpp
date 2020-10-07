#include "Transform.h"

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
    Transform::Transform()
    {
        // Identity matrix
        m_matrix[0] = 1.f; m_matrix[4] = 0.f; m_matrix[8]  = 0.f; m_matrix[12] = 0.f;
        m_matrix[1] = 0.f; m_matrix[5] = 1.f; m_matrix[9]  = 0.f; m_matrix[13] = 0.f;
        m_matrix[2] = 0.f; m_matrix[6] = 0.f; m_matrix[10] = 1.f; m_matrix[14] = 0.f;
        m_matrix[3] = 0.f; m_matrix[7] = 0.f; m_matrix[11] = 0.f; m_matrix[15] = 1.f;
    }
    
    
    ////////////////////////////////////////////////////////////
    Transform::Transform(float a00, float a01, float a02,
                         float a10, float a11, float a12,
                         float a20, float a21, float a22)
    {
        m_matrix[0] = a00; m_matrix[4] = a01; m_matrix[8]  = 0.f; m_matrix[12] = a02;
        m_matrix[1] = a10; m_matrix[5] = a11; m_matrix[9]  = 0.f; m_matrix[13] = a12;
        m_matrix[2] = 0.f; m_matrix[6] = 0.f; m_matrix[10] = 1.f; m_matrix[14] = 0.f;
        m_matrix[3] = a20; m_matrix[7] = a21; m_matrix[11] = 0.f; m_matrix[15] = a22;
    }
    
    Transform Transform::getInverse() const
    {
        // Compute the determinant
        float det = m_matrix[0] * (m_matrix[15] * m_matrix[5] - m_matrix[7] * m_matrix[13]) -
        m_matrix[1] * (m_matrix[15] * m_matrix[4] - m_matrix[7] * m_matrix[12]) +
        m_matrix[3] * (m_matrix[13] * m_matrix[4] - m_matrix[5] * m_matrix[12]);
        
        // Compute the inverse if the determinant is not zero
        // (don't use an epsilon because the determinant may *really* be tiny)
        if (det != 0.f)
        {
            return Transform( (m_matrix[15] * m_matrix[5] - m_matrix[7] * m_matrix[13]) / det,
                             -(m_matrix[15] * m_matrix[4] - m_matrix[7] * m_matrix[12]) / det,
                             (m_matrix[13] * m_matrix[4] - m_matrix[5] * m_matrix[12]) / det,
                             -(m_matrix[15] * m_matrix[1] - m_matrix[3] * m_matrix[13]) / det,
                             (m_matrix[15] * m_matrix[0] - m_matrix[3] * m_matrix[12]) / det,
                             -(m_matrix[13] * m_matrix[0] - m_matrix[1] * m_matrix[12]) / det,
                             (m_matrix[7]  * m_matrix[1] - m_matrix[3] * m_matrix[5])  / det,
                             -(m_matrix[7]  * m_matrix[0] - m_matrix[3] * m_matrix[4])  / det,
                             (m_matrix[5]  * m_matrix[0] - m_matrix[1] * m_matrix[4])  / det);
        }
        else
        {
            return Transform();
        }
    }
    
Vec2 Transform::transformPoint(float x, float y) const
{
    return Vec2(m_matrix[0] * x + m_matrix[4] * y + m_matrix[12],
                    m_matrix[1] * x + m_matrix[5] * y + m_matrix[13]);
}


////////////////////////////////////////////////////////////
Vec2 Transform::transformPoint(const Vec2& point) const
{
    return transformPoint(point.x, point.y);
}


////////////////////////////////////////////////////////////
Bounds Transform::transformRect(const Bounds& rectangle) const
{
    // Transform the 4 corners of the rectangle
    const Vec2 points[] =
    {
        transformPoint(rectangle.x, rectangle.y),
        transformPoint(rectangle.x, rectangle.y + rectangle.height),
        transformPoint(rectangle.x + rectangle.width, rectangle.y),
        transformPoint(rectangle.x + rectangle.width, rectangle.y + rectangle.height)
    };
    
    // Compute the bounding rectangle of the transformed points
    float left = points[0].x;
    float top = points[0].y;
    float right = points[0].x;
    float bottom = points[0].y;
    for (int i = 1; i < 4; ++i)
    {
        if      (points[i].x < left)   left = points[i].x;
        else if (points[i].x > right)  right = points[i].x;
        if      (points[i].y < top)    top = points[i].y;
        else if (points[i].y > bottom) bottom = points[i].y;
    }
    
    return Bounds(left, top, right - left, bottom - top);
}


////////////////////////////////////////////////////////////
Transform& Transform::combine(const Transform& transform)
{
    const float* a = m_matrix;
    const float* b = transform.m_matrix;
    
    *this = Transform(a[0] * b[0]  + a[4] * b[1]  + a[12] * b[3],
                      a[0] * b[4]  + a[4] * b[5]  + a[12] * b[7],
                      a[0] * b[12] + a[4] * b[13] + a[12] * b[15],
                      a[1] * b[0]  + a[5] * b[1]  + a[13] * b[3],
                      a[1] * b[4]  + a[5] * b[5]  + a[13] * b[7],
                      a[1] * b[12] + a[5] * b[13] + a[13] * b[15],
                      a[3] * b[0]  + a[7] * b[1]  + a[15] * b[3],
                      a[3] * b[4]  + a[7] * b[5]  + a[15] * b[7],
                      a[3] * b[12] + a[7] * b[13] + a[15] * b[15]);
    
    return *this;
}


////////////////////////////////////////////////////////////
Transform& Transform::translate(float x, float y)
{
    Transform translation(1, 0, x,
                          0, 1, y,
                          0, 0, 1);
    
    return combine(translation);
}


////////////////////////////////////////////////////////////
Transform& Transform::translate(const Vec2& offset)
{
    return translate(offset.x, offset.y);
}


////////////////////////////////////////////////////////////
Transform& Transform::rotate(float angle)
{
    float rad = angle * 3.141592654f / 180.f;
    float cos = std::cos(rad);
    float sin = std::sin(rad);
    
    Transform rotation(cos, -sin, 0,
                       sin,  cos, 0,
                       0,    0,   1);
    
    return combine(rotation);
}


////////////////////////////////////////////////////////////
Transform& Transform::scale(float scaleX, float scaleY)
{
    Transform scaling(scaleX, 0,      0,
                      0,      scaleY, 0,
                      0,      0,      1);
    
    return combine(scaling);
}


////////////////////////////////////////////////////////////
Transform& Transform::scale(const Vec2& factors)
{
    return scale(factors.x, factors.y);
}


////////////////////////////////////////////////////////////
Transform operator *(const Transform& left, const Transform& right)
{
    return Transform(left).combine(right);
}


////////////////////////////////////////////////////////////
Transform& operator *=(Transform& left, const Transform& right)
{
    return left.combine(right);
}


////////////////////////////////////////////////////////////
Vec2 operator *(const Transform& left, const Vec2& right)
{
    return left.transformPoint(right);
}
    
    const float* Transform::getMatrix() const {
        return m_matrix;
    }

}
