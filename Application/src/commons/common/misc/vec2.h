#pragma once

#include <misc/defines.h>

namespace cmn {
    typedef float Float2_t;
    struct size_members { Float2_t width, height; };
    struct vec_members { Float2_t x, y; };
    
    template<bool is_size>
    class Float2 : public std::conditional<is_size, size_members, vec_members>::type {
    public:
        typedef typename std::conditional<is_size, size_members, vec_members>::type member_base;
        typedef Float2<is_size> self_type;
        
    public:
#define F2_IS_VECTOR template<typename S = member_base, typename std::enable_if<std::is_same<vec_members, S>::value, int>::type = 0>
#define F2_IS_SIZE template<typename S = member_base, typename std::enable_if<std::is_same<size_members, S>::value, int>::type = 0>
        
        F2_IS_VECTOR inline constexpr const Float2_t& A() const { return member_base::x; }
        F2_IS_VECTOR inline constexpr const Float2_t& B() const { return member_base::y; }
        
        F2_IS_VECTOR inline constexpr Float2_t& A() { return member_base::x; }
        F2_IS_VECTOR inline constexpr Float2_t& B() { return member_base::y; }
        
        F2_IS_SIZE inline constexpr const Float2_t& A() const { return member_base::width; }
        F2_IS_SIZE inline constexpr const Float2_t& B() const { return member_base::height; }
        
        F2_IS_SIZE inline constexpr Float2_t& A() { return member_base::width; }
        F2_IS_SIZE inline constexpr Float2_t& B() { return member_base::height; }
        
        F2_IS_SIZE Float2(const cv::Size& v) : Float2(v.width, v.height) {}
        
#undef F2_IS_VECTOR
#undef F2_IS_SIZE
        
    public:
        template<bool K>
        explicit constexpr Float2(const Float2<K>& other) noexcept : Float2(other.A(), other.B()) {}
        constexpr Float2() noexcept : Float2(0, 0) {}
        explicit constexpr Float2(Float2_t a) noexcept : Float2(a, a) {}
        constexpr Float2(Float2_t a, Float2_t b) noexcept : member_base{a, b} { }
        
#if CMN_WITH_IMGUI_INSTALLED
        Float2(const ImVec2& v) noexcept : Float2(v.x, v.y) {}
        operator ImVec2() const { return ImVec2(A(), B()); }
#endif
        
        template<typename T>
        Float2(const cv::Point_<T>& v) noexcept : Float2(v.x, v.y) {}
        
        template<bool K = is_size>
        explicit Float2(const typename std::enable_if<K, cv::Mat>::type& matrix) noexcept
            : Float2(matrix.cols, matrix.rows)
        {}
        
        template<typename T>
        operator cv::Point_<T>() const { return cv::Point_<T>((int)A(), (int)B()); }
        
        operator const Float2<!is_size>&() const { return *((const Float2<!is_size>*)this); }
        operator Float2<!is_size>&() { return *((Float2<!is_size>*)this); }
        
        constexpr Float2_t min() const { return cmn::min(A(), B()); }
        constexpr Float2_t max() const { return cmn::max(A(), B()); }
        constexpr Float2 abs() const { return Float2(cmn::abs(A()), cmn::abs(B())); }
        
        constexpr bool empty() const { return operator==(Float2(0)); }
        
        //! Element-wise multiplication
        template<bool K>
        constexpr Float2 mul(const Float2<K>& other) const {
            return Float2(A() * other.A(), B() * other.B());
        }
        template<bool K>
        constexpr Float2 div(const Float2<K>& other) const {
            return Float2(A() / other.A(), B() / other.B());
        }
        
        constexpr Float2 mul(const Float2_t& other) const {
            return Float2(A() * other, B() * other);
        }
        constexpr Float2 div(const Float2_t& other) const {
            return Float2(A() / other, B() / other);
        }
        
        constexpr Float2 T() const {
            return Float2(B(), A());
        }
        
        template<Float2_t (*F)(Float2_t)>
        constexpr Float2 map() const {
            return Float2(F(A()), F(B()));
        }

        constexpr Float2 map(std::function<Float2_t(Float2_t)>&& fn) const {
            return Float2(fn(A()), fn(B()));
        }
        
        static constexpr inline bool almost_equal(Float2_t a, Float2_t b) {
            constexpr auto epsilon = Float2_t(0.001);
            return cmn::abs(a - b) <= epsilon;
        }
        
        template<bool K>
        constexpr bool Equals(const Float2<K>& other) const {
            return almost_equal(A(), other.A()) && almost_equal(B(), other.B());
        }
        
        template<bool K>
        constexpr bool operator==(const Float2<K>& other) const {
            return other.A() == A() && other.B() == B();
        }
        template<bool K>
        constexpr bool operator!=(const Float2<K>& other) const {
            return other.A() != A() || other.B() != B();
        }
        
        template<bool K>
        constexpr bool operator<(const Float2<K>& other) const {
            return other.B() < B() || (other.B() == B() && other.A() < A());
        }
        
        template<bool K>
        constexpr bool operator>(const Float2<K>& other) const {
            return other.B() > B() || (other.B() == B() && other.A() > A());
        }
        
        template<bool K>
        constexpr Float2& operator+=(const Float2<K>& other) {
            A() += other.A();
            B() += other.B();
            return *this;
        }
        
        template<typename K>
        constexpr Float2& operator+=(const cv::Point_<K>& other) {
            A() += other.x;
            B() += other.y;
            return *this;
        }
        
        template<bool K>
        constexpr Float2& operator-=(const Float2<K>& other) {
            A() -= other.A();
            B() -= other.B();
            return *this;
        }
        
        template<typename K>
        constexpr Float2& operator-=(const cv::Point_<K>& other) {
            A() -= other.x;
            B() -= other.y;
            return *this;
        }
        
        template<bool K>
        constexpr Float2_t dot(const Float2<K>& other) const {
            return A()*other.A() + B()*other.B();
        }
        
        constexpr Float2 perp() const {
            return Float2(B(), -A());
        }
        
        constexpr Float2_t length() const {
            return cmn::sqrt(sqlength());
        }
        
        template<typename K>
        static constexpr K square(K k) {
            return k * k;
        }
        
        constexpr Float2_t sqlength() const {
            return square(A()) + square(B());
        }
        
        constexpr Float2 normalize() const {
            auto L = length();
            return Float2_t(L != 0) * (*this / (Float2_t(L == 0) + L));
        }
        
        constexpr Float2 clip(Float2_t start, Float2_t end) const {
            return Float2(saturate(A(), start, end), saturate(B(), start, end));
        }
        
        constexpr Float2 clip(const Float2& start, const Float2& end) const {
            return Float2(saturate(A(), start.A(), end.A()), saturate(B(), start.B(), end.B()));
        }
        
        //! Element-wise reciprocal (1/x)
        constexpr Float2 reciprocal() const {
            return Float2(1.f / A(), 1.f / B());
        }
        
        constexpr Float2& operator+=(Float2_t other) { A() += other; B() += other; return *this; }
        constexpr Float2& operator-=(Float2_t other) { A() -= other; B() -= other; return *this; }
        constexpr Float2& operator*=(Float2_t other) { A() *= other; B() *= other; return *this; }
        constexpr Float2& operator/=(Float2_t other) { A() /= other; B() /= other; return *this; }
    };
    
    typedef Float2<true> Size2;
    typedef Float2<false> Vec2;
    
#define TEMPLATE_FLOAT2_OTHER template<typename K, typename T, typename std::enable_if<std::is_base_of<vec_members, T>::value || std::is_base_of<size_members, T>::value, int>::type = 0>
#define TEMPLATE_FLOAT2 template<typename T, typename std::enable_if<std::is_base_of<vec_members, T>::value || std::is_base_of<size_members, T>::value, int>::type = 0>
#define TEMPLATE_FLOAT2_SEPERATE template<typename T0, typename T1, typename std::enable_if<(std::is_base_of<vec_members, T0>::value || std::is_base_of<size_members, T0>::value) && (std::is_base_of<vec_members, T1>::value || std::is_base_of<size_members, T1>::value), int>::type = 0>
    
#define ScalarFloat2operator(SIGN) \
TEMPLATE_FLOAT2 \
constexpr inline T operator SIGN(const T& v, Float2_t s) { return T(v.A() SIGN s, v.B() SIGN s); }
    
#define RScalarFloat2operator(SIGN) \
TEMPLATE_FLOAT2 \
constexpr inline T operator SIGN(Float2_t s, const T& v) { return T(s SIGN v.A(), s SIGN v.B()); }
    
    ScalarFloat2operator(+)
    ScalarFloat2operator(-)
    ScalarFloat2operator(*)
    RScalarFloat2operator(*)
    ScalarFloat2operator(/)
    
    TEMPLATE_FLOAT2
    constexpr inline T operator -(const T& v) { return T(-v.A(), -v.B()); }
    
    TEMPLATE_FLOAT2
    constexpr inline T operator +(const T& v) { return v; }
    
    TEMPLATE_FLOAT2_SEPERATE
    constexpr inline T0 operator+(const T0& v, const T1& w) {
        return T0(v.A() + w.A(), v.B() + w.B());
    }
    
    TEMPLATE_FLOAT2_SEPERATE
    constexpr inline T0 operator-(const T0& v, const T1& w) {
        return T0(v.A() - w.A(), v.B() - w.B());
    }
    
    TEMPLATE_FLOAT2_OTHER
    constexpr inline T operator+(const cv::Point_<K>& v, const T& w) {
        return T(v.x + w.A(), v.y + w.B());
    }
    
    TEMPLATE_FLOAT2_OTHER
    constexpr inline T operator-(const cv::Point_<K>& v, const T& w) {
        return T(v.x - w.A(), v.y - w.B());
    }
    
    TEMPLATE_FLOAT2_OTHER
    constexpr inline T operator+(const T& w, const cv::Point_<K>& v) {
        return T(v.x + w.A(), v.y + w.B());
    }
    
    TEMPLATE_FLOAT2_OTHER
    constexpr inline T operator-(const T& w, const cv::Point_<K>& v) {
        return T(w.A() - v.x, w.B() - v.y);
    }
    
    TEMPLATE_FLOAT2
    inline std::ostream &operator <<(std::ostream &os, const T& obj) {
        uint _x = (uint)roundf(obj.A()), _y = (uint)roundf(obj.B());
        //assert(obj.A() >= SHRT_MIN && obj.B() >= SHRT_MIN && obj.A() <= SHRT_MAX && obj.B() <= SHRT_MAX);
        
        uint both = (_x << 16) & 0xFFFF0000;
        both |= _y & 0xFFFF;
        
        return os << both;
    }
    
#undef TEMPLATE_FLOAT2_SEPERATE
#undef TEMPLATE_FLOAT2
#undef RScalarFloat2operator
#undef ScalarFloat2operator
#undef TEMPLATE_FLOAT2_OTHER
    
    template<bool T>
    inline Float2_t atan2(const Float2<T>& vector) {
        return ::atan2(vector.y, vector.x);
    }
    
    class Bounds {
    public:
        union {
            struct { Float2_t x, y; };
            Vec2 _pos;
        };
        
        union {
            struct { Float2_t width, height; };
            Size2 _size;
        };
        
    public:
        Bounds(Bounds&& other) = default;
        
        constexpr Bounds(const Bounds& other)
            : Bounds(other.x, other.y, other.width, other.height)
        {}
        
        constexpr Bounds(Float2_t _x = 0, Float2_t _y = 0, Float2_t w = 0, Float2_t h = 0)
            : x(_x), y(_y), width(w), height(h)
        {}
        
        constexpr Bounds(const Vec2& pos,
                         const Size2& dim = Size2())
            : Bounds(pos.x, pos.y, dim.width, dim.height)
        {}
        
        constexpr Bounds(const Size2& dim)
            : Bounds(0, 0, dim.width, dim.height)
        {}
        
        explicit Bounds(const cv::Mat& matrix) : Bounds(0, 0, matrix.cols, matrix.rows) {}
        
        template<typename T>
        Bounds(const cv::Rect_<T>& rect) : Bounds(rect.x, rect.y, rect.width, rect.height) {}
        
        template<typename T>
        operator cv::Rect_<T>() const { return cv::Rect_<T>((int)x, (int)y, (int)width, (int)height); }
        
        constexpr const Vec2& pos() const { return this->_pos; }
        constexpr Vec2& pos() { return this->_pos; }
        
        constexpr const Size2& size() const { return _size; }
        constexpr Size2& size() { return _size; }
        
        constexpr void operator=(const Bounds& other) {
            x = other.x;
            y = other.y;
            width = other.width;
            height = other.height;
        }
        
        constexpr bool operator<(const Bounds& other) const {
            return pos() < other.pos() || (pos() == other.pos() && size() < other.size());
        }
        
        constexpr bool Equals(const Bounds& other) const {
            return pos().Equals(other.pos()) && size().Equals(other.size());
        }
        constexpr bool operator==(const Bounds& other) const {
            return x == other.x && y == other.y && width == other.width && height == other.height;
        }
        constexpr bool operator!=(const Bounds& other) const {
            return x != other.x || y != other.y || width != other.width || height != other.height;
        }
        
        constexpr bool contains(const Vec2& point) const { return contains(point.x, point.y); }
        constexpr bool contains(const Float2_t x, const Float2_t y) const {
            return x >= this->x && x < this->x+width && y >= this->y && y < this->y+height;
        }
        constexpr bool contains(const Bounds& other) const {
            return contains(other.pos()) || contains(other.pos()+other.size());
        }
        
        void restrict_to(const Bounds& bounds);
        void insert_point(const Vec2& pt) {
            if(pt.x < x) x = pt.x;
            if(pt.y < y) y = pt.y;
            if(pt.x > width) width = pt.x;
            if(pt.y > height) height = pt.y;
        }
        
        template<bool K>
        constexpr Bounds mul(const Float2<K>& other) const {
            return Bounds(x * other.A(), y * other.B(), width * other.A(), height * other.B());
        }
        
        //! Calculate the bounding-box of combined this and other
        constexpr void combine(const Bounds& other) {
            // dont combine empty rects (for example Lines, which dont support bounds)
            if(other.width == 0 && other.height == 0)
                return;
            
            if(other.x < x) {
                if(x != FLT_MAX)
                    width += x - other.x;
                x = other.x;
            }
            
            if(other.y < y) {
                if(y != FLT_MAX)
                    height += y - other.y;
                y = other.y;
            }
            
            width  = max(x + width,  other.x+other.width) - x;
            height = max(y + height, other.y+other.height) - y;
        }
        
        constexpr bool overlaps(const Bounds& other) const {
            const auto r = x + width, oR = other.x + other.width;
            const auto b = y + height, oB = other.y + other.height;
            
            return (oR >= x && other.x <= r && oB >= y && other.y <= b)
                || (r >= other.x && x <= oR && b >= other.y && y <= oB);
        }
        
        constexpr bool intersects(const Bounds& other) const {
            return x < other.x + other.width  && x + width > other.x
                && y > other.y + other.height && y + height < other.y;
        }
        
        constexpr bool empty() const { return width == 0 && height == 0; }
        
        Float2_t distance(const Vec2& p) const;
    };
    
    //! Calculates the angle between two given vectors.
    //  The result has a value range from [0, pi] (acos) and thus does
    //  not provide information about signedness.
    inline Float2_t angle_between_vectors(const Vec2& v0, const Vec2& v1) {
        return acosf(v0.normalize().dot(v1.normalize()));
    }
        
    inline bool pnpoly(const std::vector<Vec2>& pts, const Vec2& pt)
    {
        size_t npol = pts.size();
        size_t i, j;
        bool c = false;
        for (i = 0, j = npol-1; i < npol; j = i++) {
            if ((((pts[i].y <= pt.y) && (pt.y < pts[j].y)) ||
                 ((pts[j].y <= pt.y) && (pt.y < pts[i].y))) &&
                (pt.x < (pts[j].x - pts[i].x) * (pt.y - pts[i].y) / (pts[j].y - pts[i].y) + pts[i].x))
                c = !c;
        }
        return c;
    }
        
    inline std::shared_ptr<std::vector<Vec2>> poly_convex_hull(const std::vector<Vec2>* _vertices)
    {
        /**
            SOURCE: https://github.com/RandyGaul/ImpulseEngine
         
             Copyright (c) 2013 Randy Gaul http://RandyGaul.net

               This software is provided 'as-is', without any express or implied
               warranty. In no event will the authors be held liable for any damages
               arising from the use of this software.

               Permission is granted to anyone to use this software for any purpose,
               including commercial applications, and to alter it and redistribute it
               freely, subject to the following restrictions:
                 1. The origin of this software must not be misrepresented; you must not
                    claim that you wrote the original software. If you use this software
                    in a product, an acknowledgment in the product documentation would be
                    appreciated but is not required.
                 2. Altered source versions must be plainly marked as such, and must not be
                    misrepresented as being the original software.
                 3. This notice may not be removed or altered from any source distribution.
         */
        
        uint32_t count = (uint32_t)_vertices->size();
        if(count > 2 && count <= 64) {
            // No hulls with less than 3 vertices (ensure actual polygon)
            assert( count > 2 && count <= 64 );
            count = std::min( count, 64u );
            auto _points = std::make_shared<std::vector<Vec2>>();
            _points->resize(count);
            
            // Find the right most point on the hull
            uint32_t rightMost = 0;
            double highestXCoord = (*_vertices)[0].x;
            for(uint32_t i = 1; i < count; ++i)
            {
                double x = (*_vertices)[i].x;
                if(x > highestXCoord)
                {
                    highestXCoord = x;
                    rightMost = i;
                }
                
                // If matching x then take farthest negative y
                else if(x == highestXCoord)
                    if((*_vertices)[i].y < (*_vertices)[rightMost].y)
                        rightMost = i;
            }
            
            uint32_t hull[64];
            uint32_t outCount = 0;
            uint32_t indexHull = rightMost;
            
            for (;;)
            {
                hull[outCount] = indexHull;
                
                // Search for next index that wraps around the hull
                // by computing cross products to find the most counter-clockwise
                // vertex in the set, given the previos hull index
                uint32_t nextHullIndex = 0;
                for(uint32_t i = 1; i < count; ++i)
                {
                    // Skip if same coordinate as we need three unique
                    // points in the set to perform a cross product
                    if(nextHullIndex == indexHull)
                    {
                        nextHullIndex = i;
                        continue;
                    }
                    
                    // Cross every set of three unique vertices
                    // Record each counter clockwise third vertex and add
                    // to the output hull
                    // See : http://www.oocities.org/pcgpe/math2d.html
                    Vec2 e1 = (*_vertices)[nextHullIndex] - (*_vertices)[hull[outCount]];
                    Vec2 e2 = (*_vertices)[i] - (*_vertices)[hull[outCount]];
                    double c = cross( e1, e2 );
                    if(c < 0.0f)
                        nextHullIndex = i;
                    
                    // Cross product is zero then e vectors are on same line
                    // therefor want to record vertex farthest along that line
                    if(c == 0.0f && e2.sqlength() > e1.sqlength())
                        nextHullIndex = i;
                }
                
                ++outCount;
                indexHull = nextHullIndex;
                
                // Conclude algorithm upon wrap-around
                if(nextHullIndex == rightMost)
                {
                    _points->resize(outCount);
                    //m_vertexCount = outCount;
                    break;
                }
            }
            
            // Copy vertices into shape's vertices
            for(uint32_t i = 0; i < _points->size(); ++i)
                (*_points)[i] = (*_vertices)[hull[i]];
            
            return _points;
        }
        
        return nullptr;
    }
}
