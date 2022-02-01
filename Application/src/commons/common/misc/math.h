#pragma once

namespace cmn {
    template<typename T = double>
    inline T cos(const T& s) {
        return ::cos(s);
    }
    
    template<>
    inline float cos(const float& s) {
        return ::cosf(s);
    }
    
    template<typename T = double>
    inline T sin(const T& s) {
        return ::sin(s);
    }
    
    template<>
    inline float sin(const float& s) {
        return ::sinf(s);
    }
    
    template<typename T = double>
    inline T sqrt(const T& s) {
        return ::sqrt(s);
    }
    
    template<>
    inline float sqrt(const float& s) {
        return ::sqrtf(s);
    }
    
    template<typename T0>
    inline bool isnan(const T0& x, typename std::enable_if<std::is_floating_point<T0>::value || std::is_integral<T0>::value, bool>::type * =NULL) {
        return std::isnan(x);
    }

#ifdef _WIN32
    template<>
    inline bool isnan(const size_t& x, bool *) {
        return std::isnan<double>(static_cast<double>(x));
    }
#endif

    template<typename T0>
    inline bool isnan(const T0& x, typename std::enable_if<std::is_same<decltype(x.x), decltype(x.y)>::value, bool>::type * =NULL) {
        return std::isnan(x.x) || std::isnan(x.y);
    }

#ifdef WIN32
    template<typename T>
    inline bool isinf(T s, typename std::enable_if<std::is_floating_point<T>::value, bool>::type * = NULL) {
        return std::isinf(s);
    }
    
    template<typename T>
    inline bool isinf(T s, typename std::enable_if<std::is_integral<T>::value, bool>::type * = NULL) {
        return !_finite(s);
    }
#else
    template<typename T>
    inline bool isinf(T s, typename std::enable_if<std::is_floating_point<T>::value || std::is_integral<T>::value, bool>::type * = NULL) {
        return std::isinf(s);
    }
#endif
    
    namespace Detail
    {
        template <typename T>
        constexpr T sqrt_helper(T x, T lo, T hi)
        {
            if (lo == hi)
                return lo;
            
            const T mid = (lo + hi + 1) / 2;
            
            if (x / mid < mid)
                return sqrt_helper<T>(x, lo, mid - 1);
            else
                return sqrt_helper(x, mid, hi);
        }
        
        template<typename T>
        T constexpr sqrtNewtonRaphson(T x, T curr, T prev)
        {
            return curr == prev
                ? curr
                : sqrtNewtonRaphson<T>(x, 0.5 * (curr + x / curr), curr);
        }
    }
    
    template <typename T>
    constexpr T ct_int_sqrt(T x)
    {
        return Detail::sqrt_helper<T>(x, 0, x / 2 + 1);
    }
    
    /*
     * Constexpr version of the square root
     * Return value:
     *    - For a finite and non-negative value of "x", returns an approximation for the square root of "x"
     *   - Otherwise, returns NaN
     */
    template<typename T>
    T constexpr ct_sqrt(T x)
    {
        return x >= 0 && x < std::numeric_limits<T>::infinity()
            ? Detail::sqrtNewtonRaphson<T>(x, x, 0)
            : std::numeric_limits<T>::quiet_NaN();
    }
    
    template<typename T = double>
    constexpr inline T csqrt(const T& s, typename std::enable_if<std::is_floating_point<T>::value, bool>::type * = NULL) {
        return ct_sqrt<T>(s);
    }
    
    template<typename T = double>
    constexpr inline T csqrt(const T& s, typename std::enable_if<std::is_integral<T>::value, bool>::type * = NULL) {
        return ct_int_sqrt<T>(s);
    }
    
    template<typename T = double>
    inline T atan2(const T& y, const T& x) {
        return ::atan2(y, x);
    }
    
    template<>
    inline float atan2(const float& y, const float& x) {
        return ::atan2f(y, x);
    }
    
    template<typename T, typename K = typename std::remove_reference<typename std::remove_cv<T>::type>::type>
        requires (!std::unsigned_integral<K>)
    constexpr inline auto abs(T&& x, typename std::enable_if_t< std::is_arithmetic<K>::value> * = NULL)
    {
        return std::abs(std::forward<T>(x));
    }

    template<typename T>
        requires std::unsigned_integral<T>
    constexpr inline auto abs(T x)
    {
        return x;
    }

    namespace check_abs_detail {
        template<typename T>
        concept has_coordinates = requires(T t) {
            { t.x } -> std::convertible_to<float>;
        };

        template<typename T>
        concept has_get = requires(T t) {
            { t.get() } -> std::convertible_to<float>;
        };
    }


    template<typename T>
        requires check_abs_detail::has_coordinates<T>
    inline T abs(const T& x) {
        return T(cmn::abs(x.x), cmn::abs(x.y));
    }

    template<typename T>
        requires check_abs_detail::has_get<T>
    inline T abs(const T& x) {
        return T(std::abs(x.get()));
    }
    
    template <typename T0 = cv::Point2f>
    inline float dot(const T0& point0, const T0& point1) {
        return point0.dot(point1);
    }
    
    /**
     * ======================
     * LENGTH methods
     * ======================
     */
    
    template <typename _Tp = ScalarType, int m>
    ScalarType length(const cv::Matx<_Tp, m, 1>& v) {
        ScalarType value(0.0);
        for(int i = 0; i < m; i++) {
            value += v(i) * v(i);
        }
        return cmn::sqrt(value);
    }
    
    template<typename T, typename V = typename std::enable_if<std::is_same<float, decltype(T::x)>::value || true, bool>::type>
    inline ScalarType length(const T& v) {
        return cmn::sqrt(v.x*v.x+v.y*v.y);
    }
    
    template<typename T>
    inline ScalarType length(const T& v, typename std::enable_if<std::is_same<float, decltype(T::width)>::value || true, bool>::type * = NULL) {
        return cmn::sqrt(v.width*v.width+v.height*v.height);
    }
    
    template<typename T>
    inline ScalarType length(const T& v, typename std::enable_if<std::is_same<decltype(v.r), decltype(v.g * v.b)>::value || true, bool>::type* = NULL) {
        return cmn::sqrt(v.r*v.r+v.g*v.g+v.b*v.b);
    }
    
    template<typename T, typename V = typename std::enable_if<std::is_same<float, decltype(T::x)>::value || true, bool>::type>
    inline ScalarType sqlength(const T& v) {
        return v.x*v.x+v.y*v.y;
    }
    
    template<typename T>
    inline ScalarType sqlength(const T& v, typename std::enable_if<std::is_same<float, decltype(T::width)>::value || true, bool>::type * = NULL) {
        return v.width*v.width+v.height*v.height;
    }
    
    template<typename T>
    inline ScalarType sqlength(const T& v, typename std::enable_if<std::is_same<decltype(v.r), decltype(v.g * v.b)>::value || true, bool>::type* = NULL) {
        return v.r*v.r+v.g*v.g+v.b*v.b;
    }
    
    template<typename T, typename V>
    inline ScalarType cross(const T& v, const V& w, typename std::enable_if<std::is_same<decltype(v.x), decltype(w.y)>::value, bool>::type* = NULL) {
        return v.x * w.y - v.y * w.x;
    }
    
    template<typename T, typename V>
    constexpr inline T cross(const T& v, V w, typename std::enable_if<std::is_floating_point<V>::value, bool>::type* = NULL) {
        return T( w * v.y, -w * v.x );
    }
    
    template<typename T, typename V>
    constexpr inline T cross(V w, const T& v, typename std::enable_if<std::is_floating_point<V>::value, bool>::type* = NULL) {
        return T( -w * v.y, w * v.x );
    }
    
    template <typename T0, typename T1>
    auto sqdistance(const T0& point0, const T1& point1) -> typename std::remove_reference<decltype(point1.x)>::type {
        return SQR(point1.x - point0.x) + SQR(point1.y - point0.y);
    }
    
    template <typename T0, typename T1>
    auto euclidean_distance(const T0& point0, const T1& point1) -> typename std::remove_reference<decltype(point1.x)>::type {
        return cmn::sqrt(sqdistance(point0, point1));
    }
    
    template <typename T0, typename T1>
    auto manhattan_distance(const T0& point0, const T1& point1) -> typename std::remove_reference<decltype(point1.x)>::type {
        return cmn::abs(point1.x - point0.x) + cmn::abs(point1.y - point0.y);
    }
    
    template <typename T0, typename T1, typename T2>
    auto distance_to_line(const T0& point0, const T1& point1, const T2& compare) -> typename std::remove_reference<decltype(point1.x)>::type {
        return cmn::abs((point1.y - point0.y) * compare.x - (point1.x - point0.x) * compare.y + point1.x * point0.y - point1.y * point0.x)
        / euclidean_distance(point0, point1);
    }
    
    //! Returns t0 and t1 for circle line intersections. Circle has center p and
    //  radius r. Line is from v to w.
    template <typename T0, typename T1, typename T2>
    std::pair<float, float> t_circle_line(const T0& v, const T1& w, const T2& p, float r) {
        const float& h = p.x;
        const float& k = p.y;
        const float& x0 = v.x;
        const float& x1 = w.x;
        const float& y0 = v.y;
        const float& y1 = w.y;
        
        float a=SQR(x1-x0)+SQR(y1-y0);
        float b=2*(x1-x0)*(x0-h)+2*(y1-y0)*(y0-k);
        float c=SQR(x0-h)+SQR(y0-k)-SQR(r);
        
        float disc = (SQR(b)-4*a*c);
        if(disc < 0) {
            return {-1,-1};
        }
        
        disc = sqrt(disc);
        
        float t0 = (-b + disc) / (2*a);
        float t1 = (-b - disc) / (2*a);
        
        return {t0, t1};
    }
    
    template <typename T0, typename T1, typename T2>
    float minimum_distance_to_line(const T0& v, const T1& w, const T2& p) {
        // Return minimum distance between line segment vw and point p
        const float l2 = sqdistance(v, w);  // i.e. |w-v|^2 -  avoid a sqrt
        if (l2 == 0.0) return sqdistance(p, v);   // v == w case
        // Consider the line extending the segment, parameterized as v + t (w - v).
        // We find projection of point p onto the line.
        // It falls where t = [(p-v) . (w-v)] / |w-v|^2
        // We clamp t from [0,1] to handle points outside the segment vw.
        const float raw_t = cmn::dot(p - v, w - v) / l2;
        //if(raw_t > 1.0f || raw_t < 0.0f)
        //    return DBL_MAX;
        const float t = cmn::max(0.0f, cmn::min(1.0f, raw_t));
        const auto projection = v + t * (w - v);  // Projection falls on the segment
        return sqdistance(p, projection);
    }
    
    inline cv::Point2f operator*(cv::Mat_<float> M, const cv::Point2f& p)
    {
        cv::Mat_<float> src(3/*rows*/,1 /* cols */);
        
        src(0,0)=p.x;
        src(1,0)=p.y;
        src(2,0)=1.0;
        
        cv::Mat_<float> dst = M*src; //USE MATRIX ALGEBRA
        return cv::Point2f(dst(0,0),dst(1,0));
    }
    
    template<typename T>
    float slope(const T& v0, const T& v1) {
        return (v1.y - v0.y) / (v1.x - v0.x);
    }
    
    template<typename T>
    inline float line_crosses_height(const T& v0, const T& v1, float height = 0) {
        return (v0.y - height) / (v0.y - v1.y);
    }
    
    //! Returns the percentage from (x_0,y_0) + percentage * ((x_1,y_1)-(x_0,y_0))
    //  where the line crosses zero.
    template<typename T>
    inline float crosses_zero(T y0, T y1) {
        return y1 / (y1 - y0);
    }
    
    template<typename T>
    inline int crosses_abs_height(const T& v0, const T& v1, float height = 0) {
        float pos0 = line_crosses_height(v0, v1,  height),
        pos1 = line_crosses_height(v0, v1, -height);
        
        return (pos0 >= 0 && pos0 <= 1) ?  1
        : ((pos1 >= 0 && pos1 <= 1) ? -1 : 0);
    }

template<typename T>
inline constexpr auto infinity()
    -> typename std::enable_if<std::is_floating_point<T>::value, T>::type
{
    return std::numeric_limits<T>::infinity();
}

template<typename T>
inline constexpr auto infinity()
    -> typename std::enable_if<std::is_integral<T>::value, T>::type
{
    return std::numeric_limits<T>::max();
}

template <typename T>
constexpr T next_pow2 (T n)
{
    if(n == T{1}) return 1;
    
    static_assert(sizeof(T) <= 64, "Cannot use this for >64bit.");
    T clz = 0;
    if constexpr (sizeof(T) <= 32)
        clz = __builtin_clzl(n-1); // unsigned long
    else
        clz = __builtin_clzll(n-1); // unsigned long long
    
    return T{1} << (CHAR_BIT * sizeof(T) - clz);
}
}

