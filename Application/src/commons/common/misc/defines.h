#pragma once

#ifdef _MSC_VER
#pragma warning(push, 0)
#endif

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wdeprecated"
#pragma GCC diagnostic ignored "-Wextra"
#endif

#ifdef __llvm__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wall"
#pragma clang diagnostic ignored "-Wextra"
#pragma clang diagnostic ignored "-Wdeprecated"
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wimplicit-int-conversion"
#pragma clang diagnostic ignored "-Wimplicit-float-conversion"
#pragma clang diagnostic ignored "-Wfloat-conversion"
#endif

#include <commons/common/commons.pc.h>

#ifdef _MSC_VER
#define CV_STATIC_ANALYSIS 0
#define CV_ErrorNoReturn(code, msg) cv::errorNoReturn( code, msg, CV_Func, __FILE__, __LINE__ )

#include <opencv2/opencv.hpp>
#endif

#ifndef CMN_WITH_IMGUI_INSTALLED
    #if __has_include ( <imgui/imgui.h> )
    #include <imgui/imgui.h>
        #define CMN_WITH_IMGUI_INSTALLED true
    #else
        #define CMN_WITH_IMGUI_INSTALLED false
    #endif
#endif

#ifdef __llvm__
#pragma clang diagnostic pop
#endif

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <misc/MetaObject.h>
#include <misc/EnumClass.h>

#ifdef WIN32
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#ifdef TEXT
#undef TEXT
#endif
#ifdef small
#undef small
#endif

#define __attribute__(X)
#endif

#if CV_MAJOR_VERSION >= 3
#define USE_GPU_MAT
#else
static_assert(false, "OpenCV version insufficient.");
#endif

namespace cmn {

template< class T >
struct remove_cvref {
    typedef std::remove_cv_t<std::remove_reference_t<T>> type;
};

template< class T >
using remove_cvref_t = typename remove_cvref<T>::type;

#if __has_cpp_attribute(deprecated)
#define DEPRECATED [[deprecated]]
#else
#define DEPRECATED
#endif
    
#ifdef USE_GPU_MAT
    typedef cv::UMat gpuMat;
#else
    typedef cv::Mat gpuMat;
#endif
    
    typedef float ScalarType;
    
    typedef cv::Matx<ScalarType, 1, 3> Mat13;
    
    typedef cv::Matx<ScalarType, 3, 4> Mat34;
    typedef cv::Matx<ScalarType, 3, 3> Mat33;
    typedef cv::Matx<ScalarType, 5, 1> Mat51;
    typedef cv::Matx<ScalarType, 4, 1> Mat41;
    typedef cv::Matx<ScalarType, 3, 1> Mat31;
    typedef cv::Matx<ScalarType, 2, 1> Mat21;
    typedef cv::Matx<ScalarType, 4, 4> Mat44;
    
#define DEGREE(radians) ((radians) * (cmn::ScalarType(1.0) / cmn::ScalarType(M_PI) * cmn::ScalarType(180)))
#define RADIANS(degree) ((degree) * (cmn::ScalarType(1.0) / cmn::ScalarType(180) * cmn::ScalarType(M_PI)))
#define SQR(X) ((X)*(X))
    
#define GETTER_CONST(TYPE, VAR) protected: TYPE _##VAR; public: inline TYPE& VAR() const { return _##VAR; } protected:
#define GETTER(TYPE, VAR) protected: TYPE _##VAR; public: const TYPE& VAR() const { return _##VAR; } protected:
#define GETTER_I(TYPE, VAR, INIT) protected: TYPE _##VAR = INIT; public: const TYPE& VAR() const { return _##VAR; } protected:
#define GETTER_NCONST(TYPE, VAR) protected: TYPE _##VAR; public: inline const TYPE& VAR() const { return _##VAR; } inline TYPE& VAR() { return _##VAR; } protected:
#define GETTER_PTR(TYPE, VAR) protected: TYPE _##VAR; public: inline TYPE VAR() const { return _##VAR; } inline TYPE & VAR() { return _##VAR; } protected:
#define GETTER_CONST_PTR(TYPE, VAR) protected: TYPE _##VAR; public: inline const TYPE VAR() const { return _##VAR; } protected:
#define GETTER_NREF(TYPE, VAR) protected: TYPE _##VAR; public: inline const TYPE VAR() const { return _##VAR; } inline TYPE VAR() { return _##VAR; } protected:
#define GETTER_SETTER(TYPE, VAR) protected: TYPE _##VAR; public: inline const TYPE& VAR() const { return _##VAR; } inline void set_##VAR(const TYPE& value) { _##VAR = value; } protected:
#define GETTER_SETTER_I(TYPE, VAR, INIT) protected: TYPE _##VAR = INIT; public: inline const TYPE& VAR() const { return _##VAR; } inline void set_##VAR(const TYPE& value) { _##VAR = value; } protected:
#define GETTER_SETTER_PTR(TYPE, VAR) protected: TYPE _##VAR; public: inline TYPE VAR() const { return _##VAR; } inline void set_##VAR(TYPE value) { _##VAR = value; } protected:
#define IMPLEMENT(VAR) decltype( VAR ) VAR

    template<typename T0, typename T1,
        typename T0_ = typename remove_cvref<T0>::type,
        typename T1_ = typename remove_cvref<T1>::type,
        typename Result = typename std::conditional<(sizeof(T0_) > sizeof(T1_)), T0_, T1_>::type >
    constexpr inline auto min(T0&& x, T1&& y)
        -> typename std::enable_if< std::is_signed<T0_>::value == std::is_signed<T1_>::value
                                && (std::is_integral<T0_>::value || std::is_floating_point<T0_>::value)
                                && (std::is_integral<T1_>::value || std::is_floating_point<T1_>::value), Result>::type
    {
        return std::min(Result(x), Result(y));
    }
    
    template<typename T0, typename T1>
    constexpr inline auto min(const T0& x, const T1& y)
        -> typename std::enable_if<std::is_same<decltype(x.x), decltype(x.y)>::value, T0>::type
    {
        return T0(std::min(decltype(x.x+y.x)(x.x), decltype(x.x+y.x)(y.x)),
                  std::min(decltype(x.y+y.y)(x.y), decltype(x.y+y.y)(y.y)));
    }
    
    template<typename T0, typename T1>
    constexpr inline T0 min(const T0& x, const T1& y, typename std::enable_if<std::is_same<decltype(x.width), decltype(x.height)>::value, bool>::type * =NULL) {
        return T0(std::min(decltype(x.width+y.width)(x.width), decltype(x.width+y.width)(y.width)),
                  std::min(decltype(x.height+y.height)(x.height), decltype(x.height+y.height)(y.height)));
    }
    
    template<typename T0, typename T1, typename T2>
    constexpr inline auto min(const T0& x, const T1& y, const T2& z) -> decltype(x+y+z) {
        return std::min(decltype(x+y+z)(x), std::min(decltype(x+y+z)(y), decltype(x+y+z)(z)));
    }
        
    template<typename T0, typename T1,
        typename T0_ = typename remove_cvref<T0>::type,
        typename T1_ = typename remove_cvref<T1>::type,
        typename Result = typename std::conditional<(sizeof(T0_) > sizeof(T1_)), T0_, T1_>::type >
    constexpr inline auto max(T0&& x, T1&& y)
        -> typename std::enable_if< std::is_signed<T0_>::value == std::is_signed<T1_>::value
                                && (std::is_integral<T0_>::value || std::is_floating_point<T0_>::value)
                                && (std::is_integral<T1_>::value || std::is_floating_point<T1_>::value), Result>::type
    {
        return std::max(Result(x), Result(y));
    }
    
    template<typename T0, typename T1>
    constexpr inline auto max(const T0& x, const T1& y)
        -> typename std::enable_if<std::is_same<decltype(x.x), decltype(x.y)>::value, T0>::type
    {
        return T0(std::max(decltype(x.x+y.x)(x.x), decltype(x.x+y.x)(y.x)),
                  std::max(decltype(x.y+y.y)(x.y), decltype(x.y+y.y)(y.y)));
    }
    
    template<typename T0, typename T1>
    constexpr inline T0 max(const T0& x, const T1& y, typename std::enable_if<std::is_same<decltype(x.width), decltype(y.height)>::value, bool>::type * =NULL) {
        return T0(std::max(decltype(x.width+y.width)(x.width), decltype(x.width+y.width)(y.width)),
                  std::max(decltype(x.height+y.height)(x.height), decltype(x.height+y.height)(y.height)));
    }
    
    template<typename T0, typename T1>
    constexpr inline T0 max(const T0& x, const T1& y, typename std::enable_if<std::is_same<decltype(x.width), decltype(y.y)>::value, bool>::type * =NULL) {
        return T0(std::max(decltype(x.width+y.x)(x.width), decltype(x.width+y.x)(y.x)),
                  std::max(decltype(x.height+y.y)(x.height), decltype(x.height+y.y)(y.y)));
    }

    template<typename T0, typename T1, typename T2>
    constexpr inline auto max(const T0& x, const T1& y, const T2& z) -> decltype(x+y+z) {
        return std::max(decltype(x+y+z)(x), std::max(decltype(x+y+z)(y), decltype(x+y+z)(z)));
    }
    
    /**
     * ======================
     * COMMON custom STL selectors and iterators
     * ======================
     */
    
    /*template<typename Func, typename T>
    void foreach(Func callback, std::vector<T> &v) {
        std::for_each(v.begin(), v.end(), callback);
    }
    
    template<typename Func, typename T, typename... Args>
    void foreach(Func callback, std::vector<T> &v, Args... args) {
        std::for_each(v.begin(), v.end(), callback);
        return foreach(callback, args...);
    }*/
    
    template<typename Func, typename T>
    void foreach(Func callback, T &v) {
        std::for_each(v.begin(), v.end(), callback);
    }
    
    template<typename Func, typename T, typename... Args>
    void foreach(Func callback, T &v, Args... args) {
        std::for_each(v.begin(), v.end(), callback);
        return foreach(callback, args...);
    }
    
    template<class T> struct is_container : public std::false_type {};
    
    template<class T, class Alloc>
    struct is_container<std::vector<T, Alloc>> : public std::true_type {};
    
    template<class T, size_t Size>
    struct is_container<std::array<T, Size>> : public std::true_type {};
    
    template<class T> struct is_set : public std::false_type {};
    template<class T, class Alloc>
    struct is_set<std::set<T, Alloc>> : public std::true_type {};
    template<class T, class Alloc>
    struct is_set<std::multiset<T, Alloc>> : public std::true_type {};
    template<class T, class Alloc>
    struct is_set<std::unordered_set<T, Alloc>> : public std::true_type {};
    
    template<class T> struct is_map : public std::false_type {};
    template<class T, class Compare, class Alloc>
    struct is_map<std::map<T, Compare, Alloc>> : public std::true_type {};
    template<class T, class Compare, class Alloc>
    struct is_map<std::unordered_map<T, Compare, Alloc>> : public std::true_type {};

    template<class T> struct is_queue : public std::false_type {};
    template<class T, class Container>
    struct is_queue<std::queue<T, Container>> : public std::true_type {};
    template<class T, class Allocator>
    struct is_queue<std::deque<T, Allocator>> : public std::true_type {};

    template<class T> struct is_deque : public std::false_type {};
    template<class T, class Allocator>
    struct is_deque<std::deque<T, Allocator>> : public std::true_type {};
    
    template< class T >
    struct is_pair : public std::false_type {};
    
    template< class T1 , class T2 >
    struct is_pair< std::pair< T1 , T2 > > : public std::true_type {};
    
    class IndexedDataTransport {
    protected:
        GETTER_SETTER(long_t, index)
        
    public:
        virtual ~IndexedDataTransport() {}
    };
    
    template<typename T>
    typename T::value_type percentile(const T& values, float percent, typename std::enable_if<is_set<T>::value || is_container<T>::value, T>::type* = NULL)
    {
        using C = typename std::conditional<std::is_floating_point<typename T::value_type>::value,
            typename T::value_type, double>::type;
        
        auto start = values.begin();
        
        if(values.empty())
            return std::numeric_limits<typename T::value_type>::max();
        
        C stride = C(values.size()-1) * percent;
        std::advance(start, (int64_t)stride);
        C A = *start;
        
        auto second = start;
        ++second;
        if(second != values.end())
            start = second;
        C B = *start;
        C p = stride - C(size_t(stride));
        
        return A * (1 - p) + B * p;
    }
    
    template<typename T, typename K>
    std::vector<typename T::value_type> percentile(const T& values, const std::initializer_list<K>& tests, typename std::enable_if<(is_set<T>::value || is_container<T>::value) && std::is_floating_point<typename T::value_type>::value, T>::type* = NULL)
    {
        std::vector<typename T::value_type> result;
        for(auto percent : tests)
            result.push_back(percentile(values, percent));
        
        return result;
    }
}

#include <misc/math.h>
