#pragma once

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
namespace cv {
    typedef Rect_<int> Rect2i;
    typedef Rect_<float> Rect2f;
}
#endif

#if CV_MAJOR_VERSION <= 2
namespace cv {
    const int FILLED = -1;
    
    /**
     From OpenCV 4.0.1, for backwards compatibility.
     */
    /** @brief %VideoCapture generic properties identifier.
     
     Reading / writing properties involves many layers. Some unexpected result might happens along this chain.
     Effective behaviour depends from device hardware, driver and API Backend.
     @sa videoio_flags_others, VideoCapture::get(), VideoCapture::set()
     */
    enum VideoCaptureProperties {
        CAP_PROP_POS_MSEC       =0, //!< Current position of the video file in milliseconds.
        CAP_PROP_POS_FRAMES     =1, //!< 0-based index of the frame to be decoded/captured next.
        CAP_PROP_POS_AVI_RATIO  =2, //!< Relative position of the video file: 0=start of the film, 1=end of the film.
        CAP_PROP_FRAME_WIDTH    =3, //!< Width of the frames in the video stream.
        CAP_PROP_FRAME_HEIGHT   =4, //!< Height of the frames in the video stream.
        CAP_PROP_FPS            =5, //!< Frame rate.
        CAP_PROP_FOURCC         =6, //!< 4-character code of codec. see VideoWriter::fourcc .
        CAP_PROP_FRAME_COUNT    =7, //!< Number of frames in the video file.
        CAP_PROP_FORMAT         =8, //!< Format of the %Mat objects returned by VideoCapture::retrieve().
        CAP_PROP_MODE           =9, //!< Backend-specific value indicating the current capture mode.
        CAP_PROP_BRIGHTNESS    =10, //!< Brightness of the image (only for those cameras that support).
        CAP_PROP_CONTRAST      =11, //!< Contrast of the image (only for cameras).
        CAP_PROP_SATURATION    =12, //!< Saturation of the image (only for cameras).
        CAP_PROP_HUE           =13, //!< Hue of the image (only for cameras).
        CAP_PROP_GAIN          =14, //!< Gain of the image (only for those cameras that support).
        CAP_PROP_EXPOSURE      =15, //!< Exposure (only for those cameras that support).
        CAP_PROP_CONVERT_RGB   =16, //!< Boolean flags indicating whether images should be converted to RGB.
        CAP_PROP_WHITE_BALANCE_BLUE_U =17, //!< Currently unsupported.
        CAP_PROP_RECTIFICATION =18, //!< Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently).
        CAP_PROP_MONOCHROME    =19,
        CAP_PROP_SHARPNESS     =20,
        CAP_PROP_AUTO_EXPOSURE =21, //!< DC1394: exposure control done by camera, user can adjust reference level using this feature.
        CAP_PROP_GAMMA         =22,
        CAP_PROP_TEMPERATURE   =23,
        CAP_PROP_TRIGGER       =24,
        CAP_PROP_TRIGGER_DELAY =25,
        CAP_PROP_WHITE_BALANCE_RED_V =26,
        CAP_PROP_ZOOM          =27,
        CAP_PROP_FOCUS         =28,
        CAP_PROP_GUID          =29,
        CAP_PROP_ISO_SPEED     =30,
        CAP_PROP_BACKLIGHT     =32,
        CAP_PROP_PAN           =33,
        CAP_PROP_TILT          =34,
        CAP_PROP_ROLL          =35,
        CAP_PROP_IRIS          =36,
        CAP_PROP_SETTINGS      =37, //!< Pop up video/camera filter dialog (note: only supported by DSHOW backend currently. The property value is ignored)
        CAP_PROP_BUFFERSIZE    =38,
        CAP_PROP_AUTOFOCUS     =39,
        CAP_PROP_SAR_NUM       =40, //!< Sample aspect ratio: num/den (num)
        CAP_PROP_SAR_DEN       =41, //!< Sample aspect ratio: num/den (den)
        CAP_PROP_BACKEND       =42, //!< Current backend (enum VideoCaptureAPIs). Read-only property
        CAP_PROP_CHANNEL       =43, //!< Video input or Channel Number (only for those cameras that support)
        CAP_PROP_AUTO_WB       =44, //!< enable/ disable auto white-balance
        CAP_PROP_WB_TEMPERATURE=45, //!< white-balance color temperature
#ifndef CV_DOXYGEN
        CV__CAP_PROP_LATEST
#endif
    };
    
    //! Distance types for Distance Transform and M-estimators
    //! @see distanceTransform, fitLine
    enum DistanceTypes {
        DIST_USER    = -1,  //!< User defined distance
        DIST_L1      = 1,   //!< distance = |x1-x2| + |y1-y2|
        DIST_L2      = 2,   //!< the simple euclidean distance
        DIST_C       = 3,   //!< distance = max(|x1-x2|,|y1-y2|)
        DIST_L12     = 4,   //!< L1-L2 metric: distance = 2(sqrt(1+x*x/2) - 1))
        DIST_FAIR    = 5,   //!< distance = c^2(|x|/c-log(1+|x|/c)), c = 1.3998
        DIST_WELSCH  = 6,   //!< distance = c^2/2(1-exp(-(x/c)^2)), c = 2.9846
        DIST_HUBER   = 7    //!< distance = |x|<c ? x^2/2 : c(|x|-c/2), c=1.345
    };
}
#endif

namespace cmn {
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
#define GETTER_NCONST(TYPE, VAR) protected: TYPE _##VAR; public: inline const TYPE& VAR() const { return _##VAR; } inline TYPE& VAR() { return _##VAR; } protected:
#define GETTER_PTR(TYPE, VAR) protected: TYPE _##VAR; public: inline TYPE VAR() const { return _##VAR; } inline TYPE & VAR() { return _##VAR; } protected:
#define GETTER_CONST_PTR(TYPE, VAR) protected: TYPE _##VAR; public: inline const TYPE VAR() const { return _##VAR; } protected:
#define GETTER_NREF(TYPE, VAR) protected: TYPE _##VAR; public: inline const TYPE VAR() const { return _##VAR; } inline TYPE VAR() { return _##VAR; } protected:
#define GETTER_SETTER(TYPE, VAR) protected: TYPE _##VAR; public: inline const TYPE& VAR() const { return _##VAR; } inline void set_##VAR(const TYPE& value) { _##VAR = value; } protected:
#define GETTER_SETTER_PTR(TYPE, VAR) protected: TYPE _##VAR; public: inline TYPE VAR() const { return _##VAR; } inline void set_##VAR(TYPE value) { _##VAR = value; } protected:
#define IMPLEMENT(VAR) decltype( VAR ) VAR
    
    template<typename T0>
    inline bool isnan(const T0& x, typename std::enable_if<std::is_floating_point<T0>::value || std::is_integral<T0>::value, bool>::type * =NULL) {
        return std::isnan(x);
    }
    
    template<typename T0>
    inline bool isnan(const T0& x, typename std::enable_if<std::is_same<decltype(x.x), decltype(x.y)>::value, bool>::type * =NULL) {
        return std::isnan(x.x) || std::isnan(x.y);
    }

    template<typename T0, typename T1>
    constexpr inline auto min(const T0& x, const T1& y,
                              typename std::enable_if<(!(std::is_unsigned<T0>::value ^ std::is_unsigned<T1>::value)
                                                       && std::is_integral<T0>::value) // allow only same-signedness
                              || std::is_floating_point<T0>::value, bool>::type * = NULL)
    -> decltype(x+y)
    {
        return std::min(decltype(x+y)(x), decltype(x+y)(y));
    }
    
    template<typename T0, typename T1>
    constexpr inline T0 min(const T0& x, const T1& y, typename std::enable_if<std::is_same<decltype(x.x), decltype(x.y)>::value, bool>::type * =NULL) {
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
    
    template<typename T0, typename T1>
    constexpr inline auto max(const T0& x, const T1& y,
                              typename std::enable_if<(!(std::is_unsigned<T0>::value ^ std::is_unsigned<T1>::value)
                                                       && std::is_integral<T0>::value) // allow only same-signedness
                              || std::is_floating_point<T0>::value, bool>::type * = NULL)
    -> decltype(x+y)
    {
        return std::max(decltype(x+y)(x), decltype(x+y)(y));
    }
    
    template<typename T0, typename T1>
    constexpr inline T0 max(const T0& x, const T1& y, typename std::enable_if<std::is_same<decltype(x.x), decltype(x.y)>::value, bool>::type * =NULL) {
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
            typename T::value_type, float>::type;
        
        auto start = values.begin();
        
        if(values.empty())
            return std::numeric_limits<typename T::value_type>::max();
        
        C stride = C(values.size()-1) * percent;
        std::advance(start, stride);
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
