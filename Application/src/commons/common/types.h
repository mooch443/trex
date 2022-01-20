#ifndef VIDEO_TYPES_H
#define VIDEO_TYPES_H

#ifndef WIN32
    #if __cplusplus <= 199711L
      #error This library needs at least a C++11 compliant compiler
    #endif

    #define TREX_EXPORT
    #define EXPIMP_TEMPLATE

#else
    #ifdef TREX_EXPORTS
        #define TREX_EXPORT __declspec(dllexport)
        #define EXPIMP_TEMPLATE
    #else
        #define TREX_EXPORT __declspec(dllimport)
        #define EXPIMP_TEMPLATE extern
    #endif
#endif

#include <misc/detail.h>

#ifdef _MSC_VER
#include <intrin.h>

DWORD __inline __builtin_ctz(uint32_t value)
{
    DWORD trailing_zero = 0;
    _BitScanForward(&trailing_zero, value);
    return trailing_zero;
}

#endif

/**
 * ======================
 * THREAD-SAFE methods
 * ======================
 */

namespace cmn {
typedef std::vector<std::tuple<std::shared_ptr<std::vector<HorizontalLine>>, std::shared_ptr<std::vector<uchar>>>> blobs_t;
constexpr int CV_MAX_THICKNESS = 32767;
}


#define DEBUG_CV(COMMAND) try { COMMAND; } catch(const cv::Exception& e) { Except("OpenCV Exception ('%s':%d): %s\n%s", __FILE__, __LINE__, #COMMAND, e.what()); }

namespace tf {
    void imshow(const std::string& name, const cv::Mat& mat, std::string label = "");
    void show();
    void waitKey(std::string name);
}

namespace gui {
    using namespace cmn;
}

namespace track {
    using namespace cmn;
}

#endif
