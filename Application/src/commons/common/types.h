#ifndef VIDEO_TYPES_H
#define VIDEO_TYPES_H

#ifndef WIN32
#if __cplusplus <= 199711L
  #error This library needs at least a C++11 compliant compiler
#endif
#endif

#include <misc/detail.h>

/**
 * ======================
 * THREAD-SAFE methods
 * ======================
 */

namespace cmn {
typedef std::vector<std::tuple<std::shared_ptr<std::vector<HorizontalLine>>, std::shared_ptr<std::vector<uchar>>>> blobs_t;
constexpr int CV_MAX_THICKNESS = 32767;

template< class T >
struct remove_cvref {
    typedef std::remove_cv_t<std::remove_reference_t<T>> type;
};

template< class T >
using remove_cvref_t = typename remove_cvref<T>::type;
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
    typedef long_t idx_t;
}

#include <misc/vec2.h>

#endif
