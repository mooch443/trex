#include "Grid.h"
#include <misc/checked_casts.h>

namespace cmn {
namespace grid {

PixelGrid::PixelGrid(uint n, const Size2& resolution, const cv::Mat& bg)
: Grid2D<uint8_t>(resolution, n), background(bg)
{
}

uint8_t PixelGrid::query(float x, float y) const {
    auto v = Grid2D<uint8_t>::query(x, y);
    return v ? v : (x > 0 && y > 0 && x < background.cols && y < background.rows ? background.at<uchar>(narrow_cast<uint8_t>(y), narrow_cast<uint8_t>(x)) : uint8_t(0));
}

}
}
