#include "Grid.h"

namespace cmn {
namespace grid {

PixelGrid::PixelGrid(uint n, const Size2& resolution, const cv::Mat& bg)
: Grid2D<uint8_t>(resolution, n), background(bg)
{
}

uint8_t PixelGrid::query(float x, float y) const {
    auto v = Grid2D<uint8_t>::query(x, y);
    return v ? v : (x > 0 && y > 0 && x < background.cols && y < background.rows ? background.at<uchar>(y, x) : 0);
}

}
}
