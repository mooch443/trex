#pragma once

#include "types.h"

namespace cmn {
    namespace CPULabeling {
        enum Method {
            WITH_PIXELS,
            NO_PIXELS
        };
        /**
         * Given a binary image, this function searches it for POI (pixels != 0),
         * finds all other spatially connected POIs and combines them into Blobs.
         *
         * @param image a binary image in CV_8UC1 format
         * @param min_size minimum number of pixels per Blob, 
         *        smaller Blobs will be deleted
         * @return an array of the blobs found in image
         */
        blobs_t run_fast(const cv::Mat &image, bool enable_threads = false, Method = WITH_PIXELS);
        blobs_t run_fast(const std::vector<HorizontalLine>& lines, std::shared_ptr<std::vector<uchar>> pixels = nullptr, Method = WITH_PIXELS/*, size_t min_size = 0*/);
    }
}
