#pragma once

#include "types.h"

namespace cmn {
    namespace CPULabeling {
        /**
         * Given a binary image, this function searches it for POI (pixels != 0),
         * finds all other spatially connected POIs and combines them into Blobs.
         *
         * @param image a binary image in CV_8UC1 format
         * @param enable_threads when set to true, this function will use threads to extract horizontal lines (default false)
         * @return an array of the blobs found in image
         */
        blobs_t run(const cv::Mat &image, bool enable_threads = false);
    
        /**
         * Given a set of horizontal lines, this function will extract all connected components and return them as a list of Blobs.
         * @param lines a list of HorizontalLines
         * @param pixels all pixels in the same order as the lines in lines (each from x0 to x1+1).
         * @return an array of the blobs found in image
         */
        blobs_t run(const std::vector<HorizontalLine>& lines, const std::vector<uchar>& pixels);
    }
}
