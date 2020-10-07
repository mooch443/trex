#ifndef _RAWPROCESSING_H
#define _RAWPROCESSING_H

#include <types.h>
#include <misc/Blob.h>
#include <misc/GlobalSettings.h>

namespace cmn {
	class RawProcessing;
}

/**
 * The task of this class is to provide functionality that prepares 
 * raw images for further processing. For example, it converts raw 
 * images into binary images containing only the fish (and some noise).
 * It basically does (or calls) most of the image processing that's 
 * relevant for this program.
 */
class cmn::RawProcessing {
    gpuMat _buffer0, _buffer1;
    const gpuMat* _average;
    //const LuminanceGrid *_grid;
    //gpuMat _binary;
    //gpuMat _difference;
    //cv::Mat _hsv;

public:
    RawProcessing(const gpuMat &average, const LuminanceGrid* grid = NULL);
    ~RawProcessing() {
        _buffer0.release();
        _buffer1.release();
    }

    void generate_binary(const gpuMat& input, cv::Mat& output);
    cv::Mat get_binary() const;
};


#endif
