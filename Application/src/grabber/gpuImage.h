#pragma once

#include <types.h>
#include <misc/Image.h>

namespace cmn {
/*struct gpuImage {
    typedef cv::Mat mat_t;
    GETTER(long_t, index)
    
    typedef Image* Imageptr;
    GETTER_NCONST(Imageptr, mat)
    GETTER(uint64_t, timestamp)
    
public:
    int rows, cols;
    
    ~gpuImage() {
        if(_mat)
            delete _mat;
        
    }
    gpuImage(int rows = 0, int cols = 0) : _index(-1), _mat(new Image(rows, cols, 1)), rows(rows), cols(cols) {
        _timestamp = std::chrono::time_point_cast<std::chrono::microseconds>(cmn::Image::clock_::now()).time_since_epoch().count();
    }
    
    void set(long_t index, const mat_t& other, uint64 timestamp = std::chrono::time_point_cast<std::chrono::microseconds>(cmn::Image::clock_::now()).time_since_epoch().count()) {
        _timestamp = timestamp;
        _index = index;
        
        assert(_mat);
        if(_mat->data() != other.data) {
            if(_mat->cols != other.cols || _mat->rows != other.rows || _mat->dims != other.channels()) {
                Debug("Creating %dx%d from %dx%d", other.rows, other.cols, _mat->rows, _mat->cols);
                _mat->create(other.rows, other.cols);
            }
            assert(_mat->cols == other.cols && _mat->rows == other.rows && _mat->dims == other.channels());
            other.copyTo(_mat->get());
        }
        rows = other.rows;
        cols = other.cols;
    }
    
    void set_timestamp(uint64_t t) {
        _timestamp = t;
    }
    
    void set_index(long_t index) {
        _index = index;
    }
    
    mat_t get() const {
        return _mat->get();
    }
};*/

class ImagePair : public Image {
    GETTER_PTR(Image*, mask)
    
public:
    void set_mask(Image* mask) {
        if(_mask)
            delete _mask;
        _mask = mask;
    }
    ImagePair(const ImagePair& other) : Image(other) {
        if(other._mask)
            _mask = new Image(*other._mask);
    }
    ImagePair() : Image(), _mask(nullptr) {
        
    }
    ImagePair(uint rows, uint cols, uint dims = 1) : Image(rows, cols, dims), _mask(nullptr) {
        
    }
    ImagePair& operator=(ImagePair&& other) {
        set(std::move(other));
        return *this;
    }
    
    ~ImagePair() {
        if(_mask)
            delete _mask;
    }
private:
    void set(ImagePair&& pair) {
        //Image::operator=((Image&&)pair);
        Image::set(std::move(pair));
        std::swap(pair._mask, _mask);
    }
};

typedef ImagePair gpuImage;
typedef gpuImage Image_t;
}
