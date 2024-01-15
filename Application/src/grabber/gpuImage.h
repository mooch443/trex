#pragma once

#include <commons.pc.h>
#include <misc/Image.h>

namespace cmn {
/*struct gpuImage {
    typedef cv::Mat mat_t;
    GETTER(long_t, index);
    
    typedef Image* Imageptr;
    GETTER_NCONST(Imageptr, mat);
    GETTER(uint64_t, timestamp);
    
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
                print("Creating ",other.rows,"x",other.cols," from ",_mat->rows,"x",_mat->cols);
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

class ImagePair {
    GETTER_NCONST(Image, image);
    GETTER_PTR_I(Image*, mask, nullptr);
    GETTER_I(long_t, index, -1);
    GETTER(timestamp_t, timestamp);
    
public:
    void set_mask(Image* mask) {
        if(_mask)
            delete _mask;
        _mask = mask;
    }
    /*ImagePair(const ImagePair& other) : Image(other) {
        if(other._mask)
            _mask = new Image(*other._mask);
    }*/
    ImagePair(ImagePair&&) = delete;
    ImagePair(const ImagePair&) = delete;
    ImagePair() = default;
    ImagePair(uint rows, uint cols, uint dims = 1)
        : _image(rows, cols, dims)
    {
        set_timestamp(_image.timestamp());
    }
    ImagePair& operator=(const ImagePair&) = delete;
    ImagePair& operator=(ImagePair&& other) {
        set(std::move(other));
        return *this;
    }
    
    void set_index(long_t index) {
        _index = index;
        _image.set_index(index);
    }
    
    cv::Mat get() const {
        return _image.get();
    }
    void set_timestamp(timestamp_t t) {
        _timestamp = t;
        _image.set_timestamp(t);
    }
    
    ~ImagePair() {
        if(_mask)
            delete _mask;
    }
private:
    void set(ImagePair&& pair) {
        _image.set(std::move(pair.image()));
        _mask = pair._mask;
        _timestamp = pair.timestamp();
        _index = pair.index();
        pair._mask = nullptr;
    }
};

typedef ImagePair gpuImage;
typedef gpuImage Image_t;
using ImagePtr = std::unique_ptr<Image_t>;

template<typename... Args>
ImagePtr ImageMake(Args&&...args) {
    return std::make_unique<Image_t>(std::forward<Args>(args)...);
}
}
