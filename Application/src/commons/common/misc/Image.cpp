#include "Image.h"
#include <png.h>
#include <misc/metastring.h>
#include <gui/colors.h>

namespace cmn {
    std::string Image::toStr() const {
        return "("+Meta::toStr(cols)+"x"+Meta::toStr(rows)+"x"+Meta::toStr(dims)+" "+Meta::toStr(DurationUS{std::chrono::time_point_cast<std::chrono::microseconds>(clock_::now()).time_since_epoch().count() - _timestamp})+" ago)";
    }
    
    Image::Image()
        : _data(NULL), _size(0), _custom_data(nullptr), cols(0), rows(0), dims(0)
    { set_index(-1); }
    
    Image::Image(uint rows, uint cols, uint dims, int index, uint64_t timestamp)
        : _size(rows*cols*dims*sizeof(uchar)), _custom_data(nullptr),
        cols(cols), rows(rows), dims(dims), _timestamp(timestamp)
    {
        set_index(index);
        
        if(_size)
            _data = (uchar*)malloc(_size);
        else
            _data = NULL;
        
        _array_size = _size;
        
        reset_stamp();
    }
    
    Image::Image(const Image& other)
        : Image(other.rows, other.cols, other.dims)
    {
        assert(_size == other.size());
        
        _index = other.index();
        _timestamp = other.timestamp();
        if(other._custom_data)
            U_EXCEPTION("Cannot copy custom data from one image to another.");
        
        assert(_data);
        std::copy(other.data(), other.data() + other.size(), _data);
    }
    
    Image::Image(const cv::Mat& image, int index, uint64_t timestamp)
        : Image(image.rows, image.cols, image.channels(), index, timestamp)
    {
        set(index, image, timestamp);
    }
    
    Image::Image(uint rows, uint cols, uint dims, const uchar* data)
        : Image(rows, cols, dims)
    {
        set(0, data);
    }

    void Image::clear() {
        if(_data) {
            free(_data);
            _data = NULL;
        }
        
        set_index(-1);
        _size = cols = rows = dims = 0;
        _timestamp = 0;
        
        if(_custom_data) {
            delete _custom_data;
            _custom_data = nullptr;
        }
    }

    uchar Image::at(uint y, uint x, uint channel) const {
        assert(y < rows);
        assert(x < cols);
        assert(channel < dims);
        return data()[x * dims + channel + y * cols * dims];
    }

    uchar* Image::ptr(uint y, uint x) const {
        assert(y < rows);
        assert(x < cols);
        return data() + (x * dims + y * cols * dims);
    }

    void Image::set_pixel(uint x, uint y, const gui::Color& color) const {
        assert(y < rows);
        assert(x < cols);
        
        auto ptr = data() + x * dims + y * cols * dims;
        switch (dims) {
            case 4:
                *(ptr + 3) = color.a;
            case 3:
                *(ptr + 2) = color.b;
            case 2:
                *(ptr + 2) = color.g;
            case 1:
                *(ptr + 2) = color.r;
                
            default:
                break;
        }
    }

    Image::~Image() {
        if(_data)
            free(_data);
        if(_custom_data)
            delete _custom_data;
    }
    
    void Image::create(uint rows, uint cols, uint dims) {
        size_t N = cols * rows * dims;
        if(_data
           && _array_size >= N
           && _array_size < N * 2)
        {
            // keep the array, since its not too big, also big enough
            
        } else {
            // have to reallocate
            if(_data)
                free(_data);
            _data = NULL;
            _array_size = 0;
        }
        
        this->cols = cols;
        this->rows = rows;
        this->dims = dims;
        this->_size = N;
        reset_stamp();
        
        if(_size && !_data) {
            _data = (uchar*)malloc(_size);
            _array_size = _size;
        }
    }

    void Image::create(uint rows, uint cols, uint dims, const uchar* data) {
        create(rows, cols, dims);
        set(0, data);
    }
    
    void Image::create(const cv::Mat &matrix) {
        create(matrix.rows, matrix.cols, matrix.channels());
        set(_index, matrix);
    }

void Image::set(Image&& other) {
    std::swap(other._index, _index);
    std::swap(other._timestamp, _timestamp);
    std::swap(other._custom_data, _custom_data);
    std::swap(other._data, _data);
    std::swap(other.cols, cols);
    std::swap(other.rows, rows);
    std::swap(other.dims, dims);
    std::swap(other._size, _size);
    std::swap(other._array_size, _array_size);
}
    
    /*Image& Image::operator=(const Image& other) {
        if (&other == this)
            return *this;
        
        assert(_size == other.size());
        
        _index = other.index();
        _timestamp = other.timestamp();
        if(other._custom_data)
            U_EXCEPTION("Cannot copy custom data from one image to another.");
        
        if(_data)
            std::memcpy(_data, other.data(), _size);
        return *this;
    }*/
    
    /*Image& Image::operator=(const cv::Mat& matrix) {
        set(_index, matrix);
        reset_stamp();
        return *this;
    }*/

    /*void Image::operator=(Image &&image) {
        if(image.cols == cols && image.rows == rows && image.dims == dims && _data)
        {
            std::swap(image._data, _data);
            _timestamp = image._timestamp;
            _index = image._index;
        } else {
            set(image.index(), image.get(), image.timestamp());
        }
        
#ifndef NDEBUG
        if(image._custom_data)
            Warning("Cannot copy custom data from one image to another.");
#endif
    }*/
    
    void Image::set(long_t idx, const cv::Mat& matrix, uint64_t stamp) {
        assert(int(rows) == matrix.rows);
        assert(int(cols) == matrix.cols);
        assert(int(dims) == matrix.channels());
        assert(matrix.isContinuous());
        
        if(!_data && _size) {
            create(matrix);
            return;
        }
        
        _index = idx;
        _timestamp = stamp;
        //reset_stamp();
        if(_size)
            std::memcpy(_data, matrix.data, _size);
    }
    
    void Image::set(long_t idx, const uchar* matrix, uint64_t stamp) {
        assert(_data);
        assert(matrix);
        
        _index = idx;
        _timestamp = stamp;
        std::memcpy(_data, matrix, _size);
    }

    void Image::set_to(uchar value) {
        std::fill(data(), data()+size(), value);
    }

    void Image::set_channels(const uchar *source, const std::set<uint> &channels) {
        assert(!channels.empty());
#ifndef NDEBUG
        for(auto c : channels)
            assert(c < dims);
#endif
        
        auto ptr = _data;
        auto end = _data + size();
        auto m = source;
        for(; ptr<end; ptr+=dims, ++m)
            for(auto c : channels)
                *(ptr + c) = *m;
    }
    
    void Image::set_channel(size_t idx, const uchar* matrix) {
        assert(_data && idx < dims);
        reset_stamp();
        
        auto ptr = _data + idx;
        auto m = matrix;
        for(; ptr<_data + _size; ptr+=dims, ++m)
            *ptr = *m;
    }
    
    void Image::set_channel(size_t idx, uchar value) {
        assert(_data && idx < dims);
        reset_stamp();
        
        auto ptr = _data + idx;
        for(; ptr<_data + _size; ptr+=dims)
            *ptr = value;
    }
    
    void Image::set_channel(size_t idx, const std::function<uchar(size_t)>& value) {
        assert(_data && idx < dims);
        reset_stamp();
        
        auto ptr = _data + idx;
        size_t i=0;
        for(; ptr<_data + _size; ptr+=dims, ++i)
            *ptr = value(i);
    }
    
    void Image::get(cv::Mat& matrix) const {
        assert(int(rows) == matrix.rows && int(cols) == matrix.cols && int(dims) == matrix.channels());
        assert(matrix.isContinuous());
        std::memcpy(matrix.data, _data, _size);
    }
    
    cv::Mat Image::get() const {
        assert(_size == rows * cols * dims * sizeof(uchar));
        return cv::Mat(rows, cols, CV_8UC(dims), _data);//, cv::Mat::AUTO_STEP);
    }
    
    struct PNGGuard {
        png_struct *p;
        png_info *info;
        PNGGuard(png_struct *p) : p(p)  {}
        ~PNGGuard() { if(p) {  png_destroy_write_struct(&p, info ? &info : NULL); } }
    };
    
    static void PNGCallback(png_structp  png_ptr, png_bytep data, png_size_t length) {
        std::vector<uchar> *p = (std::vector<uchar>*)png_get_io_ptr(png_ptr);
        p->insert(p->end(), data, data + length);
    }
    
    void to_png(const Image& _input, std::vector<uchar>& output) {
        if(_input.dims < 4 && _input.dims != 1 && _input.dims != 2)
            U_EXCEPTION("Currently, only RGBA and GRAY is supported.");
        
        Image::UPtr tmp;
        const Image *input = &_input;
        if (_input.dims == 2) {
            std::vector<cv::Mat> vector;
            cv::split(_input.get(), vector);
            cv::Mat image;
            cv::merge(std::vector<cv::Mat>{vector[0], vector[0], vector[0], vector[1]}, image);
            tmp = Image::Make(image);
            input = tmp.get();
        }
        
        output.clear();
        output.reserve(sizeof(png_byte) * input->cols * input->rows * input->dims);
        
        png_structp p = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
        if(!p)
            U_EXCEPTION("png_create_write_struct() failed");
        PNGGuard PNG(p);
        
        png_infop info_ptr = png_create_info_struct(p);
        if(!info_ptr) {
            U_EXCEPTION("png_create_info_struct() failed");
        }
        PNG.info = info_ptr;
        if(0 != setjmp(png_jmpbuf(p)))
            U_EXCEPTION("setjmp(png_jmpbuf(p) failed");
        png_set_IHDR(p, info_ptr, input->cols, input->rows, 8,
                     input->dims == 4 ? PNG_COLOR_TYPE_RGBA : PNG_COLOR_TYPE_GRAY,
                     PNG_INTERLACE_NONE,
                     PNG_COMPRESSION_TYPE_DEFAULT,
                     PNG_FILTER_TYPE_DEFAULT);
        png_set_compression_level(p, 1);
        std::vector<uchar*> rows(input->rows);
        for (size_t y = 0; y < input->rows; ++y)
            rows[y] = input->data() + y * input->cols * input->dims;
        png_set_rows(p, info_ptr, rows.data());
        png_set_write_fn(p, &output, PNGCallback, NULL);
        png_write_png(p, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);
    }
    
    Image::UPtr from_png(const file::Path& path) {
        int width, height;
        png_byte color_type;
        png_byte bit_depth;
        png_bytep *row_pointers;
        
        FILE *fp = fopen(path.str().c_str(), "rb");
        
        png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
        if(!png) abort();
        
        png_infop info = png_create_info_struct(png);
        if(!info) abort();
        
        if(setjmp(png_jmpbuf(png))) abort();
        
        png_init_io(png, fp);
        
        png_read_info(png, info);
        
        width      = png_get_image_width(png, info);
        height     = png_get_image_height(png, info);
        color_type = png_get_color_type(png, info);
        bit_depth  = png_get_bit_depth(png, info);
        
        // Read any color_type into 8bit depth, RGBA format.
        // See http://www.libpng.org/pub/png/libpng-manual.txt
        
        if(bit_depth == 16)
            png_set_strip_16(png);
        
        if(color_type == PNG_COLOR_TYPE_PALETTE)
            png_set_palette_to_rgb(png);
        
        // PNG_COLOR_TYPE_GRAY_ALPHA is always 8 or 16bit depth.
        if(color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
            png_set_expand_gray_1_2_4_to_8(png);
        
        if(png_get_valid(png, info, PNG_INFO_tRNS))
            png_set_tRNS_to_alpha(png);
        
        // These color_type don't have an alpha channel then fill it with 0xff.
        if(color_type == PNG_COLOR_TYPE_RGB ||
           color_type == PNG_COLOR_TYPE_GRAY ||
           color_type == PNG_COLOR_TYPE_PALETTE)
            png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
        
        if(color_type == PNG_COLOR_TYPE_GRAY ||
           color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
            png_set_gray_to_rgb(png);
        
        png_read_update_info(png, info);
        
        row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
        for(int y = 0; y < height; y++) {
            row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png,info));
        }
        
        png_read_image(png, row_pointers);
        
        fclose(fp);
        
        static_assert(sizeof(png_byte) == sizeof(uchar), "Must be the same.");
        
        auto ptr = Image::Make(height, width, 4);
        for(int y = 0; y < height; y++) {
            png_bytep row = row_pointers[y];
            memcpy(ptr->data() + y * width * 4, row, width * 4);
            /*for(int x = 0; x < width; x++) {
                png_bytep px = &(row[x * 4]);
                // Do something awesome for each pixel here...
                printf("%4d, %4d = RGBA(%3d, %3d, %3d, %3d)\n", x, y, px[0], px[1], px[2], px[3]);
            }*/
        }
        //memcpy(ptr->data(), row_pointers, height * width * 4);
        
        png_destroy_read_struct(&png, &info, NULL);
        return ptr;
    }
    
    cv::Mat restrict_image_keep_ratio(const Size2& max_size, const cv::Mat& input) {
        using namespace gui;
        Size2 image_size(input);
        if(image_size.width <= max_size.width && image_size.height <= max_size.height)
            return input; // everything is fine
        
        float ratio = image_size.width / image_size.height;
        
        if(image_size.width > max_size.width)
            image_size = Size2(max_size.width, max_size.width / ratio);
            
        if(image_size.height > max_size.height)
            image_size = Size2(max_size.height * ratio, max_size.height);
        
        if(image_size.width > input.cols) {
            image_size.width = input.cols;
            image_size.height = input.cols / ratio;
        }
        
        if(image_size.height > input.rows) {
            image_size.height = input.rows;
            image_size.width = input.rows * ratio;
        }
        
        cv::Mat image;
        cv::resize(input, image, cv::Size(image_size), 0, 0, cv::INTER_AREA);
        return image;
    }
}
