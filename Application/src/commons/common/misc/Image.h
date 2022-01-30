#pragma once

#include <misc/defines.h>
#include <misc/vec2.h>
#include <file/Path.h>
#include <misc/checked_casts.h>
#include <gui/colors.h>

namespace cmn {
    /**
     * A structure that maps from custom malloc() calls to cv::Mat and back.
     * Also saves an "ID" or index of the image.
     */
    class Image : public IndexedDataTransport {
    public:
        typedef std::chrono::steady_clock clock_;
        typedef std::chrono::duration<double, std::ratio<1> > second_;
        using Ptr  = std::shared_ptr<Image>;
        using UPtr = std::unique_ptr<Image>;
        
        template<typename... Args>
        static UPtr Make(Args&&...args) {
            return std::make_unique<Image>(std::forward<Args>(args)...);
        }
        
        class CustomData {
        public:
            virtual ~CustomData() {}
        };
        
    private:
        GETTER_PTR(uchar*, data)
        GETTER(size_t, size)
        GETTER(size_t, array_size)
        GETTER_SETTER(uint64_t, timestamp)
        GETTER_SETTER_PTR(CustomData*, custom_data)
        
    public:
        uint cols, rows, dims;
        
    public:
        Image(Image&&) = delete;
        Image(const Image& other);
        Image(uint rows, uint cols, uint dims, const uchar* datat);
        Image(uint rows, uint cols, uint dims = 1, int index = -1, uint64_t timestamp = 0);
        explicit Image(const cv::Mat& image, int index = -1, uint64_t timestamp = 0);
        Image();
        ~Image();
        
        void create(const cv::Mat& matrix);
        void create(uint rows, uint cols, uint dims = 1);
        void create(uint rows, uint cols, uint dims, const uchar* data);
        void clear();
        
        Image& operator=(const Image& other) = delete;
        Image& operator=(Image&& other) = delete;
        //Image& operator=(const Image& other);
        //Image& operator=(const cv::Mat& matrix);
        
        void set(long_t idx, const cv::Mat& matrix, uint64_t stamp = (uint64_t)std::chrono::time_point_cast<std::chrono::microseconds>(clock_::now()).time_since_epoch().count());
        /*template<typename T>
        void set(T&& image, std::enable_if_t<std::is_base_of<Image, T>::value> * = nullptr) {
            static_assert(false, "Use the move constructor!");
        }*/
        void set(long_t idx, const uchar* matrix, uint64_t stamp = (uint64_t)std::chrono::time_point_cast<std::chrono::microseconds>(clock_::now()).time_since_epoch().count());
        void set(Image&&);
        
        //! copy one channel from a 1-d matrix of equal size
        void set_channel(size_t idx, const uchar* matrix);
        
        //! set a whole channel of the image to a constant value
        void set_channel(size_t idx, uchar value);
        //! set whole image to constant value
        void set_to(uchar value);
        
        //! set a whole channel with a generative function
        void set_channel(size_t idx, const std::function<uchar(size_t)>& value);
        
        //! pass a 1-dim image and set all channels of `channels` to these values
        void set_channels(const uchar* source, const std::set<uint>&channels);
        
        //! access pixel at y,x and channel
        uchar at(uint y, uint x, uint channel = 0) const;
        uchar* ptr(uint y, uint x) const;
        
        void set_pixel(uint x, uint y, const gui::Color& color) const;
        
        void get(cv::Mat& matrix) const;
        cv::Mat get() const;
        bool empty() const { return _data == NULL; }
        Bounds bounds() const { return Bounds(0, 0, static_cast<Float2_t>(cols), static_cast<Float2_t>(rows)); }
        Size2 dimensions() const { return Size2(static_cast<Float2_t>(cols), static_cast<Float2_t>(rows)); }
        
        uint64_t stamp() const { return _timestamp; }//return std::chrono::time_point_cast<std::chrono::microseconds>(_timestamp).time_since_epoch().count(); }
        
        bool operator==(const Image& other) const {
            return other.cols == cols && other.rows == rows && other.dims == dims && (_data == other._data || (_size == other._size && memcmp(_data, other._data, _size) == 0));
        }
        
        std::string toStr() const;
        static std::string class_name() {
            return "Image";
        }
        
    private:
        void reset_stamp() {
            _timestamp = (uint64_t)std::chrono::time_point_cast<std::chrono::microseconds>(clock_::now()).time_since_epoch().count();
        }
    };

    void to_png(const Image& input, std::vector<uchar>& output);
    Image::UPtr from_png(const file::Path& path);
    
    cv::Mat restrict_image_keep_ratio(const Size2& max_size, const cv::Mat& input);
}
