#ifndef _DATAFORMAT_H
#define _DATAFORMAT_H

#include <types.h>
#include <sys/stat.h>
#include <file/Path.h>
#include <misc/CropOffsets.h>

namespace cmn {
#if defined(WIN32)
    struct mappedRegionS {
        void * addr;
        uint64_t length;
    };

    typedef struct mappedRegionS* mappedRegion;
#endif

    class Data {
    protected:
        bool _supports_fast;
        
    public:
        Data() : _supports_fast(false) {}
        virtual ~Data() {}
        
        template <typename T>
        void read(T& val) {
            constexpr auto size = sizeof(T);
            if(_supports_fast) {
                std::memcpy(&val, read_data_fast(size), size);
                return;
            }
            
            const uint64_t read_size = read_data(size, (char*)&val);
            if(read_size != size)
                U_EXCEPTION("Read unexpected number of bytes (%d/%d).", read_size, size);
        }
        
        template <typename T, typename K>
        void read_convert(K& val) {
            T v;
            read<T>(v);
            val = narrow_cast<K>(v);
        }
        
        template <typename T>
        void read(T& val, uint64_t pos) {
            uint64_t old = tell();
            
            seek(pos);
            read(val);
            seek(old);
        }
        
        template<typename T>
        uint64_t write(const T& val) {
            const uint64_t size = sizeof(T);
            const char *buffer = (char*)&val;
            
            return write_data(size, buffer);
        }
        
        template<typename T>
        uint64_t write(const T& val, uint64_t pos) {
            uint64_t old = tell();
            //Debug("Position %ld, new %ld (%ld)", old, pos, _file_offset);
            
            seek(pos);
            uint64_t p = write(val);
            seek(old);
            
            return p;
        }
        
        virtual void seek(uint64_t pos) = 0;
        virtual uint64_t tell() const = 0;
        virtual uint64_t read_data(uint64_t num_bytes, char *buffer) = 0;
        virtual uint64_t write_data(uint64_t num_bytes, const char *buffer) = 0;

        virtual const char* read_data_fast(uint64_t) { U_EXCEPTION("Not supported."); }
        
        uint64_t read_data(uint64_t pos, uint64_t num_bytes, char *buffer) {
            auto old = tell();
            
            seek(pos);
            uint64_t p = read_data(num_bytes, buffer);
            seek(old);
            
            return p;
        }
        uint64_t write_data(uint64_t pos, uint64_t num_bytes, const char *buffer) {
            auto old = tell();
            
            seek(pos);
            uint64_t p = write_data(num_bytes, buffer);
            seek(old);
            
            return p;
        }
    };

    class ReadonlyMemoryWrapper : public Data {
        uchar *_data;
#ifndef NDEBUG
        uint64_t _capacity;
#endif
        uint64_t pos;
        
    public:
        ReadonlyMemoryWrapper(uchar *memory, uint64_t
#ifndef NDEBUG
                              size
#endif
                              )
            : _data(memory),
#ifndef NDEBUG
              _capacity(size),
#endif
              pos(0)
        {
            _supports_fast = true;
        }
        
        virtual uint64_t write_data(uint64_t, const char *) override {
            U_EXCEPTION("This class is read-only.");
        }
        
        virtual uint64_t read_data(uint64_t num_bytes, char *buffer) override {
            assert(pos+num_bytes <= _capacity);
            memcpy(buffer, _data + pos, num_bytes);
            seek(pos + num_bytes);
            
            return num_bytes;
        }
        
        virtual const char* read_data_fast(uint64_t) override;
        
        virtual uint64_t tell() const override { return pos; }
        virtual void seek(uint64_t p) override {
            pos = p;
        }
    };

    class DataPackage : public Data {
        uint64_t pos;
        uchar *_data;
        uint64_t _capacity;
        
        const std::function<void(uint64_t)> *_change_position_callback;
        
    public:
        DataPackage(uint64_t size = 0, const decltype(_change_position_callback) callback = NULL) : pos(0), _data(NULL), _capacity(0), _change_position_callback(callback)
        {
            resize(size);
        }
        
        ~DataPackage() {
            if(_data)
                free(_data);
        }
        
        //! resize it to the given size (if necessary).
        void resize(uint64_t size, bool move = true) {
            if(size > _capacity) {
                if(move) {
                    _data = (uchar*)realloc(_data, size);
                } else {
                    if(_data)
                        free(_data);
                    _data = (uchar*)malloc(size);
                }
                _capacity = size;
            }
        }
        
        virtual uint64_t write_data(uint64_t num_bytes, const char *buffer) override;
        
        virtual uint64_t read_data(uint64_t num_bytes, char *buffer) override {
            assert(pos+num_bytes <= _capacity);
            memcpy(buffer, _data + pos, num_bytes);
            seek(pos + num_bytes);
            
            return num_bytes;
        }
        
        virtual uint64_t tell() const override { return pos; }
        virtual void seek(uint64_t p) override {
            pos = p;
            
            if(_change_position_callback)
                (*_change_position_callback)(pos);
        }
        
        char * data() const { return (char*)_data; }
        uint64_t size() const { return pos; }
        void reset_offset() { pos = 0; }
    };

    class DataFormat : public Data {
    private:
        std::string _project_name;
        const file::Path _filename;
        FILE *f;
        uint64_t _file_offset;
        bool _mmapped;
        int fd;
        char *_data;
        std::mutex _internal_modification;
        
        uint64_t _reading_file_size;
#if defined(WIN32)
        mappedRegion reg;
#endif
        
    protected:
        bool _open_for_writing, _open_for_modifying, _header_written;
        
    public:
        virtual ~DataFormat();
        
        //void read_file();
        //void write_file(bool overwrite = false);
        
        void start_reading();
        void start_modifying();
        void start_writing(bool overwrite = false);
        virtual void stop_writing();
        void close();
        
        bool open() const { return f != NULL || _mmapped; }
        
        const file::Path& filename() const { return _filename; }
        const std::string& project_name() const { return _project_name; }
        
        //long read_size() const { if(!_mmapped) U_EXCEPTION("Must be mmapped."); return _reading_file_size; }
		uint64_t current_offset() const;
		virtual uint64_t tell() const override;
        
    public:
        DataFormat(const file::Path& filename, const std::string& proj_name = "untitled")
        : _project_name(proj_name), _filename(filename), f(NULL), _file_offset(0), _mmapped(false), fd(0), _open_for_writing(false), _open_for_modifying(false), _header_written(false)
        {}
        
        // reading some data from the opened file
        virtual uint64_t read_data(uint64_t num_bytes, char *buffer) override;
        // writing data to the opened file, returning current position
        virtual uint64_t write_data(uint64_t num_bytes, const char *buffer) override;
        
        const char* read_data_fast(uint64_t num_bytes) override;
        void set_project_name(const std::string& name) { _project_name = name; }
        
        virtual void seek(uint64_t pos) override {
            if(!_mmapped) {
                if(!f)
                    U_EXCEPTION("File not open.");
#ifdef WIN32
				_fseeki64(f, (int64_t)pos, SEEK_SET);
#else
                fseeko(f, (off_t)pos, SEEK_SET);
#endif
            }
            _file_offset = pos;
        }

    protected:
        virtual void _read_header() {}
        virtual void _write_header() {}
    };

    template<> void Data::read(uint8_t&);

    template<> void Data::read(cv::Size&);
    template<> uint64_t Data::write(const cv::Size& val);

    template<> void Data::read(std::string&);
    template<> uint64_t Data::write(const std::string& val);

    template<> void Data::read(cv::Point&);
    template<> uint64_t Data::write(const cv::Point& val);

    template<> void Data::read(cv::Point2f&);
    template<> uint64_t Data::write(const cv::Point2f& val);
    
    template<> void Data::read(Vec2&);
    template<> uint64_t Data::write(const Vec2& val);

    template<> void Data::read(Size2&);
    template<> uint64_t Data::write(const Size2& val);

    template<> void Data::read(cv::Rect2i&);
    template<> uint64_t Data::write(const cv::Rect2i& val);

    template<> void Data::read(CropOffsets&);
    template<> uint64_t Data::write(const CropOffsets& val);

    template<> uint64_t Data::write(const DataPackage& val);
}

#endif
