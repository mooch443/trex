#ifndef _PV_H
#define _PV_H

#include <types.h>
#include <file/DataFormat.h>
#include <video/GenericVideo.h>
#include <file/Path.h>
#include <misc/Image.h>
#include <misc/PVBlob.h>

namespace pv {
    using namespace cmn;
    class File;
    class Blob;

    struct LegacyShortHorizontalLine {
    private:
        //! starting and end position on x
        //  the last bit of _x1 is a flag telling the program
        //  whether this line is the last line on the current y coordinate.
        //  the following lines are on current_y + 1.
        uint16_t _x0 = 0, _x1 = 0;
        
    public:
        //! compresses an array of HorizontalLines to an array of ShortHorizontalLines
        static std::vector<LegacyShortHorizontalLine> compress(const std::vector<cmn::HorizontalLine>& lines);
        //! uncompresses an array of ShortHorizontalLines back to HorizontalLines
        static std::shared_ptr<std::vector<cmn::HorizontalLine>> uncompress(uint16_t start_y, const std::vector<LegacyShortHorizontalLine>& compressed);
        
    public:
        constexpr LegacyShortHorizontalLine() {}
        
        constexpr LegacyShortHorizontalLine(uint16_t x0, uint16_t x1, bool eol = false)
            : _x0(x0), _x1(uint16_t(x1 << 1) + uint16_t(eol))
        {
            assert(x1 < 32768); // MAGIC NUMBERZ (uint16_t - 1 bit)
        }
        
        constexpr uint16_t x0() const { return _x0; }
        constexpr uint16_t x1() const { return (_x1 & 0xFFFE) >> 1; }
        
        //! returns true if this is the last element on the current y coordinate
        //  if true, the following lines are on current_y + 1.
        //  @note stored in the last bit of _x1
        constexpr bool eol() const { return _x1 & 0x1; }
        void eol(bool v) { _x1 = v ? (_x1 | 0x1) : (_x1 & 0xFFFE); }
        
        constexpr operator pv::ShortHorizontalLine() const {
            return pv::ShortHorizontalLine(x0(), x1(), eol());
        }
    };
    
    enum Version {
        V_1 = 0,
        V_2,
        V_3,
        
        /**
         * Version 4 adds proper handling of really long files.
         * Timestamps might have gotten bigger than UINT32_MAX, so now
         * PV uses UINT64 instead in all cases.
         */
        V_4,
        
        /**
         * Version 5 adds metadata to PV files. They are basically a JSON
         * array located in the header.
         */
        V_5,
        
        /**
         * Version 6 adds on-the-fly compression per frame.
         * Big files can be almost 30% smaller with this.
         */
        V_6,
        
        /** Changed format of ShortHorizontalLine */
        V_7,
        
        /** Added flags per object */
        V_8,
        
        //! current
        current = V_8
    };
    
    class Frame : public IndexedDataTransport {
    private:
        //! time since movie start in microseconds
        GETTER_SETTER(uint64_t, timestamp)
        //! number of mask/pixel arrays
        GETTER(uint16_t, n)
        GETTER_SETTER(float, loading_time)
        
        GETTER_NCONST(std::vector<blob::line_ptr_t>, mask)
        GETTER_NCONST(std::vector<blob::pixel_ptr_t>, pixels)
        GETTER_NCONST(std::vector<uint8_t>, flags)
        
    public:
        //! Initialize copy
        //Frame(const Frame& other);
        Frame(Frame&& other);
        void operator=(const Frame& other);
        void operator=(Frame&& other);
        
        //! initialize empty object
        Frame() : Frame(0, 0) {}
        
        //! create a new one from scratch
        Frame(const uint64_t& timestamp, decltype(_n) n);
        
        //! read from a file
        Frame(File& ref, long_t idx);
        
        ~Frame() {
            //for(auto m: _mask)
            //    delete m;
            //for(auto p: _pixels)
            //    delete p;
        }
        
        void read_from(File& ref, long_t idx);
        
        void add_object(const std::vector<HorizontalLine>& mask, const cv::Mat& full_image, uint8_t flags);
        std::unique_ptr<pv::Blob> blob_at(size_t i) const;
        std::vector<pv::BlobPtr> get_blobs() const;
        
        /**
         * Adds a new object to this frame.
         * ! takes ownership of both arrays
         **/
        void add_object(blob::Pair&& pair);
        void add_object(const std::vector<HorizontalLine>& mask, const std::vector<uchar>& pixels, uint8_t flags);

        uint64_t size() const;
        void clear();
        void serialize(DataPackage&, bool& compressed) const;
        
        std::string toStr() const {
            return "pv::Frame<"+std::to_string(index())+">";
        }
        
    protected:
        friend class File;
        
        //void update_timestamp(DataPackage&) const;
    };
        
    struct Header {
        typedef ShortHorizontalLine line_type;
        friend class File;
        
    public:
        //! Fileformat version
        Version version;
        
        //! Name of the project
        std::string name;
        
        //! Metadata string associated with this file
        std::string metadata;
        
        //! Number of channels per pixel
        uchar channels;
        
        //! Size of a horizontal line struct
        //  in the mask images in bytes
        uchar line_size;
        
        //! Resolution of the video frames (constant)
        cv::Size resolution;
        
        //! Number of frames in the video
        uint32_t num_frames;
        
        //! Offset of the index table at the end of the file
        uint64_t index_offset;
        
        //! Timestamp in microseconds since 1970 of when the recording started
        //  (all following frames have delta-timestamps)
        uint64_t timestamp;
        
        //! Contains an index for each frame, pointing
        //  to its location in the file
        std::vector<uint64_t> index_table;
        
        //! Full-size average image
        Image *average;
        
        //! Binary mask applied to image (or NULL)
        Image *mask;
        
        //! Offsets for cutting on all sides (left, top, right, bottom)
        CropOffsets offsets;
        
        //! Contains average time delta between frames
        double average_tdelta;
        
        void write(DataFormat& ref);
        void read(DataFormat& ref);
        
        void update(DataFormat& ref);
        
        Header(const std::string& n)
        : version(current), name(n), channels(1), line_size(sizeof(line_type)), resolution(0, 0), num_frames(0), index_offset(0), timestamp(0), average(NULL), mask(NULL), average_tdelta(0), _num_frames_offset(0), _average_offset(0), _running_average_tdelta(0)
        { }
        
        ~Header() {
            if(average)
                delete average;
        }
        
        std::string generate_metadata() const;
        
    private:
        uint64_t _num_frames_offset;
        uint64_t _average_offset;
        uint64_t _index_offset;
        uint64_t _timestamp_offset;
        double _running_average_tdelta;
        GETTER(uint64_t, meta_offset)
    };

    struct TaskSentinel;
    
    class File : public cmn::DataFormat, public cmn::GenericVideo {
    protected:
        std::mutex _lock;
        Header _header;
        cv::Mat _average, _mask;
        GETTER(file::Path, filename)
        uint64_t _prev_frame_time;
        
        // debug compression
        GETTER_I(std::atomic<double>, compression_ratio, 0.0)
        double _compression_value = 0;
        uint32_t _compression_samples = 0;
        
        pv::Frame _last_frame;
        std::mutex _task_list_mutex;
        std::unordered_map<std::thread::id, TaskSentinel*> _task_list;
        std::condition_variable _task_variable;
        
        friend struct pv::TaskSentinel;
        
    public:
        File(const file::Path& filename = "")
            : DataFormat(filename.add_extension("pv"), filename.str()),
                _header(filename.str()),
                _filename(filename),
                _prev_frame_time(0)
        { }
        
        ~File();
        
        const pv::Frame& last_frame();
        
        std::vector<float> calculate_percentiles(const std::initializer_list<float>& percent);
        std::string get_info(bool full = true);
        std::string get_info_rich_text(bool full = true);
        void print_info() { print(get_info().c_str()); }
        
        virtual CropOffsets crop_offsets() const override {
            return _header.offsets;
        }
        
        virtual void set_offsets(const CropOffsets& offsets) override {
            _header.offsets = offsets;
        }
        
        void add_individual(Frame&& frame);
        void add_individual(const Frame& frame, DataPackage& pack, bool compressed);
        
        void read_frame(Frame& frame, uint64_t frameIndex);
        void read_next_frame(Frame& frame, uint64_t frame_to_read);
        
        virtual void stop_writing() override;
        void set_resolution(const Size2& size) { _header.resolution = (cv::Size)size; }
        void set_average(const cv::Mat& average) {
            if(average.type() != CV_8UC1) {
                auto str = getImgType(average.type());
                throw U_EXCEPTION("Average image is of type ",str," != 'CV_8UC1'.");
            }
            
            if(!_header.resolution.width && !_header.resolution.height) {
                _header.resolution.width = average.cols;
                _header.resolution.height = average.rows;
            }
            else if(average.cols != _header.resolution.width || average.rows != _header.resolution.height) {
                throw U_EXCEPTION("Average image is of size ",average.cols,"x",average.rows," but has to be ",_header.resolution.width,"x",_header.resolution.height,"");
            }
            
            if(_header.average)
                delete _header.average;
            
            _header.average = new Image(average);
            this->_average = _header.average->get();
            
            if(_open_for_modifying) {
                cmn::Data::write_data(header()._average_offset, header().average->size(), (char*)header().average->data());
            }
        }
        const Header& header() const { return _header; }
        Header& header() { return _header; }
        const cv::Mat& average() const override { assert(_header.average); return _average; }
        
        void set_mask(const cv::Mat& mask) {
            if(_header.mask)
                delete _header.mask;
            _header.mask = new Image(mask);
            this->_mask = _header.mask->get();
        }
        bool has_mask() const override { return _header.mask != NULL; }
        const cv::Mat& mask() const override { assert(_header.mask != NULL); return _mask; }
        
        /**
         * ### GENERICVIDEO INTERFACE ###
         **/
        const cv::Size& size() const override { return _header.resolution; }
        uint32_t length() const override { return _header.num_frames; }
        void frame(uint64_t frameIndex, cv::Mat& output, cmn::source_location loc = cmn::source_location::current()) override;
#ifdef USE_GPU_MAT
        void frame(uint64_t frameIndex, gpuMat& output, cmn::source_location loc = cmn::source_location::current()) override;
#endif
        void frame_optional_background(uint64_t frameIndex, cv::Mat& output, bool with_background);
        bool supports_multithreads() const override { return false; }
        
        void try_compress();
        void update_metadata();
        void set_start_time(std::chrono::time_point<std::chrono::system_clock>);
        
        virtual bool has_timestamps() const override {
            return true;
        }
        virtual timestamp_t timestamp(uint64_t, cmn::source_location loc = cmn::source_location::current()) const override;
        virtual timestamp_t start_timestamp() const override;
        virtual short framerate() const override;
        double generate_average_tdelta();
        
        std::string summary() const {
            return "pv::File<V" + Meta::toStr(_header.version+1) + ", " + filesize() + ", " + Meta::toStr(filename()) + ", " + Meta::toStr(_header.resolution) + ", " + Meta::toStr(_header.num_frames) + " frames, " + (_header.mask ? "with mask" : "no mask") + ">";
        }
        
        std::string toStr() const;
        static std::string class_name() {
            return "pv::File";
        }
        std::string filesize() const;
        
    protected:
        virtual void _write_header() override;
        virtual void _read_header() override;
    };
    
    //! Tries to find irregular frames (timestamp smaller than timestamp from previous frame)
    //  and removes them from the file.
    void fix_file(File& file);
    
    //! prefixes the given path so that it points to either
    //  the output folder, or the input folder.
    //file::Path prefixed_path(DataType type, const file::Path& filename);
    //file::Path prefixed_path(DataType type, const std::string& filename);
}

#endif
