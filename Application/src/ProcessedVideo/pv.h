#ifndef _PV_H
#define _PV_H

#include <commons.pc.h>
#include <file/DataFormat.h>
#include <video/GenericVideo.h>
#include <file/Path.h>
#include <misc/Image.h>
#include <misc/PVBlob.h>
#include <misc/frame_t.h>

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
        
        /** Added source frame index */
        V_9,
        
        /** Adding Prediction + Pose */
        V_10,
        
        /** Adding Outlines to Prediction */
        V_11,
        
        /** Adding encoding **/
        V_12,
        
        //! current
        current = V_12
    };
    
    class Frame {
    private:
        GETTER_SETTER(Frame_t, index);
        
        //! time since movie start in microseconds
        GETTER_SETTER(timestamp_t, timestamp);
        //! number of mask/pixel arrays
        GETTER_I(uint16_t, n, 0u);
        GETTER_SETTER_I(float, loading_time, 0.f);
        GETTER_SETTER(Frame_t, source_index);
        GETTER_SETTER(uint8_t, channels) = 1u;
        GETTER_SETTER(meta_encoding_t::Class, encoding) = meta_encoding_t::gray;
        
        GETTER_NCONST(std::vector<blob::line_ptr_t>, mask);
        GETTER_NCONST(std::vector<blob::pixel_ptr_t>, pixels);
        GETTER_NCONST(std::vector<uint8_t>, flags);
        
        //! predictions either empty or same size as _mask
        GETTER_NCONST(std::vector<blob::Prediction>, predictions);
        
    public:
        Frame& operator=(const Frame& other) = delete;
        Frame& operator=(Frame&& other) = default;
        
        //! initialize empty object
        Frame() = default;
        Frame(Frame&&) noexcept = default;
        explicit Frame(const Frame&);
        
        //! create a new one from scratch
        Frame(const timestamp_t& timestamp, decltype(_n) n, uint8_t channels);
        
        //! read from a file
        Frame(File& ref, Frame_t idx);
        
        void read_from(File& ref, Frame_t idx, meta_encoding_t::Class mode);
        
        void add_object(const std::vector<HorizontalLine>& mask, const cv::Mat& full_image, uint8_t flags);
        std::unique_ptr<pv::Blob> blob_at(size_t i) const;
        std::unique_ptr<pv::Blob> steal_blob(size_t i);
        std::vector<pv::BlobPtr> get_blobs() const;
        std::vector<pv::BlobPtr> steal_blobs() &&;
        
        /**
         * Adds a new object to this frame.
         * ! takes ownership of both arrays
         **/
        void add_object(blob::Pair&& pair);
        void add_object(const std::vector<HorizontalLine>& mask, const std::vector<uchar>& pixels, uint8_t flags, const cmn::blob::Prediction&);

        uint64_t size() const noexcept;
        void clear();
        void serialize(DataPackage&, bool& compressed) const;
        
        std::string toStr() const {
            return "pv::Frame<"+index().toStr()+">";
        }
        
    protected:
        friend class File;
        
        //void update_timestamp(DataPackage&) const;
    };
        
    struct Header {
        typedef ShortHorizontalLine line_type;
        friend class File;
        
    public:
        /**
         ==============================
                Can be read
                from file directly
         ==============================
         */
        
        //! Fileformat version
        Version version{current};
        
        //! Name of the project
        std::string name;
        
        //! Metadata string associated with this file
        std::string metadata;
        
        //! Number of channels per pixel
        uchar channels{1u};
        
        meta_encoding_t::Class encoding{meta_encoding_t::gray};
        
        //! Size of a horizontal line struct
        //  in the mask images in bytes
        uchar line_size{narrow_cast<uchar>(sizeof(line_type))};
        
        //! Resolution of the video frames (constant)
        cv::Size resolution{0, 0};
        
        //! Number of frames in the video
        uint32_t num_frames{0u};
        
        //! Offset of the index table at the end of the file
        uint64_t index_offset{0u};
        
        //! Timestamp in microseconds since 1970 of when the recording started
        //  (all following frames have delta-timestamps)
        timestamp_t timestamp;
        
        //! Contains an index for each frame, pointing
        //  to its location in the file
        std::vector<uint64_t> index_table;
        
        //! Full-size average image
        Image *average{nullptr};
        
        //! Binary mask applied to image (or NULL)
        Image *mask{nullptr};
        
        //! Offsets for cutting on all sides (left, top, right, bottom)
        CropOffsets offsets;
        
    public:
        /**
         ==============================
                Calculated at
                load time
         ==============================
         */
        
        //! The width of the arena from left to right edge
        //! of the video frame (in cm).
        float meta_real_width;
        
        //! Contains average time delta between frames
        double average_tdelta;
        
    private:
        /**
         ==============================
            Calculated at
            runtime while writing
         ==============================
         */
        uint64_t _num_frames_offset{0u};
        uint64_t _average_offset{0u};
        uint64_t _index_offset{0u};
        uint64_t _timestamp_offset{0u};
        double _running_average_tdelta{0.0};
        GETTER_I(uint64_t, meta_offset, 0u);
        
    public:
        void write(DataFormat& ref);
        void read(DataFormat& ref);
        
        void update(DataFormat& ref);
        
    public:
        Header() = default;
        Header(Header&&) = default;
        Header(const std::string& n, uint8_t channels, meta_encoding_t::Class encoding)
            : name(n), channels(channels), encoding(encoding)
        { }
        
        ~Header() {
            if(average)
                delete average;
        }
        
        std::string generate_metadata() const;
        
        static Header move(Header&& src) {
            Header dest = std::move(src);
            src.average = nullptr;
            src.mask = nullptr;
            return dest;
        }
    };

    struct TaskSentinel;

    enum class FileMode : std::uint8_t {
        READ      = 0b00000001,
        WRITE     = 0b00000010,
        OVERWRITE = 0b00000100,
        MODIFY    = 0b00001000
    };

    inline constexpr FileMode operator|(FileMode lhs, FileMode rhs) {
        return static_cast<FileMode>(
            static_cast<std::underlying_type_t<FileMode>>(lhs) |
            static_cast<std::underlying_type_t<FileMode>>(rhs)
        );
    }

    inline constexpr FileMode operator&(FileMode lhs, FileMode rhs) {
        return static_cast<FileMode>(
            static_cast<std::underlying_type_t<FileMode>>(lhs) &
            static_cast<std::underlying_type_t<FileMode>>(rhs)
        );
    }
    
    class File : public cmn::DataFormat, public cmn::GenericVideo {
    protected:
        std::mutex _lock;
        Header _header;
        cv::Mat _average, _mask;
        GETTER(file::Path, filename);
        timestamp_t _prev_frame_time;
        
        // debug compression
        GETTER_I(std::atomic<double>, compression_ratio, 0.0);
        double _compression_value = 0;
        uint32_t _compression_samples = 0;
        
        pv::Frame _last_frame;
        std::mutex _task_list_mutex;
        std::unordered_map<std::thread::id, TaskSentinel*> _task_list;
        std::condition_variable _task_variable;
        
        friend struct pv::TaskSentinel;
        
        const FileMode _mode;
        void _check_opened() const;
        mutable bool _tried_to_open{false};
        
        using DataFormat::start_writing;
        using DataFormat::start_reading;
        using DataFormat::start_modifying;
        
    public:
        bool is_read_mode() const override;
        bool is_write_mode() const override;
        //void start_writing(bool overwrite) override;
        //void start_reading() override;
        
    public:
        File(const file::Path& filename) 
            : File(filename, FileMode::READ, 0)
        { }
        
        template<FileMode Mode = FileMode::READ>
            requires (bool((int)Mode & (int)FileMode::READ))
        static std::unique_ptr<File> Make(const file::Path& filename) {
            return std::unique_ptr<File>(new File(filename, Mode, 1));
        }
        template<FileMode Mode>
            requires (bool((int)Mode & (int)FileMode::WRITE)
                      || bool((int)Mode & (int)FileMode::MODIFY))
        static std::unique_ptr<File> Make(const file::Path& filename, uint8_t channels) {
            return std::unique_ptr<File>(new File(filename, Mode, channels));
        }
        
        template<FileMode Mode = FileMode::READ>
            requires (bool((int)Mode & (int)FileMode::READ))
        static File Read(const file::Path& filename) {
            return File(filename, Mode, 1);
        }
        template<FileMode Mode = FileMode::WRITE>
            requires (bool((int)Mode & (int)FileMode::WRITE)
                      || bool((int)Mode & (int)FileMode::MODIFY))
        static File Write(const file::Path& filename, uint8_t channels) {
            return File(filename, Mode, channels);
        }
        
    private:
        File(const file::Path& filename, FileMode mode, uint8_t channels);
        
    public:
        File(File&&) noexcept;
        ~File();
        
        void close() override;
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
        
        template<meta_encoding_t::Class mode>
        void read_frame(Frame& frame, Frame_t frameIndex) {
            //static_assert(is_in(mode, ImageMode::RGB, ImageMode::GRAY), "Reading from pv is only supported in either RGB or GRAY mode.");
            read_frame(frame, frameIndex, mode);
        }
        
        void read_frame(Frame& frame, Frame_t frameIndex);
        
    private:
        void read_frame(Frame& frame, Frame_t frameIndex, meta_encoding_t::Class mode);
        
    public:
        void read_next_frame(Frame& frame, Frame_t frame_to_read);
        
    private:
        virtual void stop_writing() override;
        
    public:
        void set_resolution(const Size2& size) { _header.resolution = Size2((cv::Size)size); }
        void set_average(const cv::Mat& average);
        const Header& header() const; //{ return _header; }
        Header& header(); //{ return _header; }
        const cv::Mat& average() const override { _check_opened(); assert(_header.average); return _average; }
        
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
        Frame_t length() const override { return Frame_t(_header.num_frames); }
        
        using GenericVideo::frame;
        void frame(Frame_t frameIndex, cv::Mat& output, cmn::source_location loc = cmn::source_location::current());
#ifdef USE_GPU_MAT
        void frame(Frame_t frameIndex, gpuMat& output, cmn::source_location loc = cmn::source_location::current()) override;
#endif
        bool frame(Frame_t, Image&, cmn::source_location = cmn::source_location::current()) override {
            throw InvalidArgumentException("Method not implemented.");
        }
        void frame_optional_background(Frame_t frameIndex, cv::Mat& output, bool with_background);
        bool supports_multithreads() const override { return false; }
        
        void try_compress();
        void update_metadata();
        void set_start_time(std::chrono::time_point<std::chrono::system_clock>);
        
        virtual bool has_timestamps() const override {
            return true;
        }
        virtual timestamp_t timestamp(Frame_t, cmn::source_location loc = cmn::source_location::current()) const override;
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
        
        meta_encoding_t::Class color_mode() const;
        
    protected:
        virtual void _write_header() override;
        virtual void _read_header() override;
        void _update_global_settings();
        
    protected:
        // declared here so memory doesnt have to be freed/allocated all the time
        // when reading a frame and we can make use of the vectors existing capacity
        DataPackage frame_pixels;
        std::vector<Header::line_type> frame_mask_cache{cmn::NoInitializeAllocator<Header::line_type>{}};
        std::vector<LegacyShortHorizontalLine> frame_mask_legacy{cmn::NoInitializeAllocator<LegacyShortHorizontalLine>{}};
        DataPackage frame_compressed_block, frame_uncompressed_block;
        friend class pv::Frame;
        //
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
