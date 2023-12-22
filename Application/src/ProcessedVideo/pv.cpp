#include "pv.h"
#include <minilzo.h>
#include <sys/stat.h>
#include <misc/GlobalSettings.h>
#include <misc/Timer.h>
#include <misc/PVBlob.h>
#include <misc/checked_casts.h>
#include <misc/ranges.h>
#include <misc/SpriteMap.h>
#include <file/DataLocation.h>
#include <regex>

/**
 * =============================
 * ProcessedVideo IMPLEMENTATION
 * -----------------------------
 **/

using namespace file;

namespace cmn {

template<> void Data::read(blob::Prediction& pred) {
    auto version = (pv::Version)pred.clid;
    
    read<uint8_t>(pred.clid);
    read<uint8_t>(pred.p);
    
    if(version >= pv::Version::V_10) {
        uint8_t N{0u};
        read<uint8_t>(N);
        assert(N % 2u == 0);
        //print("pose::N = ", N / 2u);
        pred.pose.points.resize(N / 2u);
        for(size_t i=0; i<N / 2u; i++) {
            read<uint16_t>(pred.pose.points[i].x);
            read<uint16_t>(pred.pose.points[i].y);
        }
        
    } else {
        uint8_t trash;
        read<uint8_t>(trash);
        read<uint8_t>(trash); // trash
    }
}

template<> uint64_t Data::write(const blob::Prediction& val) {
    uint64_t pos =
    write<uint8_t>(val.clid);
    write<uint8_t>(val.p);
    //write<uint8_t>(val._reserved0);
    //write<uint8_t>(val._reserved1);
    
    write<uint8_t>(val.pose.size() * 2);
    for(auto &pt : val.pose.points) {
        write<uint16_t>(pt.x);
        write<uint16_t>(pt.y);
    }
    return pos;
}

}

namespace pv {
    // used to register for global settings updates
    static std::atomic_bool use_differences(false);
    static CallbackCollection _callback;

    /**
     * If there is a task that is async (and can be run read-only for example) and e.g. continously calls "read_frame", then a task sentinel can be registered. This prevents the file from being destroyed until the task is done.
     */
    struct TaskSentinel {
        pv::File *ptr = nullptr;
        std::atomic<bool> _please_terminate{false};
        
        bool terminate() const {
            return _please_terminate;
        }
        
        TaskSentinel(pv::File* file) : ptr(file) {
            std::lock_guard guard(file->_task_list_mutex);
            file->_task_list[std::this_thread::get_id()] = this;
        }
        
        ~TaskSentinel() {
            {
                std::lock_guard guard(ptr->_task_list_mutex);
                assert(ptr->_task_list.find(std::this_thread::get_id()) != ptr->_task_list.end());
                ptr->_task_list.erase(std::this_thread::get_id());
            }
            
            ptr->_task_variable.notify_all();
        }
    };

File::File(File&& other) noexcept
    : cmn::DataFormat(std::move(other)),       // Move-construct base class
      cmn::GenericVideo(std::move(other)),     // Move-construct another base class
      _lock(),                                // Mutexes cannot be moved, so we'll initialize a new one
      _header(Header::move(std::move(other._header))),
      _average(std::move(other._average)),
      _mask(std::move(other._mask)),
      _filename(std::move(other._filename)),
      _prev_frame_time(other._prev_frame_time),
      _compression_ratio(other._compression_ratio.load()), // Atomic types can't be moved. Copy the value.
      _compression_value(other._compression_value),
      _compression_samples(other._compression_samples),
      _last_frame(std::move(other._last_frame)),
      _task_list_mutex(),                      // Mutexes cannot be moved
      _task_list(std::move(other._task_list)),
      _mode(other._mode),
      _tried_to_open(other._tried_to_open)
{
    // Ensure the moved-from object is left in a safe state
    other._prev_frame_time = 0;
    other._compression_value = 0;
    other._compression_samples = 0;
    other._tried_to_open = false;
}

File::File(const file::Path& filename, FileMode mode)
    : DataFormat(filename.add_extension("pv"), filename.str()),
        _header(filename.str()),
        _filename(filename),
        _mode(mode)
{
    
}

    File::~File() {
        {
            std::unique_lock guard(_task_list_mutex); // try to lock once to sync
            for(auto & [i, ptr] : _task_list)
                ptr->_please_terminate = true;
            
            while(!_task_list.empty())
                _task_variable.wait_for(guard, std::chrono::milliseconds(1));
        }
        
        close();
    }

    Frame::Frame(const Frame &other)
        : _index(other._index),
          _timestamp(other._timestamp),
          _n(other._n),
          _loading_time(other._loading_time),
          _source_index(other._source_index),
          _flags(other._flags),
          _predictions(other._predictions)
    {
        _mask.reserve(other.n());
        _pixels.reserve(other.n());
        
        //! perform deep-copy
        for (size_t i=0; i<other.n(); ++i) {
            _mask.emplace_back(new blob::line_ptr_t::element_type(*other._mask[i]));
            _pixels.emplace_back(new blob::pixel_ptr_t::element_type(*other._pixels[i]));
        }
    }

    Frame::Frame(const timestamp_t& timestamp, decltype(_n) n)
        : _timestamp(timestamp)
    {
        _mask.reserve(n);
        _pixels.reserve(n);
        _flags.reserve(n);
    }
    
    Frame::Frame(File& ref, Frame_t idx) {
        read_from(ref, idx);
    }

    std::unique_ptr<pv::Blob> Frame::blob_at(size_t i) const {
        return std::make_unique<pv::Blob>(*_mask[i], *_pixels[i], _flags[i], _predictions.empty() ? blob::Prediction{} : _predictions[i]);
    }

    std::unique_ptr<pv::Blob> Frame::steal_blob(size_t i) {
        return std::make_unique<pv::Blob>(std::move(_mask[i]), std::move(_pixels[i]), _flags[i], _predictions.empty() ? blob::Prediction{} : std::move(_predictions[i]));
    }

    std::vector<pv::BlobPtr> Frame::steal_blobs() && {
        //if(_blobs.empty())
        std::vector<pv::BlobPtr> _blobs;
        {
            _blobs.clear();
            _blobs.reserve(n());
            for (uint32_t i=0; i<n(); i++)
                _blobs.emplace_back(steal_blob(i));
        }
        
        return _blobs;
    }
    
    std::vector<pv::BlobPtr> Frame::get_blobs() const {
        //if(_blobs.empty())
        std::vector<pv::BlobPtr> _blobs;
        {
            _blobs.clear();
            _blobs.reserve(n());
            for (uint32_t i=0; i<n(); i++)
                _blobs.push_back(blob_at(i));
        }
        
        return _blobs;
    }
    
    void Frame::clear() {
        for(auto &m : _mask) {
            if(m)
                buffers().move_back(std::move(m));
        }
        
        _mask.clear();
        _pixels.clear();
        _flags.clear();
        _predictions.clear();
        _n = 0;
        _timestamp = 0;
        _loading_time = 0;
        //_blobs.clear();
        
        set_index({});
    }


    
    void Frame::read_from(pv::File &ref, Frame_t idx) {
        //for(auto m: _mask)
        //    delete m;
        //for(auto p: _pixels)
        //   delete p;
        if(ref.is_write_mode())
            throw U_EXCEPTION("Cannot read from writing file ", ref);
        
        clear();
        set_index(idx);
        
        Data* ptr = &ref;
        ReadonlyMemoryWrapper *compressed = NULL;
        
        static std::once_flag flag;
        std::call_once(flag, [](){
            use_differences = GlobalSettings::map().has("use_differences") ? SETTING(use_differences).value<bool>() : false;
            _callback = GlobalSettings::map().register_callbacks({"use_differences"}, [](auto) {
                use_differences = SETTING(use_differences).value<bool>();
            });
            GlobalSettings::map().register_shutdown_callback([](auto){
                _callback.reset();
            });
        });
        
        if(ref.header().version >= V_6) {
            uchar compression_flag;
            ref.read<uchar>(compression_flag);
            
            if(compression_flag) {
                uint32_t size;
                ref.read<uint32_t>(size);
                uint32_t uncompressed_size;
                ref.read<uint32_t>(uncompressed_size);
                
                
                static DataPackage compressed_block, uncompressed_block;
                compressed_block.resize(size, false);
                uncompressed_block.resize(uncompressed_size, false);
                
                ref.read_data(size, compressed_block.data());
                
                lzo_uint new_len;
                if(lzo1x_decompress((uchar*)compressed_block.data(),size,(uchar*)uncompressed_block.data(),&new_len,NULL) == LZO_E_OK)
                {
                    assert(new_len == uncompressed_size);
                    compressed = new ReadonlyMemoryWrapper((uchar*)uncompressed_block.data(), new_len);
                    ptr = compressed;
                    
                } else {
                    FormatError("Failed to decode frame ", idx," from file ", *this);
                }
            }
        }
        
        // read relative timestamp (relative to timestamp in header)
        if(ref.header().version < V_4) {
            ptr->read_convert<uint32_t>(_timestamp);
        } else {
            ptr->read<timestamp_t>(_timestamp);
        }
        
        ptr->read<uint16_t>(_n);
        
        if(ref.header().version >= V_9) {
            int32_t original;
            ptr->read<int32_t>(original);
            if(original >= 0)
                _source_index = Frame_t(original);
            else
                _source_index.invalidate();
        }
        
        _mask.reserve(_n);
        _pixels.reserve(_n);
        _flags.reserve(_n);
        
        // declared outside so memory doesnt have to be freed/allocated all the time
        static DataPackage pixels;
        static std::vector<Header::line_type> mask((NoInitializeAllocator<Header::line_type>()));
        static std::vector<LegacyShortHorizontalLine> mask_legacy((NoInitializeAllocator<LegacyShortHorizontalLine>()));
        
        for(int i=0; i<_n; i++) {
            uint16_t start_y, mask_size;
            uint8_t flags = 0;
            
            ptr->read<uint16_t>(start_y);
            if(ref.header().version >= V_8) {
                ptr->read<uint8_t>(flags);
            }
            ptr->read<uint16_t>(mask_size);
            
            if(ref.header().version < V_7) {
                mask_legacy.resize(mask_size);
                mask.clear();
                mask.reserve(mask_legacy.size());
                
                assert(ref.header().line_size == sizeof(LegacyShortHorizontalLine));
                
                ptr->read_data(mask_size * ref.header().line_size, (char*)mask_legacy.data());
                std::copy(mask_legacy.begin(), mask_legacy.end(), std::back_inserter(mask));
                
            } else {
                mask.resize(mask_size);
                ptr->read_data(mask_size * ref.header().line_size, (char*)mask.data());
            }
            
            uint64_t num_pixels = 0;
            for(auto &l : mask) {
                num_pixels += l.x1() - l.x0() + 1;
            }
            
            if(num_pixels >= std::numeric_limits<uint32_t>::max()) {
                FormatWarning("Something is happening here ", index(), " ", num_pixels, " ", uint64_t(-1));
            }
            pixels.resize(num_pixels, false);
            ptr->read_data(num_pixels, pixels.data());
            
            auto uncompressed = buffers().get(source_location::current());
            Header::line_type::uncompress(*uncompressed, start_y, mask);
            
            if(use_differences) {
                uint64_t idx = 0;
                uchar *ptr = (uchar*)pixels.data();
                for (auto &l : *uncompressed) {
                    for (int x=l.x0; x<=l.x1; x++) {
                        auto &p = ptr[idx++];
                        p = (uchar)saturate(int(ref.average().at<uchar>(l.y, x)) - int(p));
                    }
                }
            }
            
            _mask.emplace_back(std::move(uncompressed));
            auto v = std::make_unique<std::vector<uchar>>((uchar*)pixels.data(),
                                                 (uchar*)pixels.data()+num_pixels);
            _pixels.emplace_back(std::move(v));
            _flags.push_back(flags);
        }
        
        _mask.shrink_to_fit();
        _pixels.shrink_to_fit();
        
        if(ref.header().version >= V_9) {
            uint16_t n_predictions;
            ptr->read<uint16_t>(n_predictions);
            assert(n_predictions <= _n);
            
            if(n_predictions > 0) {
                _predictions.resize(_n);
                for(int i=0; i<_n; ++i) {
                    _predictions[i].clid = (uint8_t)ref.header().version;
                    ptr->read<blob::Prediction>(_predictions[i]);
                }
            }
        }
        
        if(compressed)
            delete compressed;
    }
    
    void Frame::add_object(blob::Pair&& pair) {
        assert(pair.lines->size() < UINT16_MAX);
        
#ifndef NDEBUG
        HorizontalLine prev = pair.lines->empty() ? HorizontalLine() : pair.lines->front();
        
        uint64_t pixel_count = 0;
        for (auto &line : *pair.lines) {
            if(!(prev == line) && !(prev < line))
                FormatWarning("Lines not properly ordered, or overlapping in x [",prev.x0,"-",prev.x1,"] < [",line.x0,"-",line.x1,"] (",prev.y,"/",line.y,").");
            prev = line;
            pixel_count += line.x1 - line.x0 + 1;
        }

        assert(pixel_count == pair.pixels->size());
#endif
        
        _mask.emplace_back(std::move(pair.lines));
        _pixels.push_back(std::move(pair.pixels));
        _flags.push_back(pair.extra_flags);
        //if(pair.pred.valid() or not _predictions.empty()) {
            _predictions.resize(_flags.size());
            _predictions.back() = std::move(pair.pred);
        //}
        
        _n++;
    }

void Frame::add_object(const std::vector<HorizontalLine>& mask, const std::vector<uchar>& pixels, uint8_t flags, const blob::Prediction& pred)
{
    assert(mask.size() < UINT16_MAX);
#ifndef NDEBUG
    HorizontalLine prev = mask.empty() ? HorizontalLine() : mask.front();

    uint64_t count = 0, pixel_count = 0;
    for (auto& line : mask) {
        if (!(prev == line) && !(prev < line))
            FormatWarning("Lines not properly ordered, or overlapping in x [", prev.x0, "-", prev.x1, "] < [", line.x0, "-", line.x1, "] (", prev.y, "/", line.y, ").");
        prev = line;
        pixel_count += line.x1 - line.x0 + 1;
        ++count;
    }

    assert(pixel_count == pixels.size());
#endif

    _mask.emplace_back(new blob::line_ptr_t::element_type(mask));
    _pixels.push_back(std::make_unique<blob::pixel_ptr_t::element_type>(pixels));
    _flags.push_back(flags);
    //if(pred.valid()) {
        _predictions.resize(_flags.size());
        _predictions.back() = pred;
    //}
    
    _n++;
}
    
    void Frame::add_object(const std::vector<HorizontalLine> &mask_, const cv::Mat &full_image, uint8_t flags) {
        assert(full_image.rows > 0 && full_image.cols > 0);
        assert(!mask_.empty());
        
        auto mask = std::make_unique<std::vector<HorizontalLine>>(mask_);
        
        ptr_safe_t L = (ptr_safe_t)mask_.size();
        //uint64_t offset = 0;
        
        // calculate overall bytes
        ptr_safe_t overall = 0;
        auto line_ptr = mask->data();
        
        assert(full_image.cols-1 < UINT16_MAX);
        assert(full_image.rows-1 < UINT16_MAX);
        
        for(ptr_safe_t i=0; i<L; i++) {
            // restrict to image dimensions
            assert(line_ptr->x1 < full_image.cols);
            assert(line_ptr->y < full_image.rows && line_ptr->x0 < full_image.cols && line_ptr->x1 >= line_ptr->x0);
            /*if(line_ptr->x1 >= full_image.cols) {
                line_ptr->x1 = full_image.cols - 1;
            }
            
            if(line_ptr->y >= full_image.rows || line_ptr->x0 >= full_image.cols || line_ptr->x1 < line_ptr->x0)
            {
                throw U_EXCEPTION("x1 < x0 in ",line_ptr->x1,"-",line_ptr->x0,",",line_ptr->y," ",L,"u 0x",line_ptr,"");
                //mask->erase(mask->begin()+i-offset);
                //offset++;
                continue;
            }*/
            
            overall += ptr_safe_t(line_ptr->x1) - ptr_safe_t(line_ptr->x0) + 1;
            line_ptr++;
        }
        
        // copy grey values to pixels array
        auto pixels = std::make_unique<std::vector<uchar>>();
        pixels->resize(overall);
        //uchar *pixels = (uchar*)malloc(overall);
        
        auto pixel_ptr = pixels->data();
        line_ptr = mask->data(); // reset ptr
        L = mask->size();
        
        for (ptr_safe_t i=0; i<L; i++, line_ptr++) {
            auto ptr = full_image.ptr(line_ptr->y, line_ptr->x0);
            assert(line_ptr->x1 >= line_ptr->x0);
            auto N = ptr_safe_t(line_ptr->x1) - ptr_safe_t(line_ptr->x0) + ptr_safe_t(1);
            memcpy(pixel_ptr, ptr, sign_cast<size_t>(N));
            pixel_ptr += N;
        }
        
        add_object(blob::Pair{std::move(mask), std::move(pixels), flags});
        //free(pixels);
    }
    
    uint64_t Frame::size() const noexcept {
        uint64_t bytes = sizeof(_timestamp) + sizeof(_n) + _mask.size() * sizeof(uint16_t);
        uint64_t elem_size = sizeof(Header::line_type);
        
        for (auto &m : _mask)
            bytes += sizeof(uint16_t) + sizeof(uint8_t) + m->size() * elem_size;
        
        for (auto &m : _pixels)
            bytes += m->size() * sizeof(char);

        bytes += sizeof(uint16_t) + sizeof(uint8_t) * 4u * _predictions.size();
        return bytes;
    }

    //void Frame::update_timestamp(DataPackage& pack) const {
    //    pack.write(_timestamp, 0u); // just update the timestamp
    //}
    
    void Frame::serialize(DataPackage &pack, bool& compressed) const {
        uint64_t bytes = size();
        uint64_t elem_size = sizeof(Header::line_type);
        compressed = false;
        
        pack.resize(bytes);
        pack.reset_offset();
        
        assert(_timestamp.valid());
        pack.write<timestamp_t>(_timestamp);
        pack.write<uint16_t>(_n);
        //static_assert(std::same_as<int32_t, Frame_t::number_t>, "Assuming int32_t here. Please fix.");
        pack.write<Frame_t::number_t>(_source_index.valid() ? _source_index.get() : Frame_t::number_t(-1));
        
        for(uint16_t i=0; i<_n; i++) {
            auto &mask = _mask.at(i);
            auto &pixels = _pixels.at(i);
            auto flags = _flags.at(i);
            
            auto compressed = Header::line_type::compress(*mask);
            pack.write(uint16_t(mask->empty() ? 0 : mask->front().y));
            pack.write(uint8_t(flags));
            pack.write(uint16_t(compressed.size()));
            pack.write_data(compressed.size() * elem_size, (char*)compressed.data());
            pack.write_data(pixels->size() * sizeof(char), (char*)pixels->data());
        }
        
        //! prediction information (if available)
        pack.write<uint16_t>(narrow_cast<uint16_t>(_predictions.size()));
        //if(not _predictions.empty()) {
            assert(_predictions.size() == _mask.size());

            for(uint16_t i=0; i<_n; ++i) {
                pack.write<cmn::blob::Prediction>(_predictions.at(i));
            }
        //}

        // see whether this frame is worth compressing (size-threshold)
        if (pack.size() >= 1500) {
#define OUT_LEN(L)     (L + L / 16 + 64 + 3)


            /* Work-memory needed for compression. Allocate memory in units
             * of 'lzo_align_t' (instead of 'char') to make sure it is properly aligned.
             */

#define HEAP_ALLOC(var,size) \
    lzo_align_t __LZO_MMODEL var [ ((size) + (sizeof(lzo_align_t) - 1)) / sizeof(lzo_align_t) ]

            HEAP_ALLOC(wrkmem, LZO1X_1_MEM_COMPRESS);

            lzo_uint out_len = 0;
            assert(pack.size() < UINT32_MAX);
            uint32_t in_len = (uint32_t)pack.size();
            uint64_t reserved_size = OUT_LEN(in_len);

            DataPackage out;
            out.resize(reserved_size);

            // lock for wrkmem
            if (lzo1x_1_compress((uchar*)pack.data(), in_len, (uchar*)out.data(), &out_len, wrkmem) == LZO_E_OK)
            {
                uint64_t size = out_len + sizeof(uint32_t) * 2;
                /*if (size < in_len) {
                    if (_compression_samples > 1000) {
                        _compression_value = _compression_value / _compression_samples;
                        _compression_samples = 1;
                    }

                    _compression_value = _compression_value + size / float(in_len);
                    _compression_samples++;
                    _compression_ratio = _compression_value / double(_compression_samples);
                }*/

                if (size < in_len) {
                    pack.reset_offset();
                    compressed = true;

                    assert(out_len < UINT32_MAX);
                    pack.write<uint32_t>((uint32_t)out_len);
                    pack.write<uint32_t>(in_len);
                    pack.write_data(out_len, out.data());
                }

            }
            else {
                print("Compression of ",pack.size()," bytes failed.");
            }
        }
    }

    /*void File::start_writing(bool overwrite) {
        DataFormat::start_writing(overwrite);
    }*/

    void File::_write_header() {
        if (not (bool(_mode & FileMode::WRITE)
              || bool(_mode & FileMode::MODIFY))
            || not is_open())
            throw U_EXCEPTION("File not opened when writing header ", _filename, ".");
        
        _update_global_settings();
        _header.write(*this);
    }
    void File::_read_header() {
        if (not (bool(_mode & FileMode::READ)
              || bool(_mode & FileMode::MODIFY))
            || not is_open())
            throw U_EXCEPTION("File not opened when reading header ", _filename, ".");
        
        _header.read(*this);

        _average = _header.average->get();
        if (has_mask())
            _mask = _header.mask->get();

        //std::chrono::microseconds ns_l, ns_e;
        uint64_t fps_l = 0;

        if (!_open_for_writing) {
            uint64_t idx = length().get() / 2u;
            //uint64_t edx = length()-1;
            if (idx < length().get()) {
                pv::Frame lastframe;
                //read_frame(lastframe, edx);

                //ns_e = std::chrono::microseconds(lastframe.timestamp());

                read_frame(lastframe, Frame_t(idx));
                //ns_l = std::chrono::microseconds(lastframe.timestamp());

                if (idx >= 1) {
                    auto last = lastframe.timestamp();
                    if(last.valid()) {
                        read_frame(lastframe, Frame_t(idx - 1));
                        fps_l = last.get() - lastframe.timestamp().get();
                    } else fps_l = 0;
                }
            }

        }

        _header.average_tdelta = fps_l;
        _update_global_settings();
    }
    
    void Header::read(DataFormat& ref) {
        std::string version_str;
        ref.read<std::string>(version_str);
        
        if(version_str.length() > 2) {
            std::regex re("PV(\\d+)");
            std::smatch match;
            if (std::regex_search(version_str, match, re)) {
                int nr = std::stoi(match[1]);
                version = static_cast<Version>(nr - 1);
            } else {
                version = Version::V_1;
            }
            
        } else {
            version = Version::V_1;
        }
        
        if(version > Version::current)
            throw U_EXCEPTION("Unknown version '",version,"'.");
        
        if(version == Version::V_2) {
            // must read settings from file before loading...
            if(!file::DataLocation::is_registered("settings"))
                throw U_EXCEPTION("You have to register a DataLocation for 'settings' before using pv files (usually the same folder the video is in + exchange the .pv name with .settings).");
            auto settings_file = file::DataLocation::parse("settings");
            if (settings_file.exists())
                GlobalSettings::load_from_file({}, settings_file.str(), AccessLevelType::PUBLIC);
        }
        
        
        ref.read<uchar>(channels);
        ref.read<cv::Size>(resolution);
        
        // added offsets
        if (version >= Version::V_3) {
            ref.read(offsets);
            
        } else if(version == Version::V_2) {
            offsets = GlobalSettings::has("crop_offsets") ? SETTING(crop_offsets) : CropOffsets();
        }
        
        ref.read(line_size);
        _num_frames_offset = ref.current_offset();
        ref.read(num_frames);
        _index_offset = ref.current_offset();
        ref.read(index_offset);
        _timestamp_offset = ref.current_offset();
        ref.read(timestamp);
        
        ref.read<std::string>(name);
        
        // check values for sanity
        if(channels != 1)
            throw U_EXCEPTION("Only 1 channel currently supported (",this->channels," provided)");
        
        if(line_size != sizeof(line_type))
            throw U_EXCEPTION("The used line format in this file (",line_size," bytes) differs from the expected ",sizeof(line_type)," bytes.");
        
        if(average)
            delete average;
        
        average = new Image((uint)this->resolution.height, (uint)this->resolution.width, channels);
        _average_offset = ref.current_offset();
        ref.read_data(average->size(), (char*)average->data());
        
        if(mask)
            delete mask;
        mask = NULL;
        
        if(version >= Version::V_2) {
            uint64_t mask_size;
            ref.read<uint64_t>(mask_size);
            if(mask_size) {
                // does it use a mask?
                mask = new Image((uint)this->resolution.height /*- (offsets.y + offsets.height)*/, (uint)this->resolution.width /*- (offsets.x + offsets.width)*/, channels);
                
                ref.read_data(mask->size(), (char*)mask->data());
                
                for(uint64_t i=0; i<mask->size(); i++)
                    if(mask->data()[i] > 1) {
                        mask->get() /= mask->data()[i];
                        break;
                    }
            }
        }
        
        // read the index table
        index_table.resize(num_frames);
        auto len = sizeof(decltype(index_table)::value_type) * num_frames;
        ref.Data::read_data(index_offset, len, (char*)index_table.data());
        
        if(version >= Version::V_5) {
            _meta_offset = index_offset+len;
            ref.read(metadata, _meta_offset);
            
            try {
                sprite::Map map;
                map["quiet"] = true;
                map["meta_real_width"] = float();
                sprite::parse_values(map, metadata);
                /*for(auto key : map.keys()) {
                    print("Key: ", key, " Value: ", map[key].get().valueString());
                }*/
                if(map.has("meta_real_width"))
                    meta_real_width = map["meta_real_width"].value<float>();
            } catch(...) {
                FormatExcept("Error parsing settings metadata from ", ref.filename(), ".");
            }
        }
    }

    void Header::write(DataFormat& ref) {
        /**
         * [HEADER SECTION]
         * (string) "PV" + (version_nr)
         * (byte)   channels
         * (uint16) width
         * (uint16) height
         * (Rect2i) four ushorts with the mask-offsets left,top,right,bottom
         * (uchar)  sizeof(HorizontalLine)
         * (uint32) number of frames
         * (uint64) pointer to index at the end of file
         * (uint64) timestamp (time since 1970 in microseconds)
         * (string) project name
         * (byte*)  average img ([width x height] x channels)
         * (uint64_t) mask present / mask size in bytes (if 0 no mask)
         * [byte*]  mask, but only present if mask_size != NULL
         *
         * [DATA SECTION]
         * for each frame:
         *   (uchar) compression flag (if 1, the whole frame is compressed)
         *   if compressed:
         *      (uint32) original size
         *      (uint32) compressed size
         *      (byte*) lzo1x compressed data (see below for uncompressed)
         *   else:
         *      [UNCOMPRESSED DATA PER FRAME] {
         *          (uint32) timestamp (in microseconds) since start of movie
         *          (uint16) number of individual cropped images
         *
         *          for each individual:
         *              (uint16) number of HorizontalLine structs
         *              (byte*)  n * sizeof(HorizontalLine)
         *              (byte*)  original image pixels ordered exactly as in HorizontalLines (BGR, CV_8UC(n))
         *      }
         *
         * [INDEX TABLE]
         * for each frame
         *   (uint64) frame start position in file
         *
         * [METADATA]
         * (string) JSONized metadata array
         */
        
        // set offsets from global settings
        
        ref.write("PV" + std::to_string((int)Version::current + 1));
        
        ref.write(channels);
        
        if(!resolution.width && !resolution.height)
            throw U_EXCEPTION("Resolution of the video has not been set.");
        ref.write(resolution);
        ref.write(offsets);
        
        ref.write(line_size);
        _num_frames_offset = ref.write(decltype(this->num_frames)(0));
        _index_offset = ref.write(decltype(index_offset)(0));
        _timestamp_offset = ref.write(timestamp);
        
        ref.write<std::string>((std::string)file::Path(name).filename());
        
        if(average)
            _average_offset = ref.write_data(average->size(), (char*)average->data());
        else {
            Image tmp((uint)resolution.height, (uint)resolution.width, 1);
            _average_offset = ref.write_data(tmp.size(), (char*)tmp.data());
        }
        
        if(mask) {
            ref.write(uint64_t(mask->size()));
            ref.write_data(mask->size(), (char*)mask->data());
            print("Written mask with ", mask->cols,"x",mask->rows);
        }
        else {
            ref.write(uint64_t(0));
        }
    }

    void Header::update(DataFormat& ref) {
        // write index table
        index_offset = ref.current_offset();
        print("Index table is ",FileSize(index_table.size() * sizeof(decltype(index_table)::value_type))," big.");
        
        for (auto index : index_table) {
            ref.write<decltype(index_table)::value_type>(index);
        }
        
        metadata = generate_metadata();
        _meta_offset = ref.write(metadata);
        
        ref.write(this->num_frames, _num_frames_offset);
        ref.write(this->index_offset, _index_offset);
        ref.write(this->timestamp, _timestamp_offset);
        
        if(average) {
            ref.Data::write_data(_average_offset, average->size(), (char*)average->data());
        }
        
        print("Updated number of frames with ",this->num_frames,", index offset ",this->index_offset,", timestamp ",this->timestamp,", ", _meta_offset);
    }
    
    std::string Header::generate_metadata() const {
        std::stringstream ss;
        
        std::vector<std::string> write_these = GlobalSettings::map().has("meta_write_these") ? SETTING(meta_write_these) : std::vector<std::string>();
        for (uint64_t i=0; i<write_these.size(); i++) {
            auto &name = write_these.at(i);
            auto val = GlobalSettings::get(name).get().valueString();
            ss << "\""<< name <<"\": "<<val;
            if(i<write_these.size()-1)
                ss << ", ";
        }
        
        std::string ret = ss.str();
        if(ret.empty()) {
            print("Metadata empty.");
        } else {
            ret = "{"+ret+"}";
            print("Metadata: ",ret.c_str());
        }
        
        return ret;
    }
    
    Header& File::header() {
        _check_opened();
        return _header;
    }

    const Header& File::header() const {
        if(not bool(_mode & FileMode::WRITE))
            _check_opened();
        return _header;
    }

    void File::close() {
        if(is_open()) {
            if(bool(_mode & FileMode::WRITE))
                stop_writing();
        }
        DataFormat::close();
        _tried_to_open = false;
    }

    std::string File::get_info(bool full) {
        auto str = get_info_rich_text(full);
        str = utils::find_replace(str, "<b>", "");
        str = utils::find_replace(str, "</b>", "");
        return str;
    }
    
    std::string File::filesize() const {
        uint64_t bytes = 0;
        if(bool(_mode & FileMode::WRITE)) {
            if(!_open_for_writing)
                throw U_EXCEPTION("File has to be opened for filesize() to work in WRITE mode.");
            
            bytes = current_offset();
        } else {
#if defined(__EMSCRIPTEN__)
            bytes = reading_file_size();
#else
            bytes = _filename.add_extension("pv").file_size();
#endif
        }
        
        return Meta::toStr(FileSize(bytes));
    }
    
    std::string File::get_info_rich_text(bool full) {
        _check_opened();
        
        std::stringstream ss;
        ss << this->summary() << "\n";
        
        /**
         * Display time related information.
         */
        std::chrono::microseconds ns_l, ns_e;
        
        if(bool(_mode & FileMode::READ)
           or bool(_mode & FileMode::MODIFY))
        {
            Frame_t idx = length() / 2_f;
            Frame_t edx = length() - 1_f;
            if(idx < length()) {
                pv::Frame lastframe;
                read_frame(lastframe, edx);
                
                ns_e = std::chrono::microseconds(lastframe.timestamp());
                
                read_frame(lastframe, idx);
                ns_l = std::chrono::microseconds(lastframe.timestamp());
            }
            
        } else {
            ns_l = std::chrono::microseconds(0);
        }
        
        auto ns = std::chrono::microseconds(header().timestamp);
        auto s = std::chrono::duration_cast<std::chrono::seconds>(ns);
        auto now = std::chrono::time_point<std::chrono::system_clock>(s);
        
        auto now_c = std::chrono::system_clock::to_time_t(now);
        
        ss << std::endl;
        
        ss << "<b>crop_offsets:</b> " << Meta::toStr(crop_offsets()) << std::endl;
#ifndef NO_PUT_TIME
        ss << "<b>Time of recording:</b> '" << std::put_time(std::localtime(&now_c), "%c") << "'"
        << std::endl;
#endif
        
        ss << "<b>Length of recording:</b> '";
        duration_to_string(ss, ns_e);
        ss << "'" << std::endl;
        
        ss << "<b>Framerate:</b> " << std::setw(0) << framerate() << "fps (" << float(_header.average_tdelta) / 1000.f << "ms)";
        ss << std::endl;
        
        ss << std::endl;
        
        if(full) {
            if(header().metadata.empty())
                ss << ("<b>Metadata empty.</b>");
            else
                ss << "<b>Metadata:</b> " << header().metadata ;
        }
        
        return ss.str();
    }
    
    void File::set_start_time(std::chrono::time_point<std::chrono::system_clock> tp) {
        assert(!is_open());
        _header.timestamp = narrow_cast<uint64_t>(std::chrono::time_point_cast<std::chrono::microseconds>(tp).time_since_epoch().count());
    }
    
    const pv::Frame& File::last_frame() {
        std::unique_lock<std::mutex> lock(_lock);
        return _last_frame;
    }

    void File::add_individual(const Frame& frame, DataPackage& pack, bool compressed) {
        _check_opened();
        
        assert(_open_for_writing);
        assert(_header.timestamp != 0); // start time has to be set

        std::unique_lock<std::mutex> lock(_lock);

        _header.num_frames++;
        assert(!_prev_frame_time.valid() || frame._timestamp > _prev_frame_time);
        //if(frame._timestamp >= _header.timestamp)
        //    frame._timestamp -= _header.timestamp; // make timestamp relative to start of video

        if (_prev_frame_time.valid() && frame._timestamp <= _prev_frame_time) {
            throw U_EXCEPTION("Should be dropping frame because ",frame._timestamp," <= ",_prev_frame_time,".");
        }

        if(_prev_frame_time.valid()) {
            _header._running_average_tdelta += frame._timestamp.get() - _prev_frame_time.get();
            _header.average_tdelta = _header._running_average_tdelta / (_header.num_frames > 0 ? double(_header.num_frames) : 1);
        } else {
            _header._running_average_tdelta = 0;
            _header.average_tdelta = 0;
        }
        _prev_frame_time = frame._timestamp;

        // resets the offset and writes frame content to the pack
        auto index = tell();
        if (!compressed)
            this->write<uchar>(0);
        else
            this->write<uchar>(1);
        
        this->write(pack);
        _header.index_table.push_back(index);
        
        // cache last_frame
        _last_frame = Frame(frame);
    }

    void File::_check_opened() const {
        if(_tried_to_open)
            return;
        
        _tried_to_open = true;
        
        if(bool(_mode & FileMode::MODIFY))
            const_cast<pv::File*>(this)->start_modifying();
        else if(bool(_mode & FileMode::READ))
            const_cast<pv::File*>(this)->start_reading();
        else
            const_cast<pv::File*>(this)->start_writing(bool(_mode & FileMode::OVERWRITE));
    }

    void File::_update_global_settings() {
    }

    void File::add_individual(Frame&& frame) {
        static auto pack_mutex = LOGGED_MUTEX("File::pack_mutex");
        static DataPackage pack;

        auto g = LOGGED_LOCK(pack_mutex);
        _check_opened();
        bool compressed;
        frame.serialize(pack, compressed);
        add_individual(std::move(frame), pack, compressed);
    }
        
    void File::stop_writing() {
        if(not is_open()
           || not bool(_mode & FileMode::WRITE))
            throw U_EXCEPTION("Do not stop writing on a file that was not open for writing (",_filename,").");
        write(uint64_t(0));
        _header.update(*this);
        print_info();
    }
    
    void File::read_frame(Frame& frame, Frame_t frameIndex) {
        std::unique_lock<std::mutex> guard(_lock);
        _check_opened();
        
        assert(!_open_for_writing);
        assert(frameIndex.valid());
        
        if(frameIndex.get() >= _header.num_frames)
           throw U_EXCEPTION("Frame index ", frameIndex," out of range.");
        
        uint64_t pos = _header.index_table.at(frameIndex.get());
        uint64_t old = current_offset();
        
        if(old != pos)
            seek(pos);
        
        frame.read_from(*this, frameIndex);
    }
    
    void File::read_next_frame(Frame& frame, Frame_t frame_to_read) {
        _check_opened();
        assert(!_open_for_writing);
        
        std::unique_lock<std::mutex> guard(_lock);
        if(frame_to_read.get() >= _header.num_frames)
           throw U_EXCEPTION("Frame index ", frame_to_read," out of range.");
        
        frame.read_from(*this, frame_to_read);
    }
#ifdef USE_GPU_MAT
    void File::frame(Frame_t frameIndex, gpuMat &output, cmn::source_location) {
        cv::Mat local;
        frame_optional_background(frameIndex, local, true);
        local.copyTo(output);
    }
#endif
    void File::frame(Frame_t frameIndex, cv::Mat &output, cmn::source_location) {
        frame_optional_background(frameIndex, output, true);
    }
    
    void File::frame_optional_background(Frame_t frameIndex, cv::Mat& output, bool with_background) {
        _check_opened();
        
        Frame frame;
        read_frame(frame, frameIndex);
        
        if(with_background)
            average().copyTo(output);
        else
            output = cv::Mat::zeros(header().resolution.height, header().resolution.width, CV_8UC1);
        
        for (uint16_t i=0; i<frame.n(); i++) {
            uint64_t index = 0;
            auto &mask = frame.mask().at(i);
            auto &pixels = frame.pixels().at(i);
            
            for(const HorizontalLine &line : *mask) {
                for(int x=line.x0; x<=line.x1; x++) {
                    output.at<uchar>(line.y, x) = (uchar)pixels->at(index++);
                }
            }
        }
    }

    void fix_file(File& file) {
        print("Starting file copy and fix (",file.filename(),")...");
        
        File copy(file.filename()+"_fix", FileMode::WRITE | FileMode::OVERWRITE);
        copy.set_resolution(file.header().resolution);
        copy.set_offsets(file.crop_offsets());
        copy.set_average(file.average());
        
        auto keys = sprite::parse_values(file.header().metadata).keys();
        sprite::parse_values(GlobalSettings::map(), file.header().metadata);
        SETTING(meta_write_these) = keys;
        
        if(file.has_mask())
            copy.set_mask(file.mask());
        
        copy.header().timestamp = file.header().timestamp;
        
        timestamp_t raw_prev_timestamp;
        timestamp_t last_reset;
        Frame_t last_reset_idx;
        uint64_t last_difference = 0;
        
        for (Frame_t idx = 0_f; idx < Frame_t(file.length()); ++idx) {
            pv::Frame frame;
            file.read_frame(frame, idx);
            
            //frame.set_timestamp(file.header().timestamp + frame.timestamp());
            
            if (raw_prev_timestamp.valid() && frame.timestamp() < raw_prev_timestamp) {
                last_reset = raw_prev_timestamp.get() + last_difference;
                last_reset_idx = idx;
                
                FormatWarning("Fixing frame ",idx," because timestamp ",frame.timestamp()," < ",last_reset," -> ",last_reset + frame.timestamp().get());
            } else {
            	last_difference = frame.timestamp().get() - raw_prev_timestamp.get();
            }
            
            raw_prev_timestamp = frame.timestamp();
            
            if(last_reset_idx.valid()) {
                frame.set_timestamp(last_reset + frame.timestamp());
            }
            
            copy.add_individual(std::move(frame));
            
            if (idx.get() % 1000 == 0) {
                print("Frame ", idx," / ", file.length()," (",copy.compression_ratio() * 100,"% compression ratio)...");
            }
        }
        
        copy.close();
        
        print("Written fixed file.");
    }
    
    void File::try_compress() {
        _check_opened();
        
        File copy(filename()+"_test", FileMode::WRITE);
        copy.set_resolution(header().resolution);
        copy.set_offsets(crop_offsets());
        copy.set_average(average());
        copy.header().timestamp = header().timestamp;
        
        if(has_mask())
            copy.set_mask(mask());
        
        copy.start_writing(true);
        
        std::vector<std::string> save = GlobalSettings::map().has("meta_write_these") ? SETTING(meta_write_these) : std::vector<std::string>{};
        
        sprite::Map map;
        if(!header().metadata.empty())
            sprite::parse_values(map, header().metadata);
        SETTING(meta_write_these) = map.keys();
        
        pv::Frame frame;
        for (Frame_t i=0_f; i<length(); ++i) {
            read_frame(frame, i);
            frame.set_timestamp(header().timestamp + frame.timestamp());
            
            copy.add_individual(std::move(frame));
            
            if (i.get() % 1000 == 0) {
                print("Frame ", i," / ",length(),"...");
            }
        }
        
        copy.stop_writing();
        
        print("Written");
        
        {
            print_info();
            
            File test(filename()+"_test", FileMode::READ);
            test.start_reading();
        }
        
        SETTING(meta_write_these) = save;
    }
    
    void File::update_metadata() {
        _check_opened();
        
        if(not bool(_mode & FileMode::MODIFY)
           || not is_open())
            throw U_EXCEPTION("Must be open for writing.");
    
        print("Updating metadata...");
        auto metadata = _header.generate_metadata();
        write(metadata, _header.meta_offset());
    }
    
    short File::framerate() const {
        if(_header.average_tdelta == 0)
            return -1;
        return short(1000. * 1000. / _header.average_tdelta);
    }

    bool File::is_read_mode() const {
        return bool(_mode & FileMode::READ);
    }

    bool File::is_write_mode() const {
        return bool(_mode & FileMode::WRITE);
    }

    std::vector<float> File::calculate_percentiles(const std::initializer_list<float> &percent) {
        _check_opened();
        
        if(_open_for_writing)
            throw U_EXCEPTION("Cannot calculate percentiles when file is opened for writing.");
        Timer timer;
        TaskSentinel sentinel(this);
        
        std::vector<float> pixel_values;
#ifdef NDEBUG
        std::unique_lock guard(_lock);
        Image average(_average);
        guard.unlock();
        
        if(average.bounds().size().max() <= 4000) {
            // also take into account background image?
            pixel_values.insert(pixel_values.end(), average.data(), average.data() + average.cols * average.rows);
        }
#endif
        
        uint64_t num_frames = 0;
        Frame_t start_frame = 0_f;
        std::set<Frame_t> samples;
        
        while(!sentinel.terminate()
              && timer.elapsed() < 1
              && pixel_values.size() < 10000000
              && start_frame < length())
        {
            auto range = arange<Frame_t>{
                start_frame,
                length(),
                max(1_f, Frame_t(Frame_t::number_t(length().get() * 0.1)))
            };
            
            pv::Frame frame;
            uint64_t big_loop_size = samples.size();
            
            for (auto frameIndex : range) {
                if(frameIndex >= length())
                    continue;
                
                uint64_t prev_size = samples.size();
                samples.insert(frameIndex);
                if(samples.size() == prev_size)
                    continue;
                
                read_frame(frame, frameIndex);
                for(uint64_t i=0; i<frame.n(); ++i) {
                    pixel_values.insert(pixel_values.end(), frame.pixels().at(i)->begin(), frame.pixels().at(i)->end());
                }
                ++num_frames;
                samples.insert(frameIndex);
                
                if(timer.elapsed() >= 2)
                    break;
            }
            
            if(big_loop_size == samples.size())
                break;
            
            ++start_frame;
        }
        
        std::sort(pixel_values.begin(), pixel_values.end());
        auto p = percentile(pixel_values, percent);
        print("Took ", timer.elapsed(),"s to calculate percentiles in ",num_frames," frames.");
        //auto str = Meta::toStr(samples);
        return p;
    }

void File::set_average(const cv::Mat& average) {
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
    
    if(bool(_mode & FileMode::MODIFY)) {
        _check_opened();
        if(not is_open())
            throw U_EXCEPTION("Not open for modifying.");
        
        cmn::Data::write_data(header()._average_offset, header().average->size(), (char*)header().average->data());
    }
}
    
    double File::generate_average_tdelta() {
        _check_opened();
        
        if(!_open_for_writing && _header.average_tdelta == 0 && length().valid()) {
            // readable
            double average = 0;
            uint64_t samples = 0;
            const Frame_t step = max(1_f, (length().try_sub(1_f)) / 10_f);
            pv::Frame frame;
            for (Frame_t i=1_f; i<length(); i+=step) {
                read_frame(frame, i);
                double stamp = frame.timestamp().get();
                if(i < length()-1_f) {
                    read_frame(frame, i+1_f);
                    stamp = double(frame.timestamp().get()) - stamp;
                }
                average += stamp;
                ++samples;
            }
            average /= double(samples);
            header().average_tdelta = average;
        }
        
        return _header.average_tdelta;
    }
    
    timestamp_t File::timestamp(Frame_t frameIndex, cmn::source_location loc) const {
        if(_open_for_writing)
            throw _U_EXCEPTION(loc, "Cannot get timestamps for video while writing.");
        
        if(frameIndex >= Frame_t(header().num_frames))
            throw _U_EXCEPTION(loc, "Access out of bounds ",frameIndex,"/",header().num_frames,".");
        
        return header().index_table[frameIndex.get()];
    }
    
    timestamp_t File::start_timestamp() const {
        if(_open_for_writing)
            throw U_EXCEPTION("Cannot get timestamps for video while writing.");
        
        return header().timestamp;
    }
    
    std::string File::toStr() const {
        return "pv::File<"+filename().str()+">";
    }
}
