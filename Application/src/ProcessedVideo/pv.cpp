#include "pv.h"
#include <minilzo.h>
#include <sys/stat.h>
#include <misc/GlobalSettings.h>
#include <misc/Timer.h>
#include <misc/PVBlob.h>
#include <misc/ranges.h>
#include <misc/SpriteMap.h>
#include <file/DataLocation.h>

/**
 * =============================
 * ProcessedVideo IMPLEMENTATION
 * -----------------------------
 **/

namespace cmn {
using namespace cmn::file;

template<> void Data::read(blob::Prediction& pred) {
    auto version = (pv::Version)pred.clid;
    
    read<uint8_t>(pred.clid);
    read<uint8_t>(pred.p);
    
    if(version >= pv::Version::V_10) {
        uint8_t N{0u};
        read<uint8_t>(N);
        assert(N % 2u == 0);
        //Print("pose::N = ", N / 2u);
        pred.pose.points.resize(N / 2u);
        for(size_t i=0; i<N / 2u; i++) {
            read<uint16_t>(pred.pose.points[i].x);
            read<uint16_t>(pred.pose.points[i].y);
        }
        
        if(version >= pv::Version::V_11) {
            /// read the outline points
            std::vector<int32_t> result;
            read<uint8_t>(N);
            for(size_t i=0; i<N; ++i) {
                uint32_t M;
                read<uint32_t>(M);
                result.resize(M);
                read_data(M * sizeof(int32_t), (char*)result.data());
                pred.outlines.lines.emplace_back(blob::SegmentedOutlines::Outline{
                    ._points = std::move(result)
                });
            }
        }
        
        if(version >= pv::Version::V_13) {
            uint32_t N;
            read<uint32_t>(N);
            if(N > 0) {
                std::vector<int32_t> result{N, NoInitializeAllocator<int32_t>()};
                read_data(N * sizeof(int32_t), (char*)result.data());
                
                pred.outlines.original_outline = blob::SegmentedOutlines::Outline{
                    ._points = std::move(result)
                };
            }
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
    
    write<uint8_t>(narrow_cast<uint8_t>(val.pose.size() * 2, tag::fail_on_error{}));
    for(auto &pt : val.pose.points) {
        write<uint16_t>(pt.x);
        write<uint16_t>(pt.y);
    }
    
    write<uint8_t>(val.outlines.lines.size());
    for(auto &line : val.outlines.lines) {
        write<uint32_t>(narrow_cast<uint32_t>(line._points.size(), tag::fail_on_error{}));
        write_data(line._points.size() * sizeof(int32_t), (char*)line._points.data());
    }
    
    if(val.outlines.original_outline.has_value()) {
        auto& line = *val.outlines.original_outline;
        write<uint32_t>(narrow_cast<uint32_t>(line._points.size(), tag::fail_on_error{}));
        write_data(line._points.size() * sizeof(int32_t), (char*)line._points.data());
        
    } else {
        write<uint32_t>(0);
    }
    
    return pos;
}

}

namespace pv {
    // used to register for global settings updates
    static std::atomic_bool use_differences(false);
    static sprite::CallbackFuture _callback;

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
      _real_color_average(std::move(other._real_color_average)),
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

File::File(const file::Path& filename, FileMode mode, std::optional<meta_encoding_t::Class> encoding)
    : DataFormat(filename.add_extension("pv"), filename.str()),
        _header(filename.str(), *encoding),
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
          _encoding(other._encoding),
          _flags(other._flags),
          _predictions(other._predictions)
    {
        _mask.reserve(other.n());
        if(other.encoding() != meta_encoding_t::binary)
            _pixels.reserve(other.n());
        
        //! perform deep-copy
        for (size_t i=0; i<other.n(); ++i) {
            _mask.emplace_back(new blob::line_ptr_t::element_type(*other._mask[i]));
            if(other.encoding() != meta_encoding_t::binary)
                _pixels.emplace_back(new blob::pixel_ptr_t::element_type(*other._pixels[i]));
        }
    }

    Frame::Frame(const timestamp_t& timestamp, decltype(_n) n, cmn::meta_encoding_t::Class e)
        : _timestamp(timestamp), _encoding(e)
    {
        _mask.reserve(n);
        if(_encoding != meta_encoding_t::binary)
            _pixels.reserve(n);
        _flags.reserve(n);
    }
    
    Frame::Frame(File& ref, Frame_t idx) {
        read_from(ref, idx, ref.color_mode());
    }

    std::unique_ptr<pv::Blob> Frame::blob_at(size_t i) const {
        return std::make_unique<pv::Blob>(*_mask[i],
                encoding() != meta_encoding_t::binary
                    ? *_pixels[i]
                    : std::vector<uchar>{},
                _flags[i],
                _predictions.empty()
                    ? blob::Prediction{}
                    : _predictions[i]);
    }

    std::unique_ptr<pv::Blob> Frame::steal_blob(size_t i) {
        return std::make_unique<pv::Blob>(std::move(_mask[i]),
                encoding() != meta_encoding_t::binary
                    ? std::move(_pixels[i]) : nullptr,
                _flags[i],
                _predictions.empty()
                    ? blob::Prediction{}
                    : std::move(_predictions[i]));
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


    
    void Frame::read_from(pv::File &ref, Frame_t idx, meta_encoding_t::Class mode) {
        //for(auto m: _mask)
        //    delete m;
        //for(auto p: _pixels)
        //   delete p;
        if(ref.is_write_mode())
            throw U_EXCEPTION("Cannot read from writing file ", ref);
        
        clear();
        set_index(idx);
        
        /// set number of channels
        //_channels = mode == meta_encoding_t::rgb8 ? 3 : (mode == meta_encoding_t::binary ? 0 : 1);
        _encoding = mode;
        
        Data* ptr = &ref;
        ReadonlyMemoryWrapper *compressed = NULL;
        
        static std::once_flag flag;
        std::call_once(flag, [](){
            use_differences = GlobalSettings::map().has("use_differences") ? SETTING(use_differences).value<bool>() : false;
            _callback = GlobalSettings::map().register_callbacks({"use_differences"}, [](auto) {
                use_differences = SETTING(use_differences).value<bool>();
            });
            GlobalSettings::map().register_shutdown_callback([](auto){
                _callback.collection.reset();
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
                
                ref.frame_compressed_block.resize(size, false);
                ref.frame_uncompressed_block.resize(uncompressed_size, false);
                
                ref.read_data(size, ref.frame_compressed_block.data());
                
                lzo_uint new_len;
                if(lzo1x_decompress((uchar*)ref.frame_compressed_block.data(),size,(uchar*)ref.frame_uncompressed_block.data(),&new_len,NULL) == LZO_E_OK)
                {
                    assert(new_len == uncompressed_size);
                    compressed = new ReadonlyMemoryWrapper((uchar*)ref.frame_uncompressed_block.data(), new_len);
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
        
        const auto target_channels = mode == meta_encoding_t::rgb8 ? 3 : (mode == meta_encoding_t::binary ? 0 : 1);
        
        _mask.reserve(_n);
        if(target_channels > 0)
            _pixels.reserve(_n);
        _flags.reserve(_n);
        
        for(int i=0; i<_n; i++) {
            uint16_t start_y, mask_size;
            uint8_t flags = 0;
            
            ptr->read<uint16_t>(start_y);
            if(ref.header().version >= V_8) {
                ptr->read<uint8_t>(flags);
            }
            ptr->read<uint16_t>(mask_size);
            
            if(ref.header().version < V_7) {
                ref.frame_mask_legacy.resize(mask_size);
                ref.frame_mask_cache.clear();
                ref.frame_mask_cache.reserve(ref.frame_mask_legacy.size());
                
                assert(ref.header().line_size == sizeof(LegacyShortHorizontalLine));
                
                ptr->read_data(mask_size * ref.header().line_size, (char*)ref.frame_mask_legacy.data());
                std::copy(ref.frame_mask_legacy.begin(), ref.frame_mask_legacy.end(), std::back_inserter(ref.frame_mask_cache));
                
            } else {
                ref.frame_mask_cache.resize(mask_size);
                ptr->read_data(mask_size * ref.header().line_size, (char*)ref.frame_mask_cache.data());
            }
            
            uint64_t num_pixels = 0;
            for(auto &l : ref.frame_mask_cache) {
                num_pixels += l.x1() - l.x0() + 1;
            }
            
            if(num_pixels >= std::numeric_limits<uint32_t>::max()) {
                FormatWarning("Something is happening here ", index(), " ", num_pixels, " ", uint64_t(-1));
            }
            
            const auto channels = required_storage_channels(ref.header().encoding);
            if(ref.header().encoding != meta_encoding_t::binary) {
                ref.frame_pixels.resize(num_pixels * channels, false);
                ptr->read_data(num_pixels * channels, ref.frame_pixels.data());
            }
            
            auto uncompressed = buffers().get(source_location::current());
            Header::line_type::uncompress(*uncompressed, start_y, ref.frame_mask_cache);
            
            if(use_differences && target_channels > 0) {
                uint64_t idx = 0;
                uchar *ptr = (uchar*)ref.frame_pixels.data();
                for (auto &l : *uncompressed) {
                    for (int x=l.x0; x<=l.x1; x++) {
                        auto &p = ptr[idx++];
                        p = (uchar)saturate(int(ref.average().at<uchar>(l.y, x)) - int(p));
                    }
                }
            }
            
            _mask.emplace_back(std::move(uncompressed));
            if(target_channels > 0
               && channels == target_channels)
            {
                auto v = std::make_unique<std::vector<uchar>>((uchar*)ref.frame_pixels.data(),
                                                              (uchar*)ref.frame_pixels.data() + num_pixels * channels);
                _pixels.emplace_back(std::move(v));
                
            } else if(target_channels > 0) {
                auto v = std::make_unique<std::vector<uchar>>();
                
                call_image_mode_function(InputInfo{
                    .channels = channels,
                    .encoding = ref.header().encoding
                }, OutputInfo{
                    .channels = static_cast<uint8_t>(target_channels),
                    .encoding = mode
                }, [&]<InputInfo input, OutputInfo output, DifferenceMethod>(){
                    static_assert(is_in(input.channels, 0, 1, 3), "Only 0, 1 or 3 channels input is supported.");
                    static_assert(is_in(output.channels, 1, 3), "Only 1 or 3 channels output is supported.");
                    
                    if constexpr(input.channels == 0) {
                        v->resize(output.channels * num_pixels, 255);
                        
                    } else {
                        v->resize(output.channels * num_pixels);
                        
                        const auto istart = (uchar*)ref.frame_pixels.data();
                        const auto iend = istart + num_pixels * input.channels;
                        
                        for(auto iptr = istart, optr = v->data(); iptr < iend; iptr += input.channels, optr += output.channels)
                        {
                            assert(optr < v->data() + v->size());
                            auto value = diffable_pixel_value<input, output>(iptr);
                            write_pixel_value<output>(optr, value);
                        }
                    }
                });
                
                _pixels.emplace_back(std::move(v));
            }
            
            Blob::set_flag(flags, Blob::Flags::is_rgb, target_channels == 3);
            Blob::set_flag(flags, Blob::Flags::is_r3g3b2, mode == meta_encoding_t::r3g3b2);
            Blob::set_flag(flags, Blob::Flags::is_binary, target_channels == 0);//mode == meta_encoding_t::binary);
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
        if(pair.lines->empty()) {
            /// the blob is empty. we do not accept empty objects
#ifndef NDEBUG
            FormatWarning("Empty object passed to pv::Frame. Please dont.");
#endif
            return;
        }
        
#ifndef NDEBUG
        HorizontalLine prev = pair.lines->empty() ? HorizontalLine() : pair.lines->front();
        
        uint64_t pixel_count = 0;
        for (auto &line : *pair.lines) {
            if(!(prev == line) && !(prev < line))
                FormatWarning("Lines not properly ordered, or overlapping in x [",prev.x0,"-",prev.x1,"] < [",line.x0,"-",line.x1,"] (",prev.y,"/",line.y,").");
            prev = line;
            pixel_count += line.x1 - line.x0 + 1;
        }

        assert((pixel_count * required_storage_channels(encoding()) == 0u && not pair.pixels) || pixel_count * required_storage_channels(encoding()) == pair.pixels->size());
#endif
        
        Blob::set_flag(pair.extra_flags, Blob::Flags::is_rgb, _encoding == meta_encoding_t::rgb8);
        Blob::set_flag(pair.extra_flags, Blob::Flags::is_r3g3b2, _encoding == meta_encoding_t::r3g3b2);
        Blob::set_flag(pair.extra_flags, Blob::Flags::is_binary, _encoding == meta_encoding_t::binary);
        
        _mask.emplace_back(std::move(pair.lines));
        if(pair.pixels && encoding() != meta_encoding_t::binary)
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
    if(mask.empty()) {
        /// the blob is empty. we do not accept empty objects
#ifndef NDEBUG
        FormatWarning("Empty object passed to pv::Frame. Please dont.");
#endif
        return;
    }
    
#ifndef NDEBUG
    HorizontalLine prev = mask.empty() ? HorizontalLine() : mask.front();

    uint64_t pixel_count = 0;
    for (auto& line : mask) {
        if (!(prev == line) && !(prev < line))
            FormatWarning("Lines not properly ordered, or overlapping in x [", prev.x0, "-", prev.x1, "] < [", line.x0, "-", line.x1, "] (", prev.y, "/", line.y, ").");
        prev = line;
        pixel_count += line.x1 - line.x0 + 1;
    }

    if(auto c = required_storage_channels(encoding());
       c > 0)
    {
        assert(pixel_count * c == pixels.size());
    } else {
        assert(pixel_count == pixels.size());
    }
#endif
    
    Blob::set_flag(flags, Blob::Flags::is_rgb, _encoding == meta_encoding_t::rgb8);
    Blob::set_flag(flags, Blob::Flags::is_r3g3b2, _encoding == meta_encoding_t::r3g3b2);
    Blob::set_flag(flags, Blob::Flags::is_binary, _encoding == meta_encoding_t::binary);

    _mask.emplace_back(new blob::line_ptr_t::element_type(mask));
    if(_encoding != meta_encoding_t::binary)
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
        if(mask_.empty()) {
            /// the blob is empty. we do not accept empty objects
    #ifndef NDEBUG
            FormatWarning("Empty object passed to pv::Frame. Please dont.");
    #endif
            return;
        }
        
        const auto channels = required_storage_channels(encoding());
        //const auto input_channels = full_image.channels();
        assert(full_image.channels() == required_image_channels(encoding()));
        
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
        pixels->resize(overall * channels);
        //uchar *pixels = (uchar*)malloc(overall);
        
        auto pixel_ptr = pixels->data();
        line_ptr = mask->data(); // reset ptr
        L = mask->size();
        
        for (ptr_safe_t i=0; i<L; i++, line_ptr++) {
            auto ptr = full_image.ptr(line_ptr->y, line_ptr->x0);
            assert(line_ptr->x1 >= line_ptr->x0);
            auto N = (ptr_safe_t(line_ptr->x1) - ptr_safe_t(line_ptr->x0) + ptr_safe_t(1)) * channels;
            memcpy(pixel_ptr, ptr, sign_cast<size_t>(N));
            pixel_ptr += N;
        }
        
        Blob::set_flag(flags, Blob::Flags::is_rgb, _encoding == meta_encoding_t::rgb8);
        Blob::set_flag(flags, Blob::Flags::is_r3g3b2, _encoding == meta_encoding_t::r3g3b2);
        Blob::set_flag(flags, Blob::Flags::is_binary, _encoding == meta_encoding_t::binary);
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
            auto flags = _flags.at(i);
            
            assert(not Blob::is_flag(flags, Blob::Flags::is_rgb) || _encoding == meta_encoding_t::rgb8);
            assert(not Blob::is_flag(flags, Blob::Flags::is_binary) || _encoding == meta_encoding_t::binary);
            assert(not Blob::is_flag(flags, Blob::Flags::is_r3g3b2) || _encoding == meta_encoding_t::r3g3b2);
            
            auto compressed = Header::line_type::compress(*mask);
            pack.write(uint16_t(mask->empty() ? 0 : mask->front().y));
            pack.write(uint8_t(flags));
            pack.write(uint16_t(compressed.size()));
            if(not compressed.empty())
                pack.write_data(compressed.size() * elem_size, (char*)compressed.data());
            
            if(encoding() != meta_encoding_t::binary
               && not compressed.empty())
            {
                auto &pixels = _pixels.at(i);
                pack.write_data(pixels->size() * sizeof(char), (char*)pixels->data());
            }
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
        if (encoding() == meta_encoding_t::rgb8
            || pack.size() >= 15000)
        {
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
                
                static std::atomic<double> _compression_ratio = 0.0;
                static double _compression_value = 0;
                static uint32_t _compression_samples = 0;
                
                uint64_t size = out_len + sizeof(uint32_t) * 2;
                //if (size < in_len)
                {
                    if (_compression_samples > 1000) {
                        _compression_value = _compression_value / _compression_samples;
                        _compression_samples = 1;
                    }

                    _compression_value = _compression_value + size / float(in_len);
                    _compression_samples++;
                    _compression_ratio = _compression_value / double(_compression_samples);
                }

                if (size < in_len) {
                    pack.reset_offset();
                    compressed = true;

                    assert(out_len < UINT32_MAX);
                    pack.write<uint32_t>((uint32_t)out_len);
                    pack.write<uint32_t>(in_len);
                    pack.write_data(out_len, out.data());
                    
                    //Print("Compression ratio ", double(out_len) / double(in_len) * 100,"%");
                }

            }
            else {
                Print("Compression of ",pack.size()," bytes failed.");
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

        assert(required_storage_channels(_header.encoding) == _header.average->channels());
        _average = _header.average->get();
        if(_header.encoding == meta_encoding_t::r3g3b2) {
            convert_from_r3g3b2(_header.average->get(), _real_color_average);
        } else {
            _real_color_average = _average;
        }
        
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
        
        /*if(version == Version::V_2) {
            // must read settings from file before loading...
            if(!file::DataLocation::is_registered("settings"))
                throw U_EXCEPTION("You have to register a DataLocation for 'settings' before using pv files (usually the same folder the video is in + exchange the .pv name with .settings).");
            auto settings_file = file::DataLocation::parse("settings");
            if (settings_file.exists())
                GlobalSettings::load_from_file({}, settings_file.str(), AccessLevelType::PUBLIC);
        }*/
        
        if(version >= Version::V_14) {
            std::string encoding_name;
            ref.read<std::string>(encoding_name);
            encoding = meta_encoding_t::Class::fromStr(encoding_name);
            
        } else {
            uchar channels;
            ref.read<uchar>(channels);
            
            if(version >= Version::V_12) {
                uint8_t index;
                ref.read<uint8_t>(index);
                if(index < meta_encoding_t::values.size()) {
                    encoding = meta_encoding_t::values.at(index);
                    
                    if(required_storage_channels(encoding) != channels)
                        throw InvalidArgumentException("Read illegal number of channels (",channels,") for encoding ", encoding,".");
                    
                } else {
                    FormatExcept("Read illegal encoding from file: ", index, " with available values ", meta_encoding_t::values);
                }
            } else {
                encoding = meta_encoding_t::gray;
            }
        }
        
        ref.read<cv::Size>(resolution);
        
        // added offsets
        if (version >= Version::V_3) {
            ref.read(offsets);
            
        } else if(version == Version::V_2) {
            offsets = GlobalSettings::has("crop_offsets") ? SETTING(crop_offsets) : CropOffsets();
        }
        
        if (version >= Version::V_15) {
            int64_t start, end;
            ref.read(start);
            ref.read(end);
            
            if(start < 0)
                conversion_range.start.reset();
            else
                conversion_range.start = narrow_cast<uint32_t>(start);
            
            if(end < 0)
                conversion_range.end.reset();
            else
                conversion_range.end = narrow_cast<uint32_t>(end);
            
            std::string src;
            ref.read(src);
            source = src;
            
        } else {
            conversion_range.start = std::nullopt;
            conversion_range.end = std::nullopt;
            
            source = {};
        }
        
        ref.read(line_size);
        if(line_size != sizeof(line_type))
            throw U_EXCEPTION("The used line format in this file (",line_size," bytes) differs from the expected ",sizeof(line_type)," bytes.");
        
        _num_frames_offset = ref.current_offset();
        ref.read(num_frames);
        _index_offset = ref.current_offset();
        ref.read(index_offset);
        _timestamp_offset = ref.current_offset();
        ref.read(timestamp);
        
        ref.read<std::string>(name);
        
        // check values for sanity
        //if(channels != 1 && channels != 3)
        //    throw U_EXCEPTION("Only 1 or 3 channel(s) are currently supported (",this->channels," provided)");
        
        if(average)
            delete average;
        
        const auto storage_channels = required_storage_channels(encoding);
        average = new Image((uint)this->resolution.height, (uint)this->resolution.width, storage_channels);
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
                mask = new Image((uint)this->resolution.height /*- (offsets.y + offsets.height)*/, (uint)this->resolution.width /*- (offsets.x + offsets.width)*/, 1);
                
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
            
            std::string metadata;
            ref.read<std::string>(metadata, _meta_offset);
            
            try {
                sprite::Map map;
                map.set_print_by_default(false);
                map["quiet"] = true;
                map["meta_real_width"] = Float2_t();
                
                if(not metadata.empty())
                    sprite::parse_values(sprite::MapSource{ ref.filename() }, map, metadata, &GlobalSettings::map(), {}, {});
                
                if(map.has("meta_real_width"))
                    meta_real_width = map["meta_real_width"].value<Float2_t>();
                if(not map.has("cm_per_pixel")
                   && version <= Version::V_14)
                {
                    if(meta_real_width == 0)
                        meta_real_width = 30;
                    
                    Float2_t cm_per_pixel = meta_real_width / resolution.width;
                    Print("Missing the `cm_per_pixel` key, adding it based on ", meta_real_width, " and ", resolution, " => ", cm_per_pixel);
                    map["cm_per_pixel"] = cm_per_pixel;
                }
                
                try {
                    std::map<std::string, std::string> jsons;
                    for(const auto& key : map.keys()) {
                        try {
                            jsons.emplace(key, map.at(key).get().valueString());
                            
                        } catch(const std::exception& ex) {
                            FormatWarning("[set_metadata] Cannot convert ", key, " to json properly.");
                        }
                    }
                    
                    std::string dump = Meta::toStr(jsons);
                    this->metadata = dump;
                    
                } catch(...) {
                    FormatWarning("[set_metadata] There was some trouble updating the metadata. Using the original one from the PV file.");
                    this->metadata = metadata;
                }
                /*for(auto key : map.keys()) {
                 Print("Key: ", key, " Value: ", map[key].get().valueString());
                 }*/
                
            } catch(const std::exception& ex) {
                FormatExcept("Error parsing settings metadata from ", ref.filename(), ": ", ex.what());
            } catch(...) {
                FormatExcept("Error parsing settings metadata from ", ref.filename(), ".");
            }
        }
    }

    void Header::write(DataFormat& ref) {
        /**
         * Writes the PV file header and associated metadata.
         *
         * [HEADER SECTION]
         *   (string) "PV" + (version_nr)
         *   (string) encoding                  // e.g. "gray", "rgb8", etc.
         *   (cv::Size) resolution (width,height)
         *   (4 x uint16_t) crop offsets        // left, top, right, bottom
         *   (int64_t) conversion_range_start   // or -1 if not used
         *   (int64_t) conversion_range_end     // or -1 if not used
         *   (string) the original source that was used
         *   (uchar) line_size                  // size of horizontal line struct
         *   (uint32_t) num_frames (0 initially, updated later)
         *   (uint64_t) index_offset (updated later)
         *   (uint64_t) timestamp (microseconds since 1970)
         *   (string) project name
         *   (byte*) average image data (width*height*channels)
         *   (uint64_t) mask_size (if 0, no mask)
         *   [byte*] mask data if mask_size > 0
         *
         * [DATA SECTION - per frame]
         *   (uchar) compression_flag (0 or 1)
         *   If compression_flag:
         *       (uint32_t) compressed_size
         *       (uint32_t) uncompressed_size
         *       (byte*) compressed frame data (LZO)
         *   Else:
         *       (timestamp_t) frame_timestamp (relative to start)
         *       (uint16_t) number_of_objects
         *       (int32_t) source_frame_index or -1
         *       For each object:
         *           (uint16_t) start_y
         *           (uint8_t) flags
         *           (uint16_t) number_of_mask_lines
         *           (byte*) mask lines (line_size * number_of_mask_lines)
         *           (byte*) pixel data (if encoding includes pixels)
         *       (uint16_t) number_of_predictions
         *       [Prediction data for each object if present]
         *
         * [INDEX TABLE]
         *   For each frame:
         *       (uint64_t) offset in file where frame starts
         *
         * [METADATA]
         *   (string) JSON-formatted metadata
         */
        
        // set offsets from global settings
        //if(average)
        //    tf::imshow("average", average->get());
        
        //if(not correct_number_channels(encoding, channels))
        //    throw InvalidArgumentException("Writing illegal number of channels (",channels,") for encoding ", encoding,".");
        
        ref.write("PV" + std::to_string((int)Version::current + 1));
        
        ref.write<std::string>(encoding.name());
        
        if(!resolution.width && !resolution.height)
            throw InvalidArgumentException("Resolution of the video has not been set.");
        
        ref.write(resolution);
        ref.write(offsets);
        
        // write conversion range if applicable
        ref.write(conversion_range.start
                  ? int64_t{conversion_range.start.value()}
                  : int64_t{-1});
        ref.write(conversion_range.end
                  ? int64_t{conversion_range.end.value()}
                  : int64_t{-1});
        
        std::string src;
        if(source)
            src = source.value();
        ref.write(src);
        
        ref.write(line_size);
        _num_frames_offset = ref.write(decltype(this->num_frames)(0));
        _index_offset = ref.write(decltype(index_offset)(0));
        _timestamp_offset = ref.write(timestamp);
        
        ref.write<std::string>((std::string)file::Path(name).filename());
        
        if(average) {
            if(required_storage_channels(encoding) != average->channels()) {
                throw InvalidArgumentException("Number of channels ",average->channels()," must match the encoding format ", encoding," (",required_storage_channels(encoding)," channels) for the average image provided ", *average, " (", average->channels()," channels).");
            }
            
            if(cv::Size(average->cols, average->rows) != resolution) {
                throw InvalidArgumentException("Wrong resolution for average image ", average->cols,"x", average->rows, " vs. ", resolution,".");
            }
            
            _average_offset = ref.write_data(average->size(), (char*)average->data());
        } else {
            Image tmp((uint)resolution.height, (uint)resolution.width, required_storage_channels(encoding));
            _average_offset = ref.write_data(tmp.size(), (char*)tmp.data());
        }
        
        if(mask) {
            ref.write(uint64_t(mask->size()));
            ref.write_data(mask->size(), (char*)mask->data());
            Print("Written mask with ", mask->cols,"x",mask->rows);
        }
        else {
            ref.write(uint64_t(0));
        }
    }

    void File::set_metadata(const sprite::Map &diff) {
        std::map<std::string, std::string> jsons;
        for(const auto& key : diff.keys()) {
            try {
                jsons.emplace(key, diff.at(key).get().valueString());
                
            } catch(const std::exception& ex) {
                FormatWarning("[set_metadata] Cannot convert ", key, " to json properly.");
            }
        }
        
        std::string dump = Meta::toStr(jsons);//glz::write_json(jsons);
        _header.metadata = dump;
    }

    void Header::update(DataFormat& ref) {
        // write index table
        index_offset = ref.current_offset();
        Print("Index table is ",FileSize(index_table.size() * sizeof(decltype(index_table)::value_type))," big.");
        
        for (auto index : index_table) {
            ref.write<decltype(index_table)::value_type>(index);
        }
        
        //metadata = generate_metadata();
        if(metadata.has_value())
            _meta_offset = ref.write(metadata.value());
        else
            _meta_offset = ref.write(std::string("{}"));
        
        ref.write(this->num_frames, _num_frames_offset);
        ref.write(this->index_offset, _index_offset);
        ref.write(this->timestamp, _timestamp_offset);
        
        if(average) {
            if(required_storage_channels(encoding) != average->channels()) {
                throw InvalidArgumentException("Number of channels ",average->channels()," must match the encoding format ", encoding," (",required_storage_channels(encoding)," channels) for the average image provided ", *average, " (", average->channels()," channels).");
            }
            
            if(cv::Size(average->cols, average->rows) != resolution) {
                throw InvalidArgumentException("Wrong resolution for average image ", average->cols,"x", average->rows, " vs. ", resolution,".");
            }
            
            ref.Data::write_data(_average_offset, average->size(), (char*)average->data());
        }
        
        Print("Updated number of frames with ",this->num_frames,", index offset ",this->index_offset,", timestamp ",this->timestamp,", ", _meta_offset);
    }
    
    /*std::string Header::generate_metadata() const {
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
            Print("Metadata empty.");
        } else {
            ret = "{"+ret+"}";
            Print("Metadata: ", no_quotes(ret));
        }
        
        return ret;
    }*/

const cv::Size& File::size() const {
    std::unique_lock lock(_lock);
    return _header.resolution;
}
Frame_t File::length() const {
    std::unique_lock lock(_lock);
    return Frame_t(_header.num_frames);
}
    
    Header& File::header() {
        if(not bool(_mode & FileMode::WRITE))
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
        std::chrono::microseconds ns_l{0}, ns_e{0};
        
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
        ss << "<b>Time of recording:</b> " << std::put_time(std::localtime(&now_c), "%c")
        << std::endl;
#endif
        
        ss << "<b>Length of recording:</b> ";
        duration_to_string(ss, ns_e);
        ss << " (" << Meta::toStr(length()) << " frames)" << std::endl;
        
        ss << "<b>Video conversion offsets:</b> ";
        if (_header.conversion_range.start.has_value() || _header.conversion_range.end.has_value()) {
            ss << "start="
               << (_header.conversion_range.start ? Meta::toStr(_header.conversion_range.start.value()) : "N/A")
               << ", end="
               << (_header.conversion_range.end ? Meta::toStr(_header.conversion_range.end.value()) : "N/A")
               << std::endl;
        } else {
            ss << "N/A" << std::endl;
        }

        ss << "<b>Video source:</b> ";
        if (_header.source.has_value()) {
            ss << utils::ShortenText(_header.source.value(), 1000) << std::endl;
        } else {
            ss << "unknown" << std::endl;
        }
        
        ss << "<b>Framerate:</b> " << std::setw(0) << framerate() << "fps (" << float(_header.average_tdelta) / 1000.f << "ms)";
        
        ss << std::endl;
        
        ss << std::endl;
        
        if(full) {
            if(not header().metadata.has_value())
                ss << ("<b>Metadata empty.</b>");
            else {
                sprite::Map map;
                map.set_print_by_default(false);
                try {
                    sprite::parse_values(sprite::MapSource{ filename() }, map, header().metadata.value(), &GlobalSettings::map(), {}, {});
                } catch(...) {
#ifndef NDEBUG
                    FormatWarning("Cannot parse metadata string: ", no_quotes(header().metadata.value()));
#endif
                }
                
                ss << "<b>Metadata:</b> {";
                bool first = true;
                for(auto key : map.keys()) {
                    if(not first)
                        ss << ",";
                    ss << "'" << key << "':" << utils::ShortenText(map.at(key).get().valueString(), 1000);
                    first = false;
                }
                ss << "}";
                
                return ss.str();
            }
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
        
#ifndef NDEBUG
        const auto channels = required_storage_channels(frame.encoding());
        if(channels > 0) {
            for(size_t i = 0; i < frame.n(); ++i) {
                assert(frame.pixels().at(i) && frame.pixels().at(i)->size() % channels == 0);
            }
        } else {
            assert(frame.pixels().empty());
        }
#endif

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

    void File::add_individual(const Frame& frame) {
        static auto pack_mutex = LOGGED_MUTEX("File::pack_mutex");
        static DataPackage pack;

        auto g = LOGGED_LOCK(pack_mutex);
        _check_opened();
        bool compressed;
        frame.serialize(pack, compressed);
        add_individual(frame, pack, compressed);
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
        read_frame(frame, frameIndex, color_mode());
    }

    meta_encoding_t::Class File::color_mode() const {
        assert(is_open());
        //assert(is_in(_header.channels, 3, 1));
        return _header.encoding;
    }

    void File::read_frame(Frame& frame, Frame_t frameIndex, meta_encoding_t::Class mode) {
        _check_opened();
        
        std::unique_lock<std::mutex> guard(_lock);
        
        assert(!_open_for_writing);
        assert(frameIndex.valid());
        
        if(frameIndex.get() >= _header.num_frames)
           throw U_EXCEPTION("Frame index ", frameIndex," out of range.");
        
        uint64_t pos = _header.index_table.at(frameIndex.get());
        uint64_t old = current_offset();
        
        if(old != pos)
            seek(pos);
        
        frame.read_from(*this, frameIndex, mode);
    }
    
    void File::read_next_frame(Frame& frame, Frame_t frame_to_read) {
        _check_opened();
        assert(!_open_for_writing);
        
        std::unique_lock<std::mutex> guard(_lock);
        if(frame_to_read.get() >= _header.num_frames)
           throw U_EXCEPTION("Frame index ", frame_to_read," out of range.");
        
        frame.read_from(*this, frame_to_read, color_mode());
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
    
    void File::frame_optional_background(Frame_t frameIndex, cv::Mat& output_image, bool with_background) {
        _check_opened();
        
        Frame frame;
        read_frame(frame, frameIndex);
        
        const int channels = required_image_channels(header().encoding);
        if(with_background)
            average().copyTo(output_image);
        else
            output_image = cv::Mat::zeros(header().resolution.height, header().resolution.width, CV_8UC(channels));
        assert(output_image.channels() == channels);
        
        for (uint16_t i=0; i<frame.n(); i++) {
            uint64_t index = 0;
            auto &mask = frame.mask().at(i);
            auto pixels = frame.pixels().empty() ? nullptr : frame.pixels().at(i).get();
            
            call_image_mode_function(InputInfo{
                .channels = required_storage_channels(header().encoding),
                .encoding = header().encoding
            }, OutputInfo{
                .channels = static_cast<uint8_t>(channels),
                .encoding = channels == 1 ? meta_encoding_t::gray : meta_encoding_t::rgb8
            }, [&]<InputInfo input, OutputInfo output, DifferenceMethod>(){
                static_assert(is_in(input.channels, 0, 1, 3), "Only 0, 1 or 3 channels input is supported.");
                static_assert(is_in(output.channels, 1, 3), "Only 1 or 3 channels output is supported.");
                
                auto istart = pixels ? (uchar*)pixels->data() : nullptr;
                
                for(const HorizontalLine &line : *mask) {
                    if constexpr(input.channels == 0) {
                        auto ostart = output_image.data + (line.y * output_image.cols + line.x0) * output.channels;
                        auto oend = output_image.data + (line.y * output_image.cols + line.x1 + 1) * output.channels;
                        std::fill(ostart, oend, 255);
                        
                    } else {
                        const auto iend = istart + (ptr_safe_t(line.x1) - ptr_safe_t(line.x0) + 1) * input.channels;
                        auto optr = output_image.data + (line.y * output_image.cols + line.x0) * output.channels;
                        
                        for(auto iptr = istart;
                            iptr < iend;
                            iptr += input.channels, optr += output.channels)
                        {
                            assert(optr < output_image.data + output_image.cols * output_image.rows * output_image.channels());
                            auto value = diffable_pixel_value<input, output>(iptr);
                            write_pixel_value<output>(optr, value);
                        }
                        
                        istart = iend;
                    }
                }
            });
            
            
        }
    }

    const cv::Mat& File::average() const {
        _check_opened();
        assert(_header.average);
        return _real_color_average;
    }

    void fix_file(File& file) {
        Print("Starting file copy and fix (",file.filename(),")...");
        
        auto copy = File::Write<FileMode::WRITE | FileMode::OVERWRITE>(
            (std::string)file.filename()+"_fix",
            file.header().encoding
        );
        copy.set_resolution(file.header().resolution);
        copy.set_offsets(file.crop_offsets());
        copy.set_average(file.average());
        
        auto keys = file.header().metadata.has_value()
            ? sprite::parse_values(sprite::MapSource{ file.filename() }, file.header().metadata.value()).keys()
            : std::vector<std::string>{};
        if(file.header().metadata.has_value())
            sprite::parse_values(sprite::MapSource{ file.filename() }, GlobalSettings::map(), file.header().metadata.value(), nullptr, {}, {});
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
                Print("Frame ", idx," / ", file.length()," (",copy.compression_ratio() * 100,"% compression ratio)...");
            }
        }
        
        copy.close();
        
        Print("Written fixed file.");
    }
    
    void File::try_compress() {
        _check_opened();
        
        auto copy = File::Write((std::string)filename()+"_test", header().encoding);
        copy.set_resolution(header().resolution);
        copy.set_offsets(crop_offsets());
        copy.set_average(average());
        copy.header().timestamp = header().timestamp;
        
        if(has_mask())
            copy.set_mask(mask());
        
        copy.start_writing(true);
        
        std::vector<std::string> save = GlobalSettings::map().has("meta_write_these") ? SETTING(meta_write_these) : std::vector<std::string>{};
        
        sprite::Map map;
        if(header().metadata.has_value())
            sprite::parse_values(sprite::MapSource{ filename() }, map, header().metadata.value(), nullptr, {}, {});
        SETTING(meta_write_these) = map.keys();
        
        pv::Frame frame;
        for (Frame_t i=0_f; i<length(); ++i) {
            read_frame(frame, i);
            frame.set_timestamp(header().timestamp + frame.timestamp());
            
            copy.add_individual(std::move(frame));
            
            if (i.get() % 1000 == 0) {
                Print("Frame ", i," / ",length(),"...");
            }
        }
        
        copy.stop_writing();
        
        Print("Written");
        
        {
            print_info();
            
            auto test = File::Read((std::string)filename()+"_test");
            test.start_reading();
        }
        
        SETTING(meta_write_these) = save;
    }
    
    void File::update_metadata() {
        _check_opened();
        
        if(not bool(_mode & FileMode::MODIFY)
           || not is_open())
            throw U_EXCEPTION("Must be open for writing.");
    
        if(_header.metadata.has_value()) {
            Print("Updating metadata...");
            //auto metadata = _header.generate_metadata();
            write(_header.metadata.value(), _header.meta_offset());
        }
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
                    if(frame.encoding() == meta_encoding_t::binary)
                        pixel_values.push_back(255);
                    else
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
        Print("Took ", timer.elapsed(),"s to calculate percentiles in ",num_frames," frames.");
        //auto str = Meta::toStr(samples);
        return p;
    }

void File::set_average(const cv::Mat& average) {
    //tf::imshow("average", average);
    
    if(required_storage_channels(_header.encoding) != average.channels()) {
        throw InvalidArgumentException("Number of channels ",average.channels()," must match the encoding format ", _header.encoding," for the average image provided.");
    }
    
    if(average.type() != CV_8UC(required_storage_channels(_header.encoding)))
    {
        auto str = getImgType(average.type());
        throw InvalidArgumentException("Average image is of type ",str," != 'CV_8UC",required_storage_channels(_header.encoding),"'.");
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
    
    if(_header.encoding == meta_encoding_t::r3g3b2) {
        _real_color_average = cv::Mat::zeros(_header.resolution.height, _header.resolution.width, CV_8UC3);
        convert_from_r3g3b2(_average, _real_color_average);
    } else {
        _real_color_average = _average;
    }
    
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
