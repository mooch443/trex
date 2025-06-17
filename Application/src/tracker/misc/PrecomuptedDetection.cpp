#include "PrecomuptedDetection.h"
#include <python/TileBuffers.h>
#include <misc/Timer.h>
#include <misc/SizeFilters.h>
#include <misc/TrackingSettings.h>
#include <file/CSVReader.h>

namespace track {

struct PrecomputedDetection::Data {
    gpuMat _gpu;
    gpuMat _float_average;
    std::optional<Background> _background;
    
    std::optional<std::future<tl::expected<PrecomputedDetectionCache, std::string>>> _frame_data_loader;
    
    
    void set(Image::Ptr&&, meta_encoding_t::Class);
    
    bool has_background() const {
        std::shared_lock guard(_background_mutex);
        return _background.has_value();
    }
    void set_background(Image::Ptr&& background, meta_encoding_t::Class meta_encoding) {
        std::unique_lock guard(_background_mutex);
        _background = Background(std::move(background), meta_encoding);
    }
    
    std::shared_mutex _data_mutex;
    std::optional<file::PathArray> _filename;
    
    std::optional<PrecomputedDetectionCache> _frame_data_cache;
    
    void set(file::PathArray&& filename) {
        std::unique_lock guard{_data_mutex};
        if(_filename != filename) {
            _filename = std::move(filename);
            _frame_data_cache.reset();
            
            if(_frame_data_loader.has_value()
               && _frame_data_loader->valid())
            {
                try {
                    _frame_data_loader->get();
                } catch(...) {
                    FormatExcept("Failed to load frame_data that was still in the queue.");
                }
            }
            
            _frame_data_loader = std::async(std::launch::async, [this]()
            {
                return preload_file();
            });
        }
    }
    
    double _time{0.0}, _samples{0.0};
    mutable std::shared_mutex _time_mutex, _background_mutex, _gpu_mutex;
    
    double fps() {
        std::shared_lock guard(_time_mutex);
        if(_samples == 0)
            return 0;
        return _time / _samples;
    }
    void add_time_sample(double sample) {
        std::unique_lock guard(_time_mutex);
        _time += sample;
        _samples++;
    }
    
    tl::expected<PrecomputedDetectionCache, std::string> preload_file();
};



PrecomputedDetectionCache::PrecomputedDetectionCache(const file::Path& csv_path)
  : _cache_path(csv_path.replace_extension("pdcache")), _df(_cache_path)
{
    uint64_t csv_hash = computeFileHash(csv_path);

    bool needs_rebuild = true;
    if (_cache_path.exists()) {
        // open header to check hash
        try {
            cmn::DataFormat tmp(_cache_path);
            tmp.start_reading();
            Header hdr;
            tmp.read(hdr);
            tmp.close();
            if (std::memcmp(hdr.magic.parts.tag,"PDC",3)==0 &&
                hdr.magic.parts.ver==1 &&
                hdr.file_hash==csv_hash) {
                needs_rebuild = false;
            }
        } catch(...) {
            /// rebuild it
        }
    }
    if (needs_rebuild) {
        buildCache(csv_path, _cache_path);
    }

    _df.start_reading();
    _df.hint_access_pattern(cmn::DataFormat::AccessPattern::Sequential);
    _map_ptr  = _df.data();
    _map_size = _df.reading_file_size();
    assert(_df.supports_fast());

    const Header* hdr = reinterpret_cast<const Header*>(_map_ptr);
    if (std::memcmp(hdr->magic.parts.tag, "PDC", 3) != 0 ||
        hdr->magic.parts.ver != 1) {
        throw RuntimeError("Invalid cache file format");
    }
    const IndexEntry* entries = reinterpret_cast<const IndexEntry*>(_map_ptr + sizeof(Header));
    _index_entries.assign(entries, entries + hdr->index_count);

    for(auto &e : _index_entries) {
        _index_map[e.frame] = Offsets{
            .offset = e.offset,
            .count = e.count
        };
    }
}

std::optional<PrecomputedDetection::frame_data_t::mapped_type> PrecomputedDetectionCache::get_frame_data(Frame_t frame) {
    if(auto it = _index_map.find(frame);
       it != _index_map.end())
    {
        const auto& offsets = it->second;
        const char* ptr = _map_ptr + offsets.offset;

        frame_data_t::mapped_type objs;
        for (uint32_t i = 0; i < offsets.count; ++i) {
            float x = *reinterpret_cast<const float*>(ptr); ptr += sizeof(float);
            float y = *reinterpret_cast<const float*>(ptr); ptr += sizeof(float);
            float w = *reinterpret_cast<const float*>(ptr); ptr += sizeof(float);
            float h = *reinterpret_cast<const float*>(ptr); ptr += sizeof(float);
            objs.emplace_back(x, y, w, h);
        }
        return objs;
    }

    return std::nullopt;
}

// Helper: quick hash based on path & size
uint64_t PrecomputedDetectionCache::computeFileHash(const file::Path& p)
{
    uint64_t sz  = p.file_size();
    uint64_t h1  = std::hash<std::string_view>{}(std::string_view(p.c_str()));
    return (h1 ^ (sz + 0x9e3779b97f4a7c15ULL + (h1<<6) + (h1>>2)));
}

void PrecomputedDetectionCache::buildCache(const file::Path& csv_path, const file::Path& cache_path) {
    uint64_t csv_hash = computeFileHash(csv_path);
    // Detect columns
    CSVStreamReader reader1(csv_path, ',', true);
    auto header1 = reader1.header();
    std::optional<size_t> x_idx, y_idx, w_idx, h_idx, frame_idx;
    std::optional<std::string_view> x_name, y_name, w_name, h_name, frame_name;
    
    for(size_t i=0; i < header1.size(); ++i) {
        auto &name = header1[i];
        if(match_name({'x'}, name, x_name)) {
            x_idx = i;
        } else if(match_name({'y'}, name, y_name)) {
            y_idx = i;
        } else if(match_name({"w", "width"}, name, w_name)) {
            w_idx = i;
        } else if(match_name({"h", "height"}, name, h_name)) {
            h_idx = i;
        } else if(match_name({"frame"}, name, frame_name)) {
            frame_idx = i;
        }
    }
    
    if (not x_idx || not y_idx)
        throw InvalidArgumentException("Missing X or Y column in CSV");
    if (not frame_idx)
        throw InvalidArgumentException("Missing frame column in CSV");
    
    bool is_centroid = true;
    if (w_idx
        && h_idx
        && not utils::contains(utils::lowercase(*x_name), "centroid"))
    {
        is_centroid = false;
    }
    
    Size2 default_size;
    if (GlobalSettings::map().is_type<Size2>("individual_image_size"))
        default_size = SETTING(individual_image_size).value<Size2>();
    else {
        default_size = Size2(80, 80);
        FormatWarning("[precomputed] individual_image_size is not set.");
    }

    // First pass: count objects per frame
    std::unordered_map<Frame_t, uint32_t> counts;
    std::vector<std::string> row;
    while (reader1.hasNext()) {
        row = reader1.nextRow();
        
        double rf = std::stod(row[*frame_idx]);
        if (rf < 0) continue;
        counts[Frame_t{static_cast<uint32_t>(rf)}]++;
    }

    // Sort frames and prepare index entries
    std::vector<Frame_t> frames;
    frames.reserve(counts.size());
    for (auto& kv : counts) frames.push_back(kv.first);
    std::sort(frames.begin(), frames.end());

    std::vector<IndexEntry> entries;
    entries.reserve(frames.size());
    const uint64_t content_offset = sizeof(Header) + sizeof(IndexEntry) * frames.size();
    uint64_t offset = content_offset;
    for (Frame_t f : frames) {
        uint32_t cnt = counts[f];
        entries.push_back({f, offset, cnt});
        offset += uint64_t(cnt) * sizeof(float) * 4;
    }
    uint32_t index_count  = static_cast<uint32_t>(entries.size());
    uint64_t total_size   = offset;

    // Preallocate and map file
#ifdef _WIN32
    HANDLE hFile = CreateFileA(
        cache_path.c_str(),
        GENERIC_WRITE | GENERIC_READ,
        0,
        NULL,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );
    if (hFile == INVALID_HANDLE_VALUE) {
        throw RuntimeError("Cannot open cache for writing");
    }
    LARGE_INTEGER li{};
    li.QuadPart = static_cast<LONGLONG>(total_size);
    if (SetFilePointerEx(hFile, li, NULL, FILE_BEGIN) == 0 || SetEndOfFile(hFile) == 0) {
        CloseHandle(hFile);
        throw RuntimeError("Cannot size cache file");
    }
    CloseHandle(hFile);
#else
    int fd = ::open(cache_path.c_str(), O_RDWR | O_CREAT, 0666);
    if (fd < 0) throw RuntimeError("Cannot open cache for writing");
    if (::ftruncate(fd, (off_t)total_size) != 0) { ::close(fd); throw RuntimeError("Cannot size cache file"); }
    ::close(fd);
#endif

    cmn::DataFormat df(cache_path);
    df.start_writing(true);

    // Write header
    Header hdr{};
    hdr.magic.parts.tag[0] = 'P';
    hdr.magic.parts.tag[1] = 'D';
    hdr.magic.parts.tag[2] = 'C';
    hdr.magic.parts.ver    = 1;
    hdr.file_hash     = csv_hash;
    hdr.index_count   = index_count;
    
    df.write(hdr);

    // Second pass: write object data
    CSVStreamReader reader2(csv_path, ',', true);
    // Build quick lookup from frame → IndexEntry
    std::unordered_map<Frame_t, const IndexEntry*> idx_lookup;
    for (const auto& e : entries)
        idx_lookup.emplace(e.frame, &e);
    
    // Write index table
    for (auto& e : entries)
        df.write(e);
    
    assert(content_offset == df.tell());

    // Remaining objects to write per frame (initialised with the counts)
    auto remaining = counts;          // copy – we will decrement

    while (reader2.hasNext()) {
        row = reader2.nextRow();

        double rf = std::stod(row[*frame_idx]);
        if (rf < 0) continue;
        Frame_t f{static_cast<uint32_t>(rf)};

        float x = std::stof(row[*x_idx]);
        float y = std::stof(row[*y_idx]);
        float w = w_idx ? std::stof(row[*w_idx]) : float(default_size.width);
        float h = h_idx ? std::stof(row[*h_idx]) : float(default_size.height);
        if (is_centroid) { x -= w * 0.5f; y -= h * 0.5f; }

        auto it = idx_lookup.find(f);
        if (it == idx_lookup.end()) continue;        // should not happen

        const IndexEntry* idx = it->second;
        uint32_t written = idx->count - remaining[f];          // objects already written
        uint64_t pos = idx->offset + uint64_t(written) * sizeof(float) * 4;

        df.seek(pos);
        df.write<float>(x);
        df.write<float>(y);
        df.write<float>(w);
        df.write<float>(h);

        if (--remaining[f] == 0)
            remaining.erase(f);
    }
    df.stop_writing();
}

PipelineManager<TileImage, true>& PrecomputedDetection::manager() {
    static auto instance = PipelineManager<TileImage, true>(1u, [](std::vector<TileImage>&& images)
    {
        /// in background subtraction case, we have to wait until the background
        /// image has been generated and hang in the meantime.
        auto start_time = std::chrono::steady_clock::now();
        auto message_time = start_time;
        while(not data().has_background()
              && not manager().is_terminated())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            auto elapsed = std::chrono::steady_clock::now() - message_time;
            if(std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() > 30) {
                FormatExcept("Background image not set in ",
                             std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time).count(),
                             " seconds. Waiting for background image...");
                message_time = std::chrono::steady_clock::now();
            }
        }
        
        start_time = std::chrono::steady_clock::now();
        while(data()._frame_data_loader.has_value()
              && not manager().is_terminated())
        {
            if(data()._frame_data_loader.has_value()
               && data()._frame_data_loader->valid())
            {
                if(data()._frame_data_loader->wait_for(std::chrono::milliseconds(1)) == std::future_status::ready)
                {
                    auto result = data()._frame_data_loader->get();
                    if(result.has_value()) {
                        PrecomputedDetectionCache obj = std::move(result.value());
                        data()._frame_data_cache = std::move(obj);
                    } else {
                        data()._frame_data_cache.reset();
                        throw InvalidArgumentException("Cannot load file ", data()._filename, " into cache: ", no_quotes(result.error()));
                    }
                    break;
                }
                
            } else {
                /// then we remove it...
                data()._frame_data_loader.reset();
            }
            
            auto elapsed = std::chrono::steady_clock::now() - message_time;
            if(std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() > 30) {
                FormatExcept("Loading the precomputed data is taking ",
                             std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time).count(),
                             " seconds already. Waiting...");
                message_time = std::chrono::steady_clock::now();
            }
        }
        
        if(not manager().is_terminated()) {
            if(images.empty())
                FormatExcept("Images is empty :(");
            
            PrecomputedDetection::apply(std::move(images));
        }
    });
    return instance;
}

PrecomputedDetection::PrecomputedDetection(file::PathArray&& path, Image::Ptr&& average, meta_encoding_t::Class meta_encoding) {
    data().set(std::move(average), meta_encoding);
    data().set(std::move(path));
}

void PrecomputedDetection::set_background(Image::Ptr && average, meta_encoding_t::Class meta_encoding) {
    data().set(std::move(average), meta_encoding);
    if(data().has_background())
        manager().set_paused(false);
}

void PrecomputedDetection::Data::set(Image::Ptr&& average, meta_encoding_t::Class meta_encoding) {
    std::scoped_lock guard(_background_mutex, _gpu_mutex);
    Print("Setting background image to ", hex(average.get()));
    if(average)
        _background = Background(std::move(average), meta_encoding);
    else
        _background.reset();
    
    if(_background) {
        _background->image().get().copyTo(_gpu);
        _gpu.convertTo(_float_average, CV_32FC(_gpu.channels()), 1.0 / 255.0);
        manager().set_paused(false);
    }
}

PrecomputedDetection::Data& PrecomputedDetection::data() {
    static Data _data;
    return _data;
}

std::future<SegmentationData> PrecomputedDetection::apply(TileImage &&tiled) {
    if(tiled.promise)
        throw U_EXCEPTION("Tiled.promise was already set.");
    tiled.promise = std::make_unique<std::promise<SegmentationData>>();
    
    auto f = tiled.promise->get_future();
    manager().enqueue(std::move(tiled));
    return f;
}

void PrecomputedDetection::deinit() {
    std::unique_lock guard{data()._data_mutex};
    data()._filename.reset();
    data()._frame_data_cache.reset();
}

double PrecomputedDetection::fps() {
    return data().fps();
}

void convert_tile_to_rgb_or_gray(const Image::Ptr& image, cv::Mat& r3, meta_encoding_t::Class mode, const std::optional<uint8_t>& color_channel)
{
    static thread_local cv::Mat split_channels[4];
    
    if (mode == meta_encoding_t::r3g3b2) {
        if (image->dims == 3)
            convert_to_r3g3b2<3>(image->get(), r3);
        else if (image->dims == 4)
            convert_to_r3g3b2<4>(image->get(), r3);
        else
            throw U_EXCEPTION("Invalid number of channels (",image->dims,") in input image for the network.");
    }
    else if (mode == meta_encoding_t::gray
             || mode == meta_encoding_t::binary)
    {
        if(is_in(image->dims, 3, 4)) {
            if(not color_channel.has_value()
               || color_channel.value() >= 4)
            {
                if(image->dims == 3) {
                    cv::cvtColor(image->get(), r3, cv::COLOR_BGR2GRAY);
                } else /*if(image->dims == 4)*/ {
                    cv::cvtColor(image->get(), r3, cv::COLOR_BGRA2GRAY);
                }
                
            } else {
                
                cv::split(image->get(), split_channels);
                r3 = split_channels[color_channel.value()];
            }
            
        } else
            throw U_EXCEPTION("Invalid number of channels (",image->dims,") in input image for the network.");
    } else if(mode == meta_encoding_t::rgb8) {
        if(image->dims == 4)
            cv::cvtColor(image->get(), r3, cv::COLOR_BGRA2BGR);
        else
            throw U_EXCEPTION("Invalid number of channels (",image->dims,") in input image for the network.");
        
    } else
        throw U_EXCEPTION("Invalid image mode ", mode);
}

void PrecomputedDetection::apply(std::vector<TileImage> &&tiled) {
    Timer timer;
    const auto mode = Background::meta_encoding();
    const bool track_background_subtraction = Background::track_background_subtraction();
    const auto cm_per_pixel = SETTING(cm_per_pixel).value<Settings::cm_per_pixel_t>();
    const auto detect_size_filter = SETTING(detect_size_filter).value<SizeFilters>();
    const Float2_t sqcm = SQR(cm_per_pixel);
    
    const auto color_channel = SETTING(color_channel).value<std::optional<uint8_t>>();
    
    const auto detect_threshold = SETTING(detect_threshold).value<int>();
    
    cmn::OutputInfo output_format{
        .channels = required_storage_channels(mode),
        .encoding = mode
    };
    
    if(not data()._frame_data_cache.has_value()) {
        FormatWarning("Frame data is empty for precomputed detection.");
        return;
    }
    
    auto &fdata = data()._frame_data_cache.value();
    
    size_t i = 0;
    for(auto &&tile : tiled) {
        std::vector<Bounds> all_objects;
        std::vector<blob::Pair> filtered, filtered_out;
        auto frame = Frame_t{tile.data.image->index()};
        
        if(auto frame_data = fdata.get_frame_data(frame);
           frame_data)
        {
            all_objects = std::move(frame_data.value());
        }
        
        /// Collect all tiles for this frame:
        assert(tile.data.tiles.empty() || tile.data.tiles.size() == tile.images.size());
        cv::Mat r3;
        
        for(size_t tdx = 0; tdx < tile.images.size(); ++tdx)
        {
            auto& image = tile.images.at(tdx);
            convert_tile_to_rgb_or_gray(image, r3, mode, color_channel);
            
            if(r3.channels() == 0) {
                FormatExcept("The resulting image is of 0 dimensions.");
                convert_tile_to_rgb_or_gray(image, r3, mode, color_channel);
            }
            
            InputInfo input_format{
                .channels = static_cast<uint8_t>(r3.channels()),
                .encoding = r3.channels() == 1 ? meta_encoding_t::gray : meta_encoding_t::rgb8
            };
            
            if(input_format.channels == 0) {
                throw U_EXCEPTION("Empty channels");
            }
            
            assert(not r3.empty());
            Bounds bds;
            if(tile.data.tiles.size() <= tdx) {
                bds = Bounds(0_F, 0_F, Float2_t(r3.cols), Float2_t(r3.rows));
            } else {
                bds = tile.data.tiles.at(tdx);
            }
            
            for(auto it = all_objects.begin(); it != all_objects.end(); ++it)
            {
                /// check whether the top-left position is inside (instead of overlap so we avoid duplicates)
                if(bds.contains(it->pos())) {
                    /// our object is inside the current tile
                    /// lets crop it out
                    auto lines = std::make_unique<std::vector<HorizontalLine>>();
                    for(double y = it->y; y < it->y + it->height && y < image->rows; ++y)
                    {
                        lines->emplace_back(y, it->x, it->x + int64_t(it->width + 0.5) - 1);
                        
                        if(lines->back().x0 >= image->cols)
                        {
                            lines->back().x0 = image->cols - 1;
                            if(lines->back().x1 > lines->back().x0)
                            {
                                lines->back().x1 = lines->back().x0;
                            }
                        }
                        
                        if(lines->back().x1 >= image->cols)
                        {
                            lines->back().x1 = image->cols - 1;
                        }
                    }
                    
                    const uint8_t flags =
                        (mode == meta_encoding_t::rgb8 ? pv::Blob::flag(pv::Blob::Flags::is_rgb) : 0)
                        | (mode == meta_encoding_t::r3g3b2 ? pv::Blob::flag(pv::Blob::Flags::is_r3g3b2) : 0)
                        | (mode == meta_encoding_t::binary ? pv::Blob::flag(pv::Blob::Flags::is_binary) : 0);
                    
                    if(detect_threshold > 0)
                    {
                        pv::Blob blob(std::move(lines), flags);
                        /// TODO: implement some version of RawProcessing here
                        blob.set_pixels(blob.calculate_pixels(input_format, output_format, r3));
                        
                        /*if(filtered.empty()) {
                            auto [pos, img] = ptr->color_image();
                            auto mat = img->get();
                            tf::imshow("object", mat);
                        }*/

                        auto ptr = blob.threshold(detect_threshold, data()._background.value());
                        if(not ptr->empty())
                            filtered.emplace_back(std::move(ptr->steal_lines()), std::move(ptr->pixels()));
                        
                    } else {
                        /*if(filtered.empty()) {
                            blob.set_pixels(*pixels);
                            auto [pos, img] = blob.color_image();
                            auto mat = img->get();
                            tf::imshow("object", mat);
                        }*/
                        
                        if(not lines->empty()) {
                            auto pixels = pv::Blob::calculate_pixels(input_format, output_format, *lines, r3, std::nullopt);
                            filtered.emplace_back(std::move(lines), std::move(pixels));
                        }
                    }
                }
            }
        }
        
        tile.data.frame.set_encoding(mode);
        
        for (auto &&b: filtered) {
            if(b.lines->size() < UINT16_MAX) {
                /// TODO: Maybe we need to add a potential offset here?
                if(b.lines->size() < UINT16_MAX)
                    tile.data.frame.add_object(std::move(b));
                else
                    FormatWarning("Lots of lines!");
            }
            else
                Print("Probably a lot of noise with ",b.lines->size()," lines!");
        }
        
        filtered.clear();
        
        tile.promise->set_value(std::move(tile.data));
        tile.promise = nullptr;
        
        try {
            if(tile.callback)
                tile.callback();
            
        } catch(...) {
            FormatExcept("Exception for tile ", i," in package of ", tiled.size(), " TileImages.");
        }
        
        for(auto &image: tile.images) {
            buffers::TileBuffers::get().move_back(std::move(image));
        }
        tile.images.clear();
        
        ++i;
    }
    
    if(not tiled.empty()) {
        data().add_time_sample(double(tiled.size()) / timer.elapsed());
    }
}

tl::expected<PrecomputedDetectionCache, std::string> PrecomputedDetection::Data::preload_file() {
    if(not _filename.has_value())
        throw RuntimeError("No filename has been set for precomputed detection. Please set the `detect_precomputed_file` parameter.");
    
    for(auto &path : _filename.value()) {
        Print(" * Loading ", path, " precomputed data (",FileSize{path.file_size()},")...");
        
        if(path.has_extension("csv")) {
            return PrecomputedDetectionCache(path);
        }
        
        Print("* Done loading ", path,".");
    }
    
    return tl::unexpected("Was not able to load any detection cache files from "+Meta::toStr(_filename)+".");
}


}
