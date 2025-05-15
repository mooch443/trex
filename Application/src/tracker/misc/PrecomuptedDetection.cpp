#include "PrecomuptedDetection.h"
#include <python/TileBuffers.h>
#include <misc/Timer.h>
#include <misc/SizeFilters.h>
#include <misc/TrackingSettings.h>
#include <file/CSVReader.h>

namespace track {

struct PrecomputedDetection::Data {
    Image::Ptr _background;
    gpuMat _gpu;
    gpuMat _float_average;
    
    using frame_data_t = std::unordered_map<Frame_t, std::vector<Bounds>>;
    std::optional<std::future<frame_data_t>> _frame_data_loader;
    
    
    void set(Image::Ptr&&);
    
    bool has_background() const {
        std::shared_lock guard(_background_mutex);
        return _background != nullptr;
    }
    void set_background(Image::Ptr&& background) {
        std::unique_lock guard(_background_mutex);
        _background = std::move(background);
    }
    
    std::shared_mutex _data_mutex;
    std::optional<file::PathArray> _filename;
    
    std::optional<frame_data_t> _frame_data;
    
    void set(file::PathArray&& filename) {
        std::unique_lock guard{_data_mutex};
        if(_filename != filename) {
            _filename = std::move(filename);
            _frame_data.reset();
            
            std::promise<frame_data_t> promise;
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
    
    frame_data_t preload_file();
};

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
                    data()._frame_data = data()._frame_data_loader->get();
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

PrecomputedDetection::PrecomputedDetection(file::PathArray&& path, Image::Ptr&& average) {
    data().set(std::move(average));
    data().set(std::move(path));
}

void PrecomputedDetection::set_background(Image::Ptr && average) {
    data().set(std::move(average));
    if(data().has_background())
        manager().set_paused(false);
}

void PrecomputedDetection::Data::set(Image::Ptr&& average) {
    std::scoped_lock guard(_background_mutex, _gpu_mutex);
    Print("Setting background image to ", hex(average.get()));
    _background = std::move(average);
    if(_background) {
        _background->get().copyTo(_gpu);
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
    data()._frame_data.reset();
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
    
    if(not data()._frame_data.has_value()) {
        FormatWarning("Frame data is empty for precomputed detection.");
        return;
    }
    
    auto &fdata = data()._frame_data.value();
    
    size_t i = 0;
    for(auto &&tile : tiled) {
        std::vector<Bounds> all_objects;
        std::vector<blob::Pair> filtered, filtered_out;
        auto frame = Frame_t{tile.data.image->index()};
        
        if(auto frame_data = fdata.find(frame);
           frame_data != fdata.end())
        {
            all_objects = frame_data->second;
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
                    if(detect_threshold > 0) {
                        /// TODO: implement some version of RawProcessing here
                    }
                    
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
                    pv::Blob blob(*lines, flags);
                    auto pixels = blob.calculate_pixels(input_format, output_format, r3);
                    
                    /*if(filtered.empty()) {
                        blob.set_pixels(*pixels);
                        auto [pos, img] = blob.color_image();
                        auto mat = img->get();
                        tf::imshow("object", mat);
                    }*/
                    
                    filtered.emplace_back(std::move(lines), std::move(pixels));
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
    
    /*std::shared_lock guard(data()._gpu_mutex);
    RawProcessing raw(data()._gpu, &data()._float_average, nullptr);
    gpuMat gpu_buffer;
    TagCache tag;
    CPULabeling::ListCache_t cache;
    const auto cm_per_pixel = SETTING(cm_per_pixel).value<Settings::cm_per_pixel_t>();
    const auto detect_size_filter = SETTING(detect_size_filter).value<SizeFilters>();
    const Float2_t sqcm = SQR(cm_per_pixel);
    cv::Mat r3;
    
    static thread_local cv::Mat split_channels[4];
    const auto color_channel = SETTING(color_channel).value<std::optional<uint8_t>>();
    
    size_t i = 0;
    for(auto && tile : tiled) {
        try {
            std::vector<blob::Pair> filtered, filtered_out;
            
            for(auto &image : tile.images) {
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
                            } else {
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
                
                gpuMat* input = &gpu_buffer;
                r3.copyTo(*input);
                
                //apply_filters(*input);
                //Print("CHannels = ", r3.channels(), " input=", input->channels());
                //Print("size = ", Size2(r3), " input=", Size2(input->cols, input->rows), " average=",Size2(data().gpu.cols, data().gpu.rows), " channels=", data().gpu.channels());
                assert(Size2(r3) == Size2(data()._gpu.cols, data()._gpu.rows));
                raw.generate_binary(r3, *input, r3, &tag);
                
                {
                    std::vector<blob::Pair> rawblobs;
            #if defined(TAGS_ENABLE)
                    if(!GRAB_SETTINGS(tags_saved_only))
            #endif
                        rawblobs = CPULabeling::run(r3, cache, true);

                    const uint8_t flags = pv::Blob::flag(pv::Blob::Flags::is_tag)
                            | pv::Blob::flag(pv::Blob::Flags::is_instance_segmentation)
                            | (mode == meta_encoding_t::rgb8 ? pv::Blob::flag(pv::Blob::Flags::is_rgb) : 0)
                            | (mode == meta_encoding_t::r3g3b2 ? pv::Blob::flag(pv::Blob::Flags::is_r3g3b2) : 0)
                            | (mode == meta_encoding_t::binary ? pv::Blob::flag(pv::Blob::Flags::is_binary) : 0);
                    for (auto& blob : tag.tags) {
                        rawblobs.emplace_back(
                            std::make_unique<blob::line_ptr_t::element_type>(*blob->lines()),
                            std::make_unique<blob::pixel_ptr_t::element_type>(*blob->pixels()),
                            flags);
                    }

            #ifdef TGRABS_DEBUG_TIMING
                    _raw_blobs = _sub_timer.elapsed();
                    _sub_timer.reset();
            #endif
                    if(filtered.capacity() == 0) {
                        filtered.reserve(rawblobs.size() / 2);
                        filtered_out.reserve(rawblobs.size() / 2);
                    }
                    
                    size_t fidx = 0;
                    size_t fodx = 0;
                    
                    size_t Ni = filtered.size();
                    size_t No = filtered_out.size();
                    
                    for(auto  &&pair : rawblobs) {
                        auto &pixels = pair.pixels;
                        auto &lines = pair.lines;
                        
                        ptr_safe_t num_pixels;
                        if(pixels)
                            num_pixels = pixels->size();
                        else {
                            num_pixels = 0;
                            for(auto &line : *lines) {
                                num_pixels += ptr_safe_t(line.x1) - ptr_safe_t(line.x0) + ptr_safe_t(1);
                            }
                        }
                        
                        if(detect_size_filter.in_range_of_one(num_pixels * sqcm))
                        {
                            //b->calculate_moments();
                            assert(lines);
                            ++fidx;
                            if(Ni <= fidx) {
                                filtered.emplace_back(std::move(pair));
                                //task->filtered.push_back({std::move(lines), std::move(pixels)});
                                ++Ni;
                            } else {
            //                    *task->filtered[fidx].lines = std::move(*lines);
            //                    *task->filtered[fidx].pixels = std::move(*pixels);
                                //std::swap(task->filtered[fidx].lines, lines);
                                //std::swap(task->filtered[fidx].pixels, pixels);
                                filtered[fidx] = std::move(pair);
                            }
                        }
                        else {
                            assert(lines);
                            ++fodx;
                            if(No <= fodx) {
                                filtered_out.emplace_back(std::move(pair));
                                //task->filtered_out.push_back({std::move(lines), std::move(pixels)});
                                ++No;
                            } else {
            //                    *task->filtered_out[fodx].lines = std::move(*lines);
            //                    *task->filtered_out[fodx].pixels = std::move(*pixels);
            //                    std::swap(task->filtered_out[fodx].lines, lines);
            //                    std::swap(task->filtered_out[fodx].pixels, pixels);
                                filtered_out[fodx] = std::move(pair);
                            }
                        }
                    }
                    
                    filtered.reserve(fidx);
                    filtered_out.reserve(fodx);
                }
            }
            
            {
                static Timing timing("adding frame");
                TakeTiming take(timing);
                
                assert(required_storage_channels(mode) == 0 || r3.channels() == required_storage_channels(mode));
                tile.data.frame.set_encoding(mode);
                
                for (auto &&b: filtered) {
                    if(b.lines->size() < UINT16_MAX) {
                        if(b.lines->size() < UINT16_MAX)
                            tile.data.frame.add_object(std::move(b));
                        else
                            FormatWarning("Lots of lines!");
                    }
                    else
                        Print("Probably a lot of noise with ",b.lines->size()," lines!");
                }
                
                filtered.clear();
            }
            
            tile.promise->set_value(std::move(tile.data));
            tile.promise = nullptr;
            
        } catch(const std::exception& ex) {
            FormatExcept("Exception! ", ex.what());
            tile.promise->set_exception(std::current_exception());
            tile.promise = nullptr;
        }
        
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
    }*/
}

PrecomputedDetection::Data::frame_data_t PrecomputedDetection::Data::preload_file() {
    if(not _filename.has_value())
        throw RuntimeError("No filename has been set for precomputed detection. Please set the `detect_precomputed_file` parameter.");
    
    auto match_name = []<typename MatchType>(
            const std::initializer_list<MatchType>& matches,
            const auto& name,
            auto& target)
    {
        using Matches = cmn::remove_cvref_t<decltype(matches)>;
        
        bool does_match{false};
        for(auto& match : matches) {
            if constexpr(std::same_as<typename Matches::value_type, char>)
            {
                if(utils::contains(utils::lowercase(name), match))
                {
                    does_match = true;
                    break;
                }
                
            } else {
                if(utils::contains(utils::lowercase(name), match))
                {
                    does_match = true;
                    break;
                }
            }
        }
        
        if(does_match) {
            if(not target.has_value()) {
                target = std::string_view(name);
            } else {
                FormatWarning("Found candidate ", name, " to be used as the ",std::vector(matches),"-column, but already have: ", target);
            }
        }
    };
    
    frame_data_t result;
    for(auto &path : _filename.value()) {
        if(path.has_extension("csv")) {
            bool is_centroid = true;
            auto table = CSVStreamReader{path, ',', true}.readNumericTableOptional<double>();
            
            std::optional<std::string_view> x_column, y_column;
            std::optional<std::string_view> w_column, h_column;
            std::optional<std::string_view> frame_column;
            
            for(auto &name : table.header) {
                match_name({'x'}, name, x_column);
                match_name({'y'}, name, y_column);
                match_name({"w", "width"}, name, w_column);
                match_name({"h", "height"}, name, h_column);
                match_name({"frame"}, name, frame_column);
            }
            
            if(not x_column || not y_column) {
                throw InvalidArgumentException("Cannot read a format that does not contain valid X and Y columns (triggered by ",path,").");
            }
            
            if(not frame_column) {
                throw InvalidArgumentException("Cannot read a format that does not contain a valid frame column (triggered by ", frame_column,").");
            }
            
            if(w_column && h_column) {
                if(not utils::contains(utils::lowercase(w_column.value()), "centroid"))
                {
                    is_centroid = false;
                }
            }
            
            const Size2 individual_image_size = SETTING(individual_image_size);
            
            for(size_t i=0; i<table.size(); ++i) {
                auto row = table[i];
                std::optional<double> x = row.get(*x_column);
                std::optional<double> y = row.get(*y_column);
                double raw_frame = row.get(*frame_column).value();
                if(raw_frame < 0) {
                    FormatWarning("Ignoring row ", row.cells, " because the frame number is ", raw_frame,".");
                    continue;
                }
                
                Frame_t frame{ static_cast<uint32_t>(raw_frame) };
                
                std::optional<double> w, h;
                if(w_column) w = row.get(*w_column);
                if(h_column) h = row.get(*h_column);
                
                if(not w)
                    w = individual_image_size.width;
                if(not h)
                    h = individual_image_size.height;
                
                Bounds bds{*x, *y, *w, *h};
                if(is_centroid) {
                    bds = bds - Vec2(*w, *h) * 0.5;
                }
                
                result[frame].push_back(bds);
            }
        }
    }
    
    return result;
}


}
