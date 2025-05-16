#include "Detection.h"
//#include <python/Yolo7InstanceSegmentation.h>
#include <python/YOLO.h>
//#include <python/Yolo7ObjectDetection.h>
#include <processing/RawProcessing.h>
#include <grabber/misc/default_config.h>
#include <misc/TrackingSettings.h>
#include <video/Video.h>
#include <processing/CPULabeling.h>
#include <misc/Timer.h>
#include <misc/AbstractVideoSource.h>
#include <python/TileBuffers.h>
#include <misc/PrecomuptedDetection.h>

namespace track {
using namespace detect;

Detection::Detection() {
    switch (detection_type()) {
    /*case ObjectDetectionType::yolo7:
        Yolo7ObjectDetection::init();
        break;

    case ObjectDetectionType::customseg:
    case ObjectDetectionType::yolo7seg:
        Yolo7InstanceSegmentation::init();
        break;*/

    case ObjectDetectionType::yolo:
        YOLO::init();
        break;
            
    case ObjectDetectionType::background_subtraction:
        BackgroundSubtraction{};
        break;
            
    case ObjectDetectionType::precomputed: {
        auto detect_precomputed_file = SETTING(detect_precomputed_file).value<file::PathArray>();
        PrecomputedDetection{
            std::move(detect_precomputed_file),
            nullptr,
            meta_encoding_t::binary
        };
        break;
    }

    default:
        throw U_EXCEPTION("Unknown detection type: ", detection_type());
    }
}

void Detection::deinit() {
    if(detection_type() == ObjectDetectionType::yolo) {
        YOLO::deinit();
        manager().clean_up();
    } else if(detection_type() == ObjectDetectionType::background_subtraction) {
        manager().clean_up();
        BackgroundSubtraction::deinit();
    } else if(detection_type() == ObjectDetectionType::precomputed) {
        manager().clean_up();
        PrecomputedDetection::deinit();
    } else {
        manager().clean_up();
    }
}

bool Detection::is_initializing() {
    if(detection_type() == ObjectDetectionType::yolo)
        return YOLO::is_initializing();
    return false;
}

double Detection::fps() {
    if(detection_type() == ObjectDetectionType::yolo)
        return YOLO::fps();
    else if(detection_type() == ObjectDetectionType::background_subtraction)
        return BackgroundSubtraction::fps();
    else if(detection_type() == ObjectDetectionType::precomputed)
        return PrecomputedDetection::fps();
    else
        return AbstractBaseVideoSource::_network_fps.load();
}

std::future<SegmentationData> Detection::apply(TileImage&& tiled) {
    if(tiled.promise)
        throw U_EXCEPTION("Promise was already created.");
    
    switch (detection_type()) {
    /*case ObjectDetectionType::yolo7: {
        if(tiled.promise)
            throw U_EXCEPTION("Promise was already created.");
        tiled.promise = std::make_unique<std::promise<SegmentationData>>();
        auto f = tiled.promise->get_future();
        manager().enqueue(std::move(tiled));
        return f;
    }

    case ObjectDetectionType::customseg:
    case ObjectDetectionType::yolo7seg: {
        std::promise<SegmentationData> p;
        auto e = Yolo7InstanceSegmentation::apply(std::move(tiled));
        try {
            p.set_value(std::move(e.value()));
        }
        catch (...) {
            p.set_exception(std::current_exception());
        }
        return p.get_future();
    }*/

    case ObjectDetectionType::yolo: {
        tiled.promise = std::make_unique<std::promise<SegmentationData>>();
        auto f = tiled.promise->get_future();
        //manager().set_weight_limit(max(1u, SETTING(detect_batch_size).value<uchar>()));
        manager().enqueue(std::move(tiled));
        return f;
    }

    case ObjectDetectionType::background_subtraction: {
        tiled.promise = std::make_unique<std::promise<SegmentationData>>();
        auto f = tiled.promise->get_future();
        //manager().set_weight_limit(max(1u, SETTING(detect_batch_size).value<uchar>()));
        manager().enqueue(std::move(tiled));
        return f;
    }
            
    case ObjectDetectionType::precomputed: {
        tiled.promise = std::make_unique<std::promise<SegmentationData>>();
        auto f = tiled.promise->get_future();
        manager().enqueue(std::move(tiled));
        return f;
    }
            
    default:
        throw U_EXCEPTION("Unknown detection type: ", detection_type());
    }
}

void Detection::apply(std::vector<TileImage>&& tiled) {
    /*if (type() == ObjectDetectionType::yolo7) {
        Yolo7ObjectDetection::apply(std::move(tiled));
        return;

    }
    else*/ if (detection_type() == ObjectDetectionType::yolo) {
        YOLO::apply(std::move(tiled));
        return;
    } else if(detection_type() == ObjectDetectionType::background_subtraction) {
        BackgroundSubtraction::apply(std::move(tiled));
        return;
    } else if(detection_type() == ObjectDetectionType::precomputed) {
        PrecomputedDetection::apply(std::move(tiled));
        return;
    }

    throw U_EXCEPTION("Unknown detection type: ", detection_type());
}

struct BackgroundSubtraction::Data {
    Image::Ptr _background;
    gpuMat _gpu;
    gpuMat _float_average;
    
    double _time{0.0}, _samples{0.0};
    mutable std::shared_mutex _time_mutex, _background_mutex, _gpu_mutex;
    
    void set(Image::Ptr&&);
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
    
    bool has_background() const {
        std::shared_lock guard(_background_mutex);
        return _background != nullptr;
    }
    void set_background(Image::Ptr&& background) {
        std::unique_lock guard(_background_mutex);
        _background = std::move(background);
    }
};

PipelineManager<TileImage, true>& BackgroundSubtraction::manager() {
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
        
        if(not manager().is_terminated()) {
            if(images.empty())
                FormatExcept("Images is empty :(");
            
            BackgroundSubtraction::apply(std::move(images));
        }
    });
    return instance;
}

PipelineManager<TileImage, true>& Detection::manager() {
    if(detection_type() ==  ObjectDetectionType::background_subtraction) {
        return BackgroundSubtraction::manager();
    } else if(detection_type() == ObjectDetectionType::precomputed) {
        return PrecomputedDetection::manager();
    } else {
        static auto instance = PipelineManager<TileImage, true>(max(1u, SETTING(detect_batch_size).value<uchar>()), [](std::vector<TileImage>&& images) {
            // do what has to be done when the queue is full
            // i.e. py::execute()
#ifndef NDEBUG
            if(images.empty())
                FormatExcept("Images is empty :(");
#endif
            Detection::apply(std::move(images));
        });
        return instance;
    }
}

BackgroundSubtraction::BackgroundSubtraction(Image::Ptr&& average) {
    data().set(std::move(average));
}

void BackgroundSubtraction::set_background(Image::Ptr && average) {
    data().set(std::move(average));
    if(data().has_background())
        manager().set_paused(false);
}

void BackgroundSubtraction::Data::set(Image::Ptr&& average) {
    std::scoped_lock guard(_background_mutex, _gpu_mutex);
    Print("Setting background image to ", hex(average.get()));
    _background = std::move(average);
    if(_background) {
        _background->get().copyTo(_gpu);
        _gpu.convertTo(_float_average, CV_32FC(_gpu.channels()), 1.0 / 255.0);
        manager().set_paused(false);
    }
}

BackgroundSubtraction::Data& BackgroundSubtraction::data() {
    static Data _data;
    return _data;
}

std::future<SegmentationData> BackgroundSubtraction::apply(TileImage &&tiled) {
    if(tiled.promise)
        throw U_EXCEPTION("Tiled.promise was already set.");
    tiled.promise = std::make_unique<std::promise<SegmentationData>>();
    
    auto f = tiled.promise->get_future();
    manager().enqueue(std::move(tiled));
    return f;
}

void BackgroundSubtraction::deinit() {
}

double BackgroundSubtraction::fps() {
    return data().fps();
}

void BackgroundSubtraction::apply(std::vector<TileImage> &&tiled) {
    Timer timer;
    const auto mode = Background::meta_encoding();
    
    std::shared_lock guard(data()._gpu_mutex);
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
                
                gpuMat* input = &gpu_buffer;
                r3.copyTo(*input);
                
                /*if (processed().has_mask()) {
                    static gpuMat mask;
                    if (mask.empty())
                        processed().mask().copyTo(mask);
                    assert(processed().mask().cols == input->cols && processed().mask().rows == input->rows);
                    cv::multiply(*input, mask, *input);
                }*/

                /*if (use_corrected && _grid) {
                    _grid->correct_image(*input, *input);
                }*/
                
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
    }
}

} // namespace track
