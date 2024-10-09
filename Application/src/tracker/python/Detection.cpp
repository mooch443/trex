#include "Detection.h"
//#include <python/Yolo7InstanceSegmentation.h>
#include <python/Yolo8.h>
//#include <python/Yolo7ObjectDetection.h>
#include <processing/RawProcessing.h>
#include <grabber/misc/default_config.h>
#include <misc/TrackingSettings.h>
#include <video/Video.h>
#include <processing/CPULabeling.h>
#include <misc/Timer.h>
#include <misc/AbstractVideoSource.h>
#include <python/TileBuffers.h>

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

    case ObjectDetectionType::yolo8:
        Yolo8::init();
        break;
            
    case ObjectDetectionType::background_subtraction:
        BackgroundSubtraction{};
        break;

    default:
        throw U_EXCEPTION("Unknown detection type: ", detection_type());
    }
}

void Detection::deinit() {
    if(detection_type() == ObjectDetectionType::yolo8)
        Yolo8::deinit();
    else if(detection_type() == ObjectDetectionType::background_subtraction)
        BackgroundSubtraction::deinit();
    
    manager().clean_up();
}

bool Detection::is_initializing() {
    if(detection_type() == ObjectDetectionType::yolo8)
        return Yolo8::is_initializing();
    return false;
}

double Detection::fps() {
    if(detection_type() == ObjectDetectionType::yolo8)
        return Yolo8::fps();
    if(detection_type() == ObjectDetectionType::background_subtraction)
        return BackgroundSubtraction::fps();
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

    case ObjectDetectionType::yolo8: {
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
            
    default:
        throw U_EXCEPTION("Unknown detection type: ", detection_type());
    }
}

void Detection::apply(std::vector<TileImage>&& tiled) {
    /*if (type() == ObjectDetectionType::yolo7) {
        Yolo7ObjectDetection::apply(std::move(tiled));
        return;

    }
    else*/ if (detection_type() == ObjectDetectionType::yolo8) {
        Yolo8::apply(std::move(tiled));
        return;
    } else if(detection_type() == ObjectDetectionType::background_subtraction) {
        BackgroundSubtraction::apply(std::move(tiled));
        return;
    }

    throw U_EXCEPTION("Unknown detection type: ", detection_type());
}

PipelineManager<TileImage, true>& BackgroundSubtraction::manager() {
    static auto instance = PipelineManager<TileImage, true>(1u, [](std::vector<TileImage>&& images)
    {
        /// in background subtraction case, we have to wait until the background
        /// image has been generated and hang in the meantime.
        while(not data().background && not manager().is_terminated())
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        if(not manager().is_terminated()) {
            if(images.empty())
                FormatExcept("Images is empty :(");
            
            BackgroundSubtraction::apply(std::move(images));
        }
    });
    return instance;
}

BasicManager<TileImage>& Detection::manager() {
    if(detection_type() ==  ObjectDetectionType::background_subtraction) {
        return BackgroundSubtraction::manager();
    } else {
        static auto instance = PipelineManager<TileImage>(max(1u, SETTING(detect_batch_size).value<uchar>()), [](std::vector<TileImage>&& images) {
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
    if(data().background)
        manager().set_paused(false);
}

void BackgroundSubtraction::Data::set(Image::Ptr&& average) {
    background = std::move(average);
    if(background) {
        background->get().copyTo(data().gpu);
        gpu.convertTo(data().float_average, CV_32FC(gpu.channels()), 1.0 / 255.0);
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

void BackgroundSubtraction::deinit() {}

double BackgroundSubtraction::fps() {
    if(data().samples == 0)
        return 0;
    return data().time / data().samples;
}

void BackgroundSubtraction::apply(std::vector<TileImage> &&tiled) {
    Timer timer;
    const auto mode = Background::meta_encoding();
    
    RawProcessing raw(data().gpu, &data().float_average, nullptr);
    gpuMat gpu_buffer;
    TagCache tag;
    CPULabeling::ListCache_t cache;
    const auto cm_per_pixel = SETTING(cm_per_pixel).value<Settings::cm_per_pixel_t>();
    const auto segment_size_filter = SETTING(segment_size_filter).value<BlobSizeRange>();
    const Float2_t sqcm = SQR(cm_per_pixel);
    cv::Mat r3;
    
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
                    if(image->dims == 3)
                        cv::cvtColor(image->get(), r3, cv::COLOR_BGR2GRAY);
                    else if(image->dims == 4)
                        cv::cvtColor(image->get(), r3, cv::COLOR_BGRA2GRAY);
                    else
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
                assert(Size2(r3) == Size2(data().gpu.cols, data().gpu.rows));
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
                        
                        if(segment_size_filter.in_range_of_one(num_pixels * sqcm))
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
        data().time +=  double(tiled.size()) / timer.elapsed();
        data().samples++;
    }
}

} // namespace track
