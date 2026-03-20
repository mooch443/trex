#include "Detection.h"
//#include <python/Yolo7InstanceSegmentation.h>
#include <python/YOLO.h>
//#include <python/Yolo7ObjectDetection.h>
#include <grabber/misc/default_config.h>
#include <core/TrackingSettings.h>
#include <video/Video.h>
#include <processing/CPULabeling.h>
#include <misc/Timer.h>
#include <core/AbstractVideoSource.h>
#include <python/TileBuffers.h>
#include <core/PrecomuptedDetection.h>
#include <python/BackgroundSubtraction.h>
#include <python/NoDetection.h>
#include <python/SAM3.h>

namespace track {
using namespace detect;

Detection::Detection() {
    auto detect_type = detection_type();
    if(not detect_type)
        return;
    
    switch (*detect_type) {
    case ObjectDetectionType::yolo:
        YOLO::init();
        break;

    case ObjectDetectionType::sam3:
        SAM3::init();
        break;
            
    case ObjectDetectionType::background_subtraction:
        BackgroundSubtraction{};
        break;
            
    case ObjectDetectionType::precomputed: {
        auto detect_precomputed_file = READ_SETTING(detect_precomputed_file, file::PathArray);
        PrecomputedDetection{
            std::move(detect_precomputed_file),
            nullptr,
            meta_encoding_t::binary
        };
        break;
    }
            
    case ObjectDetectionType::none:
        break;

    default:
        throw U_EXCEPTION("Unknown detection type: ", detection_type());
    }
}

void Detection::deinit() {
    const auto type = detection_type();
    if(type == ObjectDetectionType::yolo) {
        YOLO::deinit();
        manager().clean_up();
    } else if(type == ObjectDetectionType::sam3) {
        SAM3::deinit();
        manager().clean_up();

    } else if(type == ObjectDetectionType::background_subtraction) {
        manager().clean_up();
        BackgroundSubtraction::deinit();
    } else if(type == ObjectDetectionType::precomputed) {
        manager().clean_up();
        PrecomputedDetection::deinit();
    } else {
        manager().clean_up();
    }
}

bool Detection::is_initializing() {
    if(detection_type() == ObjectDetectionType::yolo)
        return YOLO::is_initializing();
    if(detection_type() == ObjectDetectionType::sam3)
        return SAM3::is_initializing();
    return false;
}

double Detection::fps() {
    if(detection_type() == ObjectDetectionType::yolo)
        return YOLO::fps();
    else if(detection_type() == ObjectDetectionType::sam3)
        return SAM3::fps();
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
    
    auto detect_type = detection_type();
    if(not detect_type)
        throw RuntimeError("No detect_type was set before Detection::apply.");
    
    switch (*detect_type) {
        case ObjectDetectionType::yolo:
        case ObjectDetectionType::sam3:
        case ObjectDetectionType::background_subtraction:
        case ObjectDetectionType::precomputed:
        case ObjectDetectionType::none: {
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
    } else if(detection_type() == ObjectDetectionType::sam3) {
        SAM3::apply(std::move(tiled));
        return;
    } else if(detection_type() == ObjectDetectionType::background_subtraction) {
        BackgroundSubtraction::apply(std::move(tiled));
        return;
    } else if(detection_type() == ObjectDetectionType::precomputed) {
        PrecomputedDetection::apply(std::move(tiled));
        return;
    } else if(detection_type() == ObjectDetectionType::none) {
        NoDetection::apply(std::move(tiled));
        return;
    }

    throw U_EXCEPTION("Unknown detection type: ", detection_type());
}

PipelineManager<TileImage, true>& Detection::manager() {
    if(detection_type() ==  ObjectDetectionType::background_subtraction) {
        return BackgroundSubtraction::manager();
    } else if(detection_type() == ObjectDetectionType::precomputed) {
        return PrecomputedDetection::manager();
    } /*else if(detection_type() == ObjectDetectionType::yolo) {
        return YOLO::manager();
        
    }*/ else {
        static auto instance = PipelineManager<TileImage, true>(max(1u, READ_SETTING(detect_batch_size, uchar)), [](std::vector<TileImage>&& images) {
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

} // namespace track
