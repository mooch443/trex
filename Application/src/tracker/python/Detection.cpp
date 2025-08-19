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
#include <python/BackgroundSubtraction.h>

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
        auto detect_precomputed_file = READ_SETTING(detect_precomputed_file, file::PathArray);
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
        //manager().set_weight_limit(max(1u, READ_SETTING(detect_batch_size, uchar)));
        manager().enqueue(std::move(tiled));
        return f;
    }

    case ObjectDetectionType::background_subtraction: {
        tiled.promise = std::make_unique<std::promise<SegmentationData>>();
        auto f = tiled.promise->get_future();
        //manager().set_weight_limit(max(1u, READ_SETTING(detect_batch_size, uchar)));
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

PipelineManager<TileImage, true>& Detection::manager() {
    if(detection_type() ==  ObjectDetectionType::background_subtraction) {
        return BackgroundSubtraction::manager();
    } else if(detection_type() == ObjectDetectionType::precomputed) {
        return PrecomputedDetection::manager();
    } else {
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
