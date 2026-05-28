#include "Detection.h"

#include <file/PathArray.h>
#include <python/BackgroundSubtraction.h>
#include <python/NoDetection.h>
#include <python/PipelineRegistry.h>
#include <grabber/misc/default_config.h>
#include <core/AbstractVideoSource.h>
#include <python/PrecomuptedDetection.h>
#include <core/TrackingSettings.h>
#include <core/TileBuffers.h>
namespace track {

using namespace detect;

void Detection::init() {
    auto type = detection_type();
    if(!type)
        return;

    if (const auto* hooks = detect::ensure_backend(*type);
        hooks
        && hooks->init)
    {
        hooks->init();
    }

    switch(*type) {
    case ObjectDetectionType::background_subtraction:
        BackgroundSubtraction{};
        return;
    case ObjectDetectionType::precomputed: {
        auto detect_precomputed_file = READ_SETTING(detect_precomputed_file, file::PathArray);
        PrecomputedDetection{
            std::move(detect_precomputed_file),
            nullptr,
            meta_encoding_t::binary
        };
        return;
    }
    case ObjectDetectionType::none:
        detect::register_pipeline(
            detect::ObjectDetectionType::none,
            max(1u, READ_SETTING(detect_batch_size, uchar)),
            /*start_paused=*/false,
            [](std::vector<TileImage>&& images) {
                Detection::apply(std::move(images));
            });
        return;
    case ObjectDetectionType::sam3:
    case ObjectDetectionType::yolo:
        return;
    default:
        break;
    }

    throw U_EXCEPTION("Unknown detection type: ", detection_type());
}

void Detection::deinit() {
    const auto type = detection_type();
    auto* mgr = type ? detect::try_pipeline_manager(*type) : nullptr;

    if(type == ObjectDetectionType::background_subtraction) {
        if(mgr) mgr->clean_up();
        BackgroundSubtraction::deinit();
        return;
    }
    if(type == ObjectDetectionType::precomputed) {
        if(mgr) mgr->clean_up();
        PrecomputedDetection::deinit();
        return;
    }
    if(type == ObjectDetectionType::none) {
        if(mgr) mgr->clean_up();
        detect::unregister_pipeline(detect::ObjectDetectionType::none);
        return;
    }

    if(type) {
        if(const auto* hooks = detect::ensure_backend(*type); hooks && hooks->deinit) {
            hooks->deinit();
            // The hook's deinit is responsible for clean_up() and unregistering its pipeline.
            return;
        }
    }

    // Fallback: type has no deinit hook; clean up its pipeline if registered.
    if(mgr) mgr->clean_up();
    
    /// clear out remaining tilebuffer images
    buffers::TileBuffers::clear();
}

bool Detection::is_initializing() {
    const auto type = detection_type();
    if(!type)
        return false;

    if(const auto* hooks = detect::ensure_backend(*type); hooks && hooks->is_initializing) {
        return hooks->is_initializing();
    }
    return false;
}

double Detection::fps() {
    const auto type = detection_type();
    if(type == ObjectDetectionType::background_subtraction)
        return BackgroundSubtraction::fps();
    if(type == ObjectDetectionType::precomputed)
        return PrecomputedDetection::fps();
    if(type == ObjectDetectionType::none)
        return AbstractBaseVideoSource::_network_fps.load();

    if(type) {
        if(const auto* hooks = detect::ensure_backend(*type); hooks && hooks->fps) {
            return hooks->fps();
        }
    }

    return AbstractBaseVideoSource::_network_fps.load();
}

std::future<SegmentationData> Detection::apply(TileImage&& tiled) {
    if(tiled.promise)
        throw U_EXCEPTION("Promise was already created.");

    auto type = detection_type();
    if(!type)
        throw RuntimeError("No detect_type was set before Detection::apply.");

    switch(*type) {
    case ObjectDetectionType::yolo:
    case ObjectDetectionType::sam3:
    case ObjectDetectionType::background_subtraction:
    case ObjectDetectionType::precomputed:
    case ObjectDetectionType::none:
        tiled.promise = std::make_unique<std::promise<SegmentationData>>();
        {
            auto future = tiled.promise->get_future();
            manager().enqueue(std::move(tiled));
            return future;
        }
    default:
        break;
    }

    throw U_EXCEPTION("Unknown detection type: ", detection_type());
}

void Detection::apply(std::vector<TileImage>&& tiled) {
    const auto type = detection_type();
    if(type == ObjectDetectionType::background_subtraction) {
        BackgroundSubtraction::apply(std::move(tiled));
        return;
    }
    if(type == ObjectDetectionType::precomputed) {
        PrecomputedDetection::apply(std::move(tiled));
        return;
    }
    if(type == ObjectDetectionType::none) {
        NoDetection::apply(std::move(tiled));
        return;
    }

    if(type) {
        if(const auto* hooks = detect::ensure_backend(*type); hooks && hooks->apply) {
            hooks->apply(std::move(tiled));
            return;
        }
    }

    throw U_EXCEPTION("Unknown detection type: ", detection_type());
}

void Detection::set_background(const cmn::Image::Ptr& image) {
    const auto type = detection_type();
    if(!type) {
        return;
    }

    if(const auto* hooks = detect::ensure_backend(*type); hooks && hooks->set_background) {
        hooks->set_background(image);
    }
}

PipelineManager<TileImage>& Detection::manager() {
    return detect::current_pipeline_manager();
}

} // namespace track
