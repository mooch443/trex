#include "Detection.h"

#include <file/PathArray.h>
#include <detect/BackgroundSubtraction.h>
#include <detect/NoDetection.h>
#include <grabber/misc/default_config.h>
#include <core/AbstractVideoSource.h>
#include <core/PrecomuptedDetection.h>
#include <core/TrackingSettings.h>
#include <python/PythonWrapper.h>

namespace track::detect {

namespace {

auto& backend_registry() {
    static std::unordered_map<ObjectDetectionType::Class, BackendHooks> registry;
    return registry;
}

bool is_python_backend_type(ObjectDetectionType::Class type) {
    return type == ObjectDetectionType::yolo
        || type == ObjectDetectionType::sam3;
}

}

void register_backend(ObjectDetectionType::Class type, BackendHooks hooks) {
    backend_registry()[type] = std::move(hooks);
}

void unregister_backend(ObjectDetectionType::Class type) {
    backend_registry().erase(type);
}

const BackendHooks* backend(ObjectDetectionType::Class type) {
    auto it = backend_registry().find(type);
    if(it == backend_registry().end())
        return nullptr;
    return &it->second;
}

const BackendHooks* ensure_backend(ObjectDetectionType::Class type) {
    if(const auto* hooks = backend(type)) {
        return hooks;
    }

    if(is_python_backend_type(type)) {
        Python::ensure_python_impl_loaded();
        return backend(type);
    }

    return nullptr;
}

} // namespace track::detect

namespace track {

using namespace detect;

Detection::Detection() {
    auto type = detection_type();
    if(!type)
        return;

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
        return;
    default:
        break;
    }

    if(const auto* hooks = detect::ensure_backend(*type); hooks && hooks->init) {
        hooks->init();
        return;
    }

    throw U_EXCEPTION("Unknown detection type: ", detection_type());
}

void Detection::deinit() {
    const auto type = detection_type();
    if(type == ObjectDetectionType::background_subtraction) {
        manager().clean_up();
        BackgroundSubtraction::deinit();
        return;
    }
    if(type == ObjectDetectionType::precomputed) {
        manager().clean_up();
        PrecomputedDetection::deinit();
        return;
    }
    if(type == ObjectDetectionType::none) {
        manager().clean_up();
        return;
    }

    if(type) {
        if(const auto* hooks = detect::ensure_backend(*type); hooks && hooks->deinit) {
            hooks->deinit();
            manager().clean_up();
            return;
        }
    }

    manager().clean_up();
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

PipelineManager<TileImage, true>& Detection::manager() {
    if(detection_type() == ObjectDetectionType::background_subtraction) {
        return BackgroundSubtraction::manager();
    }
    if(detection_type() == ObjectDetectionType::precomputed) {
        return PrecomputedDetection::manager();
    }

    static auto instance = PipelineManager<TileImage, true>(
        max(1u, READ_SETTING(detect_batch_size, uchar)),
        [](std::vector<TileImage>&& images) {
#ifndef NDEBUG
            if(images.empty())
                FormatExcept("Images is empty :(");
#endif
            Detection::apply(std::move(images));
        }
    );
    return instance;
}

} // namespace track
