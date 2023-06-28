#include "Detection.h"
#include <tracking/Yolo7InstanceSegmentation.h>
#include <tracking/Yolo8.h>
#include <tracking/Yolo7ObjectDetection.h>

namespace track {

Size2 get_model_image_size() {
    auto detection_resolution = Size2(SETTING(detection_resolution).value<uint16_t>());
    if (detection_type() == ObjectDetectionType::yolo8) {
        const auto meta_video_size = SETTING(meta_video_size).value<Size2>();
        const auto detection_resolution = SETTING(detection_resolution).value<uint16_t>();
        const auto region_resolution = SETTING(region_resolution).value<uint16_t>();

        Size2 size;
        const float ratio = meta_video_size.height / meta_video_size.width;
        if (region_resolution > 0 && not SETTING(region_model).value<file::Path>().empty()) {
            const auto max_w = max((float)detection_resolution, (float)region_resolution * 2);
            size = Size2(max_w, ratio * max_w);
            size = meta_video_size;//.div(4);
        }
        else
            size = Size2(detection_resolution, ratio * detection_resolution);

        //print("Using a resolution of meta_video_size = ", meta_video_size, " and detection_resolution = ", detection_resolution, " and region_resolution = ", region_resolution," gives a model image size of ", size);
        //return meta_video_size.div(2);
        return size;
    }
    else {
        return Size2(detection_resolution);
    }
}

Detection::Detection() {
    switch (type()) {
    case ObjectDetectionType::yolo7:
        Yolo7ObjectDetection::init();
        break;

    case ObjectDetectionType::customseg:
    case ObjectDetectionType::yolo7seg:
        Yolo7InstanceSegmentation::init();
        break;

    case ObjectDetectionType::yolo8:
        Yolo8::init();
        break;

    default:
        throw U_EXCEPTION("Unknown detection type: ", type());
    }
}

void Detection::deinit() {
    if(type() == ObjectDetectionType::yolo8)
        Yolo8::deinit();
}

ObjectDetectionType::Class Detection::type() {
    return SETTING(detection_type).value<ObjectDetectionType::Class>();
}

std::future<SegmentationData> Detection::apply(TileImage&& tiled) {
    switch (type()) {
    case ObjectDetectionType::yolo7: {
        auto f = tiled.promise.get_future();
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
    }

    case ObjectDetectionType::yolo8: {
        auto f = tiled.promise.get_future();
        manager().enqueue(std::move(tiled));
        return f;
    }

    default:
        throw U_EXCEPTION("Unknown detection type: ", type());
    }
}

void Detection::apply(std::vector<TileImage>&& tiled) {
    if (type() == ObjectDetectionType::yolo7) {
        Yolo7ObjectDetection::apply(std::move(tiled));
        tiled.clear();
        return;

    }
    else if (type() == ObjectDetectionType::yolo8) {
        Yolo8::apply(std::move(tiled));
        tiled.clear();
        return;
    }

    throw U_EXCEPTION("Unknown detection type: ", type());
}

} // namespace track
