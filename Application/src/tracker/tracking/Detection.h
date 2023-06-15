#pragma once
#include <commons.pc.h>
#include <misc/TileImage.h>
#include <misc/TaskPipeline.h>

namespace track {

ENUM_CLASS(ObjectDetectionType, yolo7, yolo7seg, yolo8, customseg);
static ObjectDetectionType::Class detection_type() {
    return SETTING(detection_type).value<ObjectDetectionType::Class>();
}

Size2 get_model_image_size();

template<typename T>
concept MultiObjectDetection = requires (std::vector<TileImage> tiles) {
    { T::apply(std::move(tiles)) };
    //{ T::receive(data, Vec2{}, {}) };
};

template<typename T>
concept SingleObjectDetection = requires (TileImage tiles) {
    { T::apply(std::move(tiles)) } -> std::convertible_to<tl::expected<SegmentationData, const char*>>;
    //{ T::receive(data, Vec2{}, {}) };
};

template<typename T>
concept ObjectDetection = MultiObjectDetection<T> || SingleObjectDetection<T>;


struct Detection {
    Detection();
    
    static ObjectDetectionType::Class type();
    static std::future<SegmentationData> apply(TileImage&& tiled);
    
    static void apply(std::vector<TileImage>&& tiled);

    static auto& manager() {
        static auto instance = PipelineManager<TileImage>(max(1u, SETTING(batch_size).value<uchar>()), [](std::vector<TileImage>&& images) {
            // do what has to be done when the queue is full
            // i.e. py::execute()
            Detection::apply(std::move(images));
        });
        return instance;
    }
};

} // namespace track