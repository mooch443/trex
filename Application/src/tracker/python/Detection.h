#pragma once
#include <commons.pc.h>
#include <misc/TileImage.h>
#include <misc/TaskPipeline.h>
#include <misc/DetectionTypes.h>

namespace track {

template<typename T>
concept MultiObjectDetection = requires (std::vector<TileImage> tiles) {
    { T::apply(std::move(tiles)) };
    //{ T::receive(data, Vec2{}, {}) };
};

template<typename T>
concept SingleObjectDetection = requires (TileImage tiles) {
    { T::apply(std::move(tiles)) } -> std::convertible_to<std::future<SegmentationData>>;
    //{ T::receive(data, Vec2{}, {}) };
};

template<typename T>
concept ObjectDetection = MultiObjectDetection<T> || SingleObjectDetection<T>;

struct TREX_EXPORT Detection {
    Detection();
    
    static detect::ObjectDetectionType::Class type();
    static std::future<SegmentationData> apply(TileImage&& tiled);
    static void deinit();
    static bool is_initializing();

    static auto& manager() {
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
    
private:
    static void apply(std::vector<TileImage>&& tiled);
};

struct TREX_EXPORT BackgroundSubtraction {
    BackgroundSubtraction(Image::Ptr&& = nullptr);
    static void set_background(Image::Ptr&&);
    
    static detect::ObjectDetectionType::Class type() { return detect::ObjectDetectionType::background_subtraction; }
    static std::future<SegmentationData> apply(TileImage&& tiled);
    static void deinit();

    static auto& manager() {
        static auto instance = PipelineManager<TileImage, true>(max(1u, SETTING(detect_batch_size).value<uchar>()), [](std::vector<TileImage>&& images) {
            // do what has to be done when the queue is full
            // i.e. py::execute()
            if(images.empty())
                FormatExcept("Images is empty :(");
            while(not data().background)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            BackgroundSubtraction::apply(std::move(images));
        });
        return instance;
    }
    
private:
    static void apply(std::vector<TileImage>&& tiled);
    
    struct Data {
        Image::Ptr background;
        gpuMat gpu;
        gpuMat float_average;
        
        void set(Image::Ptr&&);
    };
    
    static Data& data();
};

} // namespace track
