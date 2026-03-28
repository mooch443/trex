#include "PipelineRegistry.h"
#include <core/TrackingSettings.h>

namespace track::detect {

namespace {

struct PipelineEntry {
    std::unique_ptr<PipelineManager<TileImage>> manager;
};

// std::map gives stable references across insertions/erasures.
auto& pipeline_registry() {
    static std::map<ObjectDetectionType::Class, PipelineEntry> registry;
    return registry;
}

} // namespace

void register_pipeline(ObjectDetectionType::Class type,
                       size_t batch_size,
                       bool start_paused,
                       std::function<void(std::vector<TileImage>&&)> callback)
{
    auto& reg = pipeline_registry();
    reg.erase(type);
    reg.emplace(std::piecewise_construct,
                std::forward_as_tuple(type),
                std::forward_as_tuple(std::make_unique<PipelineManager<TileImage>>(
        double(batch_size),
        start_paused,
        std::move(callback)
    )));
}

void unregister_pipeline(ObjectDetectionType::Class type) {
    pipeline_registry().erase(type);
}

PipelineManager<TileImage>& pipeline_manager(ObjectDetectionType::Class type) {
    auto& reg = pipeline_registry();
    auto it = reg.find(type);
    if(it == reg.end())
        throw U_EXCEPTION("No pipeline manager registered for detection type: ", type);
    if(not it->second.manager)
        throw RuntimeError("Our manager is zero. How are we gonna be productive now. We need managers!");
    return *it->second.manager;
}

PipelineManager<TileImage>* try_pipeline_manager(ObjectDetectionType::Class type) {
    auto& reg = pipeline_registry();
    auto it = reg.find(type);
    if(it == reg.end() || not it->second.manager)
        return nullptr;
    return it->second.manager.get();
}

PipelineManager<TileImage>& current_pipeline_manager() {
    auto type = detection_type();
    if(!type)
        throw U_EXCEPTION("No detection type set when calling current_pipeline_manager().");
    return pipeline_manager(*type);
}

PipelineManager<TileImage>* try_current_pipeline_manager() {
    auto type = detection_type();
    if(!type)
        return nullptr;
    return try_pipeline_manager(*type);
}

} // namespace track::detect
