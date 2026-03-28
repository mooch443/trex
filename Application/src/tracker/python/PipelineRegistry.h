#pragma once
#include <commons.pc.h>
#include <core/DetectionTypes.h>
#include <core/TileImage.h>
#include <core/TaskPipeline.h>

namespace track::detect {

/// Register a pipeline manager for the given detection type.
/// The manager is created with the given batch_size, start_paused flag, and callback.
/// Calling this again for the same type replaces the existing entry.
TREX_EXPORT void register_pipeline(
    ObjectDetectionType::Class type,
    size_t batch_size,
    bool start_paused,
    std::function<void(std::vector<TileImage>&&)> callback);

/// Remove the manager entry for the given type (call at deinit, after clean_up()).
TREX_EXPORT void unregister_pipeline(ObjectDetectionType::Class type);

/// Retrieve the manager for the given type. Throws if not registered.
TREX_EXPORT PipelineManager<TileImage>& pipeline_manager(ObjectDetectionType::Class type);

/// Retrieve the manager for the given type, or nullptr if not registered.
TREX_EXPORT PipelineManager<TileImage>* try_pipeline_manager(ObjectDetectionType::Class type);

/// Retrieve the manager for the current detection_type(). Throws if not set or not registered.
TREX_EXPORT PipelineManager<TileImage>& current_pipeline_manager();

/// Retrieve the manager for the current detection_type(), or nullptr if not set or not registered.
TREX_EXPORT PipelineManager<TileImage>* try_current_pipeline_manager();

} // namespace track::detect
