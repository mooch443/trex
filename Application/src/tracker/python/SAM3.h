#pragma once

#include <commons.pc.h>
#include <misc/Image.h>
#include <core/TileImage.h>
#include <python/ModuleProxy.h>
#include <core/GPURecognitionTypes.h>

namespace track {

struct Detection;
void register_sam3_backend();

/**
 * C++ bridge for the Python SAM3 backend.
 *
 * The class keeps only backend lifecycle state on the C++ side and forwards
 * image batches plus image-aligned prompt payloads into Python.
 */
struct TREX_EXPORT SAM3 {
    SAM3(cmn::Image::Ptr&& = nullptr);

    /** No-op compatibility hook for the generic detection backend interface. */
    static void set_background(cmn::Image::Ptr&&);
    
    /**
     * Queue one tiled frame for asynchronous SAM3 processing.
     *
     * @param tiled Frame/tile payload owned by the detection pipeline.
     * @return Future resolved with segmentation output for the tile package.
     */
    static std::future<SegmentationData> apply(TileImage&& tiled);
    
    /** Recreate or refresh the Python-side SAM3 model session. */
    static void reinit(track::ModuleProxy& proxy);
    
    /** Register the backend pipeline and initialize the Python SAM3 module. */
    static void init();

    /** Tear down the backend pipeline and unload the Python SAM3 module. */
    static void deinit();
    
    /** SAM3 initializes synchronously today, so this remains false. */
    static bool is_initializing();

    /** Return the average wall-clock processing time accumulated for SAM3. */
    static double fps();

    
private:
    /** Process one scheduled batch of tile images on the Python thread. */
    static void apply(std::vector<TileImage>&& tiled);
    friend struct Detection;
    friend void register_sam3_backend();
    
    struct Data;
    
    static Data& data();
};

detect::Sam3PromptsPerImage resolve_prompts_for_input(
  const detect::YoloInput& input,
  const std::optional<detect::Sam3Prompts>& prompts_by_frame);

detect::Sam3PromptsPerImage resolve_prompts_for_tile(
  const TileImage& tile,
  const std::optional<detect::Sam3Prompts>& prompts_by_frame);


}
