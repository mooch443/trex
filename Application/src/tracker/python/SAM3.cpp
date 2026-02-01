#include "SAM3.h"

namespace track {

struct SAM3::Data {
    // Internal data members (if needed)
};

SAM3::SAM3(cmn::Image::Ptr&&) {
    // Constructor implementation (if needed)
}

void SAM3::set_background(cmn::Image::Ptr&&) {
    // Set background implementation (if needed)
}

std::future<SegmentationData> SAM3::apply(TileImage&& tiled) {
    // Apply implementation (if needed)
    return {};
}

void SAM3::reinit(track::ModuleProxy& proxy) {
    // Reinitialize implementation (if needed)
}

void SAM3::init() {
    // Initialization implementation (if needed)
}

void SAM3::deinit() {
    // Deinitialization implementation (if needed)
}

bool SAM3::is_initializing() {
    // Check if initializing implementation (if needed)
    return false;
}

double SAM3::fps() {
    // Get FPS implementation (if needed)
    return 0.0;
}

PipelineManager<TileImage, true>& SAM3::manager() {
    static auto instance = PipelineManager<TileImage, true>(1u, [](std::vector<TileImage>&& images)
    {
        /// in background subtraction case, we have to wait until the background
        /// image has been generated and hang in the meantime.
        auto start_time = std::chrono::steady_clock::now();
        auto message_time = start_time;
        while(//not data().has_background() &&
               not manager().is_terminated())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            auto elapsed = std::chrono::steady_clock::now() - message_time;
            if(std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() > 30) {
                FormatExcept("Background image not set in ",
                             std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time).count(),
                             " seconds. Waiting for background image...");
                message_time = std::chrono::steady_clock::now();
            }
        }
        
        if(not manager().is_terminated()) {
            if(images.empty())
                FormatExcept("Images is empty :(");
            
            SAM3::apply(std::move(images));
        }
    });
    return instance;
}

void SAM3::apply(std::vector<TileImage>&& tiled) {
    // Apply implementation (if needed)
}

SAM3::Data& SAM3::data() {
    static Data instance;
    return instance;
}

}
