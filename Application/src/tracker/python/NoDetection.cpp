#include "NoDetection.h"
#include <python/Detection.h>
#include <python/TileBuffers.h>

namespace track {

std::future<SegmentationData> NoDetection::apply(TileImage &&tiled) {
    if(tiled.promise)
        throw U_EXCEPTION("Tiled.promise was already set.");
    tiled.promise = std::make_unique<std::promise<SegmentationData>>();
    
    auto f = tiled.promise->get_future();
    Detection::manager().enqueue(std::move(tiled));
    return f;
}

void NoDetection::apply(std::vector<TileImage> &&tiled)
{
    const auto encoding = Background::meta_encoding();
    
    size_t i = 0;
    for(auto &&tile : tiled) {
        SegmentationData data = std::move(tile.data);
        data.frame.set_encoding(encoding);
        
        if (tile.promise) {
            tile.promise->set_value(std::move(data));
            tile.promise = nullptr;
        }
        
        try {
            if(tile.callback)
                tile.callback();
            
        } catch(...) {
            FormatExcept("Exception for tile ", i," in package of ", tiled.size(), " TileImages.");
        }
        
        for(auto &image: tile.images) {
            buffers::TileBuffers::get().move_back(std::move(image));
        }
        tile.images.clear();
        
        ++i;
    }
}

}
