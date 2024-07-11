#include "TaskPipeline.h"

namespace cmn {
namespace OverlayBuffers {

static auto& buffer_mutex() {
    static auto mutex = new LOGGED_MUTEX("OverlayBuffers::buffer_mutex");
    return *mutex;
}
static auto& buffers() {
    static std::vector<Image::Ptr> buffers;
    return buffers;
}

Image::Ptr get_buffer() {
    if (auto guard = LOGGED_LOCK(OverlayBuffers::buffer_mutex());
        not OverlayBuffers::buffers().empty())
    {
        auto ptr = std::move(OverlayBuffers::buffers().back());
        OverlayBuffers::buffers().pop_back();
        //Print("Received from buffers ", ptr->bounds());
        return ptr;
    }
    
    return Image::Make();
}

void put_back(Image::Ptr&& ptr) {
    if (not ptr)
        return;
    auto guard = LOGGED_LOCK(OverlayBuffers::buffer_mutex());
    //Print("Pushed back buffer ", ptr->bounds());
    OverlayBuffers::buffers().push_back(std::move(ptr));
}
}

SegmentationData::~SegmentationData() {
    if (image) {
        OverlayBuffers::put_back(std::move(image));
    }
}

SegmentationData& SegmentationData::operator=(SegmentationData&& other)
{
    frame = std::move(other.frame);
    tiles = std::move(other.tiles);
    predictions = std::move(other.predictions);
    //outlines = std::move(other.outlines);
    keypoints = std::move(other.keypoints);

    if (image) {
        OverlayBuffers::put_back(std::move(image));
    }
    image = std::move(other.image);
    return *this;
}

std::string SegmentationData::toStr() const {
    return "Segmentation<"+frame.index().toStr()+">";
}

Frame_t SegmentationData::original_index() const {
    return image ? Frame_t(image->index()) : Frame_t();
}

Frame_t SegmentationData::written_index() const {
    return frame.index();
}

}
