#include "TaskPipeline.h"

namespace cmn {
namespace OverlayBuffers {

inline static std::mutex buffer_mutex;
inline static std::vector<Image::Ptr> buffers;

Image::Ptr get_buffer() {
    if (std::unique_lock guard(OverlayBuffers::buffer_mutex);
        not OverlayBuffers::buffers.empty())
    {
        auto ptr = std::move(OverlayBuffers::buffers.back());
        OverlayBuffers::buffers.pop_back();
        //print("Received from buffers ", ptr->bounds());
        return ptr;
    }
    
    return Image::Make();
}

void put_back(Image::Ptr&& ptr) {
    if (not ptr)
        return;
    std::unique_lock guard(OverlayBuffers::buffer_mutex);
    //print("Pushed back buffer ", ptr->bounds());
    OverlayBuffers::buffers.push_back(std::move(ptr));
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
    outlines = std::move(other.outlines);
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
