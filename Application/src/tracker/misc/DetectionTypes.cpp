#include "DetectionTypes.h"

#include <misc/SoftException.h>
#include <misc/GlobalSettings.h>

using namespace cmn;

namespace track::detect {

DetectResolution DetectResolution::fromStr(const std::string& str) {
    if(utils::beginsWith(str, '[')) {
        auto pair = Meta::fromStr<std::vector<uint16_t>>(str);
        if(pair.empty())
            return {};
        
        uint16_t height = pair.front();
        uint16_t width = height;
        if(pair.size() > 1)
            width = pair.back();
        return {height, width};
        
    } else {
        auto width = Meta::fromStr<uint16_t>(str);
        return {width, width};
    }
}
nlohmann::json DetectResolution::to_json() const {
    auto array = nlohmann::json::array();
    array.push_back(height);
    array.push_back(width);
    return array;
}
std::string DetectResolution::toStr() const {
    std::ostringstream os;
    os << "[" << height << "," << width << "]";
    return os.str();
}
std::string DetectResolution::class_name() {
    return "DetectResolution";
}

ObjectDetectionType_t detection_type() {
    return SETTING(detect_type).value<ObjectDetectionType_t>();
}
ObjectDetectionFormat_t detection_format() {
    return SETTING(detect_format).value<ObjectDetectionFormat_t>();
}

Size2 get_model_image_size() {
    const auto detect_resolution = SETTING(detect_resolution).value<track::detect::DetectResolution>();
    const auto meta_video_size = SETTING(meta_video_size).value<Size2>();
    
    if(detection_type() == ObjectDetectionType::background_subtraction) {
        return meta_video_size;
        
    } else if (detection_type() == ObjectDetectionType::yolo8) {
        const auto region_resolution = SETTING(region_resolution).value<track::detect::DetectResolution>();

        Size2 size;
        const float ratio = meta_video_size.height / meta_video_size.width;
        if (region_resolution.width > 0 && not SETTING(region_model).value<file::Path>().empty()) {
            const auto max_w = max((float)detect_resolution.width, (float)region_resolution.width * 2);
            size = Size2(max_w, ratio * max_w);
            size = meta_video_size;//.div(4);
        }
        else
            size = Size2(detect_resolution.width, ratio * detect_resolution.height);

        //Print("Using a resolution of meta_video_size = ", meta_video_size, " and detect_resolution = ", detect_resolution, " and region_resolution = ", region_resolution," gives a model image size of ", size);
        //return meta_video_size.div(2);
        return size;
    }
    else {
        return Size2(detect_resolution.width, detect_resolution.height);
    }
}

std::string Keypoint::toStr() const {
    return "Keypoint<" + Meta::toStr(bones) + ">";
}

const Bone& Keypoint::bone(size_t index) const {
    if (index >= bones.size()) {
        throw SoftException("Index ", index, " out of bounds for array of size ", bones.size(), ".");
    }
    return bones[index];
}

blob::Pose Keypoint::toPose() const {
    std::vector<blob::Pose::Point> coords;
    for (auto& b : bones)
        coords.push_back(blob::Pose::Point(b.x, b.y));
    return blob::Pose{
        .points = std::move(coords)
    };
}

}
