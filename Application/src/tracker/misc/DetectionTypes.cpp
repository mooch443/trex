#include "DetectionTypes.h"

#include <misc/SoftException.h>
#include <misc/GlobalSettings.h>

using namespace cmn;

namespace track::detect {

namespace yolo {

namespace names {

std::mutex names_mutex;
std::optional<std::map<uint16_t, std::string>> names_owner;
std::optional<map_t> easy_cp_names_reference;
std::optional<vec_t> easy_cp_names_vector;

CallbackCollection callbacks;

void check_callbacks() {
    static std::once_flag flag;
    std::call_once(flag, [](){
        if(callbacks)
            return;
        
        callbacks = GlobalSettings::map().register_callbacks({
            "detect_classes"
        }, [](auto) {
            std::unique_lock g(names_mutex);
            names_owner = SETTING(detect_classes).value<std::map<uint16_t, std::string>>();
            easy_cp_names_vector.reset();
            easy_cp_names_reference.reset();
        });
    });
}

const std::map<uint16_t, std::string>& raw_names() {
    
    //std::unique_lock g(names_mutex);
    return names_owner.value();
}

vec_t get_vector() {
    check_callbacks();
    
    /// check if the value already exists
    /// otherwise initialise it
    if(std::unique_lock g(names_mutex);
       easy_cp_names_vector.has_value())
    {
        return easy_cp_names_vector.value();
    }
    
    std::unique_lock g(names_mutex);
    auto& names = raw_names();
    std::vector<std::string_view> cp;
    cp.reserve(names.size());
    
    for(auto &[key, value] : names)
        cp.emplace_back(value);
    
    easy_cp_names_vector = cp;
    return cp;
}

map_t get_map() {
    check_callbacks();
    
    /// check if the value already exists
    /// otherwise initialise it
    if(std::unique_lock g(names_mutex);
       easy_cp_names_reference.has_value())
    {
        return easy_cp_names_reference.value();
    }
    
    std::unique_lock g(names_mutex);
    auto& names = raw_names();
    std::map<uint16_t, std::string_view> cp;
    
    for(auto &[key, value] : names)
        cp.emplace(key, value);
    
    easy_cp_names_reference = cp;
    return cp;
}

}

bool is_valid_default_model(const std::string& filename) {
    static const std::regex pattern("^yolov\\d+([blmnxsucet]|x6|sp|lu|mu|xu|)?((\\d|[sn])+u|-(tinyu|cls|sppu|human|obb|oiv7|pose-p6|pose|seg|v8loader|[0-9]+)+)?\\.pt$");
    return std::regex_match(filename, pattern);
}

std::string default_model() {
    return "yolov10n.pt";
}

bool valid_model(const file::Path& path, const file::FilesystemInterface& fs) {
    if(is_default_model(path))
        return true;
    
    if(fs.exists(path) && path.has_extension("pt"))
        return true;
    
    return false;
}

bool is_default_model(const file::Path& path) {
    return is_valid_default_model(path.str());
}

}

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
glz::json_t DetectResolution::to_json() const {
    return { height, width };
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
