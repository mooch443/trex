#pragma once
#include <commons.pc.h>
#include <file/Path.h>

namespace track::detect::yolo::names {

using owner_map_t = std::map<uint16_t, std::string>;
using map_t = std::map<uint16_t, std::string_view>;
using vec_t = std::vector<std::string_view>;

vec_t get_vector();
map_t get_map();

}

namespace track::detect {

ENUM_CLASS(ObjectDetectionType, none, yolo, background_subtraction);
ENUM_CLASS(ObjectDetectionFormat, none, boxes, masks, poses);

using ObjectDetectionType_t = ObjectDetectionType::Class;
using ObjectDetectionFormat_t = ObjectDetectionFormat::Class;

}

namespace EnumMeta {
/// add a tag checker for whether a customparser is available for a given enum class
template<> struct HasCustomParser<track::detect::ObjectDetectionType_t> : std::true_type {
    static const track::detect::ObjectDetectionType_t& fromStr(const std::string& str);
};

}

namespace track::detect {

ObjectDetectionType_t detection_type();
ObjectDetectionFormat_t detection_format();

cmn::Size2 get_model_image_size();

class Bone {
public:
    float x;
    float y;
    //float conf;
    std::string toStr() const {
        return "Bone<" + cmn::Meta::toStr(x) + "," + cmn::Meta::toStr(y) + ">";
    }
};

class Keypoint {
public:
    std::vector<Bone> bones;
    std::string toStr() const;
    const Bone& bone(size_t index) const;
    cmn::blob::Pose toPose() const;
};

struct DetectResolution {
    uint16_t width{640}, height{640};
    
    bool operator==(const DetectResolution&) const = default;
    bool operator!=(const DetectResolution&) const = default;
    
    static DetectResolution fromStr(const std::string& str);
    glz::json_t to_json() const;
    std::string toStr() const;
    static std::string class_name();
};

namespace yolo {
/// returns true if the given filename is a valid model name
/// that can be automatically retrieved from the ultralytics repo
bool is_valid_default_model(const std::string&);

bool valid_model(const cmn::file::Path&, const cmn::file::FilesystemInterface& = cmn::file::RealFilesystem{});
bool is_default_model(const cmn::file::Path&);
std::string default_model();
}

}
