#pragma once
#include <commons.pc.h>
#include <file/Path.h>

namespace track::detect::yolo::names {

using owner_map_t = cmn::blob::ObjectClass_t;
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

struct KeypointFormat {
    uint8_t n_points{0};
    uint8_t n_dims{0};
    
    constexpr bool operator==(const KeypointFormat&) const = default;
    glz::json_t to_json() const;
    std::string toStr() const;
    static KeypointFormat fromStr(const std::string&);
    constexpr bool valid() const {
        return n_points != 0 && n_dims != 0;
    }
    static std::string class_name() { return "KeypointFormat"; }
};

struct KeypointNames {
    std::optional<std::vector<std::string>> names;
    
    std::optional<std::string> name(size_t index) const;
    
    constexpr bool operator==(const KeypointNames&) const = default;
    glz::json_t to_json() const;
    std::string toStr() const;
    static KeypointNames fromStr(const std::string&);
    bool valid() const {
        return names.has_value();
    }
    static std::string class_name() { return "KeypointNames"; }
};

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
    auto operator<=>(const DetectResolution& other) const = default;
    
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

namespace track::vi {

struct VIWeights {
    enum Status {
        NONE,
        PROGRESS,
        FINISHED
    };
    
    cmn::file::Path _path;
    std::optional<double> _uniqueness;
    bool _loaded{false};
    Status _status{NONE};
    std::optional<uint64_t> _modified;
    std::optional<detect::DetectResolution> _resolution;
    std::optional<uint8_t> _num_classes;
    
    auto operator<=>(const VIWeights& other) const = default;
    constexpr bool valid() const { return loaded() && status() != NONE; }
    constexpr bool finished() const { return loaded() && status() == FINISHED; }
    constexpr bool loaded() const { return _loaded; }
    const auto& path() const { return _path; }
    auto uniqueness() const { return _uniqueness; }
    Status status() const { return _status; }
    auto modified() const { return _modified; }
    auto resolution() const { return _resolution; }
    auto classes() const { return _num_classes; }
    
    std::string toStr() const;
    static VIWeights fromStr(const std::string&);
    static std::string class_name() { return "VIWeights"; }
    glz::json_t to_json() const;
};

}

namespace glz {
   template <>
   struct meta<track::vi::VIWeights> {
      using T = track::vi::VIWeights;
      static constexpr auto value = object(
         "path",       &T::_path,
         "uniqueness", &T::_uniqueness,
         "loaded",     &T::_loaded,
         "status",     &T::_status,
         "modified",   &T::_modified,
         "resolution", &T::_resolution,
         "num_classes", &T::_num_classes
      );
   };

    template <>
    struct from<JSON, track::detect::DetectResolution>
    {
        template <auto Opts>
        static void op(track::detect::DetectResolution& value, auto&&... args)
        {
            std::array<int,2> arr;
            parse<JSON>::op<Opts>(arr, args...);
            value.width = arr[0];
            value.height = arr[1];
        }
    };

    template <>
    struct to<JSON, track::detect::DetectResolution>
    {
        template <auto Opts>
        static void op(const track::detect::DetectResolution& value, auto&&... args) noexcept
        {
            std::array<int,2> arr{value.width, value.height};
            serialize<JSON>::op<Opts>(arr, args...);
        }
    };
}
