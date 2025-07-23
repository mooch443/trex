#include "DetectionTypes.h"

#include <misc/SoftException.h>
#include <misc/GlobalSettings.h>

using namespace cmn;

namespace track::detect {

bool PredictionFilter::allowed(uint16_t clid) const {
    if(_inverted_from)
        return not cmn::contains(*_inverted_from, clid);
    if(detect_only.empty())
        return true;
    return cmn::contains(detect_only, clid);
}

std::string PredictionFilter::toStr() const {
    if(_inverted_from)
        return "-"+Meta::toStr(_inverted_from.value());
    return Meta::toStr(detect_only);
}

glz::json_t PredictionFilter::to_json() const {
    return toStr(); /// we return a string here because it will include the - sign that is not valid JSON
}

std::vector<uint16_t> PredictionFilter::invert(const std::vector<uint16_t>& ids, const yolo::names::map_t& detect_classes) {
    std::vector<uint16_t> result;
    for(auto &[id, name] : detect_classes) {
        if(not cmn::contains(ids, id)
           && not cmn::contains(result, id))
        {
            result.push_back(id);
        }
    }
    return result;
}
std::optional<uint16_t> PredictionFilter::class_id_for(std::string_view search, const yolo::names::map_t& detect_classes) {
    for(auto &[id, name] : detect_classes) {
        if(utils::lowercase_equal_to(name, search)) {
            return id;
        }
    }
    
    return std::nullopt;
}
PredictionFilter PredictionFilter::fromStr(std::string_view sv) {
    const yolo::names::map_t detect_classes = yolo::names::get_map();
    std::vector<uint16_t> only_detect;
    
    bool invert = false;
    if(utils::beginsWith(sv, '-')) {
        sv = sv.substr(1);
        invert = true;
    }
    
    auto parts = util::parse_array_parts(util::truncate(sv));
    for(auto& part : parts) {
        if(utils::is_number_string(part)) {
            only_detect.push_back(Meta::fromStr<uint16_t>(part));
        } else if(auto id = class_id_for(part, detect_classes);
                  id)
        {
            if(not cmn::contains(only_detect, *id))
                only_detect.push_back(*id);
        } else {
            throw InvalidArgumentException("Unknown detection class: ", part);
        }
    }
    
    if(invert) {
        return PredictionFilter{
            .detect_only = PredictionFilter::invert(only_detect, detect_classes),
            ._inverted_from = std::move(only_detect)
        };
    }
    
    return PredictionFilter{
        .detect_only = std::move(only_detect),
        ._inverted_from = std::nullopt
    };
    
}

std::string KeypointNames::toStr() const {
    if(not valid())
        return "null";
    return cmn::Meta::toStr(names.value());
}

glz::json_t KeypointNames::to_json() const {
    if(not valid())
        return glz::json_t::null_t{};
    return cvt2json(names.value());
}

std::optional<std::string> KeypointNames::name(size_t index) const {
    if(not names || names->size() <= index)
        return std::nullopt;
    return names.value()[index];
}

std::string KeypointFormat::toStr() const {
    if(not valid())
        return "null";
    return "[" + cmn::Meta::toStr(n_points) + "," + cmn::Meta::toStr(n_dims) + "]";
}

glz::json_t KeypointFormat::to_json() const {
    if(not valid())
        return glz::json_t::null_t{};
    return { n_points, n_dims };
}

namespace yolo {

namespace names {

std::mutex names_mutex;
cmn::blob::MaybeObjectClass_t names_owner;
std::optional<map_t> easy_cp_names_reference;
std::optional<vec_t> easy_cp_names_vector;

sprite::CallbackFuture callbacks;

void check_callbacks() {
    static std::once_flag flag;
    std::call_once(flag, [](){
        if(callbacks)
            return;
        
        callbacks = GlobalSettings::map().register_callbacks({
            "detect_classes"
        }, [](auto) {
            std::unique_lock g(names_mutex);
            auto detect_classes = SETTING(detect_classes).value<cmn::blob::MaybeObjectClass_t>();
            if(not detect_classes) {
                names_owner = cmn::blob::ObjectClass_t{};
            } else {
                names_owner = detect_classes.value();
            }
            easy_cp_names_vector.reset();
            easy_cp_names_reference.reset();
        });
    });
}

const owner_map_t& raw_names() {
    
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

std::optional<cmn::blob::Pose::Skeleton> get_skeleton(
      uint8_t clid,
      const std::optional<cmn::blob::Pose::Skeletons>& skeletons)
{
    if(not skeletons) {
        return std::nullopt;
    }
    
    auto map = detect::yolo::names::get_map();
    std::optional<blob::Pose::Skeleton> skeleton;
    if(auto it = map.find(clid);
       it != map.end())
    {
        auto name = (std::string)it->second;
        
        if(skeletons) {
            skeleton = skeletons->get(name);
        }
    }
    
    return skeleton;
}

}

bool is_valid_default_model(const std::string& filename) {
    static const std::regex pattern(
        "^"
        "("
            // Group 1: Versions 1 to 10 with 'v' required
            "(yolov([1-9]|10))"
            "|"
            // Group 2: Versions 11 and above without 'v'
            "(yolo("
                "1[1-9]\\d*"       // Versions 11-19, and numbers like 110, 119, etc.
                "|1\\d{2,}"        // Versions 100 and above starting with '1'
                "|[2-9]\\d+"       // Versions 20 and above starting with '2'-'9'
                "|\\d{3,}"         // Any version number with 3 or more digits
            "))"
        ")"
        "([blmnxsucet]|x6|sp|lu|mu|xu)?"  // Optional suffixes
        "("
            "(\\d|[sn])+u"                // Optional pattern
            "|"
            "-(tinyu|cls|sppu|human|obb|oiv7|pose-p6|pose|seg|v8loader|\\d+)+"
        ")?"
        "(\\.pt)?"
        "$"
    );
    return std::regex_match(filename, pattern);
}

std::string default_model() {
    return "yolo11n.pt";
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
    } else if(detection_type() == ObjectDetectionType::precomputed) {
        return meta_video_size;
        
    } else if (detection_type() == ObjectDetectionType::yolo) {
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

namespace track::vi {

std::string VIWeights::toStr() const {
    return glz::write_json(to_json()).value();
}

glz::json_t VIWeights::to_json() const {
    glz::json_t json = std::initializer_list<std::pair<const char*, glz::json_t>>{
        {"path", _path.to_json()},
        {"uniqueness", _uniqueness ? glz::json_t(_uniqueness.value()) : nullptr},
        {"loaded", _loaded},
        {"status", _status},
        {"modified", _modified ? glz::json_t(_modified.value()) : nullptr},
        {"resolution", _resolution ? cvt2json(_resolution.value()) : nullptr},
        {"num_classes", _num_classes ? glz::json_t(_num_classes.value()) : nullptr}
    };
    return json;
}

VIWeights VIWeights::fromStr(const std::string &str)
{
    VIWeights weights;
    //auto s = Meta::fromStr<std::string>(str);
    auto error = glz::read_json(weights, str);
    if(error != glz::error_code::none) {
        std::string descriptive_error = glz::format_error(error, str);
        throw U_EXCEPTION("Error loading VIWeights from JSON:\n", no_quotes(descriptive_error)," full json: ", no_quotes(str));
    }

    //weights._path = unescape(weights._path.str());
    return weights;
}

}
