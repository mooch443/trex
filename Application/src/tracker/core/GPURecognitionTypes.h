#pragma once

#include <commons.pc.h>
#include <core/DetectionTypes.h>
#include <core/SoftException.h>
#include <core/idx_t.h>
#include <misc/frame_t.h>
#include <misc/Image.h>

namespace track {
using namespace cmn;

namespace detect {

enum class TREX_EXPORT ModelTaskType {
    detect,
    region
};

struct TREX_EXPORT ModelConfig {
    ModelConfig(ModelTaskType task,
                bool use_tracking,
                std::string model_path,
                DetectResolution trained_resolution = {},
                ObjectDetectionFormat::data::values output = ObjectDetectionFormat::none,
                std::optional<KeypointFormat> keypoints = std::nullopt)
        : task(task),
          use_tracking(use_tracking),
          model_path(std::move(model_path)),
          trained_resolution(trained_resolution),
          output_format(output),
          keypoint_format(keypoints)
    {
        if(!yolo::is_valid_default_model(this->model_path)) {
            std::ifstream f(this->model_path.c_str());
            if(!f.good()) {
                throw std::invalid_argument("Model path (for task) does not exist: " + this->model_path);
            }
        }
    }

    std::string toStr() const {
        std::string s =
            "ModelConfig<task=" + Meta::toStr(static_cast<int>(task)) +
            " format=" + Meta::toStr(ObjectDetectionFormat::values.at((size_t)output_format)) +
            " use_tracking=" + Meta::toStr(use_tracking) +
            " model_path='" + model_path +
            "' trained_resolution=" + Meta::toStr(trained_resolution);

        if(keypoint_format) {
            s += " keypoints=" + Meta::toStr(keypoint_format->n_points) +
                 "x" + Meta::toStr(keypoint_format->n_dims);
        }

        s += ">";
        return s;
    }

    static consteval std::string_view class_name() {
        return "detect::ModelConfig";
    }

    ModelTaskType task;
    bool use_tracking;
    std::string model_path;
    DetectResolution trained_resolution;
    ObjectDetectionFormat::data::values output_format;
    detect::yolo::names::owner_map_t classes;
    std::optional<KeypointFormat> keypoint_format;
};

struct TREX_EXPORT Rect {
    float x0;
    float y0;
    float x1;
    float y1;

    operator cmn::Bounds() const {
        return cmn::Bounds(x0, y0, x1 - x0, y1 - y0);
    }

    std::string toStr() const {
        return "Rect<" + Meta::toStr(x0) + "," + Meta::toStr(y0) + " " + Meta::toStr(x1) + "," + Meta::toStr(y1) + ">";
    }
};

struct TREX_EXPORT Row {
    Rect box;
    float conf;
    float clid;

    std::string toStr() const {
        return "Row<" + Meta::toStr(box) + " " + Meta::toStr(conf) + " " + Meta::toStr(clid) + ">";
    }
};

class TREX_EXPORT Boxes {
public:
    Boxes() = delete;
    Boxes(const Boxes&) = default;
    Boxes& operator=(const Boxes&) = default;
    Boxes(Boxes&&) = default;
    Boxes& operator=(Boxes&&) = default;

    Boxes(std::vector<float>&& data, size_t size)
        : data(std::move(data)), rows_count(size / 6u)
    {
        if(size != 0 && size % 6u != 0u)
            throw std::invalid_argument("Invalid size for Boxes constructor. Please use a size that is divisible by 6 and is a flat float array.");
        assert(size % 6u == 0u);
    }

    const Row& row(size_t index) const {
        if(index >= num_rows()) {
            throw SoftException("Index ", index, " out of bounds for array of size ", num_rows(), ".");
        }
        return reinterpret_cast<const Row*>(data.data())[index];
    }

    size_t num_rows() const {
        return rows_count;
    }

    std::string toStr() const {
        return "Boxes<" + std::to_string(num_rows()) + " rows>";
    }

    static consteval std::string_view class_name() {
        return "detect::Boxes";
    }

    const Row& operator[](size_t index) const {
        if(index >= num_rows())
            throw SoftException("Index ", index, " out of bounds for array of size ", num_rows(), ".");
        return row(index);
    }

    const Row* begin() const {
        return reinterpret_cast<const Row*>(data.data());
    }

    const Row* end() const {
        return reinterpret_cast<const Row*>(data.data()) + num_rows();
    }

private:
    std::vector<float> data;
    size_t rows_count;
};

class TREX_EXPORT MaskData {
private:
    std::vector<uint8_t> ptr;
    MaskData(std::vector<uint8_t>&& ptr, int rows, int cols, int dims = 1)
        : ptr(std::move(ptr)), mat(rows, cols, CV_8UC(dims), this->ptr.data())
    {}

    friend class Mask;

public:
    cv::Mat mat;

    MaskData() = default;
    MaskData(const MaskData& other)
        : ptr(other.ptr), mat(other.mat.rows, other.mat.cols, CV_8UC(other.mat.channels()), this->ptr.data())
    {}

    MaskData& operator=(const MaskData& other) {
        ptr = other.ptr;
        mat = cv::Mat(other.mat.rows, other.mat.cols, CV_8UC(other.mat.channels()), this->ptr.data());
        return *this;
    }

    MaskData(MaskData&& other)
        : MaskData(std::move(other.ptr), other.mat.rows, other.mat.cols, other.mat.channels())
    {}

    MaskData& operator=(MaskData&& other) {
        ptr = std::move(other.ptr);
        mat = cv::Mat(other.mat.rows, other.mat.cols, CV_8UC(other.mat.channels()), this->ptr.data());
        return *this;
    }

    std::string toStr() const {
        return "MaskData<" + std::to_string(mat.rows) + "x" + std::to_string(mat.cols) + "x" + std::to_string(mat.channels()) + ">";
    }

    static consteval std::string_view class_name() {
        return "detect::MaskData";
    }
};

class Keypoint;

class TREX_EXPORT KeypointData {
    GETTER(uint64_t, num_bones);
    GETTER(std::vector<float>, xy_conf);

public:
    KeypointData(std::vector<float>&& data, size_t bones);

    KeypointData() = default;
    KeypointData(const KeypointData&) = default;
    KeypointData& operator=(const KeypointData&) = default;
    KeypointData(KeypointData&&) = default;
    KeypointData& operator=(KeypointData&&) = default;

    Keypoint operator[](size_t index) const;

    [[nodiscard]] bool empty() const { return _xy_conf.empty(); }
    [[nodiscard]] size_t size() const { return _xy_conf.size() / _num_bones / 2u; }

    std::string toStr() const {
        return "KeypointData<" + std::to_string(_num_bones) + " " + (!_xy_conf.empty() ? Meta::toStr(_xy_conf.size() / 2u) : "null") + " triplets>";
    }

    static consteval std::string_view class_name() {
        return "detect::KeypointData";
    }
};

struct TREX_EXPORT ICXYWHR {
    float clid;
    float conf;
    float x;
    float y;
    float w;
    float h;
    float r;

    std::string toStr() const {
        return "ICXYWHR<" + Meta::toStr(clid) + "," + Meta::toStr(conf) + "," + Meta::toStr(x) + "," + Meta::toStr(y) + "," + Meta::toStr(w) + "," + Meta::toStr(h) + "," + Meta::toStr(r) + ">";
    }

    static consteval std::string_view class_name() {
        return "detect::ICXYWHR";
    }

    std::array<cmn::Vec2, 4> corners() const;
    Bounds bounding_box() const;
    static Bounds bounding_box(const std::array<cmn::Vec2, 4>&);
};

class TREX_EXPORT ObbData {
    GETTER(std::vector<float>, icxywhr);

public:
    ObbData(std::vector<float>&& data);
    ObbData() = default;
    ObbData(const ObbData&) = default;
    ObbData& operator=(const ObbData&) = default;
    ObbData(ObbData&&) = default;
    ObbData& operator=(ObbData&&) = default;

    ICXYWHR operator[](size_t index) const;

    [[nodiscard]] bool empty() const { return _icxywhr.empty(); }
    [[nodiscard]] size_t size() const { return _icxywhr.size() / 7u; }

    std::string toStr() const {
        return "ObbData<" + std::to_string(size()) + ">";
    }

    static consteval std::string_view class_name() {
        return "detect::ObbData";
    }
};

struct TREX_EXPORT ICXYR {
    float clid;
    float conf;
    float x;
    float y;
    float r;

    std::string toStr() const {
        return "ICXYR<" + Meta::toStr(clid) + "," + Meta::toStr(conf) + "," + Meta::toStr(x) + "," + Meta::toStr(y) + "," + Meta::toStr(r) + ">";
    }

    static consteval std::string_view class_name() {
        return "detect::ICXYR";
    }

    std::array<cmn::Vec2, 4> corners() const;
    Bounds bounding_box() const;
    static Bounds bounding_box(const std::array<cmn::Vec2, 4>&);
};

class TREX_EXPORT PointData {
    GETTER(std::vector<float>, icxyr);

public:
    PointData(std::vector<float>&& data);
    PointData() = default;
    PointData(const PointData&) = default;
    PointData& operator=(const PointData&) = default;
    PointData(PointData&&) = default;
    PointData& operator=(PointData&&) = default;

    ICXYR operator[](size_t index) const;

    [[nodiscard]] bool empty() const { return _icxyr.empty(); }
    [[nodiscard]] size_t size() const { return _icxyr.size() / 5u; }

    std::string toStr() const {
        return "PointData<" + std::to_string(size()) + ">";
    }

    static consteval std::string_view class_name() {
        return "detect::PointData";
    }
};

class TREX_EXPORT Result {
public:
    Result(int index,
           Boxes&& boxes,
           std::vector<MaskData>&& masks,
           KeypointData&& keypoints,
           track::detect::ObbData&& obbdata,
           track::detect::PointData&& points)
        : _index(index),
          _boxes(std::move(boxes)),
          _masks(std::move(masks)),
          _keypoints(std::move(keypoints)),
          _obbdata(std::move(obbdata)),
          _points(std::move(points))
    {
        if(_boxes.num_rows() != 0) {
            if(!_masks.empty() && _masks.size() != _boxes.num_rows())
                throw std::invalid_argument("Number of masks must be equal to number of boxes.");
        }
        if(!_obbdata.empty() && _boxes.num_rows() > 0)
            throw std::invalid_argument("Boxes must be empty if obb data is set.");
        if(!_points.empty() && _boxes.num_rows() > 0)
            throw std::invalid_argument("Boxes must be empty if points data is set.");
    }

    std::string toStr() const {
        return "Result<" + std::to_string(index()) + "," + _boxes.toStr() + "," + Meta::toStr(_masks) + "," + Meta::toStr(_keypoints) + "," + Meta::toStr(_obbdata) + "," + Meta::toStr(_points) + ">";
    }

    static consteval std::string_view class_name() {
        return "detect::Result";
    }

protected:
    GETTER(int, index);
    GETTER(Boxes, boxes);
    GETTER(std::vector<MaskData>, masks);
    GETTER(KeypointData, keypoints);
    GETTER(ObbData, obbdata);
    GETTER(PointData, points);
};

class TREX_EXPORT YoloInput {
    GETTER(std::vector<Image::Ptr>, images);
    GETTER(std::vector<Vec2>, offsets);
    GETTER(std::vector<Vec2>, scales);
    GETTER(std::vector<size_t>, orig_id);
    std::function<void(std::vector<Image::Ptr>&&)> _delete;

public:
    YoloInput(std::vector<Image::Ptr>&& images,
              std::vector<Vec2> offsets,
              std::vector<Vec2> scales,
              std::vector<size_t> orig_id,
              std::function<void(std::vector<Image::Ptr>&&)>&& deleter = nullptr)
        : _images(std::move(images)),
          _offsets(std::move(offsets)),
          _scales(std::move(scales)),
          _orig_id(std::move(orig_id)),
          _delete(std::move(deleter))
    {}

    YoloInput(const YoloInput&) = delete;
    YoloInput(YoloInput&&) = default;
    YoloInput& operator=(const YoloInput&) = delete;
    YoloInput& operator=(YoloInput&&) = default;

    ~YoloInput() {
        if(_delete)
            _delete(std::move(_images));
    }

    std::string toStr() const {
        return "YoloInput<images=" + Meta::toStr(_images) + " offsets=" + Meta::toStr(_offsets) + " scales=" + Meta::toStr(_scales) + " belongs=" + Meta::toStr(_orig_id) + ">";
    }
};

enum class TREX_EXPORT Sam3PromptType : uint8_t {
    none,
    text,
    boxes,
    points
};

/**
 * Canonical SAM3 prompt payload transported from C++ to Python.
 *
 * The payload intentionally remains lightweight and format-oriented. Prompt
 * routing and frame association are handled outside this struct.
 */
struct TREX_EXPORT Sam3PromptPayload {
    std::variant<std::monostate, std::string, std::vector<Vec2>, std::vector<Bounds>> value{};
    
    static Sam3PromptPayload fromStr(StringLike auto&& str) {
        std::string_view sv = utils::trim(utils::string_like_view(str));
        if(sv.empty()) {
            return Sam3PromptPayload{};
        }

        if((sv.front() == '"' && sv.back() == '"')
           || (sv.front() == '\'' && sv.back() == '\''))
        {
            Sam3PromptPayload payload;
            payload.value = Meta::fromStr<std::string>(std::string(sv));
            return payload;
        }

        if(sv.front() == '[' && sv.back() == ']') {
            const auto values = Meta::fromStr<std::vector<std::vector<float>>>(std::string(sv));
            if(values.empty()) {
                return Sam3PromptPayload{};
            }

            const bool all_points = std::ranges::all_of(values, [](const auto& row) {
                return row.size() == 2u;
            });
            const bool all_boxes = std::ranges::all_of(values, [](const auto& row) {
                return row.size() == 4u;
            });

            Sam3PromptPayload payload;
            if(all_points) {
                payload.value = std::vector<Vec2>{};
                payload.points().reserve(values.size());
                for(const auto& row : values) {
                    payload.points().emplace_back(row.at(0), row.at(1));
                }
                return payload;
            }

            if(all_boxes) {
                payload.value = std::vector<Bounds>{};
                payload.boxes().reserve(values.size());
                for(const auto& row : values) {
                    payload.boxes().emplace_back(
                        row.at(0),
                        row.at(1),
                        row.at(2),
                        row.at(3)
                    );
                }
                return payload;
            }

            throw InvalidArgumentException(
                "The sam3 prompt ", str,
                " must be plain text, [[x,y],...] points, or [[x0,y0,x1,y1],...] boxes.");
        }

        Sam3PromptPayload payload;
        payload.value = std::string(sv);
        return payload;
    }
    
    std::vector<Vec2>& points() { return std::get<std::vector<Vec2>>(value); }
    std::vector<Bounds>& boxes() { return std::get<std::vector<Bounds>>(value); }
    std::string& text() { return std::get<std::string>(value); }
    const std::vector<Vec2>& points() const { return std::get<std::vector<Vec2>>(value); }
    const std::vector<Bounds>& boxes() const { return std::get<std::vector<Bounds>>(value); }
    const std::string& text() const { return std::get<std::string>(value); }
    bool has_value() const { return not std::holds_alternative<std::monostate>(value); }
    
    Sam3PromptType type() const {
        if(not has_value())
            return Sam3PromptType::none;
        if(std::holds_alternative<std::string>(value))
            return Sam3PromptType::text;
        if(std::holds_alternative<std::vector<Vec2>>(value))
            return Sam3PromptType::points;
        if(std::holds_alternative<std::vector<Bounds>>(value))
            return Sam3PromptType::boxes;
        throw InvalidArgumentException("Invalid data type.");
    }

    glz::json_t to_json() const;
    std::string toStr() const;
    static consteval std::string_view class_name() { return "Sam3Prompt"; }
    bool operator==(const Sam3PromptPayload&) const = default;
    bool operator!=(const Sam3PromptPayload&) const = default;
};

/// Ordered prompts associated with one SAM3 image.
struct TREX_EXPORT Sam3PromptList : public std::vector<Sam3PromptPayload> {
    using base_t = std::vector<Sam3PromptPayload>;
    using base_t::vector; // inherit std::vector constructors

    Sam3PromptList(base_t&& v) : base_t(std::move(v))
    {}

    std::string toStr() const;
    glz::json_t to_json() const;

    static Sam3PromptList fromStr(StringLike auto&& str) {
        auto sv = utils::trim(utils::string_like_view(str));
        if(sv.empty())
            return {};

        if(sv.front() == '[' && sv.back() == ']') {
            /// remove outer [] to get the inner elements
            auto parts = util::parse_array_parts(util::truncate(sv));

            /// we might have either
            ///     1. a normal array of payloads aka [fish,human,[[1,2]],...]
            ///     2. a shortened form e.g. `fish` or `[[1,2]]`
            /// so here we test whether its a double array `[[[...` which has at least 3 brackets
            /// or something else (string)
            if(parts.empty()) {
                return {};

            } else if(parts.front().front() != '['
                      || (parts.front().length() > 2u && parts.front()[1] == '['))
            {
                /// it is either a string (not an array) or an array with at least
                /// (truncated one bracket) [|[[...
                /// not checking for closing brackets is an approximation
                Sam3PromptList result;
                result.reserve(parts.size());
                for(auto &&part : parts)
                    result.push_back(Meta::fromStr<Sam3PromptPayload>(part));

                /// its just a normal array
                return {
                    result
                };

            } else {
                /// here we have only a maximum of `[[` in the original string
                /// so it cant be an array of payloads. it must be a single payload
                /// falls through to the end
            }
        }

        return {
            Meta::fromStr<Sam3PromptPayload>(std::forward<decltype(str)>(str))
        };
    }

    static consteval std::string_view class_name() { return "Sam3PromptList"; }
  };

/// Frame-indexed prompt repository used by C++ settings/state.
struct TREX_EXPORT Sam3Prompts {
    using MapType = std::map<Frame_t, Sam3PromptList>;
    
    using iterator = MapType::iterator;
    using const_iterator = MapType::const_iterator;
    using value_type = MapType::value_type;
    using key_type = MapType::key_type;
    
    MapType map;

    Sam3Prompts() noexcept = default;
    Sam3Prompts(MapType&& map) : map(std::move(map)) {}
    Sam3Prompts(std::initializer_list<MapType::value_type>&& list) : map(std::move(list)) {}

    iterator begin() noexcept { return map.begin(); }
    iterator end() noexcept { return map.end(); }
    const_iterator begin() const noexcept { return map.begin(); }
    const_iterator end() const noexcept { return map.end(); }
    const_iterator cbegin() const noexcept { return map.cbegin(); }
    const_iterator cend() const noexcept { return map.cend(); }

    bool empty() const noexcept { return map.empty(); }
    std::size_t size() const noexcept { return map.size(); }
    
    // Map-like accessors
    Sam3PromptList& at(const key_type& key) { return map.at(key); }
    const Sam3PromptList& at(const key_type& key) const { return map.at(key); }

    Sam3PromptList& operator[](const key_type& key) { return map[key]; }
    Sam3PromptList& operator[](key_type&& key) { return map[std::move(key)]; }

    bool contains(const key_type& key) const { return map.contains(key); }

    iterator find(const Frame_t& key) { return map.find(key); }
    const_iterator find(const Frame_t& key) const { return map.find(key); }
    
    auto erase(auto&& k) { return map.erase(std::forward<decltype(k)>(k)); }
    
    std::string toStr() const;
    glz::json_t to_json() const;
    
    bool operator==(const Sam3Prompts&) const = default;
    bool operator!=(const Sam3Prompts&) const = default;
    
    static Sam3Prompts fromStr(StringLike auto&& str) {
        std::string_view sv = utils::trim(utils::string_like_view(str));
        if(sv.empty()) {
            return {};
        }
        
        if(sv.front() == '{' && sv.back() == '}') {
            /// this is actually a map
            return Sam3Prompts{
                Meta::fromStr<MapType>(std::forward<decltype(str)>(str))
            };
        }
        
        /// try to parse it as a single promptlist then for frame null:
        if(sv.front() == '[' && sv.back() == ']') {
            return Sam3Prompts {
                {
                    Frame_t{},
                    Meta::fromStr<Sam3PromptList>(std::forward<decltype(str)>(str))
                }
            };
        }
        
        /// in this case
        return Sam3Prompts {
            {
                Frame_t{},
                Sam3PromptList{
                    Meta::fromStr<Sam3PromptPayload>(std::forward<decltype(str)>(str))
                }
            }
        };
    }
    
    static consteval std::string_view class_name() { return "Sam3Prompts"; }
};


/// Prompt lists aligned one-to-one with `YoloInput::images()`.
using Sam3PromptsPerImage = std::vector<Sam3PromptList>;

/**
 * SAM3 batch input with prompts already aligned to each image.
 *
 * No session-global or frame-keyed prompt semantics are stored here. C++
 * resolves those before the object crosses into Python.
 */
class TREX_EXPORT Sam3Input {
    GETTER(YoloInput, base);
    GETTER(Sam3PromptsPerImage, prompts_per_image);

public:
    /**
     * Construct an image-aligned SAM3 inference batch.
     *
     * @param base Image batch and geometric metadata.
     * @param prompts_per_image Ordered prompts aligned with `base.images()`.
     */
    Sam3Input(YoloInput&& base, Sam3PromptsPerImage prompts_per_image = {})
        : _base(std::move(base)), _prompts_per_image(std::move(prompts_per_image))
    {
        if(!_prompts_per_image.empty()
           && _prompts_per_image.size() != _base.images().size())
        {
            throw InvalidArgumentException(
                "Sam3Input expects prompts_per_image to be empty or match images().size(), got ",
                _prompts_per_image.size(), " prompts for ", _base.images().size(), " images.");
        }
    }

    std::string toStr() const {
        return "Sam3Input<base=" + _base.toStr() + " prompts_per_image=" + Meta::toStr(_prompts_per_image.size()) + ">";
    }
};

} // namespace detect
} // namespace track
