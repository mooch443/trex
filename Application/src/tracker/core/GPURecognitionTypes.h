#pragma once

#include <commons.pc.h>
#include <core/DetectionTypes.h>
#include <core/SoftException.h>
#include <core/idx_t.h>
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

    static std::string class_name() {
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

    static std::string class_name() {
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

    static std::string class_name() {
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

    static std::string class_name() {
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

    static std::string class_name() {
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

    static std::string class_name() {
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

    static std::string class_name() {
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

    static std::string class_name() {
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

    static std::string class_name() {
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
    text,
    box,
    boxes,
    points,
    mask,
    remove_object
};

struct TREX_EXPORT Sam3PromptPayload {
    Sam3PromptType type = Sam3PromptType::text;
    int64_t frame_index = 0;
    std::optional<std::string> text;
    std::optional<int64_t> obj_id;
    std::vector<Vec2> points;
    std::vector<int32_t> point_labels;
    std::vector<std::array<float, 4>> boxes;
    std::vector<int32_t> labels;
    std::vector<uint8_t> mask;
    Size2 mask_size{};
    bool text_session_scope = false;
    bool text_skip_if_unchanged = true;

    std::string toStr() const {
        return "Sam3PromptPayload<type=" + Meta::toStr(static_cast<int>(type)) +
               " frame=" + Meta::toStr(frame_index) +
               " text=" + Meta::toStr(text) +
               " obj_id=" + Meta::toStr(obj_id) +
               " points=" + Meta::toStr(points.size()) +
               " boxes=" + Meta::toStr(boxes.size()) + ">";
    }
};

class TREX_EXPORT Sam3Input {
    GETTER(YoloInput, base);
    GETTER(std::vector<std::vector<Sam3PromptPayload>>, prompts_per_item);

public:
    Sam3Input(YoloInput&& base, std::vector<std::vector<Sam3PromptPayload>> prompts_per_item = {})
        : _base(std::move(base)), _prompts_per_item(std::move(prompts_per_item))
    {}

    std::string toStr() const {
        return "Sam3Input<base=" + _base.toStr() + " prompts=" + Meta::toStr(_prompts_per_item.size()) + ">";
    }
};

} // namespace detect
} // namespace track
