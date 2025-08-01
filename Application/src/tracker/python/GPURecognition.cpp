#define PYBIND11_CPP17
#ifdef PYBIND11_CPP14
#undef PYBIND11_CPP14
#endif

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wrange-loop-analysis"
#endif

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#include <misc/Image.h>
#include <misc/SpriteMap.h>

#include <misc/default_settings.h>
#include <misc/default_config.h>
#include <misc/GlobalSettings.h>
#include <file/DataLocation.h>
#include <misc/PythonWrapper.h>

#include <misc/DetectionTypes.h>

#include <signal.h>
typedef void (*sighandler_t)(int);

//#define TREX_PYTHON_DEBUG true

using namespace cmn;
namespace py = pybind11;

template<typename T>
bool CHECK_NONE(T obj) {
    return !obj.ptr() || obj.is_none();
}

namespace track {
pybind11::module numpy, TRex, _main, _json_module;
pybind11::dict* _locals = nullptr;
}

namespace pybind11 {
    namespace detail {
        /*template<> struct type_caster<cmn::Image::Ptr>
        {
        public:

            PYBIND11_TYPE_CASTER(cmn::Image::Ptr, _("Image::Ptr"));

            // Conversion part 1 (Python -> C++)
            bool load(py::handle , bool )
            {
                return false;
            }

            //Conversion part 2 (C++ -> Python)
            static py::handle cast(const cmn::Image::Ptr& src, py::return_value_policy, py::handle)
            {

                std::vector<size_t> shape{ src->rows, src->cols, src->dims };
                std::vector<size_t> strides{
                    sizeof(uint8_t) * src->dims * src->cols,
                    sizeof(uint8_t) * src->dims,
                    sizeof(uint8_t)
                };

                return py::buffer_info(
                   src->data(),
                   sizeof(uint8_t),
                   py::format_descriptor<uint8_t>::format(),
                   3,
                   { src->rows, src->cols, src->dims },
                   {
                       sizeof(uint8_t) * src->dims * src->cols,
                       sizeof(uint8_t) * src->dims,
                       sizeof(uint8_t)
                   }
                );
                py::array a(std::move(shape), std::move(strides), src->data());
                return a.release();
            }
            
            static py::handle cast(const cmn::Image::Ptr& src, py::return_value_policy , py::handle )
            {

                std::vector<size_t> shape{ src->rows, src->cols, src->dims };
                std::vector<size_t> strides{
                    sizeof(uint8_t) * src->dims * src->cols,
                    sizeof(uint8_t) * src->dims,
                    sizeof(uint8_t)
                };

                py::array a(std::move(shape), std::move(strides), src->data());
                return a.release();
            }
        };*/
    
        template<> struct type_caster<cmn::Image::Ptr>
        {
        public:

            PYBIND11_TYPE_CASTER(cmn::Image::Ptr, _("Image::Ptr"));

            // Conversion part 1 (Python -> C++)
            bool load(py::handle, bool)
            {
                return false;
            }

            //Conversion part 2 (C++ -> Python)
            static py::handle cast(const cmn::Image::Ptr& src, py::return_value_policy, py::handle)
            {

                std::vector<size_t> shape{ src->rows, src->cols, src->dims };
                std::vector<size_t> strides{
                    sizeof(uint8_t) * src->dims * src->cols,
                    sizeof(uint8_t) * src->dims,
                    sizeof(uint8_t)
                };

                py::array a(std::move(shape), std::move(strides), src->data());
                return a.release();
            }
        };

        template<> struct type_caster<cmn::Image::SPtr>
        {
        public:

            PYBIND11_TYPE_CASTER(cmn::Image::SPtr, _("Image::SPtr"));

            // Conversion part 1 (Python -> C++)
            bool load(py::handle, bool)
            {
                return false;
            }

            //Conversion part 2 (C++ -> Python)
            static py::handle cast(const cmn::Image::SPtr& src, py::return_value_policy, py::handle)
            {

                std::vector<size_t> shape{ src->rows, src->cols, src->dims };
                std::vector<size_t> strides{
                    sizeof(uint8_t) * src->dims * src->cols,
                    sizeof(uint8_t) * src->dims,
                    sizeof(uint8_t)
                };

                py::array a(std::move(shape), std::move(strides), src->data());
                return a.release();
            }
        };
    
    }
} // namespace pybind11::detail

/*#ifdef MESSAGE_TYPE
    #undef MESSAGE_TYPE
#endif
#define MESSAGE_TYPE(NAME, TYPE, FORCE_CALLBACK, COLOR, PREFIX) \
void NAME(const char *cmd, ...) { \
    va_list args; \
    va_start(args, cmd); \
    \
    std::string str;\
    DEBUG::ParseFormatString(str, cmd, args); \
    DEBUG::StatusMsg msg(DEBUG::DEBUG_TYPE::TYPE, str.c_str(), -1, NULL); \
    msg.color = DEBUG::CONSOLE_COLORS::COLOR; \
    msg.prefix = PREFIX; \
    msg.force_callback = false; \
    DEBUG::ParseStatusMessage(&msg); \
    \
    va_end(args); \
}

MESSAGE_TYPE(PythonLog, TYPE_INFO, false, CYAN, "python");
MESSAGE_TYPE(PythonWarn, TYPE_WARNING, false, YELLOW, "python");*/

cmn::GlobalSettings* _settings{ nullptr };
std::function<void(const std::string&, const cv::Mat&)> _mat_display = [](auto&, auto&) { };
std::function<void()> _destroy_all_windows = []() {};

#include "GPURecognition.h"
#include <pybind11/stl.h>
#include <gui/WorkProgress.h>
#include <misc/SoftException.h>

#include <misc/Timer.h>
#include <file/DataLocation.h>

namespace track::detect {
    namespace py = pybind11;

    template<typename T>
    std::shared_ptr<T> transfer_array(py::array_t<T, py::array::c_style | py::array::forcecast> input) {
        py::buffer_info buf_info = input.request();
        T* ptr = static_cast<T*>(buf_info.ptr);
        //std::size_t size = buf_info.size * sizeof(T);

        // Increase reference count to prevent Python from garbage collecting the array
        Py_INCREF(input.ptr());

        // Create shared_ptr with custom deleter that decreases the Python object's reference count
        return std::shared_ptr<T>(ptr, [cap = input.ptr()](T*) mutable {
            Py_DECREF(cap);
            });
    }

    template<typename T>
    std::vector<T> move_array(py::array_t<T, py::array::c_style | py::array::forcecast> input) {
        // Get buffer info and pointer
        py::buffer_info buf_info = input.request();
        T* ptr = static_cast<T*>(buf_info.ptr);

        // Calculate the size of the array
        std::size_t size = static_cast<std::size_t>(buf_info.size);
        return std::vector<T>(ptr, ptr + size);

        // Transfer ownership to a std::shared_ptr with a custom deleter
        /*std::shared_ptr<std::vector<T>> result(new std::vector<T>(ptr, ptr + size), [input](std::vector<T>* vec) mutable {
            // When the std::shared_ptr is deleted, we release the Python buffer
            input.release();
            delete vec;
        });

        // Return the shared pointer
        return result;*/
    }

    class Mask {
    public:
        MaskData data;

    public:
        Mask(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> mask) {
            py::buffer_info buf_info = mask.request();
            int rows = narrow_cast<int>(buf_info.shape[0]);
            int cols = narrow_cast<int>(buf_info.shape[1]);
            auto ptr = move_array<uint8_t>(mask);
            data = MaskData{
                std::move(ptr),
                rows,
                cols
            };
        }
    };

class KeypointArray {
public:
    KeypointData data;

public:
    KeypointArray(py::array_t<float, py::array::c_style | py::array::forcecast> keypoint) {
        py::buffer_info buf_info = keypoint.request();
        size_t points = buf_info.shape[0];
        if(points == 0)
            return;
        
        size_t bones = buf_info.shape[1];
        size_t dims = buf_info.shape[2];
        assert(dims == 2u);
        auto array = transfer_array<float>(keypoint);
        data = KeypointData{
            std::vector<float>(array.get(), array.get() + points * bones * dims),
            bones
        };
    }
};

    std::vector<MaskData> transfer_masks(py::list masks) {
        std::vector<MaskData> result;
        result.reserve(masks.size());

        for (py::handle h : masks) {
            auto mask = h.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
            result.emplace_back(std::move(Mask(mask).data));
        }
        return result;
    }

std::vector<KeypointData> transfer_keypoints(py::list keypoints) {
    std::vector<KeypointData> result;
    result.reserve(keypoints.size());

    for (py::handle h : keypoints) {
        py::array_t<float, py::array::c_style | py::array::forcecast> keypoint = h.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
        result.emplace_back(std::move(KeypointArray(keypoint).data));
    }
    return result;
}

KeypointData::KeypointData(std::vector<float>&& data, size_t bones)
    : _num_bones(bones), _xy_conf(std::move(data))
{
    if (data.size() % (sizeof(Bone) / sizeof(decltype(Bone::x))) != 0u)
        throw InvalidArgumentException("Invalid size for KeypointData constructor. Please use a size that is divisible by ", sizeof(Bone) / sizeof(decltype(Bone::x)), " and is a flat ", Meta::name<decltype(Bone::x)>(), " array.");
    // expecting 3 floats per row, 2 for xy, 1 for conf
    assert(data.size() % (sizeof(Bone) / sizeof(decltype(Bone::x))) == 0u);
    assert(data.size() % _num_bones == 0);
}
    
Keypoint KeypointData::operator[](size_t index) const {
    if (index * num_bones() * 2u >= xy_conf().size())
        throw OutOfRangeException("The index ", index, " is outside the keypoints arrays dimensions of ", size());
    return Keypoint{
        .bones = std::vector<Bone>{
            reinterpret_cast<const Bone*>(xy_conf().data()) + num_bones() * index,
            reinterpret_cast<const Bone*>(xy_conf().data()) + num_bones() * (index + 1)
        }
    };
}

ICXYWHR ObbData::operator[](size_t index) const {
    if (index * 7u >= icxywhr().size())
        throw OutOfRangeException("The index ", index, " is outside the OBB arrays dimensions of ", size());
    return reinterpret_cast<const ICXYWHR*>(icxywhr().data())[index];
}

ObbData::ObbData(std::vector<float>&& data)
    : _icxywhr(std::move(data))
{
    if (not _icxywhr.empty() && _icxywhr.size() % 7u != 0u)
        throw InvalidArgumentException("Invalid size for ObbData constructor. Please use a size that is divisible by 7 and is a flat ICXYWHR array.");
    // expecting 7 floats per row, 1 for id, 1 for confidence, 2 for xy, 2 for wh, 1 for r
    assert(_icxywhr.size() % 7u == 0u);
}


std::array<cmn::Vec2, 4> ICXYWHR::corners() const {
    float cos_r = std::cos(r);
    float sin_r = std::sin(r);
    float dx = w / 2.f;
    float dy = h / 2.f;
    std::array<cmn::Vec2, 4> out;
    // bottom-left
    {
        float x_local = -dx;
        float y_local = -dy;
        out[0] = {x + x_local * cos_r - y_local * sin_r, y + x_local * sin_r + y_local * cos_r};
    }
    // bottom-right
    {
        float x_local = dx;
        float y_local = -dy;
        out[1] = {x + x_local * cos_r - y_local * sin_r, y + x_local * sin_r + y_local * cos_r};
    }
    // top-right
    {
        float x_local = dx;
        float y_local = dy;
        out[2] = {x + x_local * cos_r - y_local * sin_r, y + x_local * sin_r + y_local * cos_r};
    }
    // top-left
    {
        float x_local = -dx;
        float y_local = dy;
        out[3] = {x + x_local * cos_r - y_local * sin_r, y + x_local * sin_r + y_local * cos_r};
    }
    return out;
}

Bounds ICXYWHR::bounding_box() const {
    return bounding_box(corners());
}

Bounds ICXYWHR::bounding_box(const std::array<cmn::Vec2, 4>& pts) {
    float min_x = pts[0].x;
    float min_y = pts[0].y;
    float max_x = pts[0].x;
    float max_y = pts[0].y;
    for (int i = 1; i < 4; ++i) {
        min_x = std::min(min_x, pts[i].x);
        min_y = std::min(min_y, pts[i].y);
        max_x = std::max(max_x, pts[i].x);
        max_y = std::max(max_y, pts[i].y);
    }
    return Bounds(min_x, min_y, max_x - min_x, max_y - min_y);
}

}

using namespace track::detect;

PYBIND11_EMBEDDED_MODULE(TRex, m) {
    namespace py = pybind11;
    using namespace track::detect;

    py::enum_<track::detect::ModelTaskType>(m, "ModelTaskType")
        .value("detect", track::detect::ModelTaskType::detect)
        .value("region", track::detect::ModelTaskType::region)
        .export_values();
    py::enum_<track::detect::ObjectDetectionFormat::data::values>(m, "ObjectDetectionFormat")
        .value("none", track::detect::ObjectDetectionFormat::none)
        .value("boxes", track::detect::ObjectDetectionFormat::boxes)
        .value("masks", track::detect::ObjectDetectionFormat::masks)
        .value("poses", track::detect::ObjectDetectionFormat::poses)
        .value("obb", track::detect::ObjectDetectionFormat::obb)
        .export_values();

    py::class_<DetectResolution>(m, "DetectResolution")
        .def(py::init<int, int>(),
            py::arg("width"),
            py::arg("height"))
        .def(py::init(
           [](std::pair<uint16_t, uint16_t> pair) {
               return DetectResolution{pair.first, pair.second};
           }),
           py::arg("size"))
        .def_readonly("width", &DetectResolution::width)
        .def_readonly("height", &DetectResolution::height)
        .def("__repr__", &DetectResolution::toStr)
        .def("__str__", &DetectResolution::toStr)
        .def_static("class_name", &DetectResolution::class_name);
    
    py::class_<KeypointFormat>(m, "KeypointFormat")
        .def(py::init<uint8_t, uint8_t>(),
            py::arg("n_points"),
            py::arg("n_dims"))
        .def_readonly("n_points", &KeypointFormat::n_points)
        .def_readonly("n_dims", &KeypointFormat::n_dims)
        .def("__repr__", &KeypointFormat::toStr)
        .def("__str__", &KeypointFormat::toStr)
        .def_static("class_name", &KeypointFormat::class_name);
    
    py::class_<ModelConfig>(m, "ModelConfig")
        .def(py::init<ModelTaskType, bool, std::string, DetectResolution, ObjectDetectionFormat::data::values, std::optional<KeypointFormat>
             >(),
            py::arg("task"),
            py::arg("use_tracking"),
            py::arg("model_path"),
            py::arg("trained_resolution") = DetectResolution{},
            py::arg("output") = ObjectDetectionFormat::data::values::none,
            py::arg("keypoints") = std::optional<KeypointFormat>{})
        .def_readwrite("task", &ModelConfig::task)
        .def_readonly("use_tracking", &ModelConfig::use_tracking)
        .def_readonly("model_path", &ModelConfig::model_path)
        .def_readwrite("trained_resolution", &ModelConfig::trained_resolution)
        .def_readwrite("classes", &ModelConfig::classes)
        .def_readwrite("output_format", &ModelConfig::output_format)
        .def_readwrite("keypoint_format", &ModelConfig::keypoint_format)
        .def("__repr__", &ModelConfig::toStr)
        .def("__str__", &ModelConfig::toStr)
        .def_static("class_name", &ModelConfig::class_name);


    py::class_<Rect>(m, "Rect")
        .def("__repr__", [](const Rect& v) -> std::string {
            return v.toStr();
        })
        .def_readonly("x0", &Rect::x0)
        .def_readonly("y0", &Rect::y0)
        .def_readonly("x1", &Rect::x1)
        .def_readonly("y1", &Rect::y1);

    py::class_<Row>(m, "Row")
        .def("__repr__", [](const Row& v) -> std::string {
            return v.toStr();
        })
        .def_readonly("box", &Row::box)
        .def_readonly("clid", &Row::clid)
        .def_readonly("conf", &Row::conf);
    
    py::class_<Bone>(m, "Bone")
        .def("__repr__", [](const Bone& v) -> std::string {
            return v.toStr();
        })
        .def_readonly("x", &Bone::x)
        .def_readonly("y", &Bone::y);
        //.def_readonly("conf", &Bone::conf);
    
    py::class_<Keypoint>(m, "Keypoint")
        .def("__repr__", [](const Keypoint& v) -> std::string {
            return v.toStr();
        })
        .def_readonly("bones", &Keypoint::bones);

    py::class_<Boxes>(m, "Boxes")
        .def(py::init([](py::array_t<float, py::array::c_style | py::array::forcecast> boxes_and_scores) -> Boxes {
            auto ptr = move_array<float>(boxes_and_scores);
            return Boxes{
                std::move(ptr),
                size_t(boxes_and_scores.request().size)
            };
        }))
        .def("__repr__", [](const track::detect::Boxes& boxes) -> std::string {
            return boxes.toStr();
        })
        .def("row", &Boxes::row, py::return_value_policy::reference_internal)
        .def("num_rows", &Boxes::num_rows);
    
    py::class_<KeypointData>(m, "KeypointData")
        .def(py::init([](py::array_t<float, py::array::c_style | py::array::forcecast> xy_and_scores) -> KeypointData {
            return KeypointArray(xy_and_scores).data;
        }))
        .def("__repr__", [](const track::detect::KeypointData& keypoint) -> std::string {
            return keypoint.toStr();
        })
        .def("at", &KeypointData::operator[], py::return_value_policy::reference_internal)
        .def("num_bones", &KeypointData::num_bones);
    py::class_<ObbData>(m, "ObbData")
        .def(py::init([](py::array_t<float, py::array::c_style | py::array::forcecast> icxywhr) -> ObbData {
            return ObbData(move_array<float>(icxywhr));
        }))
        .def("__repr__", [](const track::detect::ObbData& obb) -> std::string {
            return obb.toStr();
        })
        .def("at", &ObbData::operator[], py::return_value_policy::reference_internal)
        .def("size", &ObbData::size);

    py::class_<track::detect::Result>(m, "Result")
        .def(py::init([](int index,
                         track::detect::Boxes boxes_and_scores,
                         py::list masks, track::detect::KeypointData keypoints, track::detect::ObbData obb)
                -> Result
            {
                auto _masks = transfer_masks(masks);
                return track::detect::Result {
                    index,
                    std::move(boxes_and_scores),
                    std::move(_masks),
                    std::move(keypoints),
                    std::move(obb)
                };
            })
        )
        .def("__repr__", [](const track::detect::Result& result) -> std::string {
            return result.toStr();
        })
        .def("index", &Result::index)
        .def("boxes_and_scores", &Result::boxes, py::return_value_policy::reference_internal)
        .def("masks", &Result::masks)
        .def("keypoints", &Result::keypoints)
        .def("obbdata", &Result::obbdata);

    py::class_<cmn::Vec2>(m, "Vec2")
        .def(py::init<>())
        .def("__repr__", [](const cmn::Vec2& v) -> std::string {
            return v.toStr();
        })
        .def_readwrite("x", &cmn::Vec2::x)
        .def_readwrite("y", &cmn::Vec2::y);

    py::class_<cmn::file::Path>(m, "Path")
        .def(py::init<std::string>())
        .def("__repr__", [](const cmn::file::Path& v) -> std::string {
            return v.toStr();
        })
        .def("str", &cmn::file::Path::str)
        .def("exists", &cmn::file::Path::exists);
    
    using namespace track::vi;
    py::class_<VIWeights>(m, "VIWeights")
        .def(py::init([](std::string path, std::optional<double> uniqueness, bool loaded, std::string status, std::optional<uint64_t> modified, std::optional<DetectResolution> resolution, std::optional<uint8_t> classes)
        {
            return VIWeights{
                ._path = path,
                ._uniqueness = uniqueness,
                ._loaded = loaded,
                ._status = status == "FINISHED"
                    ? VIWeights::Status::FINISHED
                    : (status == "PROGRESS"
                        ? VIWeights::Status::PROGRESS
                        : VIWeights::Status::NONE),
                ._modified = modified,
                ._resolution = resolution,
                ._num_classes = classes
            };
        }), py::arg("path"), py::arg("uniqueness"), py::arg("loaded"), py::arg("status"), py::arg("modified"), py::arg("resolution"), py::arg("classes"))
        .def("__repr__", [](const VIWeights& v) -> std::string {
            return v.toStr();
        })
        .def_readwrite("path", &VIWeights::_path)
        .def_readwrite("uniqueness", &VIWeights::_uniqueness)
        .def_readwrite("loaded", &VIWeights::_loaded)
        .def_readwrite("modified", &VIWeights::_modified)
        .def_readwrite("resolution", &VIWeights::_resolution)
        .def_readwrite("classes", &VIWeights::_num_classes)
        .def("to_json", [](const VIWeights& v) -> py::object {
            return track::_json_module.attr("loads")(v.toStr());
        })
        .def("to_string", &VIWeights::toStr);

    py::class_<track::detect::YoloInput>(m, "YoloInput")
        .def(py::init<std::vector<cmn::Image::Ptr>&&, std::vector<cmn::Vec2>&&, std::vector<cmn::Vec2>&&, std::vector<size_t>&&>())
        .def("__repr__", [](const YoloInput& v) -> std::string {
            return v.toStr();
        })
        .def("images", &track::detect::YoloInput::images)
        .def("offsets", &track::detect::YoloInput::offsets)
        .def("scales", &track::detect::YoloInput::scales)
        .def("orig_id", &track::detect::YoloInput::orig_id);

    m.def("log", [](std::string text) {
        Print(fmt::clr<FormatColor::DARK_GRAY>("[py] "), text.c_str());
    });
    m.def("log", [](std::string filename, int line, std::string text) {
        Print(fmt::clr<FormatColor::DARK_GRAY>("[" + (std::string)file::Path(filename).filename() + ":"+Meta::toStr(line) + "] "), text.c_str());
     });

    auto choose_backend = []() -> std::string {
        using namespace default_config;
        if(not _settings)
            throw InvalidArgumentException("No _settings has been set.");
        
        auto torch = py::module::import("torch");
        bool is_cuda_available = torch.attr("cuda").attr("is_available")().cast<bool>();
        std::string backend;

        if (is_cuda_available)
            backend = "cuda";
        // torch.backends.mps.is_available()
        else if (torch.attr("backends").attr("mps").attr("is_available")().cast<bool>())
            backend = "mps";
        else
            backend = "cpu";
        
        return backend;
    };
    
    m.def("choose_backend", choose_backend);

    m.def("choose_device", [choose_backend]() -> std::string {
        using namespace default_config;
        if(not _settings)
            throw InvalidArgumentException("No _settings has been set.");
        
        std::string device;
        auto device_from_settings = GlobalSettings::read_value<default_config::gpu_torch_device_t::Class>("gpu_torch_device");
        if (not device_from_settings
            || *device_from_settings == gpu_torch_device_t::automatic)
        {
            device = "";
        } else {
            auto device_index = GlobalSettings::read_value<int>("gpu_torch_device_index");
            if (device_index && *device_index >= 0)
                device = device_from_settings->toStr() + ":" + Meta::toStr(*device_index);
            else
                device = device_from_settings->toStr();
            
            Print("Using device ", device, " from settings.");
        }
        
        if(device.empty())
            device = choose_backend();

        //auto torch = py::module::import("torch");
        //auto device_obj = torch.attr("device")(device);
        return device;
    });

    m.def("warn", [](std::string text) {
        py::module inspect_mod = py::module::import("inspect");
        py::list frames = inspect_mod.attr("stack")();
        py::object calling_frame = frames[0];
        py::str filename_py = calling_frame.attr("filename");
        py::int_ line_no_py = calling_frame.attr("lineno");
        auto filename = filename_py.cast<std::string>();
        auto line_no = line_no_py.cast<uint32_t>();
        
        FormatWarning(fmt::clr<FormatColor::DARK_GRAY>("[py "), fmt::clr<FormatColor::DARK_YELLOW>(no_quotes(filename)), fmt::clr<FormatColor::DARK_GRAY>(":"), fmt::clr<FormatColor::GREEN>(Meta::toStr(line_no)), fmt::clr<FormatColor::DARK_GRAY>("] "), text.c_str());
    });

    m.def("video_size", []() -> pybind11::dict {
        using namespace pybind11::literals;
        
        pybind11::dict d;
        
        auto size = GlobalSettings::read_value<cmn::Size2>("video_size");
        auto w = size ? size->width : 1,
             h = size ? size->height : 1;

        d["width"] = w;
        d["height"] = h;
        
        return d;
    });

    m.def("setting", [](const std::string& name) -> std::string {
        using namespace pybind11::literals;
        return GlobalSettings::read_value<cmn::NoType>(name).get().valueString();
    });

    m.def("setting", [](const std::string& name, const std::string& value) {
        using namespace pybind11::literals;
        
        try {
            constexpr auto accessLevel = default_config::AccessLevelType::PUBLIC;
            if (!_settings->has_access(name, accessLevel))
                FormatError("User cannot write setting ", name, " (AccessLevel::", _settings->access_level(name).name(), ").");
            else {
                GlobalSettings::write([&](Configuration& config){
                    if (config.values.has(name)) {
                        config.values[name].get().set_value_from_string(value);
                    }
                    else
                        FormatError("Setting ", name, " unknown.");
                });
            }
        }
        catch (...) {
            FormatExcept("Failed to set setting ", name, " to ", value, ".");
        }
        });

    m.def("imshow", [](std::string name, pybind11::buffer b) {
#if CMN_WITH_IMGUI_INSTALLED
        namespace py = pybind11;
        
        /* Request a buffer descriptor from Python */
        py::buffer_info info = b.request();

        /* Some sanity checks ... */
        if (info.format != py::format_descriptor<uint8_t>::format())
            throw std::runtime_error("Incompatible format: expected a uint8_t array!");

        if (info.ndim != 3 && info.ndim != 2)
            throw std::runtime_error("Incompatible buffer dimension!");

        if (not BOOL_SETTING(nowindow))
        {
            auto map = cv::Mat((int)info.shape[0], (int)info.shape[1], (int)CV_8UC(info.ndim == 2 ? 1 : info.shape[2]), static_cast<uint8_t*>(info.ptr));
            _mat_display(name, map);
            //tf::imshow(name, map);
        }
#endif
    }, pybind11::arg().none(), pybind11::arg().noconvert());

    m.def("destroyAllWindows", []() {
#if CMN_WITH_IMGUI_INSTALLED
        namespace py = pybind11;

        if (not BOOL_SETTING(nowindow))
        {
            _destroy_all_windows();
        }
#endif
    });

    py::bind_vector<std::vector<cmn::Image::Ptr>>(m, "ImageVector", "Vector of images");
    py::bind_vector<std::vector<float>>(m, "FloatVector", "Float vector");
    py::bind_vector<std::vector<std::string>>(m, "StringVector", "String vector");
    py::bind_vector<std::vector<long_t>>(m, "LongVector", "Long vector");
    py::bind_vector<std::vector<uchar>>(m, "UcharVector", "Uchar vector");
}

namespace track {
namespace py = pybind11;

std::atomic_bool& initialized() {
    static std::atomic_bool _python_initialized = false;
    return _python_initialized;
}
std::atomic_bool& initializing() {
    static std::atomic_bool _python_initializing = false;
    return _python_initializing;
}
std::atomic_bool& python_gpu_initialized() {
    static std::atomic_bool _python_gpu_initialized = false;
    return _python_gpu_initialized;
}
std::atomic_int& python_major_version() {
    static std::atomic_int _python_major_version = 0;
    return _python_major_version;
}
std::atomic_int& python_minor_version() {
    static std::atomic_int _python_minor_version = 0;
    return _python_minor_version;
}
std::atomic_int& python_uses_gpu() {
    static std::atomic_int _python_uses_gpu = false;
    return _python_uses_gpu;
}

std::string& python_init_error() {
    static std::string _python_init_error = "";
    return _python_init_error;
}
std::string& python_gpu_name() {
    static std::string _python_gpu_name;
    return _python_gpu_name;
}

py::dict json_to_pydict(const glz::json_t& j);

std::mutex module_mutex;
std::map<std::string, std::string> _module_contents;
std::map<std::string, pybind11::module> _modules;

std::shared_mutex initialize_mutex;
std::thread::id _saved_id;
std::unique_ptr<py::scoped_interpreter> _interpreter;

void PythonIntegration::set_settings(GlobalSettings* obj, file::DataLocation* instance, void *python_wrapper) {
    GlobalSettings::set_instance(obj);
    file::DataLocation::set_instance(instance);
    Python::set_instance(python_wrapper);
    _settings = obj;
}

template<typename T>
void set_function_internal(const char* name_, T&& f, const std::string& m);

void PythonIntegration::set_display_function(std::function<void(const std::string&, const cv::Mat&)> fn, std::function<void()> destroy_all_windows) {
    _mat_display = fn;
    _destroy_all_windows = destroy_all_windows;
}

void PythonIntegration::init() {
    struct SignalRestorer {
#if !defined(WIN32)
        PyOS_sighandler_t handler = PyOS_getsig(SIGINT);
        
        ~SignalRestorer() {
#ifndef NDEBUG
            Print("* handler ", hex(handler), " vs ", PyOS_getsig(SIGINT));
#endif
            PyOS_setsig(SIGINT, handler);
        }
#else
        ~SignalRestorer() {
            // after you�ve created the interpreter�
            HMODULE hExe = GetModuleHandle(NULL);                    // or the name of the module that exports it
            auto fn = (void(*)())GetProcAddress(hExe, "RehookConsoleHandler");
            if (fn)
                fn();
        }
#endif
    } _restorer;
    
    auto fail = [](const auto& e, cmn::source_location loc = cmn::source_location::current()){
        python_init_error() = e.what();
        initializing() = false;
        FormatExcept("Python runtime error (GPURecognition:", loc.line(), "): ", e.what());
        
        initialized() = false;
        initializing() = false;
    };
    
    std::unique_lock guard(initialize_mutex);
    initialized() = false;
    initializing() = true;
    
    //! set new thread ID. we expect everything to happen from this thread now.
    _saved_id = std::this_thread::get_id();
    
    if(file::DataLocation::is_registered("app"))
        file::cd(file::DataLocation::parse("app"));
    
    auto trex_init = file::DataLocation::is_registered("app")
    ? file::DataLocation::parse("app", "trex_init.py")
    : "trex_init.py";
    if(not trex_init.exists()) {
        trex_init = "trex_init.py";
    }
    
    try {
        using namespace py::literals;
        
#if defined(WIN32)
        const DWORD buffSize = 65535;
        char path[buffSize] = { 0 };
        GetEnvironmentVariable("PYTHONHOME", path, buffSize);
        Print("Inherited pythonhome: ", std::string(path));
        GetEnvironmentVariable("PYTHONPATH", path, buffSize);
        Print("Inherited pythonpath: ", std::string(path));
        GetEnvironmentVariable("PATH", path, buffSize);
        Print("Inherited path: ", std::string(path));
#endif
        
#if !defined(WIN32)
        // Store the old SIGINT handler for non-Windows systems
        sighandler_t old_sigint_handler = signal(SIGINT, SIG_DFL);
#endif
        
        _interpreter = std::make_unique<py::scoped_interpreter>();
#if !defined(WIN32)
        // Restore the old SIGINT handler
        signal(SIGINT, old_sigint_handler);
#endif
        
        _main = py::module::import("__main__");
        _main.def("set_version", [](std::string x, bool has_gpu, std::string physical_name) {
#ifndef NDEBUG
            Print("set_version called with ",x," and ",physical_name," - ",has_gpu?"gpu":"no gpu");
#endif
            auto array = utils::split(x, ' ');
            if(array.size() > 0) {
                array = utils::split(array.front(), '.');
                if(array.size() >= 1)
                    python_major_version() = Meta::fromStr<int>((std::string)array[0]);
                if(array.size() > 1)
                    python_minor_version() = Meta::fromStr<int>((std::string)array[1]);
            }
            
            python_uses_gpu() = has_gpu;
            python_gpu_name() = physical_name;
        });
        
        TRex = _main.import("TRex");
        _json_module = _main.import("json");
        //numpy = _main.import("numpy");
        _locals = new pybind11::dict();
        Print("# imported TRex module");
        
        PythonIntegration::execute("import sys\nset_version(sys.version, False, '')", false);
        
        try {
#ifdef __APPLE__
            // this is only in here because of cpuinfo being used in YOLO>= and it calls the "python executable" of our embedded program. meaning it spawns another process of our program. not great!
            // https://github.com/ultralytics/ultralytics/blame/c20d2654e95d4d8f1a42e106118f21ddb2762115/ultralytics/utils/torch_utils.py#L55
            // and https://github.com/workhorsy/py-cpuinfo/blob/f3f0fec58335b9699b9b294267c15f516045b1fe/cpuinfo/cpuinfo.py#L2753C18-L2753C18
            try {
                py::exec("import sys; sys.executable = '"+std::string(default_config::conda_environment_path() / "bin" / "python")+"'");
            } catch(...) {
                // catch problems here, but the problem is likely elsewhere so we continue...
            }
#endif
            auto cmd = trex_init.read_file();
            py::exec(cmd);
            python_gpu_initialized() = true;
            initializing() = false;
            
        } catch(const UtilsException& ex) {
            Print("Error while executing ", trex_init,". Content: ",ex.what());
            python_init_error() = ex.what();
            fail(ex);
            //return false;
            throw;
        }
        
        initialized() = true;
        initializing() = false;
        
    } catch(const UtilsException& ex) {
        fail(ex);
        throw SoftException("Error while executing ", trex_init,". Content: ",ex.what());
        
    } catch(py::error_already_set& e) {
        fail(e);
        throw SoftException(e.what());
    } catch(const std::exception& ex) {
        python_init_error() = "Cannot initialize interpreter: "+std::string(ex.what());
        initializing() = false;
        initialized() = false;
        throw SoftException("Cannot initialize interpreter: ", ex.what());
    }
    catch (...) {
        python_init_error() = "Cannot initialize interpreter.";
        initializing() = false;
        initialized() = false;
        throw SoftException("Cannot initialize interpreter.");
    }
}

void PythonIntegration::deinit() {
    try {
        track::numpy.release();
        track::TRex.release();
        track::_main.release();
        track::_json_module.release();

        {
            std::lock_guard<std::mutex> guard(module_mutex);
            _modules.clear();
        }

        if (_locals) {
            delete _locals;
            _locals = nullptr;
        }

    } catch(py::error_already_set &e) {
        throw SoftException("Python runtime error during clean-up: ", e.what());
    }
    
    try {
        _interpreter = nullptr;
        Print(fmt::clr<FormatColor::DARK_GRAY>("[py] "), "ended.");
        
    } catch(py::error_already_set &e) {
        throw SoftException("Python runtime error during clean-up: ", e.what());
    }
}

py::module_ import_module_from_file(pybind11::module& main, const file::Path& module_file_path) {
    // Extract directory path and module name from module_file_path
    file::Path directory_path = module_file_path.remove_filename();
    if(directory_path.empty()) {
        auto mod = main.import(module_file_path.c_str());
        mod.reload();
        return mod;
    }
    
    // Remove extension to get module name
    file::Path module_name_path = module_file_path.has_extension("py")
            ? module_file_path.remove_extension()
            : module_file_path;

    py::module_ sys = main.import("sys");
    py::list original_sys_path = sys.attr("path").cast<py::list>(); // Save original sys.path
    sys.attr("path").attr("append")(directory_path.str()); // Temporarily add new path

    py::module_ mod;
    try {
        auto filename = module_name_path.filename();
        mod = main.import(filename.c_str());
        mod.reload();
    } catch (py::error_already_set& e) {
        std::cerr << "Failed to import module: " << e.what() << std::endl;
        e.restore();
        mod = py::none(); // Set to None to indicate failure
    }

    sys.attr("path") = original_sys_path; // Restore original sys.path
    return mod;
}

bool PythonIntegration::has_loaded_module(const std::string &name) {
    check_correct_thread_id();
    std::unique_lock guard{module_mutex};
    if (_module_contents.contains(name) && not CHECK_NONE(_modules[name])) {
        return true;
    }
    return false;
}

bool PythonIntegration::check_module(const std::string& name,
                                     std::function<void()> unloader)
{
    check_correct_thread_id();
    
    std::unique_lock guard{module_mutex};
    bool result = false;
    
    auto cwd = file::cwd().canonical();
    auto app = file::DataLocation::parse("app").canonical();
    if(not cwd || not app || *cwd != *app) {
        FormatWarning("check_module:CWD: ", cwd, " app:", app);
        if(app)
            file::cd(*app);
    }
        
    auto filename = app ? (*app / name).add_extension("py") : file::Path(name).add_extension("py");
    if(not filename.exists())
        throw U_EXCEPTION("Cannot find the file ", filename, ". Please make sure your TRex installation is not damaged.");
    
    auto c = filename.read_file();
    if (c != _module_contents[name] || CHECK_NONE(_modules[name])) {
        auto& mod = _modules[name];

        try {
            if (CHECK_NONE(mod)) {
                mod = import_module_from_file(_main, filename);
                //mod = _main.import(name.c_str());
            } else if(unloader) {
                guard.unlock();
                unloader();
                guard.lock();
                
                mod.reload();
            }
            Print("Reloaded ",filename,".");
            result = true;
        }
        catch (pybind11::error_already_set & e) {
            FormatExcept("Python runtime exception while reloading ",name,": ", e.what());
            e.restore();
            mod.release();
        }

        _module_contents[name] = c;
    }

#ifdef _WIN32
    // after you�ve created the interpreter�
    HMODULE hExe = GetModuleHandle(NULL);                    // or the name of the module that exports it
    auto fn = (void(*)())GetProcAddress(hExe, "RehookConsoleHandler");
    if (fn) fn();
#endif

    return result;
}

std::optional<glz::json_t> result_to_json(py::handle&& result) {
    if(CHECK_NONE(result)) {
        return std::nullopt;
    }
    
    std::string str;
    try {
        str = result.cast<std::string>();
        
        glz::json_t json;
        auto error = glz::read_json(json, str);
        if(error != glz::error_code::none) {
            std::string descriptive_error = glz::format_error(error, str);
            throw SoftException("Error loading JSON response:\n", no_quotes(descriptive_error));
        }
        return json;
        
    } catch([[maybe_unused]] const py::cast_error& error) {
#ifndef NDEBUG
        FormatError("Cannot convert result to string: ", error.what());
#endif
    }
    return std::nullopt;
}

std::optional<glz::json_t> PythonIntegration::run(const std::string& module_name, const std::string& function, const std::string& parm) {
    check_correct_thread_id();
#ifdef TREX_PYTHON_DEBUG
    Print(fmt::clr<FormatColor::DARK_GRAY>("[py] "), "Running ",module_name.c_str(),"::",function.c_str());
#endif
    
    std::unique_lock<std::mutex> guard(module_mutex);

    try {
        py::handle module;
        
        if(!CHECK_NONE(_modules.at(module_name))) {
            if(function.empty())
                module = _modules.at(module_name);
            else
                module = _modules.at(module_name).attr(function.c_str());
        }
        
        if(!CHECK_NONE(module)) {
            guard.unlock();
            
            auto result = module(parm);
            return result_to_json(std::move(result));
            
        } else
            FormatExcept("Pointer of ",module_name,"::",function," is null.");
    }
    catch (pybind11::error_already_set & e) {
        e.restore();
        
        if (PyErr_Occurred()) {
            PyErr_PrintEx(0);
            PyErr_Clear(); // this will reset the error indicator so you can run Python code again
        }
        
        _modules.at(module_name).release();
        //_modules.at(module_name) = pybind11::none();
        throw SoftException("Python runtime exception while running ", module_name.c_str(),"::", function.c_str(),"(): ", e.what());
        
    } catch(const std::exception& e) {
        throw SoftException("Non-python error while running ", module_name.c_str(),"::", function.c_str(),"(): ", e.what());
        
    } catch(...) {
        throw SoftException("Unknown exception while running ", module_name.c_str(),"::", function.c_str(),"().");
    }
    
    return std::nullopt;
}

std::optional<glz::json_t> PythonIntegration::run(const std::string& module_name, const std::string& function) {
    check_correct_thread_id();
#ifdef TREX_PYTHON_DEBUG
    Print(fmt::clr<FormatColor::DARK_GRAY>("[py] "), "Running ",module_name.c_str(),"::",function.c_str());
#endif
    
    std::unique_lock<std::mutex> guard(module_mutex);

    try {
        py::handle module;
        
        if(!CHECK_NONE(_modules.at(module_name))) {
            if(function.empty())
                module = _modules.at(module_name);
            else
                module = _modules.at(module_name).attr(function.c_str());
        }
        
        if(!CHECK_NONE(module)) {
            guard.unlock();
            auto result = module();
            return result_to_json(std::move(result));
            
        } else
            FormatExcept("Pointer of ",module_name,"::",function," is null.");
    }
    catch (pybind11::error_already_set & e) {
        e.restore();
        
        if (PyErr_Occurred()) {
            PyErr_PrintEx(0);
            PyErr_Clear(); // this will reset the error indicator so you can run Python code again
        }
        
        _modules.at(module_name).release();
        //_modules.at(module_name) = pybind11::none();
        throw SoftException("Python runtime exception while running ", module_name.c_str(),"::", function.c_str(),"(): ", e.what());
        
    } catch(const std::exception& e) {
        throw SoftException("Non-python error while running ", module_name.c_str(),"::", function.c_str(),"(): ", e.what());
        
    } catch(...) {
        throw SoftException("Unknown exception while running ", module_name.c_str(),"::", function.c_str(),"().");
    }
    
    return std::nullopt;
}

std::optional<glz::json_t> PythonIntegration::run(const std::string& module_name, const std::string& function, const glz::json_t& json)
{
    check_correct_thread_id();
#ifdef TREX_PYTHON_DEBUG
    Print(fmt::clr<FormatColor::DARK_GRAY>("[py] "), "Running ",module_name.c_str(),"::",function.c_str());
#endif
    
    std::unique_lock<std::mutex> guard(module_mutex);

    try {
        py::handle module;
        
        if(!CHECK_NONE(_modules.at(module_name))) {
            if(function.empty())
                module = _modules.at(module_name);
            else
                module = _modules.at(module_name).attr(function.c_str());
        }
        
        if(!CHECK_NONE(module)) {
            guard.unlock();
            
            auto result = module(json_to_pydict(std::move(json)));
            return result_to_json(std::move(result));
            
        } else
            FormatExcept("Pointer of ",module_name,"::",function," is null.");
    }
    catch (pybind11::error_already_set & e) {
        e.restore();
        
        if (PyErr_Occurred()) {
            PyErr_PrintEx(0);
            PyErr_Clear(); // this will reset the error indicator so you can run Python code again
        }
        
        _modules.at(module_name).release();
        //_modules.at(module_name) = pybind11::none();
        throw SoftException("Python runtime exception while running ", module_name.c_str(),"::", function.c_str(),"(): ", e.what());
        
    } catch(const std::exception& e) {
        throw SoftException("Non-python error while running ", module_name.c_str(),"::", function.c_str(),"(): ", e.what());
        
    } catch(...) {
        throw SoftException("Unknown exception while running ", module_name.c_str(),"::", function.c_str(),"().");
    }
    
    return std::nullopt;
}

std::string PythonIntegration::run_retrieve_str(const std::string& module_name, const std::string& function)
{
    check_correct_thread_id();
    
    std::unique_lock<std::mutex> guard(module_mutex);
    try {
        if(!CHECK_NONE(_modules.at(module_name))) {
            auto result = _modules.at(module_name).attr(function.c_str());
            guard.unlock();
            return result().cast<std::string>();
        }
    }
    catch (pybind11::error_already_set & e) {
        e.restore();

        _modules.at(module_name).release();
        throw SoftException("Python runtime exception while running ", module_name.c_str(),"::", function.c_str(),"(): ", e.what());
    }
    
    return "";
}

std::optional<std::string> PythonIntegration::variable_to_string(const std::string &name, const std::string &module_name) {
    // Ensure we are on the correct thread
    check_correct_thread_id();
    py::object var;

    if(module_name.empty()) {
        // Look in the global _locals dictionary
        if (not py::hasattr(*_locals, name.c_str())) {
            return std::nullopt;
        }
        var = _locals->attr(name.c_str());
    }
    else {
        // Look in the specified module
        if (_modules.count(module_name) == 0)
            return std::nullopt;
        auto &mod = _modules[module_name];
        if (CHECK_NONE(mod)
            || not py::hasattr(mod, name.c_str()))
        {
            return std::nullopt;
        }
        
        var = mod.attr(name.c_str());
    }

    // Check if the retrieved object is None or invalid
    if (CHECK_NONE(var))
        return std::nullopt;

    try {
        // Convert the object to its string representation.
        std::string repr = py::str(var).cast<std::string>();
        return repr;
    }
    catch(py::error_already_set &e) {
        e.restore();
        return std::nullopt;
    }
}

template<typename T>
T get_variable_internal(const std::string& name, const std::string& m) {
    PythonIntegration::check_correct_thread_id();
    
    try {
        if(m.empty()) {
            if(_locals->contains(name.c_str())) {
                if(!CHECK_NONE(_locals->attr(name.c_str())))
                    return _locals->attr(name.c_str()).cast<T>();
            }
        } else {
            if(_modules.count(m)) {
                auto &mod = _modules[m];
                if(!CHECK_NONE(mod))
                    if(!CHECK_NONE(mod.attr(name.c_str())))
                        return mod.attr(name.c_str()).cast<T>();
            }
        }
    } catch(py::error_already_set & e) {
        FormatExcept("Python runtime error: ", e.what());
    }
    
    throw SoftException("Cannot find variable ", name," in ", m,".");
}

template<> TREX_EXPORT std::string PythonIntegration::get_variable(const std::string& name, const std::string& m) {
    return get_variable_internal<std::string>(name, m);
}
template<> TREX_EXPORT float PythonIntegration::get_variable(const std::string& name, const std::string& m) {
    return get_variable_internal<float>(name, m);
}

template<typename T>
void set_function_internal(const char* name_, T&& f, const std::string& m) {
    PythonIntegration::check_correct_thread_id();
#ifdef TREX_PYTHON_DEBUG
    Print(fmt::clr<FormatColor::DARK_GRAY>("[py] "), "defining function ",m.c_str(),"::",name_);
#endif
    
    if(m.empty()) {
        _main.def(name_, std::move(f));
    } else {
        if(_modules.count(m)) {
            auto &mod = _modules[m];
            if(!CHECK_NONE(mod)) {
                mod.def(name_, std::move(f));
                return;
            }
        }
        
        throw SoftException("Cannot define function ",fmt::clr<FormatColor::DARK_CYAN>(m.c_str()),"::", fmt::clr<FormatColor::CYAN>(name_)," because the module ",fmt::clr<FormatColor::DARK_CYAN>(m.c_str())," does not exist (you should probably have a look at previous error messages).");
    }
}

bool PythonIntegration::valid(const std::string & name_, const std::string& m) {
    return /*exists(name_, m) &&*/ !is_none(name_, m);
}

bool PythonIntegration::exists(const std::string & name_, const std::string& m) {
    PythonIntegration::check_correct_thread_id();
    
    if(m.empty()) {
        return _main.contains(name_);
    } else {
        if(_modules.count(m)) {
            auto &mod = _modules[m];
            if(!CHECK_NONE(mod)) {
                return mod.contains(name_);
            }
        }
        
        throw SoftException("Cannot define function ",fmt::clr<FormatColor::DARK_CYAN>(m.c_str()),"::", fmt::clr<FormatColor::CYAN>(name_)," because the module ",fmt::clr<FormatColor::DARK_CYAN>(m.c_str())," does not exist (you should probably have a look at previous error messages).");
    }
}

std::vector<ModelConfig> PythonIntegration::set_models(const std::vector<ModelConfig>& config, const std::string& m) {
    PythonIntegration::check_correct_thread_id();
    
    try {
        if (m.empty()) {
            return py::cast<std::vector<ModelConfig>>(_main.attr("load_yolo")(config));
        }
        else {
            if (_modules.count(m)) {
                auto& mod = _modules[m];
                if (!CHECK_NONE(mod)) {
                    return py::cast<std::vector<ModelConfig>>(mod.attr("load_yolo")(config));
                }
            }
        
            throw SoftException("Cannot call function ", fmt::clr<FormatColor::DARK_CYAN>(m.c_str()), "::", fmt::clr<FormatColor::CYAN>("load_yolo"), " because the module ", fmt::clr<FormatColor::DARK_CYAN>(m.c_str()), " does not exist (you should probably have a look at previous error messages).");
        }
    } catch(py::error_already_set& e) {
        std::string what = e.what();
        //e.restore();
        throw SoftException("Cannot call function ", fmt::clr<FormatColor::DARK_CYAN>(m.c_str()), "::", fmt::clr<FormatColor::CYAN>("load_yolo"), " because ",what.c_str(),".");
    }
}

std::vector<track::detect::Result> PythonIntegration::predict(track::detect::YoloInput&& input, const std::string& m) {
    PythonIntegration::check_correct_thread_id();

    if (m.empty()) {
        return _main.attr("predict")(std::move(input)).cast<std::vector<track::detect::Result>>();
    }
    else {
        if (_modules.count(m)) {
            auto& mod = _modules[m];
            if (!CHECK_NONE(mod)) {
                return mod.attr("predict")(std::move(input)).cast<std::vector<track::detect::Result>>();
            }
        }

        throw SoftException("Cannot call function ", fmt::clr<FormatColor::DARK_CYAN>(m.c_str()), "::", fmt::clr<FormatColor::CYAN>("predict"), " because the module ", fmt::clr<FormatColor::DARK_CYAN>(m.c_str()), " does not exist (you should probably have a look at previous error messages).");
    }
}

// Forward declaration of the function to handle recursive calls
py::object json_to_pyobject(const glz::json_t& j);

py::dict json_to_pydict(const glz::json_t& j) {
    py::dict dict;
    if(j.is_null())
        return dict;
    
    for (const auto& [key, item] : j.get_object()) {
        dict[py::str(key)] = json_to_pyobject(item);
    }
    return dict;
}

py::list json_to_pylist(const glz::json_t& j) {
    py::list list;
    for (const auto& item : j.get_array()) {
        list.append(json_to_pyobject(item));
    }
    return list;
}

py::object json_to_pyobject(const glz::json_t& j) {
    if (j.is_null()) {
        return py::none();
    } else if (j.is_boolean()) {
        return py::bool_(j.get<bool>());
    } else if (j.is_number()) {
        // Checking for integer vs. floating-point
        //if (j.is_number_integer()) {
        //    return py::int_(j.get<int>());
        //} else {
            return py::float_(j.get<double>());
        //}
    } else if (j.is_string()) {
        return py::str(j.get<std::string>());
    } else if (j.is_array()) {
        return json_to_pylist(j);
    } else if (j.is_object()) {
        return json_to_pydict(j);
    }
    throw std::runtime_error("Unsupported JSON type");
}

void PythonIntegration::set_function(const char* name_, std::function<bool()> f, const std::string &m) {
    set_function_internal(name_, f, m);
}
void PythonIntegration::set_function(const char* name_, std::function<glz::json_t()> f, const std::string &m) {
    set_function_internal(name_, [f = std::move(f)]() -> py::dict {
        //py::dict dict = json_to_pydict(f());
        try {
            glz::json_t json = f();
            if(json.is_null()) {
                return py::none();
            }
            return json_to_pydict(std::move(json));
            
        } catch(py::error_already_set& e) {
            FormatExcept("Exception: ", e.what());
            e.restore();
        }
        return py::none();
    }, m);
}
void PythonIntegration::set_function(const char* name_, std::function<float()> f, const std::string &m) {
    set_function_internal(name_, f, m);
}
void PythonIntegration::set_function(const char* name_, std::function<void(float)> f, const std::string &m) {
    set_function_internal(name_, f, m);
}
void PythonIntegration::set_function(const char* name_, std::function<void(std::string)> f, const std::string &m) {
    set_function_internal(name_, f, m);
}
void PythonIntegration::set_function(const char* name_, std::function<void(std::vector<uint64_t> Ns,
                       std::vector<float> vector,
                       std::vector<float> masks,
                       std::vector<float> meta,
                       std::vector<int>, std::vector<int>)> f,
    const std::string &m)
{
    set_function_internal(name_, f, m);
}


void PythonIntegration::set_function(const char* name_, std::function<void(std::vector<float>)> f, const std::string &m)
{
    set_function_internal(name_, f, m);
}

void PythonIntegration::set_function(const char* name_, std::function<void(const std::vector<track::detect::Result>&)> f, const std::string& m)
{
    set_function_internal(name_, f, m);
}

void PythonIntegration::set_function(const char* name_, std::function<void(const std::vector<std::vector<cv::Mat>>&)> f, const std::string& m)
{
    auto fn = [f](py::list batch) {
        std::vector<std::vector<cv::Mat>> batch_vector;
        // Each item in the batch is a list of images
        for (const py::handle& img_list_handle : batch) {
            py::list img_list = py::cast<py::list>(img_list_handle);

            std::vector<cv::Mat> image_vector;
            for (const py::handle& np_img_handle : img_list) {
                py::array_t<uint8_t> np_img = py::cast<py::array_t<uint8_t>>(np_img_handle);
                py::buffer_info buf_info = np_img.request();

                int nrows = narrow_cast<int>(buf_info.shape[0]);
                int ncols = narrow_cast<int>(buf_info.shape[1]);
                int nchannels = buf_info.ndim == 3 ? narrow_cast<int>(buf_info.shape[2]) : 1;

                cv::Mat img(nrows, ncols, nchannels == 1 ? CV_8UC1 : CV_8UC3, buf_info.ptr);

                image_vector.push_back(img);
            }
            batch_vector.push_back(image_vector);
        }
        f(batch_vector);
    };

    set_function_internal(name_, fn, m);
}

void PythonIntegration::set_function(const char* name_, std::function<void(std::vector<uchar>&)> f, const std::string& m)
{
    auto fn = [f](py::array_t<uint8_t, py::array::c_style | py::array::forcecast> input_array) {
        py::buffer_info buf_info = input_array.request();
        std::vector<uint8_t> vec(static_cast<uint8_t*>(buf_info.ptr),
            static_cast<uint8_t*>(buf_info.ptr) + buf_info.size);
        f(vec);
    };

    set_function_internal(name_, fn, m);
}

void PythonIntegration::set_function(const char* name_, std::function<void(std::vector<uint64_t>, std::vector<float>)> f, const std::string &m)
{
    set_function_internal(name_, f, m);
}

void PythonIntegration::set_function(const char* name_, std::function<void(std::vector<uchar>, std::vector<float>)> f, const std::string& m)
{
    set_function_internal(name_, f, m);
}

void PythonIntegration::set_function(const char* name_, std::function<void(std::vector<float>, std::vector<float>)> f, const std::string& m)
{
    set_function_internal(name_, f, m);
}

void PythonIntegration::set_function(const char* name_, std::function<void(std::vector<float>, std::vector<float>, std::vector<int>)> f, const std::string& m)
{
    set_function_internal(name_, f, m);
}

void PythonIntegration::set_function(const char* name_, std::function<void(std::vector<int>)> f, const std::string &m)
{
    set_function_internal(name_, f, m);
}

void PythonIntegration::set_function(const char* name_, cmn::package::F<void(std::vector<std::vector<float>>&&,std::vector<float>&&)>&& f, const std::string &m)
{
    set_function_internal(name_, [f = std::move(f)](std::vector<std::vector<float>>&& a,std::vector<float>&& b) mutable {
        f(std::move(a), std::move(b));
    }, m);
}

template<>
void PythonIntegration::set_function(const char* name_, cmn::package::F<void(std::vector<float>)>&& f, const std::string &m)
{
    set_function_internal(name_, [f = std::move(f)](std::vector<float> v) mutable {
        f(v);
    }, m);
}

template<>
void PythonIntegration::set_function(const char* name_, cmn::package::F<void(std::vector<int64_t>)>&& f, const std::string &m)
{
    set_function_internal(name_, [f = std::move(f)](std::vector<int64_t> v) mutable {
        f(v);
    }, m);
}

void PythonIntegration::set_function(const char* name_, std::function<void(std::vector<uchar>, std::vector<std::string>)> f, const std::string &m)
{
    set_function_internal(name_, f, m);
}

void PythonIntegration::unset_function(const char *name_, const std::string &m) {
    check_correct_thread_id();
#ifdef TREX_PYTHON_DEBUG
    Print(fmt::clr<FormatColor::DARK_GRAY>("[py] "), "Undefining function ",m.c_str(),"::",name_);
#endif
    if(m.empty()) {
        if(!CHECK_NONE(_main.attr(name_))) {
            _main.attr(name_) = nullptr;
        }
#ifdef TREX_PYTHON_DEBUG
        else
            FormatWarning("Cannot find ",std::string(name_)," in _main.");
#endif
    } else {
        if(_modules.count(m)) {
            auto &mod = _modules[m];
            if(!CHECK_NONE(mod)) {
                mod.attr(name_) = nullptr;
            }
        }
    }
}

bool PythonIntegration::is_none(const std::string& name, const std::string &m) {
    check_correct_thread_id();
    
    try {
        if(m.empty()) {
            if(py::hasattr(_main, name.c_str())
               && CHECK_NONE(_main.attr(name.c_str())))
            {
                return false;
            }
            
        } else {
            if(_modules.count(m)) {
                auto &mod = _modules[m];
                if(!CHECK_NONE(mod)
                   && py::hasattr(mod, name.c_str()))
                {
                    if(!CHECK_NONE(mod.attr(name.c_str())))
                        return false;
                }
            }
        }
    } catch(py::error_already_set& e) {
        e.restore();
    }
    
    return true;
}

void PythonIntegration::convert_python_exceptions(std::function<void()>&& fn)
{
    try {
        fn();
    } catch(py::error_already_set& e) {
        e.restore();
        throw SoftException(e.what());
    }
}

/*/else SOFT_EXCEPTION("Cannot find key '%S'.", &m); \*/
#define IMPL_VARIABLE_SHAPE(T) void PythonIntegration::set_variable(const std::string& name, const std::vector< T >& input, const std::string& m, const std::vector<size_t>& shapes, const std::vector<size_t>& strides) {\
    check_correct_thread_id(); \
    if(m.empty()) \
        (*_locals)[name.c_str()] = pybind11::array_t<T> ( !shapes.empty() ? shapes : std::vector<size_t>{ input.size() }, !strides.empty() ? strides : std::vector<size_t>{ sizeof(T) }, input.data() ); \
    else if(_modules.count(m)) { \
        auto &mod = _modules[m]; \
        if(mod.ptr() != nullptr) \
            mod.attr(name.c_str()) = pybind11::array_t<T> ( !shapes.empty() ? shapes : std::vector<size_t>{ input.size() }, !strides.empty() ? strides : std::vector<size_t>{ sizeof(T) }, input.data() ); \
    } \
}
#define IMPL_VARIABLE(T) void PythonIntegration::set_variable(const std::string& name, T input, const std::string& m) {\
    check_correct_thread_id(); \
    if(m.empty()) \
        (*_locals)[name.c_str()] = input; \
    else if(_modules.count(m)) { \
        auto &mod = _modules[m]; \
        if(mod.ptr() != nullptr) \
            mod.attr(name.c_str()) = input; \
    } \
}
IMPL_VARIABLE(const std::vector<Image::SPtr>&)
IMPL_VARIABLE(const std::vector<Image::Ptr>&)
IMPL_VARIABLE_SHAPE(long_t)
IMPL_VARIABLE_SHAPE(uint32_t)
IMPL_VARIABLE_SHAPE(float)
IMPL_VARIABLE(float)
IMPL_VARIABLE(long_t)
IMPL_VARIABLE(const std::string&)
IMPL_VARIABLE(const char*)
IMPL_VARIABLE(bool)
IMPL_VARIABLE(uint64_t)

void PythonIntegration::set_variable(const std::string & name, Size2 v, const std::string& m) {
    check_correct_thread_id();
    
    std::vector<Float2_t> vec{
        v.width, v.height
    };
    if(m.empty())
        (*_locals)[name.c_str()] = vec;
    else if(_modules.count(m)) {
        auto &mod = _modules[m];
        if(mod.ptr() != nullptr)
            mod.attr(name.c_str()) = vec;
    }
}

void PythonIntegration::set_variable(const std::string & name, Vec2 v, const std::string& m) {
    check_correct_thread_id();
    
    std::vector<Float2_t> vec{
        v.x, v.y
    };
    if(m.empty())
        (*_locals)[name.c_str()] = vec;
    else if(_modules.count(m)) {
        auto &mod = _modules[m];
        if(mod.ptr() != nullptr)
            mod.attr(name.c_str()) = vec;
    }
}

void PythonIntegration::set_variable(const std::string & name, const std::vector<Idx_t> & v, const std::string& m) {
    check_correct_thread_id();
    
    std::vector<uint32_t> copy(v.size());
    for(size_t i=0; i<v.size(); ++i) {
        copy[i] = v[i].get();
    }
    
    if(m.empty())
        (*_locals)[name.c_str()] = copy;
    else if(_modules.count(m)) {
        auto &mod = _modules[m];
        if(mod.ptr() != nullptr)
            mod.attr(name.c_str()) = copy;
    }
}

void PythonIntegration::set_variable(const std::string & name, const std::vector<Vec2> & v, const std::string& m) {
    check_correct_thread_id();
    
    std::vector<float> copy(v.size() * 2);
    for(size_t i=0; i<v.size(); ++i) {
        copy[i * 2u] = v[i].x;
        copy[i * 2u + 1u] = v[i].y;
    }
    py::array_t<float> a(std::vector<size_t>{v.size(), 2}, copy.data());
    
    
    if(m.empty())
        (*_locals)[name.c_str()] = copy;
    else if(_modules.count(m)) {
        auto &mod = _modules[m];
        if(mod.ptr() != nullptr)
            mod.attr(name.c_str()) = copy;
    }
}

void PythonIntegration::set_variable(const std::string & name, const std::vector<std::string> & v, const std::string& m) {
    check_correct_thread_id();
    
    if(m.empty())
        (*_locals)[name.c_str()] = v;
    else if(_modules.count(m)) {
        auto &mod = _modules[m];
        if(mod.ptr() != nullptr)
            mod.attr(name.c_str()) = v;
    }
}

bool PythonIntegration::is_correct_thread_id() {
    std::shared_lock guard(initialize_mutex);
    return std::this_thread::get_id() == _saved_id;
}

void PythonIntegration::check_correct_thread_id() {
    if(is_correct_thread_id())
        return;
    
    throw U_EXCEPTION("Executing python code in wrong thread (",get_thread_name(),").");
}

void PythonIntegration::execute(const std::string& cmd, bool safety_check)  {
    if(safety_check)
        check_correct_thread_id();
    
    try {
        pybind11::exec(cmd, pybind11::globals(), *_locals);
    }
    catch (pybind11::error_already_set & e) {
        if (e.what()) {
            throw SoftException(e.what());
        }
        else {
            throw SoftException("Unknown error message from Python.");
        }
    }
}
void PythonIntegration::import_module(const std::string& name) {
    check_correct_thread_id();
    
    std::lock_guard<std::mutex> guard(module_mutex);
    _module_contents[name] = file::Path(name + ".py").read_file();

    try {
        _modules[name] = _main.import(name.c_str());
    }
    catch (pybind11::error_already_set & e) {
        _modules[name].release();
        throw SoftException(e.what());
    }
}

void PythonIntegration::unload_module(const std::string& name) {
    check_correct_thread_id();
    
    std::lock_guard<std::mutex> guard(module_mutex);
    if(_module_contents.contains(name))
        _module_contents.erase(name);

    try {
        if(_modules.contains(name)) {
            _modules[name].release();
            _modules.erase(name);
        }
    }
    catch (pybind11::error_already_set & e) {
        throw SoftException(e.what());
    }
}

}
