#pragma once

#include <commons.pc.h>
#include <misc/Image.h>
#include <misc/SoftException.h>
#include <misc/idx_t.h>
#include <misc/PackLambda.h>
#include <misc/DetectionTypes.h>

namespace file {
    class DataLocation;
}

namespace cmn {
    class GlobalSettings;
}

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
                        int trained_resolution = 640, 
                        int min_image_size = -1,
                        int max_image_size = -1,
                        ObjectDetectionFormat::data::values output = ObjectDetectionFormat::none)
                :   task(task),
                    use_tracking(use_tracking),
                    model_path(model_path), 
                    trained_resolution(trained_resolution),
                    min_image_size(min_image_size != -1 ? min_image_size : trained_resolution), 
                    max_image_size(max_image_size != -1 ? max_image_size : min_image_size),
                    output_format(output)
            {
                // check if model path exists
                if(not is_in(model_path, 
                        "yolov8n-pose.pt", 
                        "yolov8s-pose.pt", 
                        "yolov8m-pose.pt", 
                        "yolov8l-pose.pt", 
                        "yolov8x-pose.pt",
                        "yolov8n.pt",
                        "yolov8s.pt",
                        "yolov8m.pt",
                        "yolov8l.pt",
                        "yolov8x.pt",
                        "yolov8n-seg.pt",
                        "yolov8s-seg.pt",
                        "yolov8m-seg.pt",
                        "yolov8l-seg.pt",
                        "yolov8x-seg.pt"
                    )) 
                {
                    std::ifstream f(model_path.c_str());
                    if (!f.good()) {
                        throw std::invalid_argument("Model path (for task) does not exist: " + model_path);
                    }
                    f.close();
                }
            }

            std::string toStr() const {
                std::ostringstream os;
                os << "ModelConfig<task=" << static_cast<int>(task)
                    << " model_path='" << model_path << "' trained_resolution=" << trained_resolution
                    << " min_image_size=" << min_image_size
                    << " max_image_size=" << max_image_size << ">";
                return os.str();
            }
            static std::string class_name() {
                return "detect::ModelConfig";
            }

            ModelTaskType task;
            bool use_tracking;
            std::string model_path;
            int trained_resolution;
            int min_image_size;
            int max_image_size;
            ObjectDetectionFormat::data::values output_format;
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
                return "Rect<"+Meta::toStr(x0)+","+ Meta::toStr(y0)+" "+ Meta::toStr(x1)+","+ Meta::toStr(y1)+">";
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
            const Row& row(size_t index) const {
                if (index >= num_rows()) {
                    throw SoftException("Index ", index, " out of bounds for array of size ", num_rows(), ".");
                }
                return reinterpret_cast<const Row*>(data.get())[index];
            }

            size_t num_rows() const {
                return rows_count;
            }

            Boxes() = delete;
            Boxes(const Boxes&) = default;
            Boxes& operator=(const Boxes&) = default;
            Boxes(Boxes&&) = default;
            Boxes& operator=(Boxes&&) = default;

            Boxes(std::shared_ptr<float> data, size_t size)
                : data(data), rows_count(size / 6u)
            {
                if (size % 6u != 0u)
                    throw std::invalid_argument("Invalid size for Boxes constructor. Please use a size that is divisible by 6 and is a flat float array.");
                // expecting 6 floats per row, 4 for box, 1 for class id, 1 for confidence
                assert(size % 6u == 0u);
            }

            std::string toStr() const {
                return "Boxes<"+std::to_string(num_rows())+" rows>";
            }
            static std::string class_name() {
                return "detect::Boxes";
            }

            // access operator []
            const Row& operator[](size_t index) const {
                if (index >= num_rows())
                    throw SoftException("Index ", index, " out of bounds for array of size ", num_rows(), ".");
				return row(index);
			}

            // begin and end functions for iterating data
            const Row* begin() const {
				return reinterpret_cast<const Row*>(data.get());
			}
            const Row* end() const {
                return reinterpret_cast<const Row*>(data.get()) + num_rows();
            }

        private:
            std::shared_ptr<float> data;
            size_t rows_count;
        };


        class TREX_EXPORT MaskData {
        private:
            std::shared_ptr<uint8_t> ptr;
            MaskData(std::shared_ptr<uint8_t> ptr, int rows, int cols, int dims = 1) : ptr(ptr), mat(rows, cols, CV_8UC(dims), ptr.get()) { }
            friend class Mask;

        public:
            cv::Mat mat;

            MaskData() = default;
            MaskData(const MaskData&) = default;
            MaskData& operator=(const MaskData&) = default;
            MaskData(MaskData&&) = default;
            MaskData& operator=(MaskData&&) = default;

            std::string toStr() const {
				return "MaskData<"+std::to_string(mat.rows)+"x"+std::to_string(mat.cols)+"x"+std::to_string(mat.channels())+">";
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
                return "KeypointData<"+std::to_string(_num_bones)+" "+(not _xy_conf.empty() ? Meta::toStr(_xy_conf.size() / 2u) : "null")+" triplets>";
            }
            static std::string class_name() {
                return "detect::KeypointData";
            }
        };

        class TREX_EXPORT Result {
        public:
            Result(int index, Boxes&& boxes, std::vector<MaskData> masks, KeypointData&& keypoints)
                : _index(index), _boxes(std::move(boxes)), _masks(std::move(masks)), _keypoints(std::move(keypoints))
            {
                if (_boxes.num_rows() != 0) {
                    if(not _masks.empty() && _masks.size() != _boxes.num_rows())
                        throw std::invalid_argument("Number of masks must be equal to number of boxes.");
                }
            }

            std::string toStr() const {
                return "Result<"+std::to_string(index())+","+_boxes.toStr()+","+Meta::toStr(_masks)+ ","+Meta::toStr(_keypoints)+">";
            }
            static std::string class_name() {
                return "detect::Result";
            }

        protected:
            GETTER(int, index);
            GETTER(Boxes, boxes);
            GETTER(std::vector<MaskData>, masks);
            GETTER(KeypointData, keypoints);
        };

        class TREX_EXPORT YoloInput {
            GETTER(std::vector<Image::Ptr>, images);
            GETTER(std::vector<Vec2>, offsets);
            GETTER(std::vector<Vec2>, scales);
            GETTER(std::vector<size_t>, orig_id);
            std::function<void(std::vector<Image::Ptr>&&)> _delete;

        public:
            YoloInput(std::vector<Image::Ptr>&& images, std::vector<Vec2> offsets, std::vector<Vec2> scales, std::vector<size_t> orig_id, std::function<void(std::vector<Image::Ptr>&&)>&& deleter = nullptr)
				: _images(std::move(images)), _offsets(std::move(offsets)), _scales(std::move(scales)), _orig_id(std::move(orig_id)), _delete(std::move(deleter))
			{ }

            YoloInput(const YoloInput&) = delete;
            YoloInput(YoloInput&&) = default;
            YoloInput& operator=(const YoloInput&) = delete;
            YoloInput& operator=(YoloInput&&) = default;

            ~YoloInput() {
                if (this->_delete)
                    this->_delete(std::move(_images));
            }

            std::string toStr() const {
				return "YoloInput<images="+Meta::toStr(_images)+" offsets="+Meta::toStr(_offsets)+" scales="+Meta::toStr(_scales)+" belongs="+Meta::toStr(_orig_id)+">";
			}
		};
    }

    TREX_EXPORT std::atomic_bool& initialized();
    TREX_EXPORT std::atomic_bool& initializing();
    TREX_EXPORT std::atomic_bool& python_gpu_initialized();
    TREX_EXPORT std::atomic_int& python_major_version();
    TREX_EXPORT std::atomic_int& python_minor_version();
    TREX_EXPORT std::atomic_int& python_uses_gpu();

    TREX_EXPORT std::string& python_init_error();
    TREX_EXPORT std::string& python_gpu_name();

    class TREX_EXPORT PythonIntegration {
    private:
        PythonIntegration() {}
        
    public:
        
        static void set_settings(GlobalSettings*, file::DataLocation*, void* python_wrapper);
        static void set_display_function(std::function<void(const std::string&, const cv::Mat&)>);
        
        static bool exists(const std::string&, const std::string& m = "");
        static bool valid(const std::string&, const std::string& m = "");
        
        static void set_variable(const std::string&, const std::vector<Image::SPtr>&, const std::string & m = "");
        static void set_variable(const std::string&, const std::vector<Image::Ptr>&, const std::string & m = "");
        static void set_variable(const std::string&, const std::vector<long_t>&, const std::string& m = "", const std::vector<size_t>& shape = {}, const std::vector<size_t>& strides = {});
        static void set_variable(const std::string&, const std::vector<uint32_t>&, const std::string& m = "", const std::vector<size_t>& shape = {}, const std::vector<size_t>& strides = {});
        static void set_variable(const std::string&, const std::vector<float>&, const std::string& m = "", const std::vector<size_t>& shape = {}, const std::vector<size_t>& strides = {});
        static void set_variable(const std::string&, const std::vector<std::string>&, const std::string& m = "");
        static void set_variable(const std::string&, const std::vector<Vec2>&, const std::string& m = "");
        static void set_variable(const std::string&, const std::vector<Idx_t>&, const std::string& m = "");
        static void set_variable(const std::string&, float, const std::string& m = "");
        static void set_variable(const std::string&, Vec2, const std::string& m = "");
        static void set_variable(const std::string&, Size2, const std::string& m = "");
        static void set_variable(const std::string&, long_t, const std::string& m = "");
        static void set_variable(const std::string&, const std::string&, const std::string& m = "");
        static void set_variable(const std::string&, bool, const std::string& m = "");
        static void set_variable(const std::string&, uint64_t, const std::string& m = "");
        static void set_variable(const std::string&, const char*, const std::string& m = "");
        static void set_variable(const std::string&, auto, const std::string& m = "") = delete;

        static void execute(const std::string&, bool safety_check = true);
        static void import_module(const std::string&);
        static void unload_module(const std::string&);
        static bool check_module(const std::string&);
        static bool is_none(const std::string& name, const std::string& attribute);
        static void run(const std::string& module_name, const std::string& function);
        static std::string run_retrieve_str(const std::string& module_name, const std::string& function);

        template<typename T>
        static T get_variable(const std::string&, const std::string& = "") {
            //static_assert(false, "Cant use without previously specified type.");
        }

        static std::vector<track::detect::Result> predict(track::detect::YoloInput&&, const std::string &m = "");
        static std::vector<track::detect::ModelConfig> set_models(const std::vector<track::detect::ModelConfig>&, const std::string& m = "");

        static void set_function(const char* name_, std::function<bool(void)> f, const std::string &m = "");
        static void set_function(const char* name_, std::function<float(void)> f, const std::string &m = "");
        static void set_function(const char* name_, std::function<void(float)> f, const std::string &m = "");
        static void set_function(const char* name_, std::function<void(std::string)> f, const std::string &m = "");
        static void set_function(const char* name_, std::function<void(std::vector<uchar>, std::vector<std::string>)> f, const std::string &m = "");
        static void set_function(const char* name_, std::function<void(std::vector<float>)> f, const std::string &m = "");
        static void set_function(const char* name_, std::function<void(std::vector<uchar>, std::vector<float>)> f, const std::string& m = "");
        static void set_function(const char* name_, std::function<void(std::vector<uchar>&)> f, const std::string& m = "");
        static void set_function(const char* name_, std::function<void(const std::vector<std::vector<cv::Mat>>&)> f, const std::string& m = "");
        static void set_function(const char* name_, std::function<void(const std::vector<track::detect::Result>&)> f, const std::string& m = "");
        static void set_function(const char* name_, std::function<void(std::vector<float>, std::vector<float>)> f, const std::string& m = "");
        static void set_function(const char* name_, std::function<void(std::vector<float>, std::vector<float>, std::vector<int>)> f, const std::string& m = "");
        static void set_function(const char* name_, std::function<void(std::vector<uint64_t>, std::vector<float>)> f, const std::string& m = "");
        static void set_function(const char* name_, std::function<void(std::vector<int>)> f, const std::string &m = "");
        static void set_function(const char* name_, cmn::package::F<void(std::vector<std::vector<float>>&&,std::vector<float>&&)>&& f, const std::string &m = "");
        static void set_function(const char* name_, std::function<void(std::vector<uint64_t> Ns,
                            std::vector<float> vector,
                            std::vector<float> masks,
                            std::vector<float> meta,
                            std::vector<int>,
                            std::vector<int>)> f,
                     const std::string& m = "");
        
        //! Setting a lambda function for a vector of T.
        //! @param name_ Name of the function
        //! @param f The function
        //! @param m Module name
        template<typename T>
        static void set_function(const char*,
                                 cmn::package::F<void(std::vector<T>)>&&, const std::string & = "") = delete;
        
        static void unset_function(const char* name_, const std::string &m = "");
        
    public:
        static void check_correct_thread_id();
        static bool is_correct_thread_id();
        static void init();
        static void deinit();
    };

    template<> TREX_EXPORT std::string PythonIntegration::get_variable(const std::string&, const std::string&);
    template<> TREX_EXPORT float PythonIntegration::get_variable(const std::string&, const std::string&);

template<> TREX_EXPORT
void PythonIntegration::set_function(const char* name_,
                  cmn::package::F<void(std::vector<float>)>&& f, const std::string &m);

template<> TREX_EXPORT
void PythonIntegration::set_function(const char* name_,
                  cmn::package::F<void(std::vector<int64_t>)>&& f, const std::string &m);
}
