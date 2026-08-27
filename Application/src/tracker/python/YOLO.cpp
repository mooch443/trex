#include "YOLO.h"
#include <processing/PixelTree.h>
#include <python/DetectionTilePostprocess.h>
#include <python/SegmentationPostprocess.h>
#include <python/PythonWrapper.h>
#include <grabber/misc/default_config.h>
#include <video/Video.h>
#include <misc/Timer.h>
#include <misc/ThreadPool.h>
#include <core/TrackingSettings.h>
#include <python/PipelineRegistry.h>
#include <python/GPURecognition.h>
#include <gui/GuiTypes.h>
#include <python/BackendRegistry.h>

namespace track {

static_assert(ObjectDetection<YOLO>);

using namespace cmn;

struct AcceptanceSettings {
    Float2_t sqcm;
    SizeFilters min_max;
    
    bool is_acceptable(uint64_t pixel_count) const {
        if(min_max.empty())
            return true;
        return min_max.in_range_of_one(pixel_count * sqcm);
    }
    
    static AcceptanceSettings Make() {
        auto cm_per_pixel = READ_SETTING_WITH_DEFAULT(cm_per_pixel, Settings::cm_per_pixel_t{1_F});
        if(cm_per_pixel <= 0_F)
            cm_per_pixel = 1_F;
        return AcceptanceSettings{
            .sqcm = SQR(cm_per_pixel),
            .min_max = READ_SETTING_WITH_DEFAULT(detect_size_filter, SizeFilters{})
        };
    }
};

std::mutex running_mutex;
std::shared_future<void> running_prediction;
std::promise<void> running_promise;

std::mutex init_mutex;
std::future<void> init_future;

std::atomic<bool> yolo_initialized{false};
std::atomic<double> _network_fps{0.0};
std::atomic<size_t> _network_samples{0u};

std::mutex transfer_done_mutex;
std::future<void> transferred_done;

std::vector<detect::ModelConfig> _loaded_models;
std::unique_ptr<GenericThreadPool> _pool;

std::mutex tile_log_mutex;
Size2 last_logged_tile_size{0, 0};
size_t last_logged_tile_count{0};

struct YOLO::Data {
    std::atomic<bool> _background_required;
    std::atomic<bool> _background_set;
    
    Data() {
        reset();
    }
    void reset() {
        _background_required = BOOL_SETTING(track_background_subtraction);
        _background_set = false;
    }
    
    bool has_background() const {
        return not _background_required.load() || _background_set.load();
    }
    void set_background(const Image::Ptr& background) {
        _background_set = background != nullptr;
    }
};

YOLO::Data& YOLO::data() {
    static Data _data;
    return _data;
}

void YOLO::set_background(const Image::Ptr &image) {
    data().set_background(image);
    if(data().has_background())
        detect::pipeline_manager(detect::ObjectDetectionType::yolo).set_paused(false);
}

void YOLO::reinit(ModuleProxy& proxy) {
    proxy.set_variable("model_type", Meta::toStr(detect::detection_type()));
    
    if(READ_SETTING(detect_model, file::Path).empty()) {
        Print("You can provide a model for object detection using the command-line argument -m <path>. Otherwise, we will assume YOLOv8n-pose");
        SETTING(detect_model) = file::Path("yolov8n-pose");
    }

    using namespace track::detect;
    _loaded_models.clear();
    data().reset();

    // caching here since it can be modified above
    auto path = READ_SETTING(detect_model, file::Path);
    if(detect::yolo::valid_model(path)) {
        if(not path.has_extension()) {
            path = path.add_extension("pt"); // pytorch model
        }
        
        _loaded_models.emplace_back(
            ModelTaskType::detect,
            BOOL_SETTING(yolo_tracking_enabled),
            path.str(),
            READ_SETTING(detect_resolution, DetectResolution)
        );
        _loaded_models.back().try_optimize = BOOL_SETTING(detect_try_optimize_model);
        
    } else
        throw U_EXCEPTION("This does not seem like a valid model to use: ", path,". Either we cannot find it, or it is not in a valid format. Expected is a supported PyTorch .pt or .pth model file.");

    if(READ_SETTING(region_model, file::Path).exists()) {
        _loaded_models.emplace_back(
            ModelTaskType::region,
            BOOL_SETTING(yolo_region_tracking_enabled), // region models dont have tracking
            READ_SETTING(region_model, file::Path).str(),
            READ_SETTING(region_resolution, DetectResolution)
        );
        _loaded_models.back().try_optimize = BOOL_SETTING(detect_try_optimize_model);
    }

    if(_loaded_models.empty()) {
        if(not path.empty())
            throw U_EXCEPTION("Cannot find model ", path);
        
        throw U_EXCEPTION("Please provide at least one model to use for segmentation.");
    }
    
    _loaded_models = PythonIntegration::set_models(_loaded_models, proxy.m);
    
    for(auto &config : _loaded_models) {
        if(config.task == ModelTaskType::detect) {
            SETTING(detect_format) = ObjectDetectionFormat_t(config.output_format);
            SETTING(detect_resolution) = config.trained_resolution;
            SETTING(detect_requires_exact_input_size) = config.requires_exact_input_size;
            if(auto detect_classes = READ_SETTING(detect_classes, cmn::blob::MaybeObjectClass_t);
               not detect_classes.has_value()
               || detect_classes->empty())
            {
                Print("// Loading classes from model: ", config.classes);
                SETTING(detect_classes) = cmn::blob::MaybeObjectClass_t{config.classes};
            }
            
            if(config.output_format == ObjectDetectionFormat::poses)
            {
                SETTING(detect_keypoint_format) = config.keypoint_format ? *config.keypoint_format : KeypointFormat{};
            }
            
        } else if(config.task == ModelTaskType::region) {
            SETTING(region_resolution) = config.trained_resolution;
        }
    }
    
    /*if(auto detect_format = READ_SETTING(detect_format, ObjectDetectionFormat_t);
       detect_format == ObjectDetectionFormat::boxes)
    {
        if(BOOL_SETTING(calculate_posture)) {
            FormatWarning("Disabling posture for now, since pure detection models cannot produce useful posture (everything will be rectangles).");
            SETTING(calculate_posture) = false;
        }
    }*/
}

void YOLO::init() {
    bool expected = false;
    if(yolo_initialized.compare_exchange_strong(expected, true)) {
        data().reset();

        _network_fps = _network_samples = 0;
        _pool = std::make_unique<GenericThreadPool>(3, "Yolo");

        detect::register_pipeline(
            detect::ObjectDetectionType::yolo,
            max(1u, READ_SETTING(detect_batch_size, uchar)),
            /*start_paused=*/true,
            [](std::vector<TileImage>&& images) {
#ifndef NDEBUG
                if(images.empty())
                    FormatExcept("Images is empty :(");
#endif
                YOLO::apply(std::move(images));
            });

        std::unique_lock guard(init_mutex);
        if(init_future.valid())
            init_future.get();

        Python::schedule([](){
            ModuleProxy proxy{
                ThrowAlways{},
                "bbx_saved_model",
                YOLO::reinit
            };
        }).get();

        if(data().has_background())
            detect::pipeline_manager(detect::ObjectDetectionType::yolo).set_paused(false);
        
        //! this will block everything + the GUI
        //! unfortunately currently this is the lazy solution
        //! to the model resolution not being up-to-date with
        //! the actual .pt file.
        //init_future.wait();
    }
}

void YOLO::deinit() {
    bool expected = true;
    if(yolo_initialized.compare_exchange_strong(expected, false)) {
        {
            std::unique_lock guard(transfer_done_mutex);
            if(transferred_done.valid())
                transferred_done.get();
        }
        _pool = nullptr;
        
        {
            std::unique_lock guard(running_mutex);
            if(running_prediction.valid()) {
                Print("[shutdown-trace] YOLO::deinit entering active-prediction wait. python_initialized=",
                      Python::python_initialized());
                Print("Still have an active prediction running, waiting...");
                running_prediction.get();
                Print("Got it.");
            }
            running_promise = {};
            running_prediction = {};
            
            if(not Python::python_initialized())
                throw U_EXCEPTION("Please Yolo::deinit before calling Python::deinit().");
            
            Python::schedule([](){
                track::PythonIntegration::unload_module("bbx_saved_model");
                track::PythonIntegration::unload_module("trex_yolo");
                track::PythonIntegration::unload_module("trex_rfdetr");
                track::PythonIntegration::unload_module("trex_detection_model");
            }).get();
            
            data().reset();
        }
        
        detect::pipeline_manager(detect::ObjectDetectionType::yolo).clean_up();
        detect::unregister_pipeline(detect::ObjectDetectionType::yolo);
    }
}

// Function to move outlines to the origin
void normalize_points(std::vector<std::vector<Vec2>>& points) {
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();

    for (const auto& outline : points) {
        for (const auto& point : outline) {
            min_x = min(min_x, point.x);
            min_y = min(min_y, point.y);
        }
    }

    for (auto& outline : points) {
        for (auto& point : outline) {
            point.x -= min_x;
            point.y -= min_y;
        }
    }
}

// Function to find bounding box size
std::pair<int, int> find_bounding_box_size(const std::vector<std::vector<Vec2>>& points) {
    float max_x = 0, max_y = 0;
    for (const auto& outline : points) {
        for (const auto& point : outline) {
            max_x = max(max_x, point.x);
            max_y = max(max_y, point.y);
        }
    }
    return { static_cast<int>(max_x) + 1, static_cast<int>(max_y) + 1 };
}

// Function to draw outlines on an OpenCV matrix
template<typename Vector>
void draw_outlines(const std::vector<Vector>& _points, const std::string& title = "Outlines") {
    std::vector<std::vector<Vec2>> copy;
    for(auto &pts : _points) {
        if constexpr(_is_smart_pointer<std::remove_cvref_t<decltype(pts)>>)
            copy.emplace_back(*pts);
        else
            copy.emplace_back(pts);
    }
    
    normalize_points(copy);
    auto size = find_bounding_box_size(copy);
    
    // Display the image
    cv::Mat image(size.second, size.first, CV_8UC3, cv::Scalar(0, 0, 0));

    cmn::gui::ColorWheel wheel;
    for (const auto& outline : copy) {
        auto color = wheel.next();
        for (size_t i = 0; i < outline.size(); ++i) {
            cv::Point2f start(outline[i].x, outline[i].y);
            cv::Point2f end(outline[(i + 1) % outline.size()].x, outline[(i + 1) % outline.size()].y);
            cv::line(image, start, end, color, 1);
            cv::circle(image, start, 5, color);
        }
    }
    
    tf::imshow(title, image);
}

void YOLO::receive(SegmentationData& data, track::detect::Result&& result) {
    const auto encoding = Background::meta_encoding();
    const auto mode = Background::image_mode();
    data.frame.set_encoding(encoding);
        
    cv::Mat r3;
    if (mode == ImageMode::R3G3B2) {
        if (data.image->dims == 3)
            convert_to_r3g3b2<3>(data.image->get(), r3);
        else if (data.image->dims == 4)
            convert_to_r3g3b2<4>(data.image->get(), r3);
        else
            throw U_EXCEPTION("Invalid number of channels (",data.image->dims,") in input image for the network.");
    }
    else if(mode == ImageMode::RGB) {
        if(data.image->dims == 3) {
            r3 = data.image->get();
        } else if(data.image->dims == 4) {
            cv::cvtColor(data.image->get(), r3, cv::COLOR_BGRA2BGR);
        } else
            throw U_EXCEPTION("Invalid number of channels (",data.image->dims,") in input image for the network.");
    }
    else if (mode == ImageMode::GRAY) {
        if(data.image->dims == 3)
            cv::cvtColor(data.image->get(), r3, cv::COLOR_BGR2GRAY);
        else if(data.image->dims == 4)
            cv::cvtColor(data.image->get(), r3, cv::COLOR_BGRA2GRAY);
        else if(data.image->dims == 1)
            r3 = data.image->get();
        else
            throw U_EXCEPTION("Invalid number of channels (",data.image->dims,") in input image for the network.");
    } else
        throw U_EXCEPTION("Invalid image mode ", mode);

    const auto detect_only_classes = READ_SETTING_WITH_DEFAULT(detect_only_classes, track::detect::PredictionFilter{});
    const coord_t w = max(0, r3.cols - 1);
    const coord_t h = max(0, r3.rows - 1);

    //! cache some of the high-level settings into a struct, to avoid repeated setting reads and conversions in the hot loop below
    const auto settings = AcceptanceSettings::Make();

    //! decide on whether to use masks (if available), or bounding boxes
    //! if masks are not available. for the boxes we simply copy over all
    //! of the pixels in the bounding box, for the masks we copy over only
    //! the pixels that are inside the mask.
    if (not result.masks().empty()) {
        /// yes we have masks!
        process_instance_segmentation(detect_only_classes, w, h, r3, data, result, settings);
    } else if (not result.obbdata().empty()) {
        /// we have obb data, but no masks
        process_obbs(detect_only_classes, w, h, r3, data, result, settings);
    } else if(not result.points().empty()) {
        process_points(detect_only_classes, w, h, r3, data, result, settings);
    } else {
        /// we had no instance segmentation...
        process_boxes_only(detect_only_classes, w, h, r3, data, result, settings);
    }
}

void YOLO::process_points(
       const track::detect::PredictionFilter& detect_only_classes,
       coord_t w,
       coord_t h,
       const cv::Mat& r3,
       SegmentationData &data,
       track::detect::Result &result,
       const AcceptanceSettings &settings)
{
    size_t N_rows = result.points().size();
    auto& points = result.points();

    auto process_index = [&](size_t i) {
        if(i >= N_rows)
            return;

        auto row = points[i];
        if (not detect_only_classes.allowed(row.clid))
            return;

        auto corners = row.corners();
        Bounds bounds = detect::ICXYR::bounding_box(corners);
        bounds.restrict_to(Bounds(0, 0, w, h));
        
        cmn::PixelArray_t pixels;
        std::vector<HorizontalLine> lines;
        uint64_t pixel_count = 0;
        
        int ymin = bounds.y;
        int ymax = bounds.y + bounds.height;
        /// copy a circle over, not a square
        const float halfh = (ymax - ymin) * 0.5f;
        const float ymiddle = halfh + ymin;
        const float xmiddle = bounds.x + bounds.width * 0.5f;

        for(int y = ymin; y<=ymax && y < h; ++y) {
            const float radicand = cmn::max(0.f, halfh * halfh - std::pow(y - ymiddle, 2));
            const float r = std::max(1.f, std::sqrt(radicand));
            const float fx0 = xmiddle - r;
            const float fx1 = xmiddle + r;
            
            // now round/clamp to integer pixel columns:
            int x0 = static_cast<int>(std::ceil(fx0));
            int x1 = static_cast<int>(std::floor(fx1));
            // clamp to image bounds [0..w-1]
            x0 = std::clamp(x0, 0, w-1);
            x1 = std::clamp(x1, 0, w-1);

            HorizontalLine line{
                saturate(coord_t(y), coord_t(0), coord_t(h)),
                coord_t(x0),
                coord_t(x1)
            };

            pixels.insert(pixels.end(), r3.ptr<uchar>(line.y, line.x0), r3.ptr<uchar>(line.y, line.x1 + 1));
            pixel_count += uint64_t(line.x1 - line.x0 + 1);
            lines.emplace_back(std::move(line));
        }

        if (lines.empty())
            return;

        if(not settings.is_acceptable(pixel_count))
            return;

        uint8_t flags{0};
        pv::Blob::set_flag(flags, pv::Blob::Flags::is_rgb, r3.channels() == 3);
        pv::Blob::set_flag(flags, pv::Blob::Flags::is_r3g3b2, Background::meta_encoding() == meta_encoding_t::r3g3b2);
        pv::Blob::set_flag(flags, pv::Blob::Flags::is_binary, Background::meta_encoding() == meta_encoding_t::binary);

        data.predictions.push_back({
            .clid = size_t(row.clid),
            .p = float(row.conf)
        });

        data.frame.add_object(lines, pixels, flags, blob::Prediction{
            .clid = uint8_t(row.clid),
            .p = uint8_t(float(row.conf) * 255.f)
        });
    };

    for(size_t idx = 0; idx < N_rows; ++idx)
        process_index(idx);
}

void YOLO::process_obbs(
       const track::detect::PredictionFilter& detect_only_classes,
       coord_t w,
       coord_t h,
       const cv::Mat& r3,
       SegmentationData &data,
       track::detect::Result &result,
       const AcceptanceSettings &settings)
{
    size_t N_rows = result.obbdata().size();
    auto& obbdata = result.obbdata();

    auto process_index = [&](size_t idx) {
        if(idx >= N_rows)
            return;

        auto row = obbdata[idx];
        if (not detect_only_classes.allowed(row.clid)) {
            return;
        }
        
        auto corners = row.corners();
        Bounds bounds = detect::ICXYWHR::bounding_box(corners);
        bounds.restrict_to(Bounds(0, 0, w, h));
        
        cmn::PixelArray_t pixels;
        std::vector<HorizontalLine> lines;
        uint64_t pixel_count = 0;
        
        int ymin = bounds.y;
        int ymax = bounds.y + bounds.height;
        
        for(int y = ymin; y<=ymax && y < h; ++y) {
            std::array<float, 4> intersections;
            size_t index = 0;
            
            /// go through all y and collect lines
            /// go through sides:
            for(size_t e=0; e<4; ++e) {
                Vec2 v0 = corners[e];
                Vec2 v1 = corners[(e+1)%4];
                
                // (v1 - v0) * t + v0 = (1 0) * t + (0 yb)
                //  t = (-v0.x yb-v0.y) / ((v1.x-v0.x-1 v1.y-v0.y))
                //  tx = -v0.x / (v1.x - v0.x -1)
                //  ty = (yb - v0.y) / (v1.y - v0.y)
                
                auto dy = (v1.y - v0.y);
                if(dy == 0) {
                    /// the side is parallel to the y-axis and we are on it
                    if(y == v0.y) {
                        auto xmin = std::min(v0.x, v1.x);
                        auto xmax = std::max(v0.x, v1.x);
                        intersections[index++] = xmin;
                        intersections[index++] = xmax;
                    }
                    
                } else {
                    auto ty = (y - v0.y) / dy;
                    if(ty >= 0 && ty < 1) {
                        auto xi = (v1.x - v0.x) * ty + v0.x;
                        intersections[index++] = xi;
                    }
                }
            }
            
            if(index < 2) {
                if(not lines.empty()
                   && y < ymax)
                {
                    FormatWarning("Invalid intersections: ", intersections, " (", index,") for y=", y, " with corners ", corners);
                    return;
                }
                
                continue;
            }
            
            // sort the two x‐intersections
            float xf0 = std::min(intersections[0], intersections[1]);
            float xf1 = std::max(intersections[0], intersections[1]);

            // now round/clamp to integer pixel columns:
            int x0 = static_cast<int>(std::floor(xf0));
            int x1 = static_cast<int>(std::ceil(xf1));
            
            if(x0 > x1) {
                throw RuntimeError("Skipping illegal horizontal extent of ", x0, " -> ", x1, " at ", y, " for object at ", bounds, " in frame ", data.image->index());
            }
            
            // clamp to image bounds [0..w-1]
            x0 = std::clamp(x0, 0, w-1);
            x1 = std::clamp(x1, x0, w-1);
            
            HorizontalLine line{
                saturate(coord_t(y), coord_t(0), coord_t(h)),
                coord_t(x0),
                coord_t(x1)
            };

            pixels.insert(pixels.end(), r3.ptr<uchar>(line.y, line.x0), r3.ptr<uchar>(line.y, line.x1 + 1));
            pixel_count += uint64_t(line.x1 - line.x0 + 1);
            lines.emplace_back(std::move(line));
        }

        /// exit early if we dont have an object
        /// (its empty)
        if (lines.empty()) {
            return;
        }

        if(not settings.is_acceptable(pixel_count)) {
            return;
        }

        uint8_t flags{0};
        pv::Blob::set_flag(flags, pv::Blob::Flags::is_rgb, r3.channels() == 3);
        pv::Blob::set_flag(flags, pv::Blob::Flags::is_r3g3b2, Background::meta_encoding() == meta_encoding_t::r3g3b2);
        pv::Blob::set_flag(flags, pv::Blob::Flags::is_binary, Background::meta_encoding() == meta_encoding_t::binary);

        blob::Pose pose;
        const bool has_keypoint = not result.keypoints().empty() && idx < result.keypoints().size();
        if(has_keypoint)
            pose = result.keypoints()[idx].toPose();

        /// add_object first; only record the parallel predictions/keypoints arrays
        /// once it has succeeded. Otherwise a throw here (e.g. from add_object)
        /// would leave predictions/keypoints one entry longer than data.frame's
        /// objects, and since we skip-and-continue on error that desync would
        /// mis-assign every following object's prediction/pose downstream.
        data.frame.add_object(lines, pixels, flags, blob::Prediction{
            .clid = uint8_t(row.clid),
            .p = uint8_t(float(row.conf) * 255.f),
            .pose = std::move(pose)
        });

        data.predictions.push_back({
            .clid = size_t(row.clid),
            .p = float(row.conf)
        });
        if(has_keypoint)
            data.keypoints.push_back(result.keypoints()[idx]);
    };

    for(size_t idx = 0; idx < N_rows; ++idx) {
        try {
            process_index(idx);
        } catch(const std::exception& ex) {
            /// Skip just this object and keep going. We cannot (and do not want to)
            /// special-case exception types, so surface the failure rather than
            /// swallow it. Each object is committed atomically (add_object before
            /// the predictions/keypoints push), so a skip leaves them consistent.
            FormatError("Skipping object in image ", data.image->index(), ": ", ex.what());
        } catch(...) {
            FormatError("Skipping object in image ", data.image->index(), ".");
        }
    }
}

void YOLO::process_boxes_only(
       const track::detect::PredictionFilter& detect_only_classes,
       coord_t w,
       coord_t h,
       const cv::Mat& r3,
       SegmentationData &data,
       track::detect::Result &result,
       const AcceptanceSettings &settings)
{
    auto& boxes = result.boxes();
    const size_t total_rows = boxes.num_rows();

    auto process_row = [&](const track::detect::Row& row, std::optional<size_t> idx) {
        if (not detect_only_classes.allowed(row.clid)) {
            return;
        }

        Bounds bounds = row.box;
        bounds.restrict_to(Bounds(0, 0, w, h));

        cmn::PixelArray_t pixels;
        std::vector<HorizontalLine> lines;
        uint64_t pixel_count = 0;

        for (int y = bounds.y; y < saturate(bounds.y + bounds.height, Float2_t(0), Float2_t(h)); ++y) {
            HorizontalLine line{
                saturate(coord_t(y), coord_t(0), coord_t(h-1)),
                saturate(coord_t(bounds.x), coord_t(0), coord_t(w-1)),
                saturate(coord_t(bounds.x + bounds.width), coord_t(0), coord_t(w-1))
            };
            pixels.insert(pixels.end(), r3.ptr<uchar>(line.y, line.x0), r3.ptr<uchar>(line.y, line.x1 + 1));
            pixel_count += uint64_t(line.x1 - line.x0 + 1);
            lines.emplace_back(std::move(line));
        }

        if (lines.empty()) {
            return;
        }

        if(not settings.is_acceptable(pixel_count)) {
            return;
        }

        uint8_t flags{0};
        pv::Blob::set_flag(flags, pv::Blob::Flags::is_rgb, r3.channels() == 3);
        pv::Blob::set_flag(flags, pv::Blob::Flags::is_r3g3b2, Background::meta_encoding() == meta_encoding_t::r3g3b2);
        pv::Blob::set_flag(flags, pv::Blob::Flags::is_binary, Background::meta_encoding() == meta_encoding_t::binary);

        data.predictions.push_back({
            .clid = size_t(row.clid),
            .p = float(row.conf)
        });

        blob::Pose pose;
        if(idx && not result.keypoints().empty() && *idx < result.keypoints().size()) {
            auto p = result.keypoints()[*idx];
            pose = p.toPose();
            data.keypoints.push_back(std::move(p));
        }

        data.frame.add_object(lines, pixels, flags, blob::Prediction{
            .clid = uint8_t(row.clid),
            .p = uint8_t(float(row.conf) * 255.f),
            .pose = std::move(pose)
        });
    };

    auto process_index = [&](size_t idx) {
        if(idx >= total_rows)
            return;
        process_row(boxes[idx], idx);
    };

    for(size_t idx = 0; idx < total_rows; ++idx)
        process_index(idx);
}

void YOLO::process_instance_segmentation(
      const track::detect::PredictionFilter& detect_only_classes,
      coord_t w,
      coord_t h,
      const cv::Mat& r3,
      SegmentationData &data,
      track::detect::Result &result,
      const AcceptanceSettings &settings)
{
    size_t N_rows = result.boxes().num_rows();
    auto& boxes = result.boxes();

    std::mutex mutex;

    auto process_idx = [&](size_t idx, cmn::CPULabeling::DLList& list) {
        if(idx >= N_rows) {
            return;
        }
        if(idx >= result.masks().size()) {
            return;
        }

        auto& row = boxes[idx];
        if (not detect_only_classes.allowed(row.clid)) {
            return;
        }

        auto& mask = result.masks()[idx];
        auto r = process_instance(list, w, h, r3, row, mask, settings);
        if(r) {
            auto &&[assign, pair] = r.value();

            std::unique_lock guard(mutex);
            data.predictions.emplace_back(std::move(assign));
            data.frame.add_object(std::move(pair));
        }
    };

    auto fn = [&](auto, size_t start, size_t end, auto) {
        cmn::CPULabeling::DLList list;
        for(size_t idx = start; idx != end; ++idx)
            process_idx(idx, list);
    };

    if(N_rows == 0u)
        return;

    if(N_rows > 1u && _pool) {
        distribute_indexes(fn, *_pool, size_t(0), N_rows);
    } else {
        fn(0, size_t(0), N_rows, 0);
    }
}

std::optional<std::tuple<SegmentationData::Assignment, blob::Pair>> YOLO::process_instance(
     cmn::CPULabeling::DLList& list,
     coord_t w,
     coord_t h,
     const cv::Mat &r3,
     const track::detect::Row &row,
     const track::detect::MaskData &mask,
     const AcceptanceSettings& settings)
{
    // Extract bounding box from the detection row
    Bounds bounds = row.box;

    //assert(bounds.x < mask.mat.cols && bounds.y < mask.mat.rows);
    assert(bounds.x < w && bounds.y < h);
    assert(bounds.x + bounds.width <= w);
    assert(bounds.y + bounds.height <= h);
    return process_instance_image(list, w, h, r3, row, bounds, mask.mat, settings);
}

std::optional<std::tuple<SegmentationData::Assignment, blob::Pair>> YOLO::process_instance_image(
     cmn::CPULabeling::DLList& list,
     coord_t w,
     coord_t h,
     const cv::Mat &r3,
     const track::detect::Row &row,
     cmn::Bounds bounds,
     const cv::Mat& mask_image,
     const AcceptanceSettings& settings)
{
    if(mask_image.empty())
        return std::nullopt;
    assert(mask_image.isContinuous());
    
    //tf::imshow("mask", mask_image);
    
    // Perform CPU-based connected-component labeling on the mask
    auto blobs = CPULabeling::run(list, mask_image);
    if(blobs.empty())
        // If no blobs found, skip this instance
        return std::nullopt;
    
    // Identify the largest blob by pixel count
    size_t msize = 0, midx = 0;
    for (size_t j = 0; j < blobs.size(); ++j) {
        if (blobs.at(j).pixels->size() > msize) {
            msize = blobs.at(j).pixels->size();
            midx = j;
        }
    }

    // Select the blob with the maximum pixel count for further processing
    auto&& pair = blobs.at(midx);
    uint64_t pixel_count = 0;
    // Adjust each horizontal line by bounding-box offset and clamp to image dimensions
    for (auto& line : *pair.lines) {
        auto oline = line;
        
        line.x0 = saturate(coord_t(line.x0 + bounds.x), coord_t(0), w);
        line.x1 = saturate(coord_t(line.x1 + bounds.x), line.x0, w);
        line.y = saturate(coord_t(line.y + bounds.y), coord_t(0), h);
        pixel_count += uint64_t(line.x1 - line.x0 + 1);
        
        if(oline.x0 > oline.x1 || oline.x1 + bounds.x - 1 > w
           || oline.y + bounds.y - 1 > h)
        {
            FormatWarning("Illegal line: ", oline, " => ", line, " offset:", bounds.pos());
        }
        
        if (line.x0 >= r3.cols
            || line.x1 >= r3.cols
            || line.y >= r3.rows)
            throw U_EXCEPTION("Coordinates of line ", line, " are invalid for image ", r3.cols, "x", r3.rows);
        // Now each line coordinate lies within valid image bounds
    }

    // Assign class ID and confidence to this blob prediction
    pair.pred = blob::Prediction{
        .clid = static_cast<uint8_t>(row.clid),
        .p = uint8_t(float(row.conf) * 255.f)
    };
    // Mark blob as instance segmentation and set encoding-based flags
    pair.extra_flags |= pv::Blob::flag(pv::Blob::Flags::is_instance_segmentation);
    
    const auto meta_encoding = Background::meta_encoding();
    if(meta_encoding == meta_encoding_t::r3g3b2) {
        assert(r3.channels() == 1);
        pv::Blob::set_flag(pair.extra_flags, pv::Blob::Flags::is_r3g3b2, true);
    }
    pv::Blob::set_flag(pair.extra_flags, pv::Blob::Flags::is_rgb, meta_encoding == meta_encoding_t::rgb8);
    pv::Blob::set_flag(pair.extra_flags, pv::Blob::Flags::is_binary, meta_encoding == meta_encoding_t::binary);
    assert(pv::Blob::is_flag(pair.extra_flags, pv::Blob::Flags::is_rgb) == (meta_encoding == meta_encoding_t::rgb8));

    /// Check whether the given object is acceptable regarding the current
    /// segmentation settings or not:
    if(not settings.is_acceptable(pixel_count)) {
        return std::nullopt;
    }

    // Build a Blob object for pixel extraction and outline generation.
    pv::Blob blob(std::make_unique<std::vector<HorizontalLine>>(*pair.lines), nullptr, uint8_t(pair.extra_flags), blob::Prediction{pair.pred});
    //blob.add_offset(bounds.pos());
    //Print("* processing object ", blob, " ", blob.bounds());
    
    //pv::Blob blob(*pair.lines, *pair.pixels, pair.extra_flags, pair.pred);
    // Convert the blob outline into actual pixel values from the image
    if(meta_encoding != meta_encoding_t::binary) {
        auto [o, px] = blob.calculate_pixels(r3);
        blob.set_pixels(std::make_unique<PixelArray_t>(*px));
        pair.pixels = std::move(px);
    }
    
    //auto &&[_, test_image] = blob.color_image();
    //auto _m = test_image->get();
    //tf::imshow("color image", _m);

    // Extract the outer contour points from the blob for outline construction
    auto points = pixel::find_outer_points(&blob, 0);
    // Remove any invalid or empty contour point sets
    for(auto it = points.begin(); it != points.end(); ) {
        if(not *it || (*it)->empty())
            it = points.erase(it);
        else
            ++it;
    }
    
    // Prepare assignment structure with class and probability for this detection
    SegmentationData::Assignment assign{
        .clid = size_t(row.clid),
        .p = float(row.conf)
    };
    
    // If there are contour points, process outlines and optionally compress
    if (not points.empty()) {
        // here we should likely make sure that we collect all possible lines
        // not just the outer lines?
        //Print("We have detected ", points.size(), " outlines here but only use the first one.");
        
        // Retrieve outline compression setting to reduce vertex count if needed
        /// we may have to downsample outlines
        const auto outline_compression = FAST_SETTING(outline_compression);
        
        // Containers for storing original and compressed outlines
        std::vector<std::vector<Vec2>> all;
        std::vector<Vec2> reduced;
        // If compression is enabled and the outline is large, perform downsampling
        if(outline_compression > 0
           && points.front()->size() > 1000)
        {
            reduced.reserve(points.front()->size());
            gui::reduce_vertex_line(*points.front(), reduced, 0.5);
            //Print(points.front()->size(), " reduced to ", reduced.size());
            all.emplace_back(reduced);
            
            // Store the compressed outline as the primary outline
            //data.outlines.emplace_back(*points.front());
            pair.pred.outlines.set_original(std::move(reduced));
            
            // Visualization: draw full outlines for debugging
            //draw_outlines(points);
            
        } else {
            // No compression: store original outline directly
            pair.pred.outlines.set_original(std::move(*points.front()));
        }
        
        // Remove the used first outline from the list
        points.erase(points.begin());
        
        if(outline_compression > 0) {
            // Process any remaining outlines after the first
            for(auto& pts : points) {
                reduced.clear();
                reduced.reserve(pts->size());
                
                gui::reduce_vertex_line(*pts, reduced, 0.5);
                //Print("* ",pts->size(), " reduced to ", reduced.size());
                all.emplace_back(reduced);
                
                // Append additional outlines to the prediction object
                pair.pred.outlines.add(std::move(reduced));
            }
            
            //draw_outlines(all, "Reduced");
            
        } else {
            // Process any remaining outlines after the first
            for(auto& pts : points)
                // Append additional outlines to the prediction object
                pair.pred.outlines.add(std::move(*pts));
        }
    }
    
    /*{
        auto &&[_, test_image] = blob.color_image();
        auto _m = test_image->get();
        tf::imshow("color image", _m);
    }*/
    
    return std::make_tuple(
        std::move(assign),
        std::move(pair)
    );
}

bool YOLO::is_initializing() {
    std::unique_lock guard(init_mutex);
    return init_future.valid();
}

double YOLO::fps() {
    if(_network_samples.load() == 0u)
        return 0.0;
    return _network_fps.load() / double(_network_samples.load());
}

struct YOLO::TransferData {
    std::vector<Image::Ptr> images;
    //std::vector<Image::Ptr> oimages;
    std::vector<SegmentationData> datas;
    std::vector<TileGeometry> tile_geometries;
    std::vector<size_t> orig_id;
    std::vector<std::promise<SegmentationData>> promises;
    std::vector<std::function<void()>> callbacks;
    std::vector<uint8_t> promise_completed;
    std::vector<uint8_t> callback_invoked;

    TransferData() = default;
    TransferData(const TransferData&) = delete;
    TransferData(TransferData&&) noexcept = default;
    TransferData& operator=(TransferData&&) = delete;
    TransferData& operator=(const TransferData&) = delete;

    bool is_promise_completed(size_t index) const noexcept {
        return index < promise_completed.size() && promise_completed[index];
    }

    void mark_promise_completed(size_t index) noexcept {
        if(index < promise_completed.size())
            promise_completed[index] = true;
    }

    void set_exception(size_t index, std::exception_ptr exception) noexcept {
        if(index >= promises.size() || is_promise_completed(index))
            return;

        try {
            promises[index].set_exception(std::move(exception));
        } catch(const std::exception& ex) {
            FormatWarning("Could not set exception on YOLO result promise ", index, ": ", ex.what());
        } catch(...) {
            FormatWarning("Could not set exception on YOLO result promise ", index, ".");
        }
        mark_promise_completed(index);
    }

    void set_soft_exception(size_t index, std::string_view message) noexcept {
        try {
            throw SoftException(no_quotes(message));
        } catch(...) {
            set_exception(index, std::current_exception());
        }
    }

    void set_value(size_t index, SegmentationData&& data) noexcept {
        if(index >= promises.size() || is_promise_completed(index))
            return;

        try {
            promises[index].set_value(std::move(data));
            mark_promise_completed(index);
        } catch(...) {
            set_exception(index, std::current_exception());
        }
    }

    void invoke_callback(size_t index) noexcept {
        if(index >= callbacks.size()
           || (index < callback_invoked.size() && callback_invoked[index]))
        {
            return;
        }

        if(index < callback_invoked.size())
            callback_invoked[index] = true;

        try {
            if(callbacks[index])
                callbacks[index]();
        } catch(...) {
            FormatExcept("Exception in callback of element ", index, " in python results.");
        }
    }

    void fail_all(std::string_view message) noexcept {
        for(size_t i = 0; i < promises.size(); ++i) {
            if(!is_promise_completed(i))
                set_soft_exception(i, message);
        }
        for(size_t i = 0; i < callbacks.size(); ++i)
            invoke_callback(i);
    }

    ~TransferData() {
        fail_all("YOLO prediction ended before producing a result.");
        for (auto&& img : images) {
            TileImage::move_back(std::move(img));
        }
        //thread_print("** deleting ", (uint64_t)this);
    }
};

void YOLO::StartPythonProcess(TransferData&& transfer) {
    if (not yolo_initialized) {
        // probably shutting down at the moment
        throw U_EXCEPTION("Cannot start a python process because we are shutting down.");
        /*for (size_t i = 0; i < transfer.datas.size(); ++i) {
            transfer.promises.at(i).set_exception(nullptr);

            try {
                transfer.callbacks.at(i)();
            }
            catch (...) {
                FormatExcept("Exception in callback of element ", i, " in python results.");
            }
        }
        FormatExcept("System shutting down.");
        return;*/
    }

    Timer timer;
    using py = track::PythonIntegration;
    //thread_print("** transfer of ", (uint64_t)& transfer);

    bool force = false;
    const size_t _N = transfer.datas.size();
    {
        [[maybe_unused]] ModuleProxy yolo("trex_yolo", [&force](ModuleProxy&) {
            force = true;
        }, true);
        [[maybe_unused]] ModuleProxy detection_model("trex_detection_model", [&force](ModuleProxy&){
            force = true;
        }, true);
        [[maybe_unused]] ModuleProxy rfdetr("trex_rfdetr", [&force](ModuleProxy&){
            force = true;
        }, true);
    }
    
    if(force) {
        try {
            py::unload_module("bbx_saved_model");
        } catch(...) {
            FormatWarning("Was unable to unload the module.");
        }
    }
    ModuleProxy bbx("bbx_saved_model", YOLO::reinit, true);
    //bbx.set_variable("image", transfer.images);
    //bbx.set_variable("oimages", transfer.oimages);

    std::vector<uint64_t> mask_Ns;
    std::vector<float> mask_points;

    try {
        track::detect::YoloInput input{
            std::move(transfer.images),
            transfer.tile_geometries,
            transfer.orig_id,
            [](std::vector<Image::Ptr>&& images)
            {
                for (auto&& image : images)
                    TileImage::move_back(std::move(image));
            }
        };

        //auto results = py::predict(std::move(input), bbx.m);
        //Print("C++ results = ", results);
        auto results = py::predict(std::move(input), bbx.m);
        double elapsed = timer.elapsed();
        timer.reset();
        ReceivePackage(std::move(transfer), std::move(results));
        //bbx.run("apply");
        //double cpp_elapsed = timer.elapsed();

        auto samples = _network_samples.load();
        auto fps = _network_fps.load();
        if (samples > 10u) {
            fps = fps / double(samples);
            samples = 1;
        }
        _network_fps = fps + (double(_N) / elapsed);
        _network_samples = samples + 1;
        //Print("[py] network: ", elapsed);
        //Print("[cpp] network: ", cpp_elapsed);
    }
    catch (const std::exception& ex) {
        FormatError("Exception: ", ex.what());
        transfer.fail_all(ex.what());
    }
    catch (...) {
        FormatWarning("Continue after exception...");

        throw;
    }
}

void YOLO::ReceivePackage(TransferData&& transfer, std::vector<track::detect::Result>&& results) {
    //size_t elements{0};
    //size_t outline_elements{0};
    //thread_print("Received a number of results: ", results.size());
    //thread_print("For elements: ", datas);
    //for(auto &t : transfer.oimages)
    //    TileImage::buffers.move_back(std::move(t));

    if(transfer.tile_geometries.size() != transfer.orig_id.size()) {
        const auto message = "YOLO input retained " + Meta::toStr(transfer.tile_geometries.size())
            + " tile geometries for " + Meta::toStr(transfer.orig_id.size()) + " tile frame indices.";
        FormatError(message);
        transfer.fail_all(message);
        return;
    }

    if(results.size() != transfer.tile_geometries.size()) {
        const auto message = "YOLO predict returned " + Meta::toStr(results.size())
            + " result(s) for " + Meta::toStr(transfer.tile_geometries.size()) + " detector tile(s).";
        FormatError(message);
        transfer.fail_all(message);
        return;
    }

    for(const auto frame : transfer.orig_id) {
        if(frame >= transfer.datas.size()) {
            const auto message = "YOLO retained an invalid frame index " + Meta::toStr(frame)
                + " for " + Meta::toStr(transfer.datas.size()) + " request(s).";
            FormatError(message);
            transfer.fail_all(message);
            return;
        }
    }
    
    std::unique_lock guard(transfer_done_mutex);
    if(transferred_done.valid())
        transferred_done.get();

    /// pack the function and move it into the pool
    /// (we have non-copyable stuff in there so we need to pack)
    /// this will move all the post-processing into a different
    /// thread:
    auto p = pack<void()>([transfer = std::move(transfer), results = std::move(results)]() mutable {
        const auto semantic_filter = READ_SETTING_WITH_DEFAULT(
            detect_only_classes,
            track::detect::PredictionFilter{});
        const float semantic_confidence = 1.f;//static_cast<float>(READ_SETTING(detect_conf_threshold, Float2_t));
        std::vector<size_t> tile_counts(transfer.datas.size(), 0u);
        for(const auto frame : transfer.orig_id)
            ++tile_counts[frame];

        std::vector<std::vector<detect::Result>> grouped_results(transfer.datas.size());
        std::vector<std::vector<TileGeometry>> grouped_geometries(transfer.datas.size());
        for(size_t frame = 0; frame < transfer.datas.size(); ++frame) {
            grouped_results[frame].reserve(tile_counts[frame]);
            grouped_geometries[frame].reserve(tile_counts[frame]);
        }
        for(size_t tile = 0; tile < transfer.orig_id.size(); ++tile) {
            const auto frame = transfer.orig_id[tile];
            grouped_results[frame].emplace_back(
                detail::SegmentationPostprocess::convert_semantic(
                    std::move(results[tile]),
                    transfer.tile_geometries[tile],
                    semantic_filter,
                    semantic_confidence));
            grouped_geometries[frame].emplace_back(std::move(transfer.tile_geometries[tile]));
        }
        
        const auto detect_mask_postprocess_containment = READ_SETTING_WITH_DEFAULT(detect_mask_postprocess_containment, std::optional<Float2_t>{});
        detail::SegmentationPostprocess::Settings mask_nms_settings{
            .overlap = {
                .iou = static_cast<float>(READ_SETTING(detect_mask_postprocess_iou, Float2_t)),
                .containment = detect_mask_postprocess_containment.value_or(Float2_t{2.f})
            },
            .class_agnostic = false,
            .mode = READ_SETTING_WITH_DEFAULT(detect_mask_postprocess_mode, MaskPostprocessMode::none),
            .frame = {}
        };

        for (size_t i = 0; i < transfer.datas.size(); ++i) {
            auto& data = transfer.datas.at(i);

            try {
                if(grouped_results[i].empty()) {
                    throw U_EXCEPTION("YOLO returned no detector tiles for request ", i, ".");
                }

                auto result = detail::DetectionTilePostprocess::apply(
                    std::move(grouped_results[i]),
                    grouped_geometries[i]
                );
                
                if(mask_nms_settings.mode != MaskPostprocessMode::none) {
                    mask_nms_settings.frame = Frame_t(data.image->index());
                    result = detail::SegmentationPostprocess::apply(std::move(result), mask_nms_settings);
                }

                receive(data, std::move(result));
                transfer.set_value(i, std::move(data));
            }
            catch (...) {
                FormatExcept("A promise failed for ", transfer.datas.at(i));
                transfer.set_exception(i, std::current_exception());
            }
            transfer.invoke_callback(i);
        }
    });
    
    transferred_done = _pool ? _pool->enqueue([p = std::move(p)](){
        p();
    }) : std::future<void>{};
}

void YOLO::apply(std::vector<TileImage>&& tiles) {
    while(true) {
        if(std::unique_lock guard(init_mutex);
           init_future.valid())
        {
            if(init_future.wait_for(std::chrono::milliseconds(1)) == std::future_status::ready) {
                init_future.get();
                break;
            }
        } else
            break;
    }
    
    namespace py = Python;
    TransferData transfer;

    size_t i = 0;
    for(auto&& tiled : tiles) {
        bool log_tile_info = false;
        {
            std::scoped_lock guard(tile_log_mutex);
            if(tiled.tile_size != last_logged_tile_size
               || tiled.images.size() != last_logged_tile_count)
            {
                last_logged_tile_size = tiled.tile_size;
                last_logged_tile_count = tiled.images.size();
                log_tile_info = true;
            }
        }
        if(log_tile_info) {
            const auto frame_index = tiled.data.image ? tiled.data.image->index() : -1;
            Print("YOLO tiling: sending ", tiled.images.size(), " tile(s) of ", tiled.tile_size.width, "x", tiled.tile_size.height, " pixels (frame ", frame_index, ") to python.");
        }

        transfer.images.insert(transfer.images.end(), std::make_move_iterator(tiled.images.begin()), std::make_move_iterator(tiled.images.end()));
        
        if(not tiled.promise)
            throw U_EXCEPTION("Promise was not set.");
        transfer.promises.emplace_back(std::move(*tiled.promise));
        transfer.promise_completed.emplace_back(false);
        tiled.promise = nullptr;
        transfer.callbacks.emplace_back(std::move(tiled.callback));
        transfer.callback_invoked.emplace_back(false);
        
        {
            for(size_t k = 0; k < tiled.tile_geometries().size(); ++k) {
                transfer.orig_id.push_back(i);
            }
            const auto bounds = tiled.source_tile_bounds();
            tiled.data.tiles.insert(tiled.data.tiles.end(), bounds.begin(), bounds.end());
        }
        
        const auto& geometries = tiled.tile_geometries();
        transfer.tile_geometries.insert(transfer.tile_geometries.end(), geometries.begin(), geometries.end());
        transfer.datas.emplace_back(std::move(tiled.data));
        
        ++i;
    }

    tiles.clear();
    
    auto mark_prediction_finished = []() noexcept {
        try {
            running_promise.set_value();
        } catch(const std::exception& ex) {
            FormatWarning("Could not mark the YOLO prediction as finished: ", ex.what());
        } catch(...) {
            FormatWarning("Could not mark the YOLO prediction as finished.");
        }
    };

    try {
        {
            std::unique_lock guard(running_mutex);
            if(running_prediction.valid())
                running_prediction.get();
            running_promise = {};
            running_prediction = running_promise.get_future().share();
        }

        /*Print("[shutdown-trace] YOLO::apply dispatch start requests=", transfer.datas.size(),
              " tiles=", transfer.images.size(),
              " callback_count=", transfer.callbacks.size());*/
        py::schedule([&transfer]() mutable {
            StartPythonProcess(std::move(transfer));
        }).get();

        mark_prediction_finished();
        
    } catch(const std::exception& ex) {
        mark_prediction_finished();
        transfer.fail_all(ex.what());
    } catch(...) {
        mark_prediction_finished();
        transfer.fail_all("Unknown exception while running YOLO prediction.");
    }
}

} // namespace track

namespace track {

void register_yolo_backend() {
    detect::register_backend(detect::ObjectDetectionType::yolo, detect::BackendHooks{
        .init = []() { YOLO::init(); },
        .deinit = []() { YOLO::deinit(); },
        .is_initializing = []() { return YOLO::is_initializing(); },
        .fps = []() { return YOLO::fps(); },
        .apply = [](std::vector<TileImage>&& tiles) { YOLO::apply(std::move(tiles)); },
        .set_background = [](const cmn::Image::Ptr& image) { YOLO::set_background(image); }
    });
}

} // namespace track
