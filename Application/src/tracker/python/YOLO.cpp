#include "YOLO.h"
#include <processing/PixelTree.h>
#include <python/PythonWrapper.h>
#include <grabber/misc/default_config.h>
#include <video/Video.h>
#include <misc/Timer.h>
#include <misc/ThreadPool.h>
#include <core/TrackingSettings.h>
#include <python/PipelineRegistry.h>
#include <python/GPURecognition.h>
#include <gui/GuiTypes.h>
#include <opencv2/imgproc.hpp>

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

namespace {

float rect_area(const track::detect::Rect& rect) {
    const float width = std::max(0.f, rect.x1 - rect.x0);
    const float height = std::max(0.f, rect.y1 - rect.y0);
    if(width <= 0.f || height <= 0.f)
        return 0.f;
    return width * height;
}

float intersection_area(const track::detect::Rect& a, const track::detect::Rect& b) {
    const float x0 = std::max(a.x0, b.x0);
    const float y0 = std::max(a.y0, b.y0);
    const float x1 = std::min(a.x1, b.x1);
    const float y1 = std::min(a.y1, b.y1);

    const float w = std::max(0.f, x1 - x0);
    const float h = std::max(0.f, y1 - y0);
    if(w <= 0.f || h <= 0.f)
        return 0.f;
    return w * h;
}

float rotated_rect_area(const cv::RotatedRect& rect) {
    return std::max(0.f, rect.size.width) * std::max(0.f, rect.size.height);
}

float rotated_intersection_area(const cv::RotatedRect& a, const cv::RotatedRect& b) {
    std::vector<cv::Point2f> intersection;
    const int status = cv::rotatedRectangleIntersection(a, b, intersection);
    if(status == cv::INTERSECT_NONE || intersection.size() < 3u)
        return 0.f;
    return static_cast<float>(std::max(0.0, cv::contourArea(intersection)));
}

} // namespace

namespace yolo_detail {

std::vector<TileMergeGroup> compute_tile_merge_groups(const track::detect::Boxes& boxes, float ios_threshold) {
    const size_t num_rows = boxes.num_rows();
    if(num_rows == 0)
        return {};

    ios_threshold = std::clamp(ios_threshold, 0.f, 1.f);

    // Tile detections arrive in original-image coordinates, but without row-level tile provenance.
    // GreedyNMM mirrors SAHI's sliced prediction postprocess: per-class, confidence-sorted
    // matching by intersection-over-smaller-area (IOS), with geometry/mask merging later.
    // Source indices stay original row indices so masks/keypoints remain aligned.
    // This can merge duplicate/partial seam detections when boxes overlap strongly,
    // but it deliberately cannot fuse two non-overlapping left/right seam halves into a new object.
    std::unordered_map<int, std::vector<size_t>> by_class;
    by_class.reserve(num_rows);
    for(size_t idx = 0; idx < num_rows; ++idx) {
        const auto& row = boxes[idx];
        if(rect_area(row.box) <= 0.f)
            continue;
        by_class[static_cast<int>(row.clid)].push_back(idx);
    }

    std::vector<TileMergeGroup> keep;
    keep.reserve(num_rows);

    for(auto& [clid, indices] : by_class) {
        std::sort(indices.begin(), indices.end(), [&](size_t lhs, size_t rhs) {
            if(boxes[lhs].conf == boxes[rhs].conf)
                return lhs < rhs;
            return boxes[lhs].conf > boxes[rhs].conf;
        });

        std::vector<bool> suppressed(indices.size(), false);
        for(size_t i = 0; i < indices.size(); ++i) {
            if(suppressed[i])
                continue;

            TileMergeGroup group{
                .representative = boxes[indices[i]],
                .representative_index = indices[i],
                .source_indices = DetectionRowSelection{ indices[i] }
            };
            const auto& ref_row = boxes[indices[i]];
            const float ref_area = rect_area(ref_row.box);

            for(size_t j = i + 1; j < indices.size(); ++j) {
                if(suppressed[j])
                    continue;

                const auto& candidate_row = boxes[indices[j]];
                const float intersection = intersection_area(ref_row.box, candidate_row.box);
                if(intersection <= 0.f)
                    continue;

                float containment = 0.f;
                const float candidate_area = rect_area(candidate_row.box);
                const float min_area = std::min(ref_area, candidate_area);
                if(min_area > 0.f)
                    containment = intersection / min_area;

                if(containment >= ios_threshold) {
                    suppressed[j] = true;
                    group.source_indices.push_back(indices[j]);
                }
            }

            std::sort(group.source_indices.begin(), group.source_indices.end());
            keep.push_back(std::move(group));
        }
    }

    std::sort(keep.begin(), keep.end(), [](const TileMergeGroup& lhs, const TileMergeGroup& rhs) {
        return lhs.representative_index < rhs.representative_index;
    });
    return keep;
}

DetectionRowSelection DetectionRowSelection::sequential(size_t count) {
    DetectionRowSelection selection;
    selection.indices.resize(count);
    std::iota(selection.indices.begin(), selection.indices.end(), size_t{0});
    return selection;
}

DetectionRowSelection compute_tile_nms_indices(
    const track::detect::Boxes& boxes, 
    float iou_threshold
) {
    const size_t num_rows = boxes.num_rows();
    if(num_rows == 0)
        return {};

    iou_threshold = std::clamp(iou_threshold, 0.f, 1.f);

    std::unordered_map<int, std::vector<size_t>> by_class;
    by_class.reserve(num_rows);
    for(size_t idx = 0; idx < num_rows; ++idx) {
        const auto& row = boxes[idx];
        if(rect_area(row.box) <= 0.f)
            continue;
        by_class[static_cast<int>(row.clid)].push_back(idx);
    }

    std::vector<size_t> keep;
    keep.reserve(num_rows);

    for(auto& [clid, indices] : by_class) {
        std::sort(indices.begin(), indices.end(), [&](size_t lhs, size_t rhs) {
            if(boxes[lhs].conf == boxes[rhs].conf)
                return lhs < rhs;
            return boxes[lhs].conf > boxes[rhs].conf;
        });

        std::vector<bool> suppressed(indices.size(), false);
        for(size_t i = 0; i < indices.size(); ++i) {
            if(suppressed[i])
                continue;

            keep.push_back(indices[i]);
            const auto& ref_row = boxes[indices[i]];
            const float ref_area = rect_area(ref_row.box);

            for(size_t j = i + 1; j < indices.size(); ++j) {
                if(suppressed[j])
                    continue;

                const auto& candidate_row = boxes[indices[j]];
                const float intersection = intersection_area(ref_row.box, candidate_row.box);
                if(intersection <= 0.f)
                    continue;

                const float candidate_area = rect_area(candidate_row.box);
                const float union_area = ref_area + candidate_area - intersection;
                const float iou = union_area > 0.f ? intersection / union_area : 0.f;
                if(iou >= iou_threshold)
                    suppressed[j] = true;
            }
        }
    }

    std::sort(keep.begin(), keep.end());
    keep.erase(std::unique(keep.begin(), keep.end()), keep.end());
    return DetectionRowSelection{ std::move(keep) };
}

DetectionRowSelection compute_tile_nms_indices_for_rotated_rects(
    const std::vector<cv::RotatedRect>& rects,
    const std::vector<float>& confidences,
    const std::vector<float>& classes,
    float iou_threshold
) {
    const size_t num_rows = rects.size();
    if(num_rows == 0)
        return {};
    if(confidences.size() != num_rows || classes.size() != num_rows)
        throw InvalidArgumentException("Rotated pose NMS expects matching rect/confidence/class counts.");

    iou_threshold = std::clamp(iou_threshold, 0.f, 1.f);

    std::unordered_map<int, std::vector<size_t>> by_class;
    by_class.reserve(num_rows);
    for(size_t idx = 0; idx < num_rows; ++idx) {
        if(rotated_rect_area(rects[idx]) <= 0.f)
            continue;
        by_class[static_cast<int>(classes[idx])].push_back(idx);
    }

    std::vector<size_t> keep;
    keep.reserve(num_rows);

    for(auto& [clid, indices] : by_class) {
        std::sort(indices.begin(), indices.end(), [&](size_t lhs, size_t rhs) {
            if(confidences[lhs] == confidences[rhs])
                return lhs < rhs;
            return confidences[lhs] > confidences[rhs];
        });

        std::vector<bool> suppressed(indices.size(), false);
        for(size_t i = 0; i < indices.size(); ++i) {
            if(suppressed[i])
                continue;

            keep.push_back(indices[i]);
            const float ref_area = rotated_rect_area(rects[indices[i]]);
            for(size_t j = i + 1; j < indices.size(); ++j) {
                if(suppressed[j])
                    continue;

                const float intersection = rotated_intersection_area(rects[indices[i]], rects[indices[j]]);
                if(intersection <= 0.f)
                    continue;

                const float candidate_area = rotated_rect_area(rects[indices[j]]);
                const float union_area = ref_area + candidate_area - intersection;
                const float iou = union_area > 0.f ? intersection / union_area : 0.f;
                if(iou >= iou_threshold)
                    suppressed[j] = true;
            }
        }
    }

    std::sort(keep.begin(), keep.end());
    keep.erase(std::unique(keep.begin(), keep.end()), keep.end());
    return DetectionRowSelection{ std::move(keep) };
}

std::optional<cv::RotatedRect> compute_pose_tile_rect(const track::detect::Keypoint& keypoint) {
    std::vector<cv::Point2f> points;
    points.reserve(keypoint.bones.size());
    for(const auto& bone : keypoint.bones) {
        if(std::isfinite(bone.x) && std::isfinite(bone.y))
            points.emplace_back(bone.x, bone.y);
    }

    if(points.empty())
        return std::nullopt;

    cv::RotatedRect rect;
    if(points.size() == 1u) {
        rect = cv::RotatedRect(points.front(), cv::Size2f(1.f, 1.f), 0.f);
    } else {
        rect = cv::minAreaRect(points);
    }

    static constexpr float min_padding = 2.f;
    const float span = std::max(rect.size.width, rect.size.height);
    const float padding = std::max(min_padding, span * 0.02f);
    rect.size.width = std::max(min_padding, rect.size.width) + padding * 2.f;
    rect.size.height = std::max(min_padding, rect.size.height) + padding * 2.f;
    return rect;
}

} // namespace yolo_detail

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
        
    } else
        throw U_EXCEPTION("This does not seem like a valid model to use: ", path,". Either we cannot find it, or it is not in a valid format. Expected is a pytorch .pt saved model file (as generated by training e.g. YOLOv8).");

    if(READ_SETTING(region_model, file::Path).exists())
        _loaded_models.emplace_back(
            ModelTaskType::region,
            BOOL_SETTING(yolo_region_tracking_enabled), // region models dont have tracking
            READ_SETTING(region_model, file::Path).str(),
            READ_SETTING(region_resolution, DetectResolution)
        );

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

    using namespace yolo_detail;
    DetectionRowSelection tile_selection;      // filled iff NMS tiling ran
    DetectionRowSelection sequential_storage;  // lazily filled for the non-tiled path
    std::vector<TileMergeGroup> merge_groups_storage;
    const std::vector<TileMergeGroup>* merge_groups = nullptr;
    bool have_tile_selection = false;
    if(const float detect_tile_overlap = READ_SETTING(detect_tile_overlap, float);
       detect_tile_overlap > 0.f && data.tiles.size() > 1) 
    {
        const Float2_t detect_tile_merge_iou = READ_SETTING(detect_tile_merge_iou, Float2_t);
        const Float2_t detect_tile_merge_containment = READ_SETTING(detect_tile_merge_containment, Float2_t);

        if(result.boxes().num_rows() > 0 && (not result.masks().empty() || result.keypoints().empty())) {
            merge_groups_storage = compute_tile_merge_groups(result.boxes(), detect_tile_merge_containment);
            merge_groups = &merge_groups_storage;

        } else if(result.boxes().num_rows() > 0) {
            if(READ_SETTING(detect_pose_bbx, default_config::detect_pose_bbx_t::Class) == default_config::detect_pose_bbx_t::keypoints
               && result.keypoints().size() == result.boxes().num_rows())
            {
                std::vector<cv::RotatedRect> rects;
                std::vector<float> confidences;
                std::vector<float> classes;
                rects.reserve(result.keypoints().size());
                confidences.reserve(result.keypoints().size());
                classes.reserve(result.keypoints().size());

                for(size_t idx = 0; idx < result.keypoints().size(); ++idx) {
                    auto rect = compute_pose_tile_rect(result.keypoints()[idx]);
                    if(rect) {
                        rects.push_back(*rect);
                        confidences.push_back(result.boxes()[idx].conf);
                        classes.push_back(result.boxes()[idx].clid);
                    } else {
                        Bounds bounds = result.boxes()[idx].box;
                        rects.emplace_back(
                            cv::Point2f(bounds.x + bounds.width * 0.5f, bounds.y + bounds.height * 0.5f),
                            cv::Size2f(bounds.width, bounds.height),
                            0.f);
                        confidences.push_back(result.boxes()[idx].conf);
                        classes.push_back(result.boxes()[idx].clid);
                    }
                }

                tile_selection = compute_tile_nms_indices_for_rotated_rects(rects, confidences, classes, detect_tile_merge_iou);
            } else {
                tile_selection = compute_tile_nms_indices(result.boxes(), detect_tile_merge_iou);
            }
            have_tile_selection = true;

        } else if(not result.obbdata().empty()) {
            std::vector<float> raw_boxes;
            raw_boxes.reserve(result.obbdata().size() * 6u);
            for(size_t idx = 0; idx < result.obbdata().size(); ++idx) {
                const auto row = result.obbdata()[idx];
                const Bounds bounds = row.bounding_box();
                raw_boxes.insert(raw_boxes.end(), {
                    bounds.x,
                    bounds.y,
                    bounds.x + bounds.width,
                    bounds.y + bounds.height,
                    row.conf,
                    row.clid
                });
            }
            const size_t raw_size = raw_boxes.size();
            track::detect::Boxes boxes(std::move(raw_boxes), raw_size);
            tile_selection = compute_tile_nms_indices(boxes, detect_tile_merge_iou);
            have_tile_selection = true;

        } else if(not result.points().empty()) {
            std::vector<float> raw_boxes;
            raw_boxes.reserve(result.points().size() * 6u);

            for(size_t idx = 0; idx < result.points().size(); ++idx) {
                const auto row = result.points()[idx];
                const Bounds bounds = row.bounding_box();
                raw_boxes.insert(raw_boxes.end(), {
                    bounds.x,
                    bounds.y,
                    bounds.x + bounds.width,
                    bounds.y + bounds.height,
                    row.conf,
                    row.clid
                });
            }

            const size_t raw_size = raw_boxes.size();
            track::detect::Boxes boxes(std::move(raw_boxes), raw_size);
            tile_selection = compute_tile_nms_indices(boxes, detect_tile_merge_iou);
            have_tile_selection = true;
        }
    }

    //! cache some of the high-level settings into a struct, to avoid repeated setting reads and conversions in the hot loop below
    const auto settings = AcceptanceSettings::Make();

    //! resolve the explicit set of rows each consumer should process: the tile
    //! NMS result if tiling ran, otherwise the full [0..count) row range. The
    //! merge-group path ignores the row view, so skip materializing it there.
    auto select_rows = [&](size_t count) -> DetectionRowView {
        if(merge_groups)
            return {};
        if(have_tile_selection)
            return tile_selection;
        sequential_storage = DetectionRowSelection::sequential(count);
        return sequential_storage;
    };

    //! decide on whether to use masks (if available), or bounding boxes
    //! if masks are not available. for the boxes we simply copy over all
    //! of the pixels in the bounding box, for the masks we copy over only
    //! the pixels that are inside the mask.
    if (not result.masks().empty()) {
        /// yes we have masks!
        process_instance_segmentation(detect_only_classes, w, h, r3, data, result, settings, select_rows(result.boxes().num_rows()), merge_groups);
    } else if (not result.obbdata().empty()) {
        /// we have obb data, but no masks
        process_obbs(detect_only_classes, w, h, r3, data, result, settings, select_rows(result.obbdata().size()));
    } else if(not result.points().empty()) {
        process_points(detect_only_classes, w, h, r3, data, result, settings, select_rows(result.points().size()));
    } else {
        /// we had no instance segmentation...
        process_boxes_only(detect_only_classes, w, h, r3, data, result, settings, select_rows(result.boxes().num_rows()), merge_groups);
    }
}

void YOLO::process_points(
       const track::detect::PredictionFilter& detect_only_classes,
       coord_t w,
       coord_t h,
       const cv::Mat& r3,
       SegmentationData &data,
       track::detect::Result &result,
       const AcceptanceSettings &settings,
       yolo_detail::DetectionRowView rows)
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

    for(size_t idx : rows)
        process_index(idx);
}

void YOLO::process_obbs(
       const track::detect::PredictionFilter& detect_only_classes,
       coord_t w,
       coord_t h,
       const cv::Mat& r3,
       SegmentationData &data,
       track::detect::Result &result,
       const AcceptanceSettings &settings,
       yolo_detail::DetectionRowView rows)
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
            int x0 = static_cast<int>(std::ceil(xf0));
            int x1 = static_cast<int>(std::floor(xf1));
            
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

        data.predictions.push_back({
            .clid = size_t(row.clid),
            .p = float(row.conf)
        });

        blob::Pose pose;
        if(not result.keypoints().empty() && idx < result.keypoints().size()) {
            auto p = result.keypoints()[idx];
            pose = p.toPose();
            data.keypoints.push_back(std::move(p));
        }

        data.frame.add_object(lines, pixels, flags, blob::Prediction{
            .clid = uint8_t(row.clid),
            .p = uint8_t(float(row.conf) * 255.f),
            .pose = std::move(pose)
        });
    };

    for(size_t idx : rows) {
        process_index(idx);
    }
}

void YOLO::process_boxes_only(
       const track::detect::PredictionFilter& detect_only_classes,
       coord_t w,
       coord_t h,
       const cv::Mat& r3,
       SegmentationData &data,
       track::detect::Result &result,
       const AcceptanceSettings &settings,
       yolo_detail::DetectionRowView rows,
       const std::vector<yolo_detail::TileMergeGroup>* merge_groups)
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

    if(merge_groups) {
        for(const auto& group : *merge_groups) {
            auto row = group.representative;
            Bounds bounds = row.box;
            bool have_bounds = false;
            for(size_t idx : group.source_indices) {
                if(idx >= total_rows)
                    continue;
                Bounds source_bounds = boxes[idx].box;
                if(source_bounds.width <= 0 || source_bounds.height <= 0)
                    continue;
                if(!have_bounds) {
                    bounds = source_bounds;
                    have_bounds = true;
                } else {
                    bounds.combine(source_bounds);
                }
            }
            if(!have_bounds)
                continue;
            row.box = track::detect::Rect{
                .x0 = bounds.x,
                .y0 = bounds.y,
                .x1 = bounds.x + bounds.width,
                .y1 = bounds.y + bounds.height
            };
            process_row(row, std::nullopt);
        }
    } else {
        for(size_t idx : rows) {
            process_index(idx);
        }
    }
}

void YOLO::process_instance_segmentation(
      const track::detect::PredictionFilter& detect_only_classes,
      coord_t w,
      coord_t h,
      const cv::Mat& r3,
      SegmentationData &data,
      track::detect::Result &result,
      const AcceptanceSettings &settings,
      yolo_detail::DetectionRowView rows,
      const std::vector<yolo_detail::TileMergeGroup>* merge_groups)
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

    auto process_group = [&](const yolo_detail::TileMergeGroup& group, cmn::CPULabeling::DLList& list) {
        if (not detect_only_classes.allowed(group.representative.clid)) {
            return;
        }

        Bounds merged_bounds;
        bool have_bounds = false;
        for(size_t idx : group.source_indices) {
            if(idx >= N_rows || idx >= result.masks().size())
                continue;
            Bounds source_bounds = boxes[idx].box;
            source_bounds.restrict_to(Bounds(0, 0, w, h));
            if(source_bounds.width <= 0 || source_bounds.height <= 0)
                continue;
            if(!have_bounds) {
                merged_bounds = source_bounds;
                have_bounds = true;
            } else {
                merged_bounds.combine(source_bounds);
            }
        }
        if(!have_bounds || merged_bounds.width <= 0 || merged_bounds.height <= 0)
            return;

        const int merged_cols = std::max(0, static_cast<int>(std::ceil(merged_bounds.width)));
        const int merged_rows = std::max(0, static_cast<int>(std::ceil(merged_bounds.height)));
        if(merged_cols <= 0 || merged_rows <= 0)
            return;

        cv::Mat merged_mask = cv::Mat::zeros(merged_rows, merged_cols, CV_8UC1);
        for(size_t idx : group.source_indices) {
            if(idx >= N_rows || idx >= result.masks().size())
                continue;

            const auto& mask = result.masks()[idx];
            if(mask.mat.empty())
                continue;

            Bounds source_bounds = boxes[idx].box;
            source_bounds.restrict_to(Bounds(0, 0, w, h));
            const int dst_x = static_cast<int>(source_bounds.x - merged_bounds.x);
            const int dst_y = static_cast<int>(source_bounds.y - merged_bounds.y);
            const int copy_cols = std::min({mask.mat.cols, merged_cols - dst_x, static_cast<int>(std::ceil(source_bounds.width))});
            const int copy_rows = std::min({mask.mat.rows, merged_rows - dst_y, static_cast<int>(std::ceil(source_bounds.height))});
            if(dst_x < 0 || dst_y < 0 || copy_cols <= 0 || copy_rows <= 0)
                continue;

            cv::Mat src_roi = mask.mat(cv::Rect(0, 0, copy_cols, copy_rows));
            cv::Mat dst_roi = merged_mask(cv::Rect(dst_x, dst_y, copy_cols, copy_rows));
            cv::bitwise_or(dst_roi, src_roi, dst_roi);
        }

        auto row = group.representative;
        row.box = track::detect::Rect{
            .x0 = merged_bounds.x,
            .y0 = merged_bounds.y,
            .x1 = merged_bounds.x + merged_bounds.width,
            .y1 = merged_bounds.y + merged_bounds.height
        };

        auto r = process_instance_image(list, w, h, r3, row, merged_bounds, merged_mask, settings);
        if(r) {
            auto&& [assign, pair] = r.value();

            std::unique_lock guard(mutex);
            data.predictions.emplace_back(std::move(assign));
            data.frame.add_object(std::move(pair));
        }
    };

    auto fn = [&](auto, size_t start, size_t end, auto) {
        cmn::CPULabeling::DLList list;
        for(size_t pos = start; pos != end; ++pos) {
            const size_t idx = rows[pos];
            process_idx(idx, list);
        }
    };

    auto fn_groups = [&](auto, size_t start, size_t end, auto) {
        cmn::CPULabeling::DLList list;
        for(size_t pos = start; pos != end; ++pos) {
            process_group(merge_groups->at(pos), list);
        }
    };

    if(merge_groups) {
        if(merge_groups->empty())
            return;
        if(merge_groups->size() > 1 && _pool) {
            distribute_indexes(fn_groups, *_pool, size_t(0), merge_groups->size());
        } else {
            fn_groups(0, size_t(0), merge_groups->size(), 0);
        }
        return;
    }

    if(rows.empty())
        return;

    if(rows.size() > 1 && _pool) {
        distribute_indexes(fn, *_pool, size_t(0), rows.size());
    } else {
        fn(0, size_t(0), rows.size(), 0);
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
    auto [o, px] = blob.calculate_pixels(r3);
    blob.set_pixels(std::make_unique<PixelArray_t>(*px));
    pair.pixels = std::move(px);
    
    
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
    std::vector<Vec2> scales;
    std::vector<Vec2> offsets;
    std::vector<size_t> orig_id;
    std::vector<std::promise<SegmentationData>> promises;
    std::vector<std::function<void()>> callbacks;

    TransferData() = default;
    TransferData(const TransferData&) = delete;
    TransferData(TransferData&&) = default;
    TransferData& operator=(TransferData&&) = default;
    TransferData& operator=(const TransferData&) = delete;

    ~TransferData() {
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
    }
    
    if(force) {
        try {
            py::unload_module("bbx_saved_model");
        } catch(...) {
            FormatWarning("Was unable to unload the module.");
        }
    }
    ModuleProxy bbx("bbx_saved_model", YOLO::reinit, true);
    //bbx.set_variable("offsets", std::move(transfer.offsets));
    //bbx.set_variable("image", transfer.images);
    //bbx.set_variable("oimages", transfer.oimages);

    std::vector<uint64_t> mask_Ns;
    std::vector<float> mask_points;

    try {
        track::detect::YoloInput input{
            std::move(transfer.images),
            (transfer.offsets),
            (transfer.scales),
            (transfer.orig_id),
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
        for(auto &t : transfer.promises) {
            try {
                throw SoftException(no_quotes((std::string)ex.what()));
            } catch(...) {
                t.set_exception(std::current_exception());
            }
        }
        
        transfer.promises.clear();
        ReceivePackage(std::move(transfer), {});
        
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

    if (results.empty()) {
#ifndef NDEBUG
        if (not transfer.images.empty())
            tf::imshow("ma", transfer.images.front()->get());
#endif
        if(not transfer.promises.empty()) {
            for (size_t i = 0; i < transfer.datas.size(); ++i) {
                try {
                    transfer.promises.at(i).set_value(std::move(transfer.datas.at(i)));
                }
                catch (...) {
                    FormatExcept("A promise failed for ", transfer.datas.at(i));
                    transfer.promises.at(i).set_exception(std::current_exception());
                }
                
                try {
                    transfer.callbacks.at(i)();
                }
                catch (...) {
                    FormatExcept("Exception in callback of element ", i, " in python results.");
                }
            }
        }
        FormatExcept("Empty data for ", transfer.datas, " image=", transfer.orig_id);
        return;
    }
    
    std::unique_lock guard(transfer_done_mutex);
    if(transferred_done.valid())
        transferred_done.get();

    /// pack the function and move it into the pool
    /// (we have non-copyable stuff in there so we need to pack)
    /// this will move all the post-processing into a different
    /// thread:
    auto p = pack<void()>([transfer = std::move(transfer), results = std::move(results)]() mutable {
        for (size_t i = 0; i < transfer.datas.size(); ++i) {
            auto&& result = results.at(i);
            auto& data = transfer.datas.at(i);
            
            try {
                receive(data, std::move(result));
                transfer.promises.at(i).set_value(std::move(data));
            }
            catch (...) {
                FormatExcept("A promise failed for ", transfer.datas.at(i));
                transfer.promises.at(i).set_exception(std::current_exception());
            }

            try {
                transfer.callbacks.at(i)();
            }
            catch (...) {
                FormatExcept("Exception in callback of element ", i, " in python results.");
            }
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
        tiled.promise = nullptr;
        
        //Print("Image scale: ", scale, " with tile source=", tiled.source_size, " image=", data.image->dimensions()," output_size=", READ_SETTING(output_size, Size2), " original=", tiled.original_size);
        
        {
            const Vec2 scale = tiled.original_size.div(tiled.source_size);
            for(size_t k = 0; k < tiled.offsets().size(); ++k) {
                transfer.orig_id.push_back(i);
                transfer.scales.push_back(scale);
            }
            auto scaled = tiled.scaled_tile_bounds();
            tiled.data.tiles.insert(tiled.data.tiles.end(), scaled.begin(), scaled.end());
        }
        
        auto o = tiled.offsets();
        transfer.offsets.insert(transfer.offsets.end(), o.begin(), o.end());
        transfer.datas.emplace_back(std::move(tiled.data));
        transfer.callbacks.emplace_back(tiled.callback);
        
        ++i;
    }

    tiles.clear();
    
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
        
        running_promise.set_value();
        
    } catch(...) {
        running_promise.set_value();
        for(auto &t : transfer.promises) {
            t.set_exception(std::current_exception());
        }
        //throw;
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
