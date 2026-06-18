#include "AnnotationExporter.h"
#include <video/VideoSource.h>
#include <thirdparty/fkYAML/node.hpp>
#include <core/DetectionTypes.h>
#include <misc/zipper.h>

namespace track::annotation_export {
namespace {

using namespace cmn;
namespace dataset = track::annotation_dataset;
using YamlNode = fkyaml::node;

constexpr std::string_view kSplitDir = "train";
constexpr std::string_view kImagesDir = "images";
constexpr std::string_view kLabelsDir = "labels";

Bounds annotation_bounds(const Annotation& annotation) {
    if(annotation.points.empty())
        throw InvalidArgumentException("Annotation has no points.");

    // POSE annotations may carry unlabeled keypoints, stored as invalid (0,0)
    // placeholders so keypoint indices stay aligned with the skeleton schema.
    // Those must not stretch the bounding box to the image origin. For boxes and
    // segmentations every point is meaningful, including a corner at (0,0).
    const bool skip_invalid = annotation.type == AnnotationType::POSE;

    std::optional<float> min_x, min_y, max_x, max_y;
    for(const auto& point : annotation.points) {
        if(skip_invalid && !point.valid())
            continue;
        min_x = min_x ? std::min<float>(*min_x, point.x) : float(point.x);
        min_y = min_y ? std::min<float>(*min_y, point.y) : float(point.y);
        max_x = max_x ? std::max<float>(*max_x, point.x) : float(point.x);
        max_y = max_y ? std::max<float>(*max_y, point.y) : float(point.y);
    }

    if(!min_x)
        throw InvalidArgumentException("POSE annotation has no valid keypoints.");

    return Bounds(*min_x, *min_y, *max_x - *min_x, *max_y - *min_y);
}

void validate_point(const Annotation::Point_t& point, const Size2& image_size) {
    if(point.x < 0 || point.y < 0 || point.x > image_size.width || point.y > image_size.height)
        throw InvalidArgumentException("Annotation point ", point, " is out of image bounds ", image_size, ".");
}

void validate_annotation(const Annotation& annotation, const Size2& image_size, const std::vector<std::string>& keypoint_names) {
    if(image_size.width <= 0 || image_size.height <= 0)
        throw InvalidArgumentException("Image size must be positive, got ", image_size, ".");

    switch(annotation.type) {
        case AnnotationType::BOX:
            if(annotation.points.size() != 2)
                throw InvalidArgumentException("BOX annotation needs exactly 2 points, got ", annotation.points.size(), ".");
            break;
        case AnnotationType::SEGMENTATION:
            if(annotation.points.size() < 3)
                throw InvalidArgumentException("SEGMENTATION annotation needs at least 3 points, got ", annotation.points.size(), ".");
            break;
        case AnnotationType::POSE:
            if(annotation.points.empty())
                throw InvalidArgumentException("POSE annotation needs at least one point.");
            if(annotation.points.size() > keypoint_names.size())
                throw InvalidArgumentException("POSE annotation has ", annotation.points.size(), " points, but the keypoint schema has only ", keypoint_names.size(), " names.");
            break;
    }

    for(const auto& point : annotation.points)
        validate_point(point, image_size);
}

void validate_keypoint_names(const std::vector<std::string>& names) {
    std::set<std::string> seen;
    for(const auto& name : names) {
        if(name.empty())
            throw InvalidArgumentException("Keypoint names cannot be empty.");
        if(!seen.insert(name).second)
            throw InvalidArgumentException("Duplicate keypoint name ", name, ".");
    }
}

std::vector<Frame_t> annotated_frames(const AnnotationMap& annotations) {
    std::vector<Frame_t> frames;
    frames.reserve(annotations.size());
    for(const auto& [frame, frame_annotations] : annotations) {
        if(frame.valid() && !frame_annotations.empty())
            frames.push_back(frame);
    }
    return frames;
}

size_t background_count(size_t annotated_count, float percent) {
    if(percent <= 0.f || annotated_count == 0)
        return 0;
    return narrow_cast<size_t>(std::round(float(annotated_count) * percent / 100.f));
}

file::Path coco_image_path(const file::Path& root, Frame_t frame) {
    return root / kSplitDir / (frame_stem(frame) + ".jpg");
}

file::Path yolo_image_path(const file::Path& root, Frame_t frame) {
    return root / kSplitDir / kImagesDir / (frame_stem(frame) + ".jpg");
}

file::Path yolo_label_path(const file::Path& root, Frame_t frame) {
    return root / kSplitDir / kLabelsDir / (frame_stem(frame) + ".txt");
}

void write_text(const file::Path& path, const std::string& text) {
    auto f = path.fopen("wb");
    if(!f)
        throw U_EXCEPTION("Cannot open ", path, " for writing.");
    if(!text.empty())
        fwrite(text.data(), sizeof(char), text.size(), f.get());
}

Frame_t exported_source_index(Frame_t frame, const Options& options) {
    if(options.source_start)
        return Frame_t(frame.get() + options.source_start->get());
    return frame;
}

std::string export_video_source(const Options& options) {
    if(!options.video_source_basename.empty())
        return file::Path(options.video_source_basename).filename();
    return dataset::source_basename_from_paths(options.source);
}

void ensure_folder(const file::Path& path) {
    if(!path.create_folder())
        throw U_EXCEPTION("Cannot create folder ", path, ".");
}

void validate_source_paths(const file::PathArray& source) {
    if(source.empty())
        throw U_EXCEPTION("Cannot export annotations: source is empty.");

    for(const auto& path : source) {
        if(path.empty() || !path.exists())
            throw U_EXCEPTION("Cannot open source file ", path, ".");
    }
}

void write_jpeg(VideoSource& source, Frame_t frame, const file::Path& path, int quality) {
    cv::Mat input;
    if(!source.frame(frame, input))
        throw U_EXCEPTION("Cannot retrieve frame ", frame, " from ", source.source(), ".");
    if(input.empty())
        throw U_EXCEPTION("Frame ", frame, " from ", source.source(), " is empty.");

    cv::Mat output;
    if(input.channels() == 4)
        cv::cvtColor(input, output, cv::COLOR_RGBA2BGR);
    else if(input.channels() == 3)
        cv::cvtColor(input, output, cv::COLOR_RGB2BGR);
    else
        output = input;

    if(!cv::imwrite(path.str(), output, {cv::IMWRITE_JPEG_QUALITY, quality}))
        throw U_EXCEPTION("Cannot write image ", path, ".");
}

std::string yolo_bbox(const Bounds& bounds, const Size2& image_size) {
    const float cx = (bounds.x + bounds.width * 0.5f) / image_size.width;
    const float cy = (bounds.y + bounds.height * 0.5f) / image_size.height;
    const float width = bounds.width / image_size.width;
    const float height = bounds.height / image_size.height;
    return Meta::toStr(cx) + " " + Meta::toStr(cy) + " " + Meta::toStr(width) + " " + Meta::toStr(height);
}

std::vector<glz::json_t> bbox_json(const Bounds& bounds) {
    return {
        glz::json_t(bounds.x),
        glz::json_t(bounds.y),
        glz::json_t(bounds.width),
        glz::json_t(bounds.height)
    };
}

std::vector<glz::json_t> segmentation_json(const Annotation& annotation) {
    std::vector<glz::json_t> polygon;
    polygon.reserve(annotation.points.size() * 2);
    for(const auto& point : annotation.points) {
        polygon.emplace_back(point.x);
        polygon.emplace_back(point.y);
    }
    return {glz::json_t(polygon)};
}

std::set<uint16_t> class_ids(const AnnotationMap& annotations) {
    std::set<uint16_t> ids;
    for(const auto& [frame, frame_annotations] : annotations) {
        (void)frame;
        for(const auto& annotation : frame_annotations)
            ids.insert(annotation.clid);
    }
    return ids;
}

void append_yolo_metadata(YamlNode& yaml, const AnnotationMap& annotations, const std::vector<std::string>& keypoint_names) {
    auto ids = class_ids(annotations);

    auto names = YamlNode::mapping();
    auto map = track::detect::yolo::names::get_map();
    for(auto id : ids) {
        if(map.contains(id)) {
            names[narrow_cast<int64_t>(id)] = std::string(map.at(id));
        } else {
            names[narrow_cast<int64_t>(id)] = "class_" + Meta::toStr(id);
        }
    }

    yaml["nc"] = narrow_cast<int64_t>(ids.size());
    yaml["names"] = std::move(names);
    if(!keypoint_names.empty()) {
        auto kpt_shape = YamlNode::sequence();
        kpt_shape.as_seq().emplace_back(narrow_cast<int64_t>(keypoint_names.size()));
        kpt_shape.as_seq().emplace_back(int64_t(3));
        yaml["kpt_shape"] = std::move(kpt_shape);
        auto keypoints = YamlNode::sequence();
        for(const auto& name : keypoint_names)
            keypoints.as_seq().emplace_back(name);
        yaml["keypoint_names"] = std::move(keypoints);
    }
}

}

std::string frame_stem(Frame_t frame) {
    std::ostringstream ss;
    ss << "frame_" << std::setw(6) << std::setfill('0') << frame.get();
    return ss.str();
}

namespace {

std::string yolo_frame_mapping_image_path(Frame_t frame) {
    return (file::Path(std::string(kSplitDir)) / kImagesDir / (frame_stem(frame) + ".jpg")).str();
}

std::string coco_frame_mapping_image_path(Frame_t frame) {
    return (file::Path(std::string(kSplitDir)) / (frame_stem(frame) + ".jpg")).str();
}

}

std::string build_frame_mapping_csv(const Options& options, const std::vector<Frame_t>& image_frames) {
    std::string text = "image,video_source,source_index\n";
    const auto video_source = export_video_source(options);
    for(const auto& frame : image_frames) {
        text += (options.format == annotation_dataset::format_t::yolo ? yolo_frame_mapping_image_path(frame) : coco_frame_mapping_image_path(frame)) + ","
              + video_source + ","
              + exported_source_index(frame, options).toStr() + "\n";
    }
    return text;
}

TypeCounts count_types(const AnnotationMap& annotations) {
    TypeCounts counts;
    for(const auto& [frame, frame_annotations] : annotations) {
        (void)frame;
        for(const auto& annotation : frame_annotations) {
            switch(annotation.type) {
                case AnnotationType::BOX:
                    ++counts.boxes;
                    break;
                case AnnotationType::SEGMENTATION:
                    ++counts.segmentations;
                    break;
                case AnnotationType::POSE:
                    ++counts.poses;
                    break;
            }
        }
    }
    return counts;
}

AnnotationMap filter_types(const AnnotationMap& annotations, bool boxes, bool segmentations, bool poses) {
    AnnotationMap result;
    for(const auto& [frame, frame_annotations] : annotations) {
        std::vector<Annotation> kept;
        for(const auto& annotation : frame_annotations) {
            switch(annotation.type) {
                case AnnotationType::BOX:
                    if(boxes) kept.push_back(annotation);
                    break;
                case AnnotationType::SEGMENTATION:
                    if(segmentations) kept.push_back(annotation);
                    break;
                case AnnotationType::POSE:
                    if(poses) kept.push_back(annotation);
                    break;
            }
        }
        if(!kept.empty())
            result.emplace(frame, std::move(kept));
    }
    return result;
}

std::vector<std::string> default_keypoint_names(const AnnotationMap& annotations, const std::vector<std::string>& configured_names) {
    size_t max_points = 0;
    for(const auto& [frame, frame_annotations] : annotations) {
        (void)frame;
        for(const auto& annotation : frame_annotations) {
            if(annotation.type == AnnotationType::POSE)
                max_points = std::max(max_points, annotation.points.size());
        }
    }

    std::vector<std::string> names;
    names.reserve(max_points);
    for(size_t i = 0; i < max_points; ++i) {
        if(i < configured_names.size() && !configured_names.at(i).empty())
            names.push_back(configured_names.at(i));
        else
            names.push_back("kp_" + Meta::toStr(i));
    }
    return names;
}

Summary summarize(const Options& options, std::optional<Frame_t> source_length, std::optional<Size2> source_size) {
    Summary summary{
        .format = options.format,
        .output_directory = options.output_directory,
        .annotated_frames = annotated_frames(options.annotations).size(),
        .counts = count_types(options.annotations),
        .keypoint_names = options.keypoint_names
    };
    summary.background_frames = background_count(summary.annotated_frames, options.background_percent);
    summary.total_images = summary.annotated_frames + summary.background_frames;

    if(options.annotations.empty())
        summary.errors.emplace_back("No track_annotations are available to export.");
    if(options.output_directory.empty())
        summary.errors.emplace_back("Output dataset folder is empty.");
    if(options.background_percent < 0.f)
        summary.errors.emplace_back("Background percentage cannot be negative.");
    for(const auto& [frame, frame_annotations] : options.annotations) {
        if(!frame.valid() && !frame_annotations.empty())
            summary.errors.emplace_back("An annotated frame is invalid.");
    }

    if(summary.counts.poses > 0) {
        try {
            validate_keypoint_names(options.keypoint_names);
        } catch(const std::exception& e) {
            summary.errors.emplace_back(e.what());
        }
    }

    if(source_length) {
        for(const auto& frame : annotated_frames(options.annotations)) {
            if(!frame.valid())
                summary.errors.emplace_back("An annotated frame is invalid.");
            else if(frame >= *source_length)
                summary.errors.emplace_back("Annotated frame " + frame.toStr() + " is outside source length " + source_length->toStr() + ".");
        }
        const auto available_background = sample_background_frames(options.annotations, *source_length, std::numeric_limits<size_t>::max(), options.background_seed).size();
        if(summary.background_frames > available_background) {
            summary.warnings.emplace_back("Requested " + Meta::toStr(summary.background_frames) + " background frames, but only " + Meta::toStr(available_background) + " non-annotated frames are available.");
            summary.background_frames = available_background;
            summary.total_images = summary.annotated_frames + summary.background_frames;
        }
    }

    if(source_size) {
        for(const auto& [frame, frame_annotations] : options.annotations) {
            for(const auto& annotation : frame_annotations) {
                try {
                    validate_annotation(annotation, *source_size, options.keypoint_names);
                } catch(const std::exception& e) {
                    summary.errors.emplace_back("Frame " + frame.toStr() + ": " + e.what());
                }
            }
        }
    }

    return summary;
}

std::vector<Frame_t> sample_background_frames(const AnnotationMap& annotations, Frame_t source_length, size_t count, uint32_t seed) {
    std::set<Frame_t> annotated;
    for(const auto& [frame, frame_annotations] : annotations) {
        if(frame.valid() && !frame_annotations.empty())
            annotated.insert(frame);
    }

    std::vector<Frame_t> candidates;
    if(source_length.valid()) {
        for(Frame_t frame = 0_f; frame < source_length; ++frame) {
            if(!annotated.contains(frame))
                candidates.push_back(frame);
        }
    }

    std::mt19937 rng(seed);
    std::shuffle(candidates.begin(), candidates.end(), rng);
    if(count < candidates.size())
        candidates.resize(count);
    std::sort(candidates.begin(), candidates.end());
    return candidates;
}

std::string annotation_to_yolo(const Annotation& annotation, const Size2& image_size, const std::vector<std::string>& keypoint_names) {
    validate_annotation(annotation, image_size, keypoint_names);

    std::string output = Meta::toStr(uint16_t(annotation.clid));
    switch(annotation.type) {
        case AnnotationType::BOX:
        case AnnotationType::POSE:
            output += " " + yolo_bbox(annotation_bounds(annotation), image_size);
            break;
        case AnnotationType::SEGMENTATION:
            for(const auto& point : annotation.points)
                output += " " + Meta::toStr(point.x / image_size.width) + " " + Meta::toStr(point.y / image_size.height);
            return output;
    }

    if(annotation.type == AnnotationType::POSE) {
        for(size_t i = 0; i < keypoint_names.size(); ++i) {
            // Unlabeled keypoints (invalid placeholders) are emitted with
            // visibility 0 so the slot stays aligned with the schema.
            if(i < annotation.points.size() && annotation.points.at(i).valid()) {
                const auto& point = annotation.points.at(i);
                output += " " + Meta::toStr(point.x / image_size.width) + " " + Meta::toStr(point.y / image_size.height) + " 2";
            } else {
                output += " 0 0 0";
            }
        }
    }

    return output;
}

glz::json_t build_coco_json(const AnnotationMap& annotations, const std::vector<Frame_t>& image_frames, const Size2& image_size, const std::vector<std::string>& keypoint_names) {
    std::vector<glz::json_t> images;
    std::vector<glz::json_t> annotation_json;
    std::map<uint16_t, std::vector<std::string>> category_keypoints;

    for(const auto& frame : image_frames) {
        images.push_back(glz::json_t::object_t{
            {"id", glz::json_t(frame.get())},
            {"file_name", glz::json_t(frame_stem(frame) + ".jpg")},
            {"width", glz::json_t(image_size.width)},
            {"height", glz::json_t(image_size.height)}
        });
    }

    uint64_t annotation_id = 1;
    for(const auto& [frame, frame_annotations] : annotations) {
        for(const auto& annotation : frame_annotations) {
            validate_annotation(annotation, image_size, keypoint_names);
            auto bounds = annotation_bounds(annotation);
            glz::json_t::object_t object{
                {"id", glz::json_t(annotation_id++)},
                {"image_id", glz::json_t(frame.get())},
                {"category_id", glz::json_t(uint64_t(annotation.clid))},
                {"bbox", glz::json_t(bbox_json(bounds))},
                {"area", glz::json_t(bounds.width * bounds.height)},
                {"iscrowd", glz::json_t(0)}
            };

            if(annotation.type == AnnotationType::SEGMENTATION)
                object["segmentation"] = glz::json_t(segmentation_json(annotation));
            else
                object["segmentation"] = glz::json_t(std::vector<glz::json_t>{});

            if(annotation.type == AnnotationType::POSE) {
                std::vector<glz::json_t> keypoints;
                keypoints.reserve(keypoint_names.size() * 3);
                uint64_t num_keypoints = 0;
                for(size_t i = 0; i < keypoint_names.size(); ++i) {
                    // COCO visibility: 2 = labeled, 0 = not labeled (x=y=0).
                    // Invalid placeholders stand in for unlabeled keypoints.
                    if(i < annotation.points.size() && annotation.points.at(i).valid()) {
                        keypoints.emplace_back(annotation.points.at(i).x);
                        keypoints.emplace_back(annotation.points.at(i).y);
                        keypoints.emplace_back(2);
                        ++num_keypoints;
                    } else {
                        keypoints.emplace_back(0);
                        keypoints.emplace_back(0);
                        keypoints.emplace_back(0);
                    }
                }
                object["keypoints"] = glz::json_t(keypoints);
                object["num_keypoints"] = glz::json_t(num_keypoints);
                category_keypoints[annotation.clid] = keypoint_names;
            }

            annotation_json.emplace_back(std::move(object));
        }
    }

    std::vector<glz::json_t> categories;

    auto ids = class_ids(annotations);
    std::vector<std::string> names;
    auto map = track::detect::yolo::names::get_map();
    for(auto id : ids) {
        if(map.contains(id)) {
            names.emplace_back(map.at(id));
        } else {
            names.push_back("class_" + Meta::toStr(id));
        }
    }

    for(auto [id, name] : Zip::Zip(ids, names)) {
        glz::json_t::object_t category{
            {"id", glz::json_t(uint64_t(id))},
            {"name", glz::json_t(name)},
            {"supercategory", glz::json_t("object")}
        };
        if(auto it = category_keypoints.find(id); it != category_keypoints.end()) {
            std::vector<glz::json_t> names;
            for(const auto& name : it->second)
                names.emplace_back(name);
            category["keypoints"] = glz::json_t(names);
            category["skeleton"] = glz::json_t(std::vector<glz::json_t>{});
        }
        categories.emplace_back(std::move(category));
    }

    return glz::json_t::object_t{
        {"images", glz::json_t(images)},
        {"annotations", glz::json_t(annotation_json)},
        {"categories", glz::json_t(categories)}
    };
}

Summary export_dataset(const Options& options) {
    validate_source_paths(options.source);

    VideoSource source(options.source);
    source.set_colors(ImageMode::RGB);

    auto summary = summarize(options, source.length(), source.size());
    if(!summary.can_export()) {
        throw InvalidArgumentException("Cannot export annotations: ", summary.errors);
    }

    const auto background_frames = sample_background_frames(options.annotations, source.length(), summary.background_frames, options.background_seed);
    summary.background_frames = background_frames.size();
    summary.total_images = summary.annotated_frames + summary.background_frames;

    ensure_folder(options.output_directory / kSplitDir);

    if(options.format == dataset::format_t::yolo) {
        ensure_folder(options.output_directory / kSplitDir / kImagesDir);
        ensure_folder(options.output_directory / kSplitDir / kLabelsDir);
    }

    std::vector<Frame_t> image_frames = annotated_frames(options.annotations);
    image_frames.insert(image_frames.end(), background_frames.begin(), background_frames.end());
    std::sort(image_frames.begin(), image_frames.end());

    if(options.format == dataset::format_t::yolo) {
        for(const auto& frame : image_frames)
            write_jpeg(source, frame, yolo_image_path(options.output_directory, frame), options.jpeg_quality);

        for(const auto& [frame, frame_annotations] : options.annotations) {
            std::string text;
            for(const auto& annotation : frame_annotations)
                text += annotation_to_yolo(annotation, source.size(), options.keypoint_names) + "\n";
            write_text(yolo_label_path(options.output_directory, frame), text);
        }
        for(const auto& frame : background_frames)
            write_text(yolo_label_path(options.output_directory, frame), "");

        auto yaml = YamlNode::mapping();
        yaml["train"] = "./train";
        append_yolo_metadata(yaml, options.annotations, options.keypoint_names);
        write_text(options.output_directory / "data.yaml", YamlNode::serialize(yaml));

    } else {
        for(const auto& frame : image_frames)
            write_jpeg(source, frame, coco_image_path(options.output_directory, frame), options.jpeg_quality);

        auto json = build_coco_json(options.annotations, image_frames, source.size(), options.keypoint_names);
        auto text = glz::write_json(json).value_or("{}");
        text = glz::prettify_json(text);
        write_text(options.output_directory / kSplitDir / "_annotations.coco.json", text);
    }

    write_text(options.output_directory / "frame_mapping.csv", build_frame_mapping_csv(options, image_frames));

    return summary;
}

}
