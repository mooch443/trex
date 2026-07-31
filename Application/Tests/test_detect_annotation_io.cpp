#include <gtest/gtest.h>
#include <tracking/AnnotationExporter.h>
#include <tracking/DetectAnnotationImporter.h>
#include <core/DetectionTypes.h>
#include <core/default_config.h>
#include <misc/GlobalSettings.h>

#include <chrono>
#include <filesystem>
#include <fstream>

using namespace cmn;
using namespace track;
using namespace track::detect;
using namespace track::detect::annotation_export;
using namespace track::detect::annotation_import;

namespace {

Annotation make_annotation(uint8_t clid, AnnotationType type, std::vector<Annotation::Point_t> points) {
    return Annotation{
        .uid = 0,
        .clid = clid,
        .type = type,
        .points = std::move(points)
    };
}

AnnotationMap mixed_annotations() {
    AnnotationMap annotations;
    annotations[1_f].push_back(make_annotation(0, AnnotationType::BOX, {{10, 20}, {50, 80}}));
    annotations[3_f].push_back(make_annotation(1, AnnotationType::SEGMENTATION, {{0, 0}, {100, 0}, {100, 100}}));
    annotations[5_f].push_back(make_annotation(2, AnnotationType::POSE, {{25, 25}, {75, 75}}));
    return annotations;
}

file::Path make_temp_dataset(const std::string& name) {
    auto base = std::filesystem::temp_directory_path()
              / ("trex_annotation_importer_" + name + "_"
                 + Meta::toStr(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(base);
    return file::Path(base.string());
}

void write_file(const file::Path& path, const std::string& text = {}) {
    std::filesystem::create_directories(std::filesystem::path(path.str()).parent_path());
    std::ofstream out(path.str(), std::ios::binary);
    out << text;
}

ImportOptions default_import_options(const file::Path& dataset_file) {
    ImportOptions options;
    options.dataset_file = dataset_file;
    options.video_size = Size2(200, 100);
    options.converted_length = 50_f;
    options.source_start = 100_f;
    options.source_end = 150_f;
    return options;
}

}

TEST(DetectAnnotationExporter, ConvertsBoxToYolo) {
    auto annotation = make_annotation(2, AnnotationType::BOX, {{100, 50}, {300, 150}});

    EXPECT_EQ("2 0.5 0.5 0.5 0.5",
              annotation_to_yolo(annotation, Size2(400, 200), {}));
}

TEST(DetectAnnotationExporter, ConvertsSegmentationToYolo) {
    auto annotation = make_annotation(1, AnnotationType::SEGMENTATION, {{0, 0}, {400, 0}, {400, 200}});

    EXPECT_EQ("1 0 0 1 0 1 1",
              annotation_to_yolo(annotation, Size2(400, 200), {}));
}

TEST(DetectAnnotationExporter, ConvertsPoseToYoloAndPadsMissingKeypoints) {
    auto annotation = make_annotation(3, AnnotationType::POSE, {{100, 50}, {300, 150}});
    std::vector<std::string> keypoints{"nose", "tail", "left"};

    EXPECT_EQ("3 0.5 0.5 0.5 0.5 0.25 0.25 2 0.75 0.75 2 0 0 0",
              annotation_to_yolo(annotation, Size2(400, 200), keypoints));
}

TEST(DetectAnnotationExporter, BuildsYoloFrameMappingCsvWithSourceIndex) {
    Options options;
    options.source = file::PathArray("current_video.mp4");
    options.video_source_basename = "current_video.mp4";
    options.source_start = 100_f;

    const auto frame_10 = (file::Path("train") / "images" / "frame_000010.jpg").str();
    const auto frame_12 = (file::Path("train") / "images" / "frame_000012.jpg").str();
    EXPECT_EQ(std::string("image,video_source,source_index\n")
              + frame_10 + ",current_video.mp4,110\n"
              + frame_12 + ",current_video.mp4,112\n",
              build_frame_mapping_csv(options, {10_f, 12_f}));
}

TEST(DetectAnnotationDataset, DetectsFormatFromImportFileExtension) {
    EXPECT_EQ(annotation_dataset::format_t::yolo,
              annotation_dataset::format_from_dataset_file(file::Path("data.yaml")));
    EXPECT_EQ(annotation_dataset::format_t::yolo,
              annotation_dataset::format_from_dataset_file(file::Path("data.yml")));
    EXPECT_EQ(annotation_dataset::format_t::coco,
              annotation_dataset::format_from_dataset_file(file::Path("_annotations.coco.json")));
    EXPECT_FALSE(annotation_dataset::format_from_dataset_file(file::Path("labels.txt")));
}

TEST(DetectAnnotationImporter, AnnotationMapAcceptsFlatAndSourceNestedText) {
    auto flat = AnnotationMap::fromStr("{100:[[0,1,[[100,120],[200,300],[350,400]]]]}");
    ASSERT_TRUE(flat.contains(100_f));
    EXPECT_FALSE(flat.has_sources());

    AnnotationMap nested;
    nested.sources()["current_video.mp4"][100_f].push_back(make_annotation(0, AnnotationType::BOX, {{10, 10}, {20, 20}}));
    auto reparsed = AnnotationMap::fromStr(nested.toStr());
    ASSERT_TRUE(reparsed.has_sources());
    ASSERT_TRUE(reparsed.sources().contains("current_video.mp4"));
    ASSERT_TRUE(reparsed.sources().at("current_video.mp4").contains(100_f));
    ASSERT_TRUE(reparsed.contains(100_f));
}

TEST(DetectAnnotationExporter, BuildsCocoJsonForMixedAnnotationsAndBackgroundImages) {
    auto annotations = mixed_annotations();
    std::vector<Frame_t> frames{1_f, 2_f, 3_f, 5_f};
    auto json = build_coco_json(annotations, frames, Size2(100, 100), {"nose", "tail"});
    auto text = glz::write_json(json).value();

    EXPECT_NE(std::string::npos, text.find("\"file_name\":\"frame_000002.jpg\""));
    EXPECT_NE(std::string::npos, text.find("\"annotations\""));
    EXPECT_NE(std::string::npos, text.find("\"keypoints\""));
    EXPECT_NE(std::string::npos, text.find("\"categories\""));
}

TEST(DetectAnnotationExporter, ExportsUnlabeledPoseKeypointsAsZeroVisibility) {
    // Index 1 is an unlabeled keypoint, stored as an invalid (0,0) placeholder
    // (as produced by COCO/YOLO import of a keypoint with visibility 0).
    AnnotationMap annotations;
    annotations[1_f].push_back(make_annotation(0, AnnotationType::POSE, {{100, 50}, {0, 0}, {300, 150}}));
    const std::vector<std::string> keypoints{"a", "b", "c"};

    // YOLO: the unlabeled keypoint serializes as "0 0 0" and must not drag the
    // bounding box (the first four floats) to the image origin.
    EXPECT_EQ("0 0.5 0.5 0.5 0.5 0.25 0.25 2 0 0 0 0.75 0.75 2",
              annotation_to_yolo(annotations.at(1_f).front(), Size2(400, 200), keypoints));

    // COCO: same expectations as absolute-pixel x/y/visibility triples.
    auto json = build_coco_json(annotations, {1_f}, Size2(400, 200), keypoints);
    const auto& annotation = json.get_object().at("annotations").get_array().front().get_object();

    const auto& kpts = annotation.at("keypoints").get_array();
    ASSERT_EQ(9u, kpts.size());
    EXPECT_EQ(100, kpts.at(0).get_number());
    EXPECT_EQ(50, kpts.at(1).get_number());
    EXPECT_EQ(2, kpts.at(2).get_number());
    EXPECT_EQ(0, kpts.at(3).get_number());
    EXPECT_EQ(0, kpts.at(4).get_number());
    EXPECT_EQ(0, kpts.at(5).get_number());
    EXPECT_EQ(300, kpts.at(6).get_number());
    EXPECT_EQ(150, kpts.at(7).get_number());
    EXPECT_EQ(2, kpts.at(8).get_number());
    EXPECT_EQ(2, annotation.at("num_keypoints").get_number());

    const auto& bbox = annotation.at("bbox").get_array();
    ASSERT_EQ(4u, bbox.size());
    EXPECT_EQ(100, bbox.at(0).get_number());
    EXPECT_EQ(50, bbox.at(1).get_number());
    EXPECT_EQ(200, bbox.at(2).get_number());
    EXPECT_EQ(100, bbox.at(3).get_number());
}

TEST(DetectAnnotationExporter, SummarizesCountsAndBackgroundPercentage) {
    Options options;
    options.annotations = mixed_annotations();
    options.keypoint_names = {"nose", "tail"};
    options.background_percent = 50.f;
    options.output_directory = file::Path("annotations");

    auto summary = summarize(options, 10_f, Size2(100, 100));

    EXPECT_TRUE(summary.can_export());
    EXPECT_EQ(3u, summary.annotated_frames);
    EXPECT_EQ(2u, summary.background_frames);
    EXPECT_EQ(5u, summary.total_images);
    EXPECT_EQ(1u, summary.counts.boxes);
    EXPECT_EQ(1u, summary.counts.segmentations);
    EXPECT_EQ(1u, summary.counts.poses);
}

TEST(DetectAnnotationExporter, SamplesBackgroundFramesDeterministicallyAndExcludesAnnotations) {
    auto annotations = mixed_annotations();

    auto first = sample_background_frames(annotations, 10_f, 4, 1337);
    auto second = sample_background_frames(annotations, 10_f, 4, 1337);

    EXPECT_EQ(first, second);
    EXPECT_EQ(4u, first.size());
    for(auto frame : first) {
        EXPECT_FALSE(annotations.contains(frame));
    }
    EXPECT_TRUE(std::is_sorted(first.begin(), first.end()));
}

TEST(DetectAnnotationExporter, RejectsInvalidKeypointNamesAndTooManyPosePoints) {
    Options options;
    options.annotations[1_f].push_back(make_annotation(0, AnnotationType::POSE, {{10, 10}, {20, 20}, {30, 30}}));
    options.keypoint_names = {"nose", "nose"};
    options.output_directory = file::Path("annotations");

    auto duplicate_summary = summarize(options, 10_f, Size2(100, 100));
    EXPECT_FALSE(duplicate_summary.can_export());

    options.keypoint_names = {"nose", "tail"};
    auto too_many_summary = summarize(options, 10_f, Size2(100, 100));
    EXPECT_FALSE(too_many_summary.can_export());
}

TEST(DetectAnnotationExporter, RejectsEmptyAnnotationsAndOutOfBoundsPoints) {
    Options empty;
    empty.output_directory = file::Path("annotations");
    EXPECT_FALSE(summarize(empty, 10_f, Size2(100, 100)).can_export());

    Options out_of_bounds;
    out_of_bounds.annotations[1_f].push_back(make_annotation(0, AnnotationType::BOX, {{10, 10}, {200, 200}}));
    out_of_bounds.output_directory = file::Path("annotations");
    EXPECT_FALSE(summarize(out_of_bounds, 10_f, Size2(100, 100)).can_export());
}

TEST(DetectAnnotationExporter, FailsPredictablyForMissingSource) {
    Options options;
    options.annotations[1_f].push_back(make_annotation(0, AnnotationType::BOX, {{10, 10}, {20, 20}}));
    options.output_directory = file::Path("annotations");
    options.source = file::PathArray("/tmp/trex_missing_annotation_source.mp4");

    EXPECT_THROW((void)export_dataset(options), std::exception);
}

TEST(DetectAnnotationImporter, ParsesSupportedSourceIndexFilenameFormats) {
    const std::vector<std::string> names{
        "frame_000123.jpg",
        "frame-000123.jpg",
        "source_000123.jpg",
        "source-000123.jpg",
        "source_index_000123.jpg",
        "source-index-000123.jpg",
        "000123.jpg",
        "frame_000123_aug0.jpg",
        "0123_jpg.rf.d092af8a1bcf290c06af89399f4e46a4.jpg"
    };

    for(const auto& name : names) {
        auto parsed = parse_source_index_from_image_stem(name);
        ASSERT_TRUE(parsed.has_value()) << name << ": " << parsed.error;
        EXPECT_EQ(123_f, *parsed.source_index) << name;
    }
}

TEST(DetectAnnotationImporter, RejectsAmbiguousAndUnrecognizedSourceIndexFilenameFormats) {
    EXPECT_FALSE(parse_source_index_from_image_stem("frame_000123_000456.jpg").has_value());
    EXPECT_FALSE(parse_source_index_from_image_stem("image_000123.jpg").has_value());
    EXPECT_FALSE(parse_source_index_from_image_stem("frame_-1.jpg").has_value());
}

TEST(DetectAnnotationImporter, ImportsBoxesAndMapsSourceIndexThroughConversionStart) {
    auto root = make_temp_dataset("boxes");
    write_file(root / "data.yaml", "path: .\ntrain: images\nnames:\n  0: fish\n");
    write_file(root / "images" / "frame_000110.jpg");
    write_file(root / "labels" / "frame_000110.txt", "0 0.5 0.5 0.5 0.5\n");

    auto preview = preview_yolo_import(default_import_options(root / "data.yaml"), import_scope_t::all_videos);

    ASSERT_TRUE(preview.can_import()) << Meta::toStr(preview.errors);
    ASSERT_FALSE(preview.warnings.empty());
    EXPECT_NE(std::string::npos, preview.warnings.front().find("No frame mapping CSV"));
    EXPECT_TRUE(preview.metadata.detect_format_changed);
    EXPECT_EQ(track::detect::ObjectDetectionFormat::boxes, preview.metadata.imported_detect_format);
    ASSERT_TRUE(preview.annotations.contains(10_f));
    ASSERT_EQ(1u, preview.annotations.at(10_f).size());
    const auto& annotation = preview.annotations.at(10_f).front();
    EXPECT_EQ(AnnotationType::BOX, annotation.type);
    ASSERT_EQ(4u, annotation.points.size());
    EXPECT_EQ(Annotation::Point_t(50, 25), annotation.points.at(0));
    EXPECT_EQ(Annotation::Point_t(150, 75), annotation.points.at(2));
}

TEST(DetectAnnotationImporter, ImportsSegmentationPolygons) {
    auto root = make_temp_dataset("segmentation");
    write_file(root / "data.yaml", "path: .\ntrain: images\nnames: [fish]\n");
    write_file(root / "images" / "source-000101.jpg");
    write_file(root / "labels" / "source-000101.txt", "0 0 0 1 0 1 1 0 1\n");

    auto preview = preview_yolo_import(default_import_options(root / "data.yaml"), import_scope_t::all_videos);

    ASSERT_TRUE(preview.can_import()) << Meta::toStr(preview.errors);
    EXPECT_TRUE(preview.metadata.detect_format_changed);
    EXPECT_EQ(track::detect::ObjectDetectionFormat::masks, preview.metadata.imported_detect_format);
    const auto& annotation = preview.annotations.at(1_f).front();
    EXPECT_EQ(AnnotationType::SEGMENTATION, annotation.type);
    ASSERT_EQ(4u, annotation.points.size());
    EXPECT_EQ(Annotation::Point_t(200, 100), annotation.points.at(2));
}

TEST(DetectAnnotationImporter, DoesNotSuggestDetectFormatForMixedImportedTypes) {
    auto root = make_temp_dataset("mixed_detect_format");
    write_file(root / "data.yaml", "path: .\ntrain: images\nnames: [fish]\n");
    write_file(root / "images" / "frame_000100.jpg");
    write_file(root / "images" / "frame_000101.jpg");
    write_file(root / "labels" / "frame_000100.txt", "0 0.5 0.5 0.5 0.5\n");
    write_file(root / "labels" / "frame_000101.txt", "0 0 0 1 0 1 1 0 1\n");

    auto preview = preview_yolo_import(default_import_options(root / "data.yaml"), import_scope_t::all_videos);

    ASSERT_TRUE(preview.can_import()) << Meta::toStr(preview.errors);
    EXPECT_EQ(task_t::mixed, preview.task);
    EXPECT_FALSE(preview.metadata.detect_format_changed);
    EXPECT_EQ(track::detect::ObjectDetectionFormat::none, preview.metadata.imported_detect_format);
}

TEST(DetectAnnotationImporter, ImportsPoseWithKeypointMetadata) {
    auto root = make_temp_dataset("pose");
    write_file(root / "data.yaml",
               "path: .\ntrain: images\nnames: [fish]\nkpt_shape: [2, 3]\nkeypoint_names:\n  - nose\n  - tail\n");
    write_file(root / "images" / "source_index_000102.jpg");
    write_file(root / "labels" / "source_index_000102.txt", "0 0.5 0.5 0.2 0.2 0.1 0.2 2 0.3 0.4 0\n");

    auto preview = preview_yolo_import(default_import_options(root / "data.yaml"), import_scope_t::all_videos);

    ASSERT_TRUE(preview.can_import()) << Meta::toStr(preview.errors);
    EXPECT_TRUE(preview.metadata.detect_format_changed);
    EXPECT_EQ(track::detect::ObjectDetectionFormat::poses, preview.metadata.imported_detect_format);
    EXPECT_TRUE(preview.metadata.keypoint_names_changed);
    EXPECT_EQ(std::vector<std::string>({"nose", "tail"}), preview.metadata.imported_keypoint_names);
    const auto& annotation = preview.annotations.at(2_f).front();
    EXPECT_EQ(AnnotationType::POSE, annotation.type);
    ASSERT_EQ(2u, annotation.points.size());
    EXPECT_EQ(Annotation::Point_t(20, 20), annotation.points.at(0));
    EXPECT_FALSE(annotation.points.at(1).valid());
}

TEST(DetectAnnotationImporter, ImportsYoloClassAndKeypointNamesFromDataYaml) {
    auto root = make_temp_dataset("yolo_metadata");
    write_file(root / "data.yaml",
               "path: .\n"
               "train: images\n"
               "names:\n"
               "  0: locust-98Nu\n"
               "  1: Locusta\n"
               "kpt_shape: [7, 3]\n"
               "keypoint_names:\n"
               "  - left_antenna\n"
               "  - head\n"
               "  - right_antenna\n"
               "  - QR\n"
               "  - tail\n"
               "  - left_hind\n"
               "  - right_hind\n");
    write_file(root / "images" / "frame_000100.jpg");
    write_file(root / "labels" / "frame_000100.txt",
               "1 0.5 0.5 0.5 0.5"
               " 0.1 0.1 2 0.2 0.2 2 0.3 0.3 2 0.4 0.4 2"
               " 0.5 0.5 2 0.6 0.6 2 0.7 0.7 2\n");

    auto preview = preview_yolo_import(default_import_options(root / "data.yaml"), import_scope_t::all_videos);

    ASSERT_TRUE(preview.can_import()) << Meta::toStr(preview.errors);
    ASSERT_TRUE(preview.metadata.imported_class_names.contains(0));
    ASSERT_TRUE(preview.metadata.imported_class_names.contains(1));
    EXPECT_EQ("locust-98Nu", preview.metadata.imported_class_names.at(0));
    EXPECT_EQ("Locusta", preview.metadata.imported_class_names.at(1));
    EXPECT_EQ((blob::ObjectClass_t{{0, "locust-98Nu"}, {1, "Locusta"}}), preview.metadata.imported_class_names);
    EXPECT_EQ((std::vector<std::string>{"left_antenna", "head", "right_antenna", "QR", "tail", "left_hind", "right_hind"}),
              preview.metadata.imported_keypoint_names);
    ASSERT_TRUE(preview.annotations.contains(0_f));
    EXPECT_EQ("Locusta", preview.metadata.imported_class_names.at(preview.annotations.at(0_f).front().clid));
}

TEST(DetectAnnotationImporter, UsesCsvMappingForArbitraryImageNames) {
    auto root = make_temp_dataset("csv");
    write_file(root / "data.yaml", "path: .\ntrain: images\nnames: [fish]\n");
    write_file(root / "images" / "arbitrary_name.jpg");
    write_file(root / "labels" / "arbitrary_name.txt", "0 0.5 0.5 0.5 0.5\n");
    write_file(root / "mapping.csv", "image,video_source,source_index\narbitrary_name.jpg,current_video.mp4,112\n");

    auto options = default_import_options(root / "data.yaml");
    options.frame_mapping_csv = root / "mapping.csv";
    options.current_source_basename = "current_video.mp4";
    auto preview = preview_yolo_import(options, import_scope_t::all_videos);

    ASSERT_TRUE(preview.can_import()) << Meta::toStr(preview.errors);
    EXPECT_EQ(1u, preview.mapped_from_csv);
    EXPECT_TRUE(preview.annotations.contains(12_f));
}

TEST(DetectAnnotationImporter, UsesSplitRelativeCsvMappingBeforeBasenameFallback) {
    auto root = make_temp_dataset("csv_split_paths");
    write_file(root / "data.yaml", "path: .\ntrain: train/images\nval: val/images\nnames: [fish]\n");
    write_file(root / "train" / "images" / "duplicate.jpg");
    write_file(root / "val" / "images" / "duplicate.jpg");
    write_file(root / "train" / "labels" / "duplicate.txt", "0 0.25 0.25 0.2 0.2\n");
    write_file(root / "val" / "labels" / "duplicate.txt", "0 0.75 0.75 0.2 0.2\n");
    const auto train_image = (file::Path("train") / "images" / "duplicate.jpg").str();
    const auto val_image = (file::Path("val") / "images" / "duplicate.jpg").str();
    write_file(root / "mapping.csv",
               std::string("image,video_source,source_index\n")
               + train_image + ",current_video.mp4,112\n"
               + val_image + ",current_video.mp4,113\n");

    auto options = default_import_options(root / "data.yaml");
    options.frame_mapping_csv = root / "mapping.csv";
    options.current_source_basename = "current_video.mp4";
    auto preview = preview_yolo_import(options, import_scope_t::all_videos);

    ASSERT_TRUE(preview.can_import()) << Meta::toStr(preview.errors);
    EXPECT_EQ(2u, preview.mapped_from_csv);
    EXPECT_TRUE(preview.annotations.contains(12_f));
    EXPECT_TRUE(preview.annotations.contains(13_f));
    EXPECT_EQ(Annotation::Point_t(30, 15), preview.annotations.at(12_f).front().points.at(0));
    EXPECT_EQ(Annotation::Point_t(130, 65), preview.annotations.at(13_f).front().points.at(0));
}

TEST(DetectAnnotationImporter, UsesAbsoluteImagePathInCsvMapping) {
    auto root = make_temp_dataset("csv_absolute_paths");
    write_file(root / "data.yaml", "path: .\ntrain: train/images\nnames: [fish]\n");
    auto image = root / "train" / "images" / "absolute_name.jpg";
    write_file(image);
    write_file(root / "train" / "labels" / "absolute_name.txt", "0 0.5 0.5 0.5 0.5\n");
    write_file(root / "mapping.csv",
               std::string("image,video_source,source_index\n")
               + image.str() + ",current_video.mp4,112\n");

    auto options = default_import_options(root / "data.yaml");
    options.frame_mapping_csv = root / "mapping.csv";
    options.current_source_basename = "current_video.mp4";
    auto preview = preview_yolo_import(options, import_scope_t::all_videos);

    ASSERT_TRUE(preview.can_import()) << Meta::toStr(preview.errors);
    EXPECT_EQ(1u, preview.mapped_from_csv);
    EXPECT_TRUE(preview.annotations.contains(12_f));
}

TEST(DetectAnnotationImporter, WarnsAndKeepsSourceAnnotationsOutsideConvertedRange) {
    auto root = make_temp_dataset("outside_range");
    write_file(root / "data.yaml", "path: .\ntrain: images\nnames: [fish]\n");
    write_file(root / "images" / "source_index_000099.jpg");
    write_file(root / "images" / "source_index_000151.jpg");
    write_file(root / "labels" / "source_index_000099.txt", "0 0.5 0.5 0.5 0.5\n");
    write_file(root / "labels" / "source_index_000151.txt", "0 0.5 0.5 0.5 0.5\n");

    auto preview = preview_yolo_import(default_import_options(root / "data.yaml"), import_scope_t::all_videos);

    ASSERT_TRUE(preview.can_import()) << Meta::toStr(preview.errors);
    EXPECT_TRUE(preview.errors.empty()) << Meta::toStr(preview.errors);
    EXPECT_GE(preview.warnings.size(), 2u);
    EXPECT_TRUE(preview.annotations.empty());
    ASSERT_TRUE(preview.source_annotations.contains("current"));
    EXPECT_TRUE(preview.source_annotations.at("current").contains(99_f));
    EXPECT_TRUE(preview.source_annotations.at("current").contains(151_f));
}

TEST(DetectAnnotationImporter, FiltersCsvRowsByCurrentVideoSource) {
    auto root = make_temp_dataset("csv_sources");
    write_file(root / "data.yaml", "path: .\ntrain: images\nnames: [fish]\n");
    write_file(root / "images" / "current_image.jpg");
    write_file(root / "images" / "other_image.jpg");
    write_file(root / "labels" / "current_image.txt", "0 0.5 0.5 0.5 0.5\n");
    write_file(root / "labels" / "other_image.txt", "0 0.5 0.5 0.5 0.5\n");
    write_file(root / "mapping.csv",
               "image,video_source,source_index\n"
               "current_image.jpg,current_video.mp4,112\n"
               "other_image.jpg,other_video.mp4,113\n");

    auto options = default_import_options(root / "data.yaml");
    options.frame_mapping_csv = root / "mapping.csv";
    options.current_source_basename = "current_video.mp4";
    auto preview = preview_yolo_import(options, import_scope_t::all_videos);

    ASSERT_TRUE(preview.can_import()) << Meta::toStr(preview.errors);
    EXPECT_EQ(2u, preview.mapped_from_csv);
    EXPECT_EQ(1u, preview.skipped_other_sources);
    EXPECT_EQ("current_video.mp4", preview.auto_source_basename);
    EXPECT_EQ("current_video.mp4", preview.selected_source_basename);
    ASSERT_EQ(2u, preview.source_choices.size());
    EXPECT_TRUE(preview.annotations.contains(12_f));
    EXPECT_FALSE(preview.annotations.contains(13_f));
    ASSERT_TRUE(preview.source_annotations.contains("current_video.mp4"));
    ASSERT_TRUE(preview.source_annotations.contains("other_video.mp4"));
    EXPECT_TRUE(preview.source_annotations.at("current_video.mp4").contains(112_f));
    EXPECT_TRUE(preview.source_annotations.at("other_video.mp4").contains(113_f));

    options.selected_source_basename = "other_video.mp4";
    auto overridden = preview_yolo_import(options, import_scope_t::all_videos);
    ASSERT_TRUE(overridden.can_import()) << Meta::toStr(overridden.errors);
    EXPECT_TRUE(overridden.annotations.contains(13_f));
    EXPECT_FALSE(overridden.annotations.contains(12_f));
}

TEST(DetectAnnotationImporter, RequiresVideoSourceColumnInCsvMapping) {
    auto root = make_temp_dataset("csv_missing_source");
    write_file(root / "data.yaml", "path: .\ntrain: images\nnames: [fish]\n");
    write_file(root / "images" / "arbitrary_name.jpg");
    write_file(root / "labels" / "arbitrary_name.txt", "0 0.5 0.5 0.5 0.5\n");
    write_file(root / "mapping.csv", "image,source_index\narbitrary_name.jpg,112\n");

    auto options = default_import_options(root / "data.yaml");
    options.frame_mapping_csv = root / "mapping.csv";
    options.current_source_basename = "current_video.mp4";
    auto preview = preview_yolo_import(options, import_scope_t::all_videos);

    EXPECT_FALSE(preview.errors.empty());
}

TEST(DetectAnnotationImporter, StripsCurrentVideoPrefixBeforeParsingFrameIndex) {
    auto root = make_temp_dataset("video_prefix");
    write_file(root / "data.yaml", "path: .\ntrain: images\nnames: [fish]\n");
    write_file(root / "images" / "cam2024_run7_frame_000110.jpg");
    write_file(root / "images" / "other_video_frame_000111.jpg");
    write_file(root / "labels" / "cam2024_run7_frame_000110.txt", "0 0.5 0.5 0.5 0.5\n");
    write_file(root / "labels" / "other_video_frame_000111.txt", "0 0.5 0.5 0.5 0.5\n");

    auto options = default_import_options(root / "data.yaml");
    options.current_source_basename = "cam2024_run7.mp4";
    auto preview = preview_yolo_import(options, import_scope_t::all_videos);

    ASSERT_TRUE(preview.can_import()) << Meta::toStr(preview.errors);
    EXPECT_EQ(2u, preview.mapped_from_filenames);
    EXPECT_EQ(1u, preview.skipped_other_sources);
    EXPECT_TRUE(preview.annotations.contains(10_f));
    EXPECT_FALSE(preview.annotations.contains(11_f));
}

TEST(DetectAnnotationImporter, AutoDetectsAndOverridesSourceFileFromFilenames) {
    auto root = make_temp_dataset("source_choice");
    write_file(root / "data.yaml", "path: .\ntrain: images\nnames: [fish]\n");
    write_file(root / "images" / "current_video_frame_000112.jpg");
    write_file(root / "images" / "other_video_frame_000113.jpg");
    write_file(root / "labels" / "current_video_frame_000112.txt", "0 0.5 0.5 0.5 0.5\n");
    write_file(root / "labels" / "other_video_frame_000113.txt", "0 0.5 0.5 0.5 0.5\n");

    auto options = default_import_options(root / "data.yaml");
    options.current_source_basename = "current_video.mp4";
    auto preview = preview_yolo_import(options, import_scope_t::all_videos);

    ASSERT_TRUE(preview.can_import()) << Meta::toStr(preview.errors);
    EXPECT_EQ("current_video", preview.auto_source_basename);
    EXPECT_EQ("current_video", preview.selected_source_basename);
    ASSERT_EQ(2u, preview.source_choices.size());
    EXPECT_TRUE(preview.annotations.contains(12_f));
    EXPECT_FALSE(preview.annotations.contains(13_f));
    EXPECT_TRUE(preview.source_annotations.at("other_video").contains(113_f));

    options.selected_source_basename = "other_video";
    auto overridden = preview_yolo_import(options, import_scope_t::all_videos);

    ASSERT_TRUE(overridden.can_import()) << Meta::toStr(overridden.errors);
    EXPECT_EQ("current_video", overridden.auto_source_basename);
    EXPECT_EQ("other_video", overridden.selected_source_basename);
    EXPECT_FALSE(overridden.annotations.contains(12_f));
    EXPECT_TRUE(overridden.annotations.contains(13_f));
}

TEST(DetectAnnotationImporter, StripsEncodedVideoExtensionPrefixBeforeParsingFrameIndex) {
    auto root = make_temp_dataset("encoded_video_prefix");
    write_file(root / "data.yaml", "path: .\ntrain: images\nnames: [fish]\n");
    write_file(root / "images" / "20250129_hexbug_5_long_mp4-0086_jpg.rf.d092af8a1bcf290c06af89399f4e46a4.jpg");
    write_file(root / "labels" / "20250129_hexbug_5_long_mp4-0086_jpg.rf.d092af8a1bcf290c06af89399f4e46a4.txt", "0 0.5 0.5 0.5 0.5\n");

    auto options = default_import_options(root / "data.yaml");
    options.current_source_basename = "20250129_hexbug_5_long.mp4";
    options.source_start = 0_f;
    options.source_end = 200_f;
    options.converted_length = 200_f;
    auto preview = preview_yolo_import(options, import_scope_t::all_videos);

    ASSERT_TRUE(preview.can_import()) << Meta::toStr(preview.errors);
    EXPECT_EQ(1u, preview.mapped_from_filenames);
    EXPECT_TRUE(preview.annotations.contains(86_f));
}

TEST(DetectAnnotationImporter, KeepsPlainFrameNamesWhenCurrentVideoSourceIsKnown) {
    auto root = make_temp_dataset("plain_with_source");
    write_file(root / "data.yaml", "path: .\ntrain: images\nnames: [fish]\n");
    write_file(root / "images" / "frame_000110.jpg");
    write_file(root / "labels" / "frame_000110.txt", "0 0.5 0.5 0.5 0.5\n");

    auto options = default_import_options(root / "data.yaml");
    options.current_source_basename = "current_video.mp4";
    auto preview = preview_yolo_import(options, import_scope_t::all_videos);

    ASSERT_TRUE(preview.can_import()) << Meta::toStr(preview.errors);
    EXPECT_EQ(1u, preview.mapped_from_filenames);
    EXPECT_EQ(0u, preview.skipped_other_sources);
    EXPECT_TRUE(preview.annotations.contains(10_f));
}

TEST(DetectAnnotationImporter, BlocksImportAndRequestsCsvWhenNoImagesMap) {
    auto root = make_temp_dataset("no_mapping");
    write_file(root / "data.yaml", "path: .\ntrain: images\nnames: [fish]\n");
    write_file(root / "images" / "other_video_arbitrary_name.jpg");
    write_file(root / "labels" / "other_video_arbitrary_name.txt", "0 0.5 0.5 0.5 0.5\n");

    auto options = default_import_options(root / "data.yaml");
    options.current_source_basename = "current_video.mp4";
    auto preview = preview_yolo_import(options, import_scope_t::all_videos);

    EXPECT_FALSE(preview.can_import());
    ASSERT_FALSE(preview.errors.empty());
    EXPECT_NE(std::string::npos, preview.errors.back().find("CSV mapping"));
}

TEST(DetectAnnotationImporter, ReportsMissingDataYamlAndMalformedLabels) {
    auto missing = preview_yolo_import(default_import_options(file::Path("/tmp/trex_missing_data_yaml.yaml")), import_scope_t::all_videos);
    EXPECT_FALSE(missing.errors.empty());

    auto root = make_temp_dataset("malformed");
    write_file(root / "data.yaml", "path: .\ntrain: images\nnames: [fish]\n");
    write_file(root / "images" / "frame_000100.jpg");
    write_file(root / "labels" / "frame_000100.txt", "0 0.5 nope 0.5 0.5\n");

    auto malformed = preview_yolo_import(default_import_options(root / "data.yaml"), import_scope_t::all_videos);
    EXPECT_FALSE(malformed.warnings.empty());
}

TEST(DetectAnnotationImporter, AppliesAddAndReplaceModesWithRenumberedUids) {
    auto root = make_temp_dataset("merge");
    write_file(root / "data.yaml", "path: .\ntrain: images\nnames: [fish]\n");
    write_file(root / "images" / "frame_000110.jpg");
    write_file(root / "labels" / "frame_000110.txt", "0 0.5 0.5 0.5 0.5\n");

    auto preview = preview_yolo_import(default_import_options(root / "data.yaml"), import_scope_t::all_videos);
    ASSERT_TRUE(preview.can_import()) << Meta::toStr(preview.errors);

    AnnotationMap existing;
    existing[10_f].push_back(make_annotation(1, AnnotationType::BOX, {{0, 0}, {10, 0}, {10, 10}, {0, 10}}));

    auto added = apply_dataset_import(preview, existing, merge_mode_t::add);
    ASSERT_EQ(2u, added.at(10_f).size());
    EXPECT_EQ(0u, added.at(10_f).at(0).uid);
    EXPECT_EQ(1u, added.at(10_f).at(1).uid);

    auto replaced = apply_dataset_import(preview, existing, merge_mode_t::replace);
    ASSERT_EQ(1u, replaced.at(10_f).size());
    EXPECT_EQ(0u, replaced.at(10_f).front().uid);
    EXPECT_EQ(0u, replaced.at(10_f).front().clid);
}

TEST(DetectAnnotationImporter, AppliesSourceAnnotationsInsideAnnotationMapWithRenumberedUids) {
    ImportPreview preview;
    preview.annotations[1_f].push_back(make_annotation(0, AnnotationType::BOX, {{0, 0}, {10, 10}}));
    preview.source_annotations["current_video.mp4"][101_f].push_back(make_annotation(0, AnnotationType::BOX, {{0, 0}, {10, 10}}));
    preview.source_annotations["other_video.mp4"][202_f].push_back(make_annotation(1, AnnotationType::BOX, {{5, 5}, {15, 15}}));

    AnnotationMap existing;
    existing.sources()["other_video.mp4"][202_f].push_back(make_annotation(2, AnnotationType::BOX, {{1, 1}, {2, 2}}));

    auto added = apply_dataset_import(preview, existing, merge_mode_t::add);
    ASSERT_EQ(2u, added.sources().at("other_video.mp4").at(202_f).size());
    EXPECT_EQ(0u, added.sources().at("other_video.mp4").at(202_f).at(0).uid);
    EXPECT_EQ(1u, added.sources().at("other_video.mp4").at(202_f).at(1).uid);
    EXPECT_TRUE(added.sources().at("current_video.mp4").contains(101_f));

    auto replaced = apply_dataset_import(preview, existing, merge_mode_t::replace);
    ASSERT_EQ(1u, replaced.sources().at("other_video.mp4").at(202_f).size());
    EXPECT_EQ(1u, replaced.sources().at("other_video.mp4").at(202_f).front().clid);
}

TEST(DetectAnnotationImporter, ReportsMetadataDifferencesInPreview) {
    auto root = make_temp_dataset("metadata");
    write_file(root / "data.yaml", "path: .\ntrain: images\nnames:\n  0: imported\n");
    write_file(root / "images" / "frame_000100.jpg");
    write_file(root / "labels" / "frame_000100.txt", "0 0.5 0.5 0.5 0.5\n");

    auto options = default_import_options(root / "data.yaml");
    options.current_class_names = blob::MaybeObjectClass_t{blob::ObjectClass_t{{0, "current"}}};
    auto preview = preview_yolo_import(options, import_scope_t::all_videos);

    ASSERT_TRUE(preview.can_import()) << Meta::toStr(preview.errors);
    EXPECT_TRUE(preview.metadata.class_names_changed);
    ASSERT_FALSE(preview.warnings.empty());
}

TEST(DetectAnnotationImporter, ImportsCocoBoundingBoxes) {
    auto root = make_temp_dataset("coco_boxes");
    write_file(root / "_annotations.coco.json",
               R"({"images":[{"id":1,"file_name":"frame_000110.jpg","width":200,"height":100}],)"
               R"("annotations":[{"id":1,"image_id":1,"category_id":0,"bbox":[50,25,100,50],"area":5000,"iscrowd":0}],)"
               R"("categories":[{"id":0,"name":"fish"}]})");

    auto options = default_import_options(root / "_annotations.coco.json");
    options.format = annotation_dataset::format_t::coco;
    auto preview = preview_coco_import(options);

    ASSERT_TRUE(preview.can_import()) << Meta::toStr(preview.errors);
    ASSERT_TRUE(preview.annotations.contains(10_f));
    EXPECT_TRUE(preview.metadata.detect_format_changed);
    EXPECT_EQ(track::detect::ObjectDetectionFormat::boxes, preview.metadata.imported_detect_format);
    const auto& annotation = preview.annotations.at(10_f).front();
    EXPECT_EQ(AnnotationType::BOX, annotation.type);
    ASSERT_EQ(4u, annotation.points.size());
    EXPECT_EQ(Annotation::Point_t(50, 25), annotation.points.at(0));
    EXPECT_EQ(Annotation::Point_t(150, 75), annotation.points.at(2));
    EXPECT_TRUE(preview.metadata.class_names_changed);
}

TEST(DetectAnnotationImporter, ImportsCocoWithCurrentVideoPrefixFrameNames) {
    auto root = make_temp_dataset("coco_video_prefix");
    write_file(root / "_annotations.coco.json",
               R"({"images":[{"id":1,"file_name":"20250129_hexbug_5_long_mp4-0086_jpg.rf.d092af8a1bcf290c06af89399f4e46a4.jpg","width":200,"height":100}],)"
               R"("annotations":[{"id":1,"image_id":1,"category_id":0,"bbox":[50,25,100,50],"area":5000,"iscrowd":0}],)"
               R"("categories":[{"id":0,"name":"fish"}]})");

    auto options = default_import_options(root / "_annotations.coco.json");
    options.format = annotation_dataset::format_t::coco;
    options.current_source_basename = "20250129_hexbug_5_long.mp4";
    options.source_start = 0_f;
    options.source_end = 200_f;
    options.converted_length = 200_f;
    auto preview = preview_coco_import(options);

    ASSERT_TRUE(preview.can_import()) << Meta::toStr(preview.errors);
    EXPECT_EQ(1u, preview.mapped_from_filenames);
    EXPECT_TRUE(preview.annotations.contains(86_f));
}

TEST(DetectAnnotationImporter, ImportsCocoWithCsvVideoSourceMapping) {
    auto root = make_temp_dataset("coco_csv_sources");
    write_file(root / "_annotations.coco.json",
               R"({"images":[)"
               R"({"id":1,"file_name":"current_image.jpg","width":200,"height":100},)"
               R"({"id":2,"file_name":"other_image.jpg","width":200,"height":100}],)"
               R"("annotations":[)"
               R"({"id":1,"image_id":1,"category_id":0,"bbox":[50,25,100,50],"area":5000,"iscrowd":0},)"
               R"({"id":2,"image_id":2,"category_id":0,"bbox":[50,25,100,50],"area":5000,"iscrowd":0}],)"
               R"("categories":[{"id":0,"name":"fish"}]})");
    write_file(root / "mapping.csv",
               "image,video_source,source_index\n"
               "current_image.jpg,current_video.mp4,112\n"
               "other_image.jpg,other_video.mp4,113\n");

    auto options = default_import_options(root / "_annotations.coco.json");
    options.format = annotation_dataset::format_t::coco;
    options.frame_mapping_csv = root / "mapping.csv";
    options.current_source_basename = "current_video.mp4";
    auto preview = preview_coco_import(options);

    ASSERT_TRUE(preview.can_import()) << Meta::toStr(preview.errors);
    EXPECT_EQ(2u, preview.mapped_from_csv);
    EXPECT_EQ(1u, preview.skipped_other_sources);
    EXPECT_TRUE(preview.annotations.contains(12_f));
    EXPECT_FALSE(preview.annotations.contains(13_f));
    EXPECT_TRUE(preview.source_annotations.at("current_video.mp4").contains(112_f));
    EXPECT_TRUE(preview.source_annotations.at("other_video.mp4").contains(113_f));
}

TEST(DetectAnnotationImporter, ImportsCocoClassAndKeypointNamesFromRoboflowStyleJson) {
    auto root = make_temp_dataset("coco_metadata");
    write_file(root / "_annotations.coco.json",
               R"({"images":[{"id":0,"file_name":"LEMRA02-MaNa-20250117143515-0-0082_jpg.rf.1dcffa832c7739c8a764a25d72b5116a.jpg","height":3000,"width":4096,"extra":{"name":"LEMRA02-MaNa-20250117143515-0-0082.jpg"}}],)"
               R"("annotations":[{"id":1,"image_id":0,"category_id":1,"bbox":[1323.13,1368.8,176.96,130.07],"iscrowd":0,"area":23017.187,"segmentation":[],"keypoints":[1400.784,1399.544,2,1418.989,1412.774,2,1440.947,1404.65,2,1416.332,1430.438,2,1405.406,1480.5,2,1388.956,1451.137,2,1432.749,1453.945,2],"num_keypoints":7}],)"
               R"("categories":[{"id":0,"name":"locust-98Nu","supercategory":"none"},{"id":1,"name":"Locusta","supercategory":"locust-98Nu","keypoints":["left_antenna","head","right_antenna","QR","tail","left_hind","right_hind"],"skeleton":[[1,2],[3,2],[2,4],[4,5],[6,4],[4,7]]}]})");

    auto options = default_import_options(root / "_annotations.coco.json");
    options.format = annotation_dataset::format_t::coco;
    options.current_source_basename = "LEMRA02-MaNa-20250117143515-0.mp4";
    options.source_start = 0_f;
    options.source_end = 200_f;
    options.converted_length = 200_f;
    auto preview = preview_coco_import(options);

    ASSERT_TRUE(preview.can_import()) << Meta::toStr(preview.errors);
    EXPECT_TRUE(preview.annotations.contains(82_f));
    ASSERT_TRUE(preview.metadata.imported_class_names.contains(0));
    ASSERT_TRUE(preview.metadata.imported_class_names.contains(1));
    EXPECT_EQ("locust-98Nu", preview.metadata.imported_class_names.at(0));
    EXPECT_EQ("Locusta", preview.metadata.imported_class_names.at(1));
    EXPECT_EQ((blob::ObjectClass_t{{0, "locust-98Nu"}, {1, "Locusta"}}), preview.metadata.imported_class_names);
    EXPECT_EQ((std::vector<std::string>{"left_antenna", "head", "right_antenna", "QR", "tail", "left_hind", "right_hind"}),
              preview.metadata.imported_keypoint_names);
    EXPECT_EQ("Locusta", preview.metadata.imported_class_names.at(preview.annotations.at(82_f).front().clid));
    EXPECT_EQ(AnnotationType::POSE, preview.annotations.at(82_f).front().type);
}

TEST(DetectAnnotationImporter, ImportsCocoPoseKeypointsAsAbsoluteXyvPixels) {
    auto root = make_temp_dataset("coco_absolute_keypoints");
    write_file(root / "_annotations.coco.json",
               R"({"images":[{"id":4,"file_name":"source_index_000004.jpg","width":4096,"height":3000}],)"
               R"("annotations":[{"id":28,"image_id":4,"category_id":1,"bbox":[2422,1645,164.19,169.46],"iscrowd":0,"area":27823.637,"segmentation":[],"keypoints":[2541.648,1649.159,2,2550.091,1689.419,2,2582.677,1685.275,2,2525.799,1711.029,2,2425.996,1796.538,2,2472.061,1742.661,2,2485.391,1769.304,2],"num_keypoints":7}],)"
               R"("categories":[{"id":1,"name":"Locusta","keypoints":["left_antenna","head","right_antenna","QR","tail","left_hind","right_hind"],"skeleton":[[1,2],[3,2],[2,4],[4,5],[6,4],[4,7]]}]})");

    auto options = default_import_options(root / "_annotations.coco.json");
    options.format = annotation_dataset::format_t::coco;
    options.video_size = Size2(4096, 3000);
    options.source_start = 0_f;
    options.source_end = 10_f;
    options.converted_length = 10_f;

    auto preview = preview_coco_import(options);

    ASSERT_TRUE(preview.can_import()) << Meta::toStr(preview.errors);
    const auto& annotation = preview.annotations.at(4_f).front();
    EXPECT_EQ(AnnotationType::POSE, annotation.type);
    ASSERT_EQ(7u, annotation.points.size());
    EXPECT_EQ(Annotation::Point_t(2542, 1649), annotation.points.at(0));
    EXPECT_EQ(Annotation::Point_t(2550, 1689), annotation.points.at(1));
    EXPECT_EQ(Annotation::Point_t(2583, 1685), annotation.points.at(2));
    EXPECT_EQ(Annotation::Point_t(2526, 1711), annotation.points.at(3));
    EXPECT_EQ(Annotation::Point_t(2426, 1797), annotation.points.at(4));
    EXPECT_EQ(Annotation::Point_t(2472, 1743), annotation.points.at(5));
    EXPECT_EQ(Annotation::Point_t(2485, 1769), annotation.points.at(6));

    options.video_size = Size2(2048, 1500);
    auto unchanged = preview_coco_import(options);

    ASSERT_TRUE(unchanged.can_import()) << Meta::toStr(unchanged.errors);
    const auto& unchanged_annotation = unchanged.annotations.at(4_f).front();
    ASSERT_EQ(7u, unchanged_annotation.points.size());
    EXPECT_EQ(Annotation::Point_t(2542, 1649), unchanged_annotation.points.at(0));
    EXPECT_EQ(Annotation::Point_t(2426, 1797), unchanged_annotation.points.at(4));

    ASSERT_TRUE(preview.metadata.imported_skeletons);
    auto skeleton = preview.metadata.imported_skeletons->get("Locusta");
    ASSERT_TRUE(skeleton);
    ASSERT_EQ(6u, skeleton->connections().size());
    EXPECT_EQ(0u, skeleton->connections().at(0).from);
    EXPECT_EQ(1u, skeleton->connections().at(0).to);
    EXPECT_EQ(2u, skeleton->connections().at(1).from);
    EXPECT_EQ(1u, skeleton->connections().at(1).to);
}

TEST(DetectAnnotationImporter, ImportsCocoSegmentationPolygons) {
    auto root = make_temp_dataset("coco_segmentation");
    write_file(root / "_annotations.coco.json",
               R"({"images":[{"id":1,"file_name":"source-000101.jpg","width":200,"height":100}],)"
               R"("annotations":[{"id":1,"image_id":1,"category_id":0,"segmentation":[[0,0,200,0,200,100,0,100]],"bbox":[0,0,200,100],"area":20000,"iscrowd":0}],)"
               R"("categories":[{"id":0,"name":"fish"}]})");

    auto options = default_import_options(root / "_annotations.coco.json");
    options.format = annotation_dataset::format_t::coco;
    auto preview = preview_dataset_import(options, import_scope_t::all_videos);

    ASSERT_TRUE(preview.can_import()) << Meta::toStr(preview.errors);
    EXPECT_TRUE(preview.metadata.detect_format_changed);
    EXPECT_EQ(track::detect::ObjectDetectionFormat::masks, preview.metadata.imported_detect_format);
    const auto& annotation = preview.annotations.at(1_f).front();
    EXPECT_EQ(AnnotationType::SEGMENTATION, annotation.type);
    ASSERT_EQ(4u, annotation.points.size());
    EXPECT_EQ(Annotation::Point_t(200, 100), annotation.points.at(2));
}

TEST(DetectAnnotationImporter, ImportsCocoPoseAndKeypointMetadata) {
    auto root = make_temp_dataset("coco_pose");
    write_file(root / "_annotations.coco.json",
               R"({"images":[{"id":1,"file_name":"source_index_000102.jpg","width":200,"height":100}],)"
               R"("annotations":[{"id":1,"image_id":1,"category_id":0,"bbox":[20,20,60,40],"keypoints":[20,20,2,60,40,0],"num_keypoints":1,"area":2400,"iscrowd":0}],)"
               R"("categories":[{"id":0,"name":"fish","keypoints":["nose","tail"],"skeleton":[]}]})");

    auto options = default_import_options(root / "_annotations.coco.json");
    options.format = annotation_dataset::format_t::coco;
    auto preview = preview_coco_import(options);

    ASSERT_TRUE(preview.can_import()) << Meta::toStr(preview.errors);
    EXPECT_TRUE(preview.metadata.detect_format_changed);
    EXPECT_EQ(track::detect::ObjectDetectionFormat::poses, preview.metadata.imported_detect_format);
    EXPECT_TRUE(preview.metadata.keypoint_names_changed);
    EXPECT_EQ(std::vector<std::string>({"nose", "tail"}), preview.metadata.imported_keypoint_names);
    const auto& annotation = preview.annotations.at(2_f).front();
    EXPECT_EQ(AnnotationType::POSE, annotation.type);
    ASSERT_EQ(2u, annotation.points.size());
    EXPECT_EQ(Annotation::Point_t(20, 20), annotation.points.at(0));
    EXPECT_FALSE(annotation.points.at(1).valid());
}

TEST(DetectAnnotationImporter, RejectsUnsupportedCocoRleSegmentation) {
    auto root = make_temp_dataset("coco_rle");
    write_file(root / "_annotations.coco.json",
               R"({"images":[{"id":1,"file_name":"frame_000100.jpg","width":200,"height":100}],)"
               R"("annotations":[{"id":1,"image_id":1,"category_id":0,"segmentation":{"counts":"abc","size":[100,200]},"area":10,"iscrowd":1}],)"
               R"("categories":[{"id":0,"name":"fish"}]})");

    auto options = default_import_options(root / "_annotations.coco.json");
    options.format = annotation_dataset::format_t::coco;
    auto preview = preview_coco_import(options);

    EXPECT_FALSE(preview.errors.empty());
}

TEST(DetectAnnotationImporter, RoundTripsCocoUnlabeledKeypointVisibility) {
    // The middle keypoint is unlabeled (visibility 0). Import must keep it as an
    // invalid placeholder so the indices stay aligned with the schema/skeleton,
    // and a subsequent export must keep that slot unlabeled rather than emitting
    // a spurious visible keypoint at the image origin.
    auto root = make_temp_dataset("coco_roundtrip_kp");
    write_file(root / "_annotations.coco.json",
               R"({"images":[{"id":3,"file_name":"source_index_000003.jpg","width":400,"height":200}],)"
               R"("annotations":[{"id":7,"image_id":3,"category_id":0,"bbox":[100,50,200,100],"iscrowd":0,"area":20000,"segmentation":[],"keypoints":[100,50,2,0,0,0,300,150,2],"num_keypoints":2}],)"
               R"("categories":[{"id":0,"name":"fish","keypoints":["a","b","c"],"skeleton":[[1,3]]}]})");

    auto options = default_import_options(root / "_annotations.coco.json");
    options.format = annotation_dataset::format_t::coco;
    options.video_size = Size2(400, 200);
    options.source_start = 0_f;
    options.source_end = 10_f;
    options.converted_length = 10_f;
    auto preview = preview_coco_import(options);

    ASSERT_TRUE(preview.can_import()) << Meta::toStr(preview.errors);
    const auto& imported = preview.annotations.at(3_f).front();
    ASSERT_EQ(3u, imported.points.size());
    EXPECT_TRUE(imported.points.at(0).valid());
    EXPECT_FALSE(imported.points.at(1).valid());
    EXPECT_TRUE(imported.points.at(2).valid());

    auto json = build_coco_json(preview.annotations, {3_f}, Size2(400, 200),
                                preview.metadata.imported_keypoint_names);
    const auto& annotation = json.get_object().at("annotations").get_array().front().get_object();
    const auto& kpts = annotation.at("keypoints").get_array();
    ASSERT_EQ(9u, kpts.size());
    EXPECT_EQ(2, kpts.at(2).get_number());
    EXPECT_EQ(0, kpts.at(5).get_number());
    EXPECT_EQ(2, kpts.at(8).get_number());
    EXPECT_EQ(2, annotation.at("num_keypoints").get_number());

    const auto& bbox = annotation.at("bbox").get_array();
    ASSERT_EQ(4u, bbox.size());
    EXPECT_EQ(100, bbox.at(0).get_number());
    EXPECT_EQ(50, bbox.at(1).get_number());
}

TEST(DetectAnnotationImporter, MalformedRowDropsImageAtomicallyKeepingSourceAndFrameConsistent) {
    // A label file with a valid row followed by a malformed one must contribute
    // nothing: previously the valid row was kept in source_annotations while the
    // frame was dropped from annotations, desyncing the two views (and making the
    // all_videos vs current_video scopes disagree for the very same file).
    auto root = make_temp_dataset("malformed_atomic");
    write_file(root / "data.yaml", "path: .\ntrain: images\nnames:\n  0: fish\n");

    // a clean image -> imports fully (source index 100 -> annotation frame 0)
    write_file(root / "images" / "frame_000100.jpg");
    write_file(root / "labels" / "frame_000100.txt", "0 0.5 0.5 0.5 0.5\n");

    // a valid row followed by a malformed row in the same file -> whole image dropped
    write_file(root / "images" / "frame_000101.jpg");
    write_file(root / "labels" / "frame_000101.txt", "0 0.5 0.5 0.5 0.5\n0 0.5 nope 0.5 0.5\n");

    auto preview = preview_yolo_import(default_import_options(root / "data.yaml"), import_scope_t::all_videos);

    ASSERT_TRUE(preview.can_import()) << Meta::toStr(preview.errors);

    // the clean image is imported; the malformed image is absent from BOTH views
    EXPECT_TRUE(preview.annotations.contains(0_f));
    EXPECT_FALSE(preview.annotations.contains(1_f));

    size_t source_total = 0;
    for(const auto& [source, by_index] : preview.source_annotations)
        for(const auto& [index, anns] : by_index)
            source_total += anns.size();

    size_t frame_total = 0;
    for(const auto& [frame, anns] : preview.annotations)
        frame_total += anns.size();

    // consistency: the malformed image contributes 0 to both views. Before the
    // fix this was 2 (source) vs 1 (frame) because the partial row leaked.
    EXPECT_EQ(source_total, frame_total);
    EXPECT_EQ(1u, frame_total);

    bool reported = false;
    for(const auto& w : preview.warnings)
        if(w.find("frame_000101") != std::string::npos)
            reported = true;
    EXPECT_TRUE(reported) << Meta::toStr(preview.warnings);
}

TEST(DetectAnnotationExporter, CocoExportsAllDetectClassesEvenWhenAbsentFromDataset) {
    // When detect_classes is set it is authoritative: every configured class must
    // be exported as a category, even classes that never appear in the dataset,
    // so COCO/YOLO outputs stay consistent across datasets of the same type.
    GlobalSettings::write([&](Configuration& config) {
        ::default_config::get(config);
    });
    // make sure the detect_classes -> names map callback is registered before we set it
    (void)track::detect::yolo::names::get_map();
    SETTING(detect_classes) = cmn::blob::MaybeObjectClass_t{
        cmn::blob::ObjectClass_t{{0, "fish"}, {1, "shrimp"}, {2, "crab"}}
    };

    AnnotationMap annotations;
    annotations[1_f].push_back(make_annotation(0, AnnotationType::BOX, {{10, 20}, {50, 80}})); // only class 0 is used

    auto json = build_coco_json(annotations, {1_f}, Size2(100, 100), {});

    const auto& categories = json.get_object().at("categories").get_array();
    std::set<std::string> names;
    for(const auto& c : categories)
        names.insert(c.get_object().at("name").get_string());

    EXPECT_EQ(3u, names.size());
    EXPECT_TRUE(names.contains("fish"));
    EXPECT_TRUE(names.contains("shrimp"));
    EXPECT_TRUE(names.contains("crab"));

    // restore the inferred (empty detect_classes) behavior for any other tests
    SETTING(detect_classes) = cmn::blob::MaybeObjectClass_t{};
}
