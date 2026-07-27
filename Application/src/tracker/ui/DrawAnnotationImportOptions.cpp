#include "DrawAnnotationImportOptions.h"

#include <tracking/AnnotationImporter.h>
#include <file/PathArray.h>
#include <gui/DrawStructure.h>
#include <gui/DynamicGUI.h>
#include <gui/dyn/Action.h>
#include <gui/types/Layout.h>
#include <misc/GlobalSettings.h>
#include <misc/default_settings.h>
#include <ui/Scene.h>
#include <ui/WorkProgress.h>
#include <portable-file-dialogs.h>

namespace cmn::gui {
namespace {

using namespace dyn;
using namespace track;
using namespace track::annotation_import;
namespace dataset = track::annotation_dataset;

std::string join(const std::vector<std::string>& parts, const std::string& separator) {
    std::string out;
    for(const auto& part : parts) {
        if(!out.empty())
            out += separator;
        out += part;
    }
    return out;
}

struct SourceChoiceItems {
    const ImportPreview* preview{nullptr};

    std::vector<glz::json_t> to_list() const {
        std::vector<glz::json_t> items;
        if(not preview)
            return items;

        for(const auto& source : preview->source_choices) {
            const bool selected = source == preview->selected_source_basename;
            const bool automatic = source == preview->auto_source_basename;
            std::string detail = automatic ? "automatically selected" : "dataset source";
            if(selected)
                detail = "current import source";

            items.emplace_back(glz::json_t::object_t{
                {"name", glz::json_t(source)},
                {"text", glz::json_t(std::string(selected ? "<b>" : "") + source + (selected ? "</b>" : ""))},
                {"detail", glz::json_t(detail)},
                {"selected", glz::json_t(selected)}
            });
        }
        
        std::sort(items.begin(), items.end(), [](const glz::json_t& A, const glz::json_t& B){
            return std::make_tuple(not A.get_object().at("selected").get_boolean(), A.get_object().at("name").get_string()) < std::make_tuple(not B.get_object().at("selected").get_boolean(), B.get_object().at("name").get_string());
        });
        return items;
    }
};

}

struct DrawAnnotationImportOptions::Data {
    PlaceinLayout _layout;
    Entangled parent;
    DynamicGUI _gui;

    file::Path _dataset_file;
    file::Path _mapping_csv;
    std::string _selected_source_basename;
    MergeMode _mode{merge_mode_t::add};
    ImportScope _scope{import_scope_t::current_video};
    bool _metadata_confirmed{false};
    bool _preview_dirty{true};
    ImportPreview _preview;
    glz::json_t _info;

    Data()
        : parent(Box(120, 90, 0, 0))
    {
        parent.set_draggable();
        parent.update([&](Entangled& e) {
            e.advance_wrap(_layout);
        });
    }

    AnnotationMap annotations() const {
        auto map = READ_SETTING_WITH_DEFAULT(track_annotations, AnnotationMap{});
        if(!map)
            map.init();
        return map;
    }

    ImportOptions make_options() const {
        ImportOptions options;
        if(auto format = dataset::format_from_dataset_file(_dataset_file); format)
            options.format = *format;
        options.dataset_file = _dataset_file;
        options.frame_mapping_csv = _mapping_csv;
        options.existing_annotations = annotations();
        options.selected_source_basename = _selected_source_basename;
        options.mode = _mode;
        options.video_size = READ_SETTING(video_size, Size2);

        auto meta_source = READ_SETTING(meta_source_path, std::string);
        if(!meta_source.empty()) {
            options.current_source_basename = file::Path(meta_source).filename();
        } else {
            options.current_source_basename = file::Path(file::find_basename(READ_SETTING(source, file::PathArray))).filename();
        }

        if(auto length = READ_SETTING(video_length, uint64_t); length > 0)
            options.converted_length = Frame_t(narrow_cast<Frame_t::number_t>(length));

        auto range = READ_SETTING(video_conversion_range, Range<long_t>);
        if(range.start >= 0)
            options.source_start = Frame_t(range.start);
        if(range.end >= 0)
            options.source_end = Frame_t(range.end);

        options.current_class_names = READ_SETTING(detect_classes, cmn::blob::MaybeObjectClass_t);
        auto keypoints = READ_SETTING(detect_keypoint_names, track::detect::KeypointNames);
        if(keypoints.names)
            options.current_keypoint_names = *keypoints.names;
        options.current_skeletons = READ_SETTING(detect_skeleton, std::optional<blob::Pose::Skeletons>);
        options.current_detect_format = READ_SETTING(detect_format, track::detect::ObjectDetectionFormat_t);
        return options;
    }

    glz::json_t preview_text() const {
        if(_dataset_file.empty())
            return glz::json_t{};

        std::string text;
        if(_preview.can_import()) {
            std::vector<std::string> parts;
            if(_preview.counts.boxes > 0)
                parts.push_back(Meta::toStr(_preview.counts.boxes) + " boxes");
            if(_preview.counts.segmentations > 0)
                parts.push_back(Meta::toStr(_preview.counts.segmentations) + " segmentations");
            if(_preview.counts.poses > 0)
                parts.push_back(Meta::toStr(_preview.counts.poses) + " poses");

            const auto format = dataset::format_from_dataset_file(_dataset_file);
            text = "Detected <b>" + (format ? format->str() : std::string("unknown")) + "</b> <b>" + Meta::toStr(_preview.task) + "</b> annotations.";
            text += "\nImporting <b>" + Meta::toStr(_preview.counts.total()) + "</b> annotations";
            if(!parts.empty())
                text += " (" + join(parts, ", ") + ")";
            text += " into <b>" + Meta::toStr(_preview.annotated_frames) + "</b> frames.";
            text += "\nFrame mapping: <b>" + Meta::toStr(_preview.mapped_from_filenames) + "</b> filenames";
            if(_preview.mapped_from_csv > 0)
                text += ", <b>" + Meta::toStr(_preview.mapped_from_csv) + "</b> CSV rows";
            if(_preview.skipped_other_sources > 0)
                text += ", <b>" + Meta::toStr(_preview.skipped_other_sources) + "</b> skipped";
            text += ".";
            if(_scope == import_scope_t::all_videos) {
                if(_preview.source_choices.size() > 1)
                    text += "\nScope: importing from <b>all " + Meta::toStr(_preview.source_choices.size()) + "</b> dataset sources.";
                else
                    text += "\nScope: importing from <b>all</b> dataset sources.";
            } else {
                text += "\nScope: importing only from the <b>selected source</b>.";
            }
            if(!_preview.selected_source_basename.empty()) {
                text += "\nSource: <b>" + settings::htmlify(_preview.selected_source_basename) + "</b>";
                if(!_preview.auto_source_basename.empty()
                   && _preview.selected_source_basename != _preview.auto_source_basename)
                {
                    text += " (auto: " + settings::htmlify(_preview.auto_source_basename) + ")";
                }
                text += ".";
            }
            text += "\nMode: <c>" + _mode.str() + "</c> current <c>track_annotations</c>.";
        } else {
            text = "<gray>No importable annotations have been parsed yet.</gray>";
        }

        if(_preview.metadata.has_changes()) {
            text += "\n<yellow>Optional metadata changes available: ";
            std::vector<std::string> changes;
            if(_preview.metadata.class_names_changed)
                changes.push_back("class names");
            if(_preview.metadata.keypoint_names_changed)
                changes.push_back("keypoint names");
            if(_preview.metadata.skeletons_changed)
                changes.push_back("skeletons");
            if(_preview.metadata.detect_format_changed)
                changes.push_back("detect format");
            text += join(changes, ", ") + ".</yellow>";
        }

        if(!_preview.errors.empty())
            text += "\n<red>" + settings::htmlify(_preview.errors.front()) + "</red>";
        else if(!_preview.warnings.empty())
            text += "\n<yellow>" + settings::htmlify(_preview.warnings.front()) + "</yellow>";

        return glz::json_t::val_t{text};
    }

    void refresh_preview() {
        if(!_preview_dirty)
            return;
        _preview_dirty = false;
        if(_dataset_file.empty()) {
            _preview = ImportPreview{};
            return;
        }
        auto format = dataset::format_from_dataset_file(_dataset_file);
        if(!format) {
            _preview = ImportPreview{};
            _preview.dataset_file = _dataset_file;
            _preview.errors.push_back("Select a .yaml/.yml YOLO data.yaml file or a .json COCO annotations file.");
            return;
        }
        _preview = preview_dataset_import(make_options(), _scope);
    }

    std::string file_dialog_start(const file::Path& current_file) const {
        auto current_directory = current_file.remove_filename();
        if(!current_directory.empty() && current_directory.is_folder())
            return current_directory.str();
        return {};
    }

    void update_info() {
        refresh_preview();
        const bool metadata_changes = _preview.metadata.has_changes();
        const bool can_import = _preview.can_import();
        const SourceChoiceItems source_choices{&_preview};
        const auto detected_format = dataset::format_from_dataset_file(_dataset_file);
        const auto detected_format_text = detected_format ? detected_format->str() : std::string("unknown");

        _info = glz::json_t::object_t{
            {"dataset_file", glz::json_t(_dataset_file.str())},
            {"mapping_csv", glz::json_t(_mapping_csv.str())},
            {"selected_source", glz::json_t(_preview.selected_source_basename)},
            {"auto_source", glz::json_t(_preview.auto_source_basename)},
            {"source_choices", cvt2json(source_choices.to_list())},
            {"format", glz::json_t(detected_format_text)},
            {"file_hint", glz::json_t("YOLO .yaml/.yml or COCO .json")},
            {"file_placeholder", glz::json_t("/path/to/data.yaml or /path/to/_annotations.coco.json")},
            {"mode", glz::json_t(_mode.str())},
            {"scope", glz::json_t(_scope.str())},
            {"has_dataset_file", glz::json_t(!_dataset_file.empty())},
            {"can_import", glz::json_t(can_import)},
            {"metadata_changes", glz::json_t(metadata_changes)},
            {"metadata_confirmed", glz::json_t(_metadata_confirmed)},
            
            // Preview counts
            {"counts", glz::json_t::object_t{
                {"boxes", glz::json_t(_preview.counts.boxes)},
                {"segmentations", glz::json_t(_preview.counts.segmentations)},
                {"poses", glz::json_t(_preview.counts.poses)},
                {"total", glz::json_t(_preview.counts.total())}
            }},
            
            // Preview details
            {"task", glz::json_t(_preview.task.str())},
            {"annotated_frames", glz::json_t(_preview.annotated_frames)},
            {"mapped_from_filenames", glz::json_t(_preview.mapped_from_filenames)},
            {"mapped_from_csv", glz::json_t(_preview.mapped_from_csv)},
            {"skipped_other_sources", glz::json_t(_preview.skipped_other_sources)},
            
            // Metadata change details
            {"metadata_class_names_changed", glz::json_t(_preview.metadata.class_names_changed)},
            {"metadata_keypoint_names_changed", glz::json_t(_preview.metadata.keypoint_names_changed)},
            {"metadata_skeletons_changed", glz::json_t(_preview.metadata.skeletons_changed)},
            {"metadata_detect_format_changed", glz::json_t(_preview.metadata.detect_format_changed)},
            
            {"detect_classes", cvt2json(_preview.metadata.imported_class_names) },
            {"detect_skeletons", cvt2json(_preview.metadata.imported_skeletons) },
            {"detect_keypoint_names", cvt2json(_preview.metadata.imported_keypoint_names) },
            {"detect_format", cvt2json(_preview.metadata.imported_detect_format) },
            
            // Errors and warnings
            {"has_errors", glz::json_t(!_preview.errors.empty())},
            {"first_error", glz::json_t(_preview.errors.empty() ? std::string() : _preview.errors.front())},
            {"has_warnings", glz::json_t(!_preview.warnings.empty())},
            {"first_warning", glz::json_t(_preview.warnings.empty() ? std::string() : _preview.warnings.front())},
            
            // Source count (for "all X dataset sources")
            {"source_count", glz::json_t(_preview.source_choices.size())}
        };
    }

    void choose_file(bool mapping) {
        const auto start = file_dialog_start(mapping ? _mapping_csv : _dataset_file);

        auto filters = mapping
            ? std::vector<std::string>{"CSV files", "*.csv", "All files", "*"}
            : std::vector<std::string>{"Annotation datasets", "*.yaml *.yml *.json"};

        auto files = pfd::open_file(mapping ? "Select frame mapping CSV" : "Select annotation dataset", start, filters).result();
        if(files.empty())
            return;

        if(mapping)
            _mapping_csv = file::Path(files.front());
        else
            _dataset_file = file::Path(files.front());
        _selected_source_basename = {};
        _metadata_confirmed = false;
        _preview_dirty = true;
    }

    void init_gui() {
        if(_gui)
            return;

        _gui = DynamicGUI{
            .gui = SceneManager::getInstance().gui_task_queue(),
            .path = "annotation_import_layout.json",
            .context = [&]() {
                dyn::Context context;
                context.actions = {
                    ActionFunc("set-dataset-file", [this](const Action& action) {
                        REQUIRE_EXACTLY(1, action);
                        _dataset_file = file::Path(action.first());
                        _selected_source_basename = {};
                        _metadata_confirmed = false;
                        _preview_dirty = true;
                    }),
                    ActionFunc("set-mapping-csv", [this](const Action& action) {
                        REQUIRE_EXACTLY(1, action);
                        _mapping_csv = file::Path(action.first());
                        _selected_source_basename = {};
                        _preview_dirty = true;
                    }),
                    ActionFunc("set-source", [this](const Action& action) {
                        REQUIRE_EXACTLY(1, action);
                        const auto index = Meta::fromStr<size_t>(action.first());
                        if(index < _preview.source_choices.size())
                            _selected_source_basename = _preview.source_choices.at(index);
                        _preview_dirty = true;
                    }),
                    ActionFunc("choose-dataset-file", [this](const Action&) {
                        WorkProgress::add_queue("Selecting annotation dataset file", [this]() {
                            choose_file(false);
                        });
                    }),
                    ActionFunc("choose-mapping-csv", [this](const Action&) {
                        WorkProgress::add_queue("Selecting mapping CSV", [this]() {
                            choose_file(true);
                        });
                    }),
                    ActionFunc("set-mode", [this](const Action& action) {
                        REQUIRE_EXACTLY(1, action);
                        _mode = Meta::fromStr<MergeMode>(action.first());
                    }),
                    ActionFunc("set-scope", [this](const Action& action) {
                        REQUIRE_EXACTLY(1, action);
                        _scope = Meta::fromStr<ImportScope>(action.first());
                    }),
                    ActionFunc("confirm-metadata", [this](const Action&) {
                        _metadata_confirmed = !_metadata_confirmed;
                    }),
                    ActionFunc("close", [](const Action&) {
                        SETTING(gui_show_annotation_import_options) = false;
                    }),
                    ActionFunc("import", [this](const Action&) {
                        auto options = make_options();
                        const bool apply_metadata = _metadata_confirmed;
                        const auto scope = _scope;

                        WorkProgress::add_queue("Importing annotations...", [options, apply_metadata, scope]() {
                            auto show_error = [](std::string message) {
                                SETTING(gui_show_annotation_import_options) = true;
                                SceneManager::enqueue([message = std::move(message)](IMGUIBase*, DrawStructure& graph) {
                                    graph.dialog(
                                        "Annotation import failed.\n\n" + settings::htmlify(message),
                                        "Import Error",
                                        "Okay"
                                    );
                                });
                            };

                            try {
                                if(!dataset::format_from_dataset_file(options.dataset_file))
                                    throw InvalidArgumentException("Select a .yaml/.yml YOLO data.yaml file or a .json COCO annotations file.");
                                auto preview = preview_dataset_import(options, scope);
                                auto imported = apply_dataset_import(preview, options.existing_annotations, options.mode, scope);
                                SETTING(track_annotations) = std::move(imported);
                                if(apply_metadata) {
                                    if(preview.metadata.class_names_changed)
                                        SETTING(detect_classes) = cmn::blob::MaybeObjectClass_t{preview.metadata.imported_class_names};
                                    if(preview.metadata.keypoint_names_changed)
                                        SETTING(detect_keypoint_names) = track::detect::KeypointNames{.names = preview.metadata.imported_keypoint_names};
                                    if(preview.metadata.skeletons_changed)
                                        SETTING(detect_skeleton) = preview.metadata.imported_skeletons;
                                    if(preview.metadata.detect_format_changed)
                                        SETTING(detect_format) = preview.metadata.imported_detect_format;
                                }
                                Print("Imported annotation dataset from ", options.dataset_file, ".");
                                SETTING(gui_show_annotation_import_options) = false;
                            } catch(const std::exception& e) {
                                show_error(e.what());
                            } catch(...) {
                                show_error("Unknown annotation import error.");
                            }
                        });
                    })
                };

                context.variables = {
                    VarFunc("info", [this](const VarProps&) -> const glz::json_t& {
                        return _info;
                    }),
                    VarFunc("source_choices", [this](const VarProps&) -> std::vector<glz::json_t> {
                        return SourceChoiceItems{&_preview}.to_list();
                    })
                };
                return context;
            }(),
            .base = nullptr
        };
    }

    void draw(DrawStructure& graph) {
        init_gui();
        update_info();

        graph.wrap_object(parent);
        parent.set_scale(graph.scale().reciprocal());
        _gui.update(graph, &_layout);
        parent.auto_size(Margin{10, 10});
    }
};

DrawAnnotationImportOptions::DrawAnnotationImportOptions()
    : _data(new Data)
{
}

DrawAnnotationImportOptions::~DrawAnnotationImportOptions() {
    delete _data;
}

void DrawAnnotationImportOptions::draw(DrawStructure& graph) {
    _data->draw(graph);
}

}
