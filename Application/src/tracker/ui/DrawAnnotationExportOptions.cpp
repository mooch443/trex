#include "DrawAnnotationExportOptions.h"
#include <tracking/AnnotationExporter.h>
#include <core/DetectionTypes.h>
#include <file/DataLocation.h>
#include <gui/DrawStructure.h>
#include <gui/DynamicGUI.h>
#include <gui/dyn/Action.h>
#include <gui/types/Layout.h>
#include <gui/ParseLayoutTypes.h>
#include <misc/GlobalSettings.h>
#include <misc/default_settings.h>
#include <ui/Scene.h>
#include <ui/WorkProgress.h>
#include <portable-file-dialogs.h>

namespace cmn::gui {
namespace {

using namespace dyn;
using namespace track;
using namespace track::annotation_export;
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

}

struct DrawAnnotationExportOptions::Data {
    PlaceinLayout _layout;
    Entangled parent;
    DynamicGUI _gui;

    Format _format{dataset::format_t::yolo};
    std::string _background_percent_text{"0"};
    std::string _suffix_text;
    bool _export_boxes{true};
    bool _export_segmentations{true};
    bool _export_poses{true};

    Summary _summary;
    glz::json_t _info;
    
    std::shared_ptr<pv::File> _video;

    Data(std::shared_ptr<pv::File>&& video)
        : parent(Box(120, 90, 0, 0)), _video(std::move(video))
    {
        parent.set_draggable();
        parent.update([&](Entangled& e) {
            e.advance_wrap(_layout);
        });
    }

    std::vector<std::string> configured_keypoint_names() const {
        auto configured = READ_SETTING(detect_keypoint_names, track::detect::KeypointNames);
        if(configured.names)
            return *configured.names;
        return {};
    }

    AnnotationMap annotations() const {
        auto map = READ_SETTING_WITH_DEFAULT(track_annotations, AnnotationMap{});
        if(!map)
            map.init();
        return map;
    }

    float background_percent() const {
        try {
            return Meta::fromStr<float>(_background_percent_text);
        } catch(...) {
            return -1.f;
        }
    }

    file::Path output_directory() const {
        file::Path input = READ_SETTING(filename, file::Path).filename();
        if(input.empty()) {
            auto source = READ_SETTING(source, file::PathArray);
            input = file::Path(file::find_basename(source)).filename();
        }
        if(input.has_extension("pv"))
            input = input.remove_extension();
        if(input.empty())
            input = file::Path("annotation_dataset");

        std::string folder = (std::string)input.filename() + "_annotations_" + _format.str();
        auto suffix = dataset::clean_filename_suffix(_suffix_text);
        if(!suffix.empty())
            folder += "_" + suffix;
        return file::DataLocation::parse("output", folder);
    }

    file::PathArray export_source() const {
        auto meta_source = READ_SETTING(meta_source_path, std::string);
        if(!meta_source.empty())
            return file::PathArray(meta_source);
        return READ_SETTING(source, file::PathArray);
    }

    Options make_options(const AnnotationMap& map) const {
        Options options;
        options.format = _format;
        options.annotations = filter_annotation_types(map, _export_boxes, _export_segmentations, _export_poses);
        options.source = export_source();
        options.output_directory = output_directory();
        options.video_source_basename = file::Path(file::find_basename(options.source)).filename();
        auto range = READ_SETTING(video_conversion_range, Range<long_t>);
        if(range.start >= 0)
            options.source_start = Frame_t(range.start);
        options.keypoint_names = default_keypoint_names(options.annotations, configured_keypoint_names());
        options.background_percent = background_percent();
        return options;
    }

    std::string summary_text() const {
        std::vector<std::string> parts;
        if(_summary.counts.boxes > 0)
            parts.push_back(Meta::toStr(_summary.counts.boxes) + " boxes");
        if(_summary.counts.segmentations > 0)
            parts.push_back(Meta::toStr(_summary.counts.segmentations) + " segmentations");
        if(_summary.counts.poses > 0)
            parts.push_back(Meta::toStr(_summary.counts.poses) + " poses");

        std::string text = "Exporting <b>" + Meta::toStr(_summary.counts.total()) + "</b> annotations";
        if(parts.size() > 1)
            text += " (" + join(parts, ", ") + ")";
        text += " in <b>" + Meta::toStr(_summary.annotated_frames) + "</b> images.";

        if(_summary.background_frames > 0)
            text += "\nPlus <b>" + Meta::toStr(_summary.background_frames) + "</b> background images, <b>" + Meta::toStr(_summary.total_images) + "</b> images in total.";

        if(_summary.counts.poses > 0)
            text += "\nPose keypoints: <nr>" + Meta::toStr(_summary.keypoint_names.size()) + "</nr> (" + join(_summary.keypoint_names, ", ") + ")";

        text += "\nOutput: <c><cyan>" + _summary.output_directory.str() + "</cyan></c>";

        if(!_summary.errors.empty())
            text += "\n<red>" + settings::htmlify(_summary.errors.front()) + "</red>";
        else if(!_summary.warnings.empty())
            text += "\n<yellow>" + settings::htmlify(_summary.warnings.front()) + "</yellow>";
        return text;
    }

    void update_info() {
        auto map = annotations();
        auto raw_counts = count_annotation_types(map);
        auto options = make_options(map);

        std::optional<Frame_t> source_length;
        if(auto length = READ_SETTING(video_length, uint64_t); length > 0)
            source_length = Frame_t(narrow_cast<Frame_t::number_t>(length));

        std::optional<Size2> source_size;
        if(auto size = READ_SETTING(video_size, Size2); size.width > 0 && size.height > 0)
            source_size = size;

        _summary = summarize(options, source_length, source_size);

        const int types_present = (raw_counts.boxes > 0 ? 1 : 0)
                                + (raw_counts.segmentations > 0 ? 1 : 0)
                                + (raw_counts.poses > 0 ? 1 : 0);

        _info = glz::json_t::object_t{
            {"format", glz::json_t(_format.str())},
            {"background_percent", glz::json_t(_background_percent_text)},
            {"suffix", glz::json_t(_suffix_text)},
            {"boxes", glz::json_t(Meta::toStr(raw_counts.boxes))},
            {"segmentations", glz::json_t(Meta::toStr(raw_counts.segmentations))},
            {"poses", glz::json_t(Meta::toStr(raw_counts.poses))},
            {"export_boxes", glz::json_t(_export_boxes)},
            {"export_segmentations", glz::json_t(_export_segmentations)},
            {"export_poses", glz::json_t(_export_poses)},
            {"multiple_types", glz::json_t(types_present > 1)},
            {"has_annotations", glz::json_t(raw_counts.total() > 0)},
            {"can_export", glz::json_t(_summary.can_export())},
            {"summary", glz::json_t(summary_text())}
        };
    }

    void init_gui() {
        if(_gui)
            return;

        _gui = DynamicGUI{
            .gui = SceneManager::getInstance().gui_task_queue(),
            .path = "annotation_export_layout.json",
            .context = [&]() {
                dyn::Context context;
                context.actions = {
                    ActionFunc("set-format", [this](const Action& action) {
                        REQUIRE_EXACTLY(1, action);
                        _format = Meta::fromStr<Format>(action.first());
                    }),
                    ActionFunc("set-background-percent", [this](const Action& action) {
                        REQUIRE_EXACTLY(1, action);
                        _background_percent_text = action.first();
                    }),
                    ActionFunc("set-suffix", [this](const Action& action) {
                        REQUIRE_EXACTLY(1, action);
                        _suffix_text = action.first();
                    }),
                    ActionFunc("toggle-type", [this](const Action& action) {
                        REQUIRE_EXACTLY(1, action);
                        auto which = action.first();
                        if(which == "box")
                            _export_boxes = !_export_boxes;
                        else if(which == "segmentation")
                            _export_segmentations = !_export_segmentations;
                        else if(which == "pose")
                            _export_poses = !_export_poses;
                    }),
                    ActionFunc("close", [](const Action&) {
                        SETTING(gui_show_annotation_export_options) = false;
                    }),
                    ActionFunc("choose-file", [](const Action& action) {
                        REQUIRE_AT_LEAST(1, action);
                        WorkProgress::add_queue("Selecting file", [action]() {
                            auto parm = action.parameters.front();
                            auto folder = action.parameters.size() > 1 ? action.parameters.at(1) : std::string{};
                            if(not file::Path{folder}.is_folder())
                                folder = {};

                            std::vector<std::string> filters;
                            if(action.parameters.size() > 2)
                                filters.insert(filters.end(), action.parameters.begin() + 2, action.parameters.end());

                            auto files = pfd::open_file("Select a file", folder, filters).result();
                            if(!files.empty())
                                GlobalSettings::get(parm).get().set_value_from_string(files.front());
                        });
                    }),
                    ActionFunc("export", [this](const Action&) {
                        auto options = make_options(annotations());

                        WorkProgress::add_queue("Exporting annotations...", [options]() {
                            auto show_error = [](std::string message) {
                                SETTING(gui_show_annotation_export_options) = true;

                                SceneManager::enqueue([message = std::move(message)](IMGUIBase*, DrawStructure& graph) {
                                    graph.dialog(
                                        "Annotation export failed.\n\n" + settings::htmlify(message),
                                        "Export Error",
                                        "Okay"
                                    );
                                });
                            };

                            try {
                                auto summary = export_dataset(options);
                                Print("Exported annotation dataset to ", summary.output_directory, ".");
                                SETTING(gui_show_annotation_export_options) = false;
                            } catch(const std::exception& e) {
                                show_error(e.what());
                            } catch(...) {
                                show_error("Unknown annotation export error.");
                            }
                        });
                    }),
                    ActionFunc("export-behavior", [this](const Action&) {
                        auto options = make_options(annotations());

                        WorkProgress::add_queue("Exporting behavior annotations...", [this, options]() {
                            auto show_error = [](std::string message) {
                                SETTING(gui_show_annotation_export_options) = true;

                                SceneManager::enqueue([message = std::move(message)](IMGUIBase*, DrawStructure& graph) {
                                    graph.dialog(
                                        "Export failed.\n\n" + settings::htmlify(message),
                                        "Export Error",
                                        "Okay"
                                    );
                                });
                            };

                            try {
                                export_tag_annotations(TagDatasetConfig{.video_file = _video});
                                //Print("Exported annotation dataset to ", summary.output_directory, ".");
                                SETTING(gui_show_annotation_export_options) = false;
                            } catch(const std::exception& e) {
                                show_error(e.what());
                            } catch(...) {
                                show_error("Unknown annotation export error.");
                            }
                        });
                    })
                };

                context.variables = {
                    VarFunc("info", [this](const VarProps&) -> const glz::json_t& {
                        return _info;
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

DrawAnnotationExportOptions::DrawAnnotationExportOptions(std::shared_ptr<pv::File> video)
    : _data(new Data(std::move(video)))
{
}

DrawAnnotationExportOptions::~DrawAnnotationExportOptions() {
    delete _data;
}

void DrawAnnotationExportOptions::draw(DrawStructure& graph) {
    _data->draw(graph);
}

}
