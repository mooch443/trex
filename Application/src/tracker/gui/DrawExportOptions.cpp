#include "DrawExportOptions.h"
#include <gui/types/Entangled.h>
#include <misc/GlobalSettings.h>
#include <tracking/OutputLibrary.h>
#include <gui/types/Button.h>
#include <gui/types/Textfield.h>
#include <gui/types/ScrollableList.h>
#include <gui/DrawStructure.h>
#include <gui/DynamicGUI.h>
#include <misc/Coordinates.h>
#include <gui/Scene.h>
#include <gui/dyn/Action.h>
#include <gui/ParseLayoutTypes.h>
#include <gui/WorkProgress.h>
#include <portable-file-dialogs.h>
#include <gui/TrackingState.h>

namespace cmn::gui {
using namespace dyn;

namespace doc_strings {
struct Description {
    std::string short_description;
    std::string long_description;
};

/// @brief Short (and long) documentation strings for the options in the export dialog. These are mainly the ones from OutputLibrary::Functions as well as the other ones from Library::Init(), added to _cache_func.
std::unordered_map<std::string, Description> options_doc_strings = {
    {"x", {
        "X-position (cm)",
        "The X-coordinate of the individual's position relative to the tank's center in centimeters."
    }},
    {"y", {
        "Y-position (cm)",
        "The Y-coordinate of the individual's position relative to the tank's center in centimeters."
    }},
    {"vx", {
        "X-velocity (cm/s)",
        "The velocity of the individual in the X-direction in centimeters per second."
    }},
    {"vy", {
        "Y-velocity (cm/s)",
        "The velocity of the individual in the Y-direction in centimeters per second."
    }},
    {"ax", {
        "X-acceleration (cm/s²)",
        "The acceleration of the individual in the X-direction in centimeters per second squared."
    }},
    {"ay", {
        "Y-acceleration (cm/s²)",
        "The acceleration of the individual in the Y-direction in centimeters per second squared."
    }},
    {"speed", {
        "Speed (cm/s)",
        "The individual's speed calculated from its X and Y velocity components in centimeters per second."
    }},
    {"acceleration", {
        "Acceleration (cm/s²)",
        "The magnitude of the individual's acceleration in centimeters per second squared."
    }},
    {"angle", {
        "Angle (radians)",
        "The absolute orientation angle of the individual with respect to the X-axis in radians."
    }},
    {"angular_v", {
        "Angular velocity (radians/s)",
        "The rate of change of the individual's orientation angle in radians per second."
    }},
    {"angular_a", {
        "Angular acceleration (radians/s²)",
        "The rate of change of the individual's angular velocity in radians per second squared."
    }},
    {"midline_offset", {
        "Midline offset (radians)",
        "The angular offset of the individual's midline from a reference direction, measured in radians."
    }},
    {"variance", {
        "Midline variance (radians²)",
        "The variance of the individual's midline angle, calculated over a frame window."
    }},
    {"normalized_midline", {
        "Normalized midline (radians)",
        "The average midline angle normalized over a frame window, measured in radians."
    }},
    {"midline_deriv", {
        "Midline derivative (radians/frame)",
        "The rate of change of the individual's midline angle per frame, measured in radians per frame."
    }},
    {"binary", {
        "Binary threshold crossing",
        "Indicates whether the midline angle crosses a predefined threshold (non-zero value if true, otherwise invalid)."
    }},
    {"border_distance", {
        "Distance to border (cm)",
        "The shortest distance from the individual's current position to the border of the tank in centimeters."
    }},
    {"neighbor_distance", {
        "Neighbor distance (cm)",
        "The average distance between the individual and all other individuals, measured in centimeters."
    }},
    {"time", {
        "Time (s)",
        "The time corresponding to the current frame in seconds."
    }},
    {"timestamp", {
        "Timestamp",
        "The exact timestamp of the current frame."
    }},
    {"frame", {
        "Frame number",
        "The sequential index of the current frame in the dataset."
    }},
    {"missing", {
        "Missing data flag",
        "Indicates whether data is missing for the current frame (1 if missing, 0 if available)."
    }},
    {"neighbor_vector_t", {
        "Perpendicular neighbor vector (cm)",
        "The perpendicular distance from the individual to its neighbors in centimeters."
    }},
    {"relative_angle", {
        "Relative angle (radians)",
        "The difference in orientation between the individual and its neighbors, measured in radians."
    }},
    {"l_v", {
        "Average velocity difference (cm/s)",
        "The average difference in velocity between the individual and other individuals in centimeters per second."
    }},
    {"dot_v", {
        "Velocity alignment (radians)",
        "The angular difference between the individual's velocity vector and those of other individuals in radians."
    }},
    {"tailbeat_threshold", {
        "Tailbeat detection threshold",
        "The threshold value for detecting a tailbeat event based on midline angle."
    }},
    {"tailbeat_peak", {
        "Tailbeat peak offset (radians)",
        "The minimum peak offset value of the midline angle required during a tailbeat event."
    }},
    {"threshold_reached", {
        "Tailbeat threshold crossed",
        "Indicates whether the tailbeat threshold has been crossed for the current frame (non-zero value if true, otherwise invalid)."
    }},
    {"sqrt_a", {
        "Midline offset (radians)",
        "The midline offset value of the individual, measured in radians."
    }},
    {"outline_size", {
        "Outline size (px²)",
        "The size of the individual's outline in pixels squared, representing the area of the silhouette."
    }},
    {"outline_std", {
        "Outline standard deviation (unitless)",
        "The normalized standard deviation of the individual's outline size over a window of frames."
    }},
    {"events", {
        "Behavioral events",
        "Indicates the presence of specific behavioral events such as tailbeats or bursts of activity (non-zero value if event is occurring)."
    }},
    {"event_energy", {
        "Event energy (arbitrary units)",
        "The energy associated with a detected event, representing movement intensity in arbitrary units."
    }},
    {"event_acceleration", {
        "Event acceleration (cm/s²)",
        "The acceleration measured during a behavioral event, in centimeters per second squared."
    }},
    {"detection_class", {
        "Detection class",
        "The class ID assigned to the individual by the YOLO detection algorithm."
    }},
    {"detection_p", {
        "Detection probability",
        "The confidence score of the detection from the YOLO algorithm, ranging from 0 to 1."
    }},
    {"category", {
        "Individual category",
        "A user-defined category label assigned to the individual (unsigned integer ID)."
    }},
    {"average_category", {
        "Average category",
        "The most probable category label of the individual over a series of frames."
    }},
    {"event_direction_change", {
        "Event direction change (radians)",
        "The change in movement direction during a behavioral event, measured in radians."
    }},
    {"v_direction", {
        "Velocity direction change (radians)",
        "The change in the direction of the individual's velocity vector during an event, measured in radians."
    }},
    {"midline_segment_length", {
        "Midline segment length (cm)",
        "The length of a segment of the individual's midline, measured in centimeters."
    }},
    {"consecutive", {
        "Consecutive frames count",
        "The number of consecutive frames with valid data for the individual."
    }},
    {"tracklet_id", {
        "Tracklet ID",
        "A unique identifier for tracklets with valid data."
    }},
    {"blobid", {
        "Blob ID",
        "The unique identifier for the blob corresponding to the individual in the current frame."
    }},
    {"blob_width", {
        "Blob width (px)",
        "The width of the individual's blob (its outline) in pixels."
    }},
    {"blob_height", {
        "Blob height (px)",
        "The height of the individual's blob (its outline) in pixels."
    }},
    {"blob_x", {
        "Blob X-position (px)",
        "The X-coordinate of the individual's blob in the frame, measured in pixels."
    }},
    {"blob_y", {
        "Blob Y-position (px)",
        "The Y-coordinate of the individual's blob in the frame, measured in pixels."
    }},
    {"num_pixels", {
        "Blob pixel count",
        "The number of pixels comprising the individual's blob in the current frame."
    }},
    {"pixels_squared", {
        "Blob area (px²)",
        "The area of the individual's blob in the frame, calculated as width multiplied by height in pixels."
    }},
    {"midline_x", {
        "Midline X-position (cm)",
        "The X-coordinate of the individual's midline position in the frame, measured in centimeters."
    }},
    {"midline_y", {
        "Midline Y-position (cm)",
        "The Y-coordinate of the individual's midline position in the frame, measured in centimeters."
    }},
    {"global", {
        "Average position length (px)",
        "The average length of the positions of all individuals from the origin, measured in pixels."
    }},
    {"compactness", {
        "Compactness (unitless)",
        "A unitless measure of how closely grouped the individuals are in the tank, representing the clustering."
    }},
    {"amplitude", {
        "Midline amplitude (cm)",
        "The amplitude of the individual's midline movement, measured in centimeters."
    }},
    {"midline_length", {
        "Midline length (cm)",
        "The total length of the individual's midline, measured in centimeters."
    }},
    {"qr_id", {
        "QR code ID",
        "The identifier extracted from a QR code associated with the individual (unsigned integer ID)."
    }},
    {"qr_p", {
        "QR code probability",
        "The confidence score of the detected QR code, ranging from 0 to 1."
    }}
};

} // namespace doc_strings

struct DrawExportOptions::Data {
    struct Item {
        std::string _name, _source;
        uint32_t _count = 0;
        bool _single_source{false};
        Font _font = Font(0.5, Align::Left);
        Color _color = White;

        Font font() const {
            return _font;
        }

        Color color() const {
            return _color;
        }

        std::string tooltip() const {
            std::string append;
            if (utils::endsWith(_source, "centroid")) {
                append.push_back(_source[0]);
                append += "centroid";
            }
            else
                append += _source;
            return append;
        }
        operator std::string() const {
            if(not _single_source)
                return _name + "#" + _source + (_count ? " (" + Meta::toStr(_count) + ")" : "");
            else
                return _name + (_count ? " (" + Meta::toStr(_count) + ")" : "");
        }
        bool operator!=(const Item& other) const{
            return _name != other._name || _font != other._font || _count != other._count || _source != other._source;
        }
    };
    
    PlaceinLayout _layout;
    Entangled parent;
    //ScrollableList<Item> export_options;
    DynamicGUI _gui;
    
    std::optional<std::unordered_map<std::string, std::set<std::string_view>, MultiStringHash, MultiStringEqual>> previous_graphs;
    std::optional<std::string> previous_text;
    
    std::vector<sprite::Map> _filtered_options, _selected_options;
    std::vector<std::shared_ptr<dyn::VarBase_t>> _filtered_variables, _selected_variables;
    std::vector<bool> _auto_variables;
    
    std::string _search_text;
    
    Data()
        :
            _layout(),
            parent(Box(100, 100, 800, 650))//,
            //export_options(Box(Vec2(5, 15), Size2(parent.width() - 10, 30)), ItemFont_t(0.5, Align::Left))
    {
        _layout.set(Box(0, 0, 200, parent.height()));
        parent.update([&](Entangled& e) {
            //e.advance_wrap(export_options);
            //e.advance_wrap(search);
            e.advance_wrap(_layout);
        });
        //parent.set_background(Black.alpha(200), Black.alpha(240));

        /*export_options.on_select([&](auto idx, const std::string&) {
            auto graphs = READ_SETTING(output_fields, std::vector<std::pair<std::string, std::vector<std::string>>>);
            auto& item = export_options.items().at(idx);
            Print("Removing ",item.value()._name);

            for (auto it = graphs.begin(); it != graphs.end(); ++it) {
                if (it->first == item.value()._name) {
                    graphs.erase(it);
                    SETTING(output_fields) = graphs;
                    return;
                }
            }

            graphs.push_back({ item.value()._name, {} });
            SETTING(output_fields) = graphs;
        });*/
        
            parent.set_draggable();
    }
    
    std::optional<doc_strings::Description> get_description(std::string_view param) {
        auto lc = utils::lowercase(param);
        if(auto it = doc_strings::options_doc_strings.find(lc);
           it != doc_strings::options_doc_strings.end())
        {
            return it->second;
        } else if(utils::beginsWith(param, "poseX")) {
            return doc_strings::Description{
                .short_description = "YOLO keypoint X coord",
                .long_description = "This represents the X-component of a keypoint provided by a YOLO (or similar) network. You can manually add these `poseX[n]` points, or let TRex automatically add them based on network output."
            };
        } else if(utils::beginsWith(param, "poseY")) {
            return doc_strings::Description{
                .short_description = "YOLO keypoint Y coord",
                .long_description = "This represents the Y-component of a keypoint provided by a YOLO (or similar) network. You can manually add these `poseY[n]` points, or let TRex automatically add them based on network output."
            };
        }
        
        return std::nullopt;
    }
    
    void draw(DrawStructure& base, TrackingState* state) {
        static const std::unordered_map<std::string_view, const char*> mappings {
            {"posture_centroid","PCENTROID"},
            {"weighted_centroid", "WCENTROID"}
        };
        
        static const auto find_sub = [](const std::string& _fn,
                                        const std::string& _sub,
                                        const auto& graphs)
        {
            std::string fn = utils::lowercase(_fn);
            std::string sub = utils::lowercase(_sub);
            
            for(auto it = graphs.begin(); it != graphs.end(); ++it) {
                auto [f, subs] = *it;
                
                if(utils::lowercase(f) != fn) {
                    continue;
                }
                
                for(auto kit = subs.begin(); kit != subs.end();) {
                    if(is_in(utils::lowercase(*kit), "raw", "smooth", "head")) {
                        kit = subs.erase(kit);
                    } else
                        ++kit;
                }
                
                if((sub == "head" || sub.empty())
                   && subs.empty())
                {
                    /// already added the "default"
                    return it;
                } else {
                    for(auto &n : subs) {
                        /// if yes then this is what we were searching for
                        if(utils::lowercase(n) == sub)
                            return it;
                    }
                }
            }
            
            return graphs.end();
        };
        
        if(not _gui) {
            _gui = DynamicGUI{
                .gui = SceneManager::getInstance().gui_task_queue(),
                .path = "export_options_layout.json",
                .context = [&](){
                    dyn::Context context;
                    context.actions = {
                        ActionFunc("clear_list", [](const Action& action) {
                            REQUIRE_EXACTLY(0, action);
                            SETTING(output_fields) = std::vector<std::pair<std::string, std::vector<std::string>>>{};
                        }),
                        ActionFunc("reset_options", [](const Action& action) {
                            REQUIRE_EXACTLY(0, action);
                            if(auto defaults = GlobalSettings::current_defaults("output_fields");
                               defaults)
                            {
                                SETTING(output_fields) = defaults->at("output_fields").value<std::vector<std::pair<std::string, std::vector<std::string>>>>();
                            }
                            
                        }),
                        ActionFunc("add_option", [this](const Action& action) {
                            REQUIRE_EXACTLY(1, action);
                            Print("Got action: ", action);
                            auto idx = Meta::fromStr<uint32_t>(action.parameters.front());
                            auto graphs = READ_SETTING(output_fields, std::vector<std::pair<std::string, std::vector<std::string>>>);
                            auto &item = this->_filtered_options.at(idx);
                            
                            if(auto subname = item.at("subname").value<std::string>();
                               find_sub(item.at("name").value<std::string>(),
                                            subname, graphs) == graphs.end())
                            {
                                Print("* Adding ", no_quotes(item.at("name").value<std::string>()), "#", no_quotes(subname));
                                
                                if(not mappings.contains(subname)) {
                                    graphs.emplace_back(item.at("name").value<std::string>(), std::vector<std::string>{"RAW"});
                                } else {
                                    auto real_name = mappings.at(subname);
                                    graphs.emplace_back(item.at("name").value<std::string>(), std::vector<std::string>{"RAW", real_name});
                                }
                                SETTING(output_fields) = graphs;
                                
                            } else {
                                
                                Print("* ", no_quotes(item.at("name").value<std::string>()), "#", no_quotes(item.at("subname").value<std::string>()), " already added.");
                            }
                        }),
                        ActionFunc("remove_option", [this](const Action& action) {
                            REQUIRE_EXACTLY(1, action);
                            Print("Got action: ", action);
                            auto idx = Meta::fromStr<uint32_t>(action.parameters.front());
                            auto graphs = READ_SETTING(output_fields, std::vector<std::pair<std::string, std::vector<std::string>>>);
                            auto &item = this->_selected_options.at(idx);
                            auto automatic = this->_auto_variables.at(idx);
                            
                            auto subname = item.at("subname").value<std::string>();
                            if(mappings.contains(subname)) {
                                subname = mappings.at(subname);
                            } else {
                                subname = "head";
                            }
                            
                            if(auto it = find_sub(item.at("name").value<std::string>(),
                                                  subname, graphs);
                               it != graphs.end()
                               && not automatic)
                            {
                                Print("* Removing ", no_quotes(item.at("name").value<std::string>()), "#", no_quotes(subname));
                                graphs.erase(it);
                                SETTING(output_fields) = graphs;
                                
                            } else {
                                Print("* ", no_quotes(item.at("name").value<std::string>()), "#", no_quotes(item.at("subname").value<std::string>()), " was not found.");
                            }
                        }),
                        ActionFunc("choose-folder", [](const Action& action) {
                            REQUIRE_AT_LEAST(1, action);
                            WorkProgress::add_queue("Selecting folder", [action](){
                                auto parm = action.parameters.front();
                                auto folder = action.parameters.size() == 1 ? action.parameters.back() : file::cwd().str();
                                if(not file::Path{folder}.is_folder())
                                    folder = {};
                                
                                auto dir = pfd::select_folder("Select a folder", folder).result();
                                GlobalSettings::get(parm).get().set_value_from_string(dir);
                                std::cout << "Selected "<< parm <<": " << dir << "\n";
                            });
                        }),
                        ActionFunc("export", [state](auto){
                            Print("Triggered export...");
                            
                            WorkProgress::add_queue("Saving to "+READ_SETTING(output_format, default_config::output_format_t::Class).str()+" ...", [state]()
                            {
                                state->_controller->export_tracks();
                                /// hide yourself:
                                SETTING(gui_show_export_options) = false;
                            });
                        }),
                        ActionFunc("set", [](Action action) {
                            REQUIRE_EXACTLY(2, action);
                            
                            auto parm = Meta::fromStr<std::string>(action.first());
                            if(not GlobalSettings::has_value(parm))
                                throw InvalidArgumentException("No parameter ",parm," in global settings.");
                            
                            auto value = action.last();
                            GlobalSettings::get(parm).get().set_value_from_string(value);
                        })
                    };

                    context.variables = {
                        VarFunc("window_size", [this](const VarProps&) -> Vec2 {
                            return parent.size();
                        }),
                        VarFunc("available_options", [this](const VarProps&) -> decltype(_filtered_variables)& {
                            return _filtered_variables;
                        }),
                        VarFunc("chosen_options", [this](const VarProps&) -> decltype(_selected_variables)& {
                            return _selected_variables;
                        })
                    };

                    return context;
                }(),
                .base = nullptr
            };
            
            _gui.context.custom_elements["option_search"] = std::unique_ptr<CustomElement>(new CustomElement {
                "option_search",
                [this](LayoutContext& layout) -> Layout::Ptr {
                    derived_ptr<Textfield> search{
                        new Textfield(Box(Vec2(), Size2(parent.width() - 10, 30)))
                    };
                    Placeholder_t placeholder{ layout.get(std::string("Type to filter..."), "placeholder") };
                    search->set(placeholder);
                    ClearText_t cleartext{ layout.get(std::string("<sym>⮾</sym>"), "cleartext") };
                    search->set(cleartext);
                    search->on_text_changed([this, ptr = search.get()](){
                        _search_text = ptr->text();
                    });
                    return Layout::Ptr(search);
                },
                [](Layout::Ptr&, const Context& , State& , const auto& ) -> bool {
                    return false;
                }
            });
        }
        
        auto graphs = READ_SETTING(output_fields, std::vector<std::pair<std::string, std::vector<std::string>>>);
        auto graphs_map = [&graphs]() {
            std::unordered_map<std::string, std::set<std::string_view>, MultiStringHash, MultiStringEqual> result;
            for(auto &g : graphs) {
                static const std::set<Output::Modifiers::Class> wanted{
                    Output::Modifiers::CENTROID,
                    Output::Modifiers::POSTURE_CENTROID,
                    Output::Modifiers::WEIGHTED_CENTROID
                };
                OptionsList<Output::Modifiers::Class> modifiers;
                for (auto& e : g.second) {
                    Output::Library::parse_modifiers(e, modifiers);
                }
                if(modifiers.size() == 0) {
                    // no modifiers is also okay
                    result[g.first].insert("HEAD");
                } else {
                    std::vector<Output::Modifiers::Class> interesting;
                    for (auto& e : modifiers.values()) {
                        if (wanted.contains(e))
                            interesting.push_back(e);
                    }
                    
                    if(interesting.size() == 0)
                        result[g.first].insert("HEAD");
                    else {
                        if(interesting.size() > 1)
                            FormatWarning("Key ", g.first, " has ", interesting);
                        result[g.first].insert(interesting.front().name());
                    }
                }
            }
            return result;
        }();
        
        base.wrap_object(parent);
        parent.set_scale(base.scale().reciprocal());
        //
        if(not previous_graphs
           || *previous_graphs != graphs_map
           || not previous_text
           || _search_text != *previous_text)
        {
            previous_graphs = graphs_map;
            previous_text = _search_text;
            
            auto search_word = utils::lowercase(_search_text);
            std::string search_hash = "";
            if(search_word.contains('#')) {
                auto v = utils::split(search_word, '#');
                assert(v.size() > 1);
                search_hash = v.at(1);
                search_word = v.front();
            }

            auto functions = Output::Library::functions();
            auto pose_fields = default_config::add_missing_pose_fields();
            std::set<std::string> pose_fields_set;
            if(not pose_fields.empty())
            {
                for(auto &[name, mods] : pose_fields) {
                    functions.emplace_back(name);
                    pose_fields_set.emplace(utils::lowercase(name));
                }
            }
            
            std::sort(functions.begin(), functions.end());
            
            std::vector<Item> items;
            
            //_filtered_variables.clear();
            _filtered_options.clear();
            _selected_options.clear();
            _auto_variables.clear();
            
            size_t i = 0, j = 0;
            
            static const std::unordered_map<std::string, std::string> modifier_translations {
                {"head","head"},
                {"posture_centroid","midline"},
                {"weighted_centroid", "centroid"}
            };
            
            for (auto& f : functions) {
                if (search_word.empty()
                    || utils::contains(utils::lowercase(f), search_word))
                {
                    auto it = graphs_map.find(f);
                    auto sources = Output::Library::possible_sources_for(f);
                    const auto properties = Output::properties_for(f);
                    const bool single_source = properties.centroid_only || properties.posture_only || properties.is_global;
                    
                    for(auto source : sources) {
                        auto name = utils::lowercase(source.name());
                        auto hash = modifier_translations.at(name);
                        uint32_t count = 0;
                        bool auto_generated{false};
                        
                        if(source == Output::Modifiers::HEAD
                           && pose_fields_set.contains(utils::lowercase(f)))
                        {
                            count = 1;
                            auto_generated = true;
                            
                        } else if (it != graphs_map.end()) {
                            //count = narrow_cast<uint32_t>(max(it->second.size(), 1u));
                            std::set<std::string> append;
                            for(auto a : it->second) {
                                std::string n = utils::lowercase(a);
                                if(not is_in(n, "raw", "smooth", ""))
                                    append.insert(n);
                            }
                            
                            if(append.contains(name)) {
                                count = 1;
                            }
                        }
                        
                        items.push_back(Item{
                            ._name = (std::string)f,
                            ._source = (std::string)name,
                            ._count = count,
                            ._single_source = single_source,
                            ._font = Font(0.5, count ? Style::Bold : Style::Regular, Align::Left)
                        });
                        
                        if(auto desc = get_description(f);
                           count > 0)
                        {
                            _selected_options.emplace_back();
                            _auto_variables.emplace_back(auto_generated);
                            auto &map = _selected_options.back();
                            map["name"] = (std::string)f;
                            map["sub"] = single_source || sources.size() <= 1 ? "" : (std::string)hash;
                            map["subname"] = (std::string)name;
                            map["count"] = count;
                            map["auto"] = auto_generated;
                            map["doc_tooltip"] = desc
                                ? desc->long_description
                                : "<gray><sym>❮</sym><i>missing docs</i><sym>❯</sym></gray>";
                            map["doc"] = desc
                                ? ((auto_generated ? "<gray>[auto] " : "") + desc->short_description + (auto_generated ? "</gray>" : ""))
                                : "<gray><sym>❮</sym><i>missing docs</i><sym>❯</sym></gray>";
                            
                            if(_selected_variables.size() <= j) {
                                _selected_variables.emplace_back(new Variable([j, this](const VarProps&) -> sprite::Map& {
                                    return _selected_options.at(j);
                                }));
                            }
                            
                            ++j;
                            
                        } else {
                            _filtered_options.emplace_back();
                            auto &map = _filtered_options.back();
                            map["name"] = (std::string)f;
                            map["sub"] = single_source || sources.size() <= 1 ? "" : (std::string)hash;
                            map["subname"] = (std::string)name;
                            map["count"] = count;
                            map["auto"] = auto_generated;
                            map["doc_tooltip"] = desc
                                ? desc->long_description
                                : "<gray><sym>❮</sym><i>missing docs</i><sym>❯</sym></gray>";
                            map["doc"] = desc
                                ? ((auto_generated ? "<gray>[auto] " : "") + desc->short_description + (auto_generated ? "</gray>" : ""))
                                : "<gray><sym>❮</sym><i>missing docs</i><sym>❯</sym></gray>";
                            
                            if(_filtered_variables.size() <= i) {
                                _filtered_variables.emplace_back(new Variable([i, this](const VarProps&) -> sprite::Map& {
                                    return _filtered_options.at(i);
                                }));
                            }
                            
                            ++i;
                        }
                    }
                }
            }
            
            if(i < _filtered_variables.size()) {
                _filtered_variables.resize(i);
            }
            if(j < _selected_variables.size()) {
                _selected_variables.resize(j);
            }

            //export_options.set_items(items);
            //Print("Filtering for: ",search.text());
        }
        
        _gui.update(base, &_layout);
    }
};

DrawExportOptions::DrawExportOptions()
    : _data(new Data)
{
    
}

DrawExportOptions::~DrawExportOptions() {
    delete _data;
}

void DrawExportOptions::draw(DrawStructure &base, TrackingState* state) {
    _data->draw(base, state);
}


}
