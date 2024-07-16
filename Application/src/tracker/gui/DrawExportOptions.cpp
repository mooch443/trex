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

struct DrawExportOptions::Data {
    struct Item {
        std::string _name;
        uint32_t _count = 0;
        Font _font = Font(0.5, Align::Left);
        Color _color = White;
        std::set<std::string> _sources;

        Font font() const {
            return _font;
        }

        Color color() const {
            return _color;
        }

        std::string tooltip() const {
            std::string append;
            for (auto s : _sources) {
                s = utils::lowercase(s);
                if (utils::endsWith(s, "centroid")) {
                    append.push_back(s[0]);
                    append += "centroid";
                }
                else
                    append += s;
            }
            return append;
        }
        operator std::string() const {
            return _name + (_count ? " (" + Meta::toStr(_count) + ")" : "");
        }
        bool operator!=(const Item& other) const{
            return _name != other._name || _font != other._font || _count != other._count;
        }
    };
    
    PlaceinLayout _layout;
    Entangled parent;
    ScrollableList<Item> export_options;
    DynamicGUI _gui;
    
    std::vector<sprite::Map> _filtered_options;
    std::vector<std::shared_ptr<dyn::VarBase_t>> _filtered_variables;
    
    std::string _search_text;
    
    Data()
        :
            _layout(),
            parent(Box(100, 100, 500, 550)),
            export_options(Box(Vec2(5, 15), Size2(parent.width() - 10, 30)), ItemFont_t(0.5, Align::Left))
    {
        _layout.set(Box(0, 0, 200, parent.height()));
        parent.update([&](Entangled& e) {
            //e.advance_wrap(export_options);
            //e.advance_wrap(search);
            e.advance_wrap(_layout);
        });
        parent.set_background(Black.alpha(200), Black.alpha(240));

        export_options.on_select([&](auto idx, const std::string&) {
            auto graphs = SETTING(output_graphs).value<std::vector<std::pair<std::string, std::vector<std::string>>>>();
            auto& item = export_options.items().at(idx);
            Print("Removing ",item.value()._name);

            for (auto it = graphs.begin(); it != graphs.end(); ++it) {
                if (it->first == item.value()._name) {
                    graphs.erase(it);
                    SETTING(output_graphs) = graphs;
                    return;
                }
            }

            graphs.push_back({ item.value()._name, {} });
            SETTING(output_graphs) = graphs;
        });
        
        static bool first = true;
        
        if(first) {
            parent.set_draggable();
            first = false;
        }

    }
    
    void draw(DrawStructure& base, TrackingState* state) {
        if(not _gui) {
            _gui = DynamicGUI{
                .gui = SceneManager::getInstance().gui_task_queue(),
                .path = "export_options_layout.json",
                .graph = &base,
                .context = [&](){
                    dyn::Context context;
                    context.actions = {
                        ActionFunc("add_option", [this](const Action& action) {
                            REQUIRE_EXACTLY(1, action);
                            Print("Got action: ", action);
                            auto idx = Meta::fromStr<uint32_t>(action.parameters.front());
                            auto graphs = SETTING(output_graphs).value<std::vector<std::pair<std::string, std::vector<std::string>>>>();
                            auto& item = export_options.items().at(idx);
                            Print("Removing ",item.value()._name);

                            for (auto it = graphs.begin(); it != graphs.end(); ++it) {
                                if (it->first == item.value()._name) {
                                    graphs.erase(it);
                                    SETTING(output_graphs) = graphs;
                                    return;
                                }
                            }

                            graphs.push_back({ item.value()._name, {} });
                            SETTING(output_graphs) = graphs;
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
                            
                            WorkProgress::add_queue("Saving to "+(std::string)SETTING(output_format).value<default_config::output_format_t::Class>().name()+" ...", [state]()
                            {
                                state->_controller->export_tracks();
                            });
                        }),
                        ActionFunc("set", [](Action action) {
                            REQUIRE_EXACTLY(2, action);
                            
                            auto parm = Meta::fromStr<std::string>(action.first());
                            if(not GlobalSettings::has(parm))
                                throw InvalidArgumentException("No parameter ",parm," in global settings.");
                            
                            auto value = action.last();
                            GlobalSettings::get(parm).get().set_value_from_string(value);
                        })
                    };

                    context.variables = {
                        VarFunc("window_size", [this](const VarProps&) -> Vec2 {
                            return parent.size();
                        }),
                        VarFunc("options", [this](const VarProps&) -> decltype(_filtered_variables)& {
                            return _filtered_variables;
                        })
                    };

                    return context;
                }(),
                .base = nullptr
            };
            
            _gui.context.custom_elements["option_search"] = std::unique_ptr<CustomElement>(new CustomElement {
                .name = "option_search",
                .create = [this](LayoutContext& layout) -> Layout::Ptr {
                    derived_ptr<Textfield> search = std::make_shared<Textfield>(Box(Vec2(), Size2(parent.width() - 10, 30)));
                    Placeholder_t placeholder{ layout.get(std::string("Type to filter..."), "placeholder") };
                    search->set(placeholder);
                    search->on_text_changed([this, ptr = search.get()](){
                        _search_text = ptr->text();
                    });
                    return Layout::Ptr(search);
                },
                .update = [this](Layout::Ptr& o, const Context& context, State& state, const robin_hood::unordered_map<std::string, Pattern>& patterns) -> bool {
                    return false;
                }
            });
        }
        
        auto graphs = SETTING(output_graphs).value<std::vector<std::pair<std::string, std::vector<std::string>>>>();
        auto graphs_map = [&graphs]() {
            std::map<std::string, std::set<std::string>> result;
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
                    result.try_emplace(g.first);
                } else {
                    for (auto& e : modifiers.values()) {
                        if (wanted.contains(e))
                            result[g.first].insert(e.name());
                    }
                }
            }
            return result;
        }();
        static decltype(graphs_map) previous_graphs;
        
        base.wrap_object(parent);
        parent.set_scale(base.scale().reciprocal());
        
        static std::string previous_text;
        //
        if(previous_graphs != graphs_map || _search_text != previous_text) {
            previous_graphs = graphs_map;
            previous_text = _search_text;

            auto functions = Output::Library::functions();
            std::sort(functions.begin(), functions.end());
            
            std::vector<Item> items;
            
            //_filtered_variables.clear();
            _filtered_options.clear();
            
            size_t i = 0;
            
            for (auto& f : functions) {
                if (_search_text.empty() || utils::contains(utils::lowercase(f), utils::lowercase(_search_text))) {
                    uint32_t count = 0;
                    std::set<std::string> append;
                    auto it = graphs_map.find(f);
                    if (it != graphs_map.end()) {
                        count = narrow_cast<uint32_t>(max(it->second.size(), 1u));
                        append = it->second;
                    }
                    
                    items.push_back(Item{
                        ._name = f,
                        ._count = count,
                        ._font = Font(0.5, count ? Style::Bold : Style::Regular, Align::Left),
                        ._sources = append
                    });
                    
                    _filtered_options.emplace_back();
                    auto &map = _filtered_options.back();
                    map["name"] = f;
                    map["count"] = count;
                    map["sources"] = append;
                    
                    if(_filtered_variables.size() <= i) {
                        _filtered_variables.emplace_back(new Variable([i, this](const VarProps&) -> sprite::Map& {
                            return _filtered_options.at(i);
                        }));
                    }
                    
                    ++i;
                }
            }
            
            if(i < _filtered_variables.size()) {
                _filtered_variables.resize(i);
            }

            export_options.set_items(items);
            //Print("Filtering for: ",search.text());
        }
        
        _gui.update(&_layout);
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
