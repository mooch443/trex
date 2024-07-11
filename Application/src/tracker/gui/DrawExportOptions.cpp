#include "DrawExportOptions.h"
#include <gui/types/Entangled.h>
#include <misc/GlobalSettings.h>
#include <tracking/OutputLibrary.h>
#include <gui/types/Button.h>
#include <gui/types/Textfield.h>
#include <gui/types/ScrollableList.h>
#include <gui/DrawStructure.h>

namespace cmn::gui {
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
    
    Entangled parent;
    Button close;
    Textfield search;
    ScrollableList<Item> export_options;
    
    Data()
        :   parent(Box(100, 100, 200, 500)),
            close(Str{"x"}, Box(Vec2(parent.width() - 3, 5), Size2(25, 25))),
            search(Box(Vec2(5, close.height() + 15), Size2(parent.width() - 10, 30))),
            export_options(Box(
               search.pos() + Vec2(0, search.height() + 10),
               Size2(search.width(), parent.height() - (search.pos().y + search.height() + 20))), ItemFont_t(0.5, Align::Left))
    {
        close.set_font(Font(0.5, Align::Center));
        search.set_placeholder("Type to search...");
        parent.update([&](Entangled& e) {
            e.advance_wrap(export_options);
            e.advance_wrap(search);
            e.advance_wrap(close);
        });
        parent.set_background(Black.alpha(50), Red.alpha(50));

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
            
            close.set_fill_clr(Red.exposure(0.5));
            close.on_click([](auto) {
                SETTING(gui_show_export_options) = false;
            });
            close.set_origin(Vec2(1, 0));
            
            first = false;
        }

    }
    
    void draw(DrawStructure& base) {
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
        if(previous_graphs != graphs_map || search.text() != previous_text) {
            previous_graphs = graphs_map;
            previous_text = search.text();

            auto functions = Output::Library::functions();
            std::vector<Item> items;

            for (auto& f : functions) {
                if (search.text().empty() || utils::contains(utils::lowercase(f), utils::lowercase(search.text()))) {
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
                }
            }

            export_options.set_items(items);
            //Print("Filtering for: ",search.text());
        }
    }
};

DrawExportOptions::DrawExportOptions()
    : _data(new Data)
{
    
}

DrawExportOptions::~DrawExportOptions() {
    delete _data;
}

void DrawExportOptions::draw(DrawStructure &base) {
    _data->draw(base);
}


}
