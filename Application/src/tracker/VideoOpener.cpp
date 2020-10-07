#include "VideoOpener.h"
#include <tracker/misc/default_config.h>
#include <misc/GlobalSettings.h>
#include <gui/types/Dropdown.h>
#include <pv.h>
#include <tracker/misc/Output.h>
#include <misc/default_settings.h>
#include <gui/types/StaticText.h>

namespace gui {

VideoOpener::VideoOpener() {
    _horizontal = std::make_shared<gui::HorizontalLayout>();
    _extra = std::make_shared<gui::VerticalLayout>();
    _infos = std::make_shared<gui::VerticalLayout>();
    
    _horizontal->set_policy(gui::HorizontalLayout::TOP);
    _extra->set_policy(gui::VerticalLayout::LEFT);
    _infos->set_policy(gui::VerticalLayout::LEFT);
    
    _horizontal->set_children({_infos, _extra});
    
    _settings_to_show = {
        "track_max_individuals",
        "blob_size_ranges",
        "track_threshold",
        "calculate_posture",
        "recognition_enable",
        "auto_train",
        "auto_quit",
        "output_prefix",
        "manual_matches",
        "manual_splits"
    };
    
    _output_prefix = SETTING(output_prefix).value<std::string>();
    
    _file_chooser = std::make_shared<gui::FileChooser>(SETTING(output_dir).value<file::Path>(), ".pv", [this](const file::Path& path)
    {
        if(!path.empty()) {
            auto tmp = path;
            if (tmp.has_extension() && tmp.extension() == "pv")
                tmp = tmp.remove_extension();
            SETTING(filename) = tmp;
            
            std::string str = "";
            bool first = true;
            for(auto && [key, ptr] : pointers) {
                std::string val;
                
                auto textfield = dynamic_cast<gui::Textfield*>(ptr);
                
                if(!textfield) {
                    //! assume its a checkbox:
                    auto check = dynamic_cast<gui::Checkbox*>(ptr);
                    if(check)
                        val = Meta::toStr(check->checked());
                    else {
                        auto drop = dynamic_cast<gui::Dropdown*>(ptr);
                        if(drop) {
                            auto item = drop->selected_item();
                            if(item.ID() != -1) {
                                auto name = item.search_name();
                                Debug("Selected '%S' = %S", &key, &name);
                                val = name;
                            } else
                                val = drop->text();
                            
                        } else {
                            Debug("Unknown type for field '%S'", &key);
                        }
                    }
                    
                } else {
                    val = textfield->text();
                }
                
                if(start_values[key] != val) {
                    Debug("%S = %d", &key, &val);
                    
                    if(!first)
                        str += "\n";
                    str += key + "=" + val;
                    first = false;
                }
            }
            str += "\n";
            
            if(!first)
                _result.extra_command_lines = str;
            
            if(_load_results_checkbox && _load_results_checkbox->checked()) {
                _result.load_results = true;
                _result.load_results_from = "";
            }
            
        }
        
    }, [this](auto& path){ select_file(path); }, _horizontal);
    
    _file_chooser->open();
}

void VideoOpener::select_file(const file::Path &p) {
    using namespace gui;
    using namespace file;
    
    GlobalSettings::map().dont_print("filename");
    _selected = p;
    SETTING(filename) = p;
    
    Path settings_file = pv::DataLocation::parse("settings");
    sprite::Map tmp;
    tmp.set_do_print(false);
    
    GlobalSettings::docs_map_t docs;
    default_config::get(tmp, docs, [](auto, auto){});
    
    docs.clear();
    pointers.clear();
    start_values.clear();
    
    if(settings_file.exists()) {
        try {
            GlobalSettings::load_from_string(
                default_config::deprecations(),
                tmp,
                utils::read_file(settings_file.str()),
                AccessLevelType::STARTUP,
                true);
        } catch(const cmn::illegal_syntax& e) {
            Warning("File '%S' has illegal syntax: %s", &_selected.str(), e.what());
        }
    }
    
    std::vector<Layout::Ptr> children {
        Layout::Ptr(std::make_shared<Text>("Settings", Vec2(), White, gui::Font(0.8, Style::Bold)))
    };
    for(auto &name : _settings_to_show) {
        std::string start;
        if(tmp[name].is_type<std::string>())
            start = tmp[name].value<std::string>();
        else
            start = tmp[name].get().valueString();
        
        if(tmp[name].is_type<bool>()) {
            children.push_back( Layout::Ptr(std::make_shared<Checkbox>(Vec2(), name, tmp[name].get().value<bool>(), gui::Font(0.7, Style::Bold))) );
        } else if(name == "output_prefix") {
            std::vector<std::string> folders;
            for(auto &p : _selected.remove_filename().find_files()) {
                if(p.is_folder() && p.filename() != "data" && p.filename() != "..") {
                    if(!p.find_files().empty()) {
                        folders.push_back(p.filename().to_string());
                    }
                }
            }
            
            children.push_back( Layout::Ptr(std::make_shared<Text>(name, Vec2(), White, gui::Font(0.7, Style::Bold))) );
            children.push_back( Layout::Ptr(std::make_shared<Dropdown>(Bounds(0, 0, 300, 30), folders)) );
            
        } else {
            children.push_back( Layout::Ptr(std::make_shared<Text>(name, Vec2(), White, gui::Font(0.7, Style::Bold))) );
            children.push_back( Layout::Ptr(std::make_shared<Textfield>(start, Bounds(0, 0, 300, 30))));
        }
        
        if(name == "output_prefix") {
            ((Dropdown*)children.back().get())->on_select([dropdown = ((Dropdown*)children.back().get()), this](long_t, const Dropdown::TextItem & item)
            {
                _output_prefix = item.search_name();
                dropdown->set_opened(false);
                
                _file_chooser->execute([this](){
                    SETTING(output_prefix) = _output_prefix;
                    select_file(_selected);
                });
            });
            ((Dropdown*)children.back().get())->textfield()->on_enter([dropdown = ((Dropdown*)children.back().get()), this]()
            {
                _output_prefix = dropdown->textfield()->text();
                dropdown->set_opened(false);
                
               _file_chooser->execute([this](){
                    SETTING(output_prefix) = _output_prefix;
                    select_file(_selected);
               });
            });
            
            if(_output_prefix.empty())
                ((Dropdown*)children.back().get())->select_item(-1);
            else {
                auto items = ((Dropdown*)children.back().get())->items();
                for(size_t i=0; i<items.size(); ++i) {
                    if(items.at(i).search_name() == _output_prefix) {
                        ((Dropdown*)children.back().get())->select_item(i);
                        break;
                    }
                }
                
                ((Dropdown*)children.back().get())->textfield()->set_text(_output_prefix);
            }
        }
        
        pointers[name] = children.back().get();
        start_values[name] = start;
    }
    
    _load_results_checkbox = nullptr;
    auto path = Output::TrackingResults::expected_filename();
    if(path.exists()) {
        children.push_back( Layout::Ptr(std::make_shared<Checkbox>(Vec2(), "load results", false, gui::Font(0.7, Style::Bold))) );
        _load_results_checkbox = dynamic_cast<Checkbox*>(children.back().get());
    } else
        children.push_back( Layout::Ptr(std::make_shared<Text>("No loadable results found.", Vec2(), Gray, gui::Font(0.7, Style::Bold))) );
    
    _extra->set_children(children);
    _extra->auto_size(Margin{0,0});
    _extra->update_layout();
    
    try {
        pv::File video(SETTING(filename).value<file::Path>());
        video.start_reading();
        auto text = video.get_info();
        

        gui::derived_ptr<gui::Text> info_text = std::make_shared<gui::Text>("Selected", Vec2(), gui::White, gui::Font(0.8, gui::Style::Bold));
        gui::derived_ptr<gui::StaticText> info_description = std::make_shared<gui::StaticText>(settings::htmlify(text), Vec2(), Size2(300, 600), gui::Font(0.5));
        
        _infos->set_children({
            info_text,
            info_description
        });
        
        _infos->auto_size(Margin{0, 0});
        _infos->update_layout();
        
    } catch(...) {
        Except("Caught an exception while reading info from '%S'.", &SETTING(filename).value<file::Path>().str());
    }
    
    _horizontal->auto_size(Margin{0, 0});
    _horizontal->update_layout();
    
    SETTING(filename) = file::Path();
}

}
