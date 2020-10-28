#include "FileChooser.h"
#include <misc/GlobalSettings.h>
#include <gui/types/StaticText.h>

namespace gui {

FileChooser::FileChooser(const file::Path& start, const std::string& extension,
                         std::function<void(const file::Path&, std::string)> callback,
                         std::function<void(const file::Path&, std::string)> on_select_callback)
:
    _graph(std::make_unique<DrawStructure>(1300, 820)),
    _description(std::make_shared<Text>("Please choose a file in order to continue.", Vec2(10, 10), White, Font(0.75))),
    _columns(std::make_shared<HorizontalLayout>()),
    _overall(std::make_shared<VerticalLayout>()),
    _base("Choose file", *_graph, [this](){
        using namespace gui;
        tf::show();
        
        {
            std::lock_guard<std::mutex> guard(_execute_mutex);
            while(!_execute.empty()) {
                _execute.front()();
                _execute.pop();
            }
        }
        
        if(!_list)
            return false;
        
        //_graph->wrap_object(*_textfield);
        //_graph->wrap_object(*_list);
        _graph->wrap_object(*_overall);
        if(_on_update)
            _on_update(*_graph);
        
        if(!_selected_file.empty()) {

            //auto text = _graph->text("Selected: "+_selected_file.str(), _list->bounds().pos() + Vec2(0, _list->bounds().height + 10), White, Font(0.6));
            //_button->set_pos(text->pos() + Vec2(0, text->height() + 10));
            //_graph->wrap_object(*_button);
        }
        if (SETTING(terminate))
            _running = false;

        return _running;
    }, [this](gui::Event e) {
        // --
        if(e.type == gui::WINDOW_RESIZED) {
            using namespace gui;
            {
                std::lock_guard guard(_graph->lock());
                Size2 size(e.size.width, e.size.height);
                auto scale = _base.dpi_scale();
                _graph->set_size(size.div(scale));
            }
            update_size();
        }
    }),
    _path(start),
    _running(true),
    _files([](const file::Path& A, const file::Path& B) -> bool {
        return (A.is_folder() && !B.is_folder()) || (A.is_folder() == B.is_folder() && A.str() < B.str()); //A.str() == ".." || (A.str() != ".." && ((A.is_folder() && !B.is_folder()) || (A.is_folder() == B.is_folder() && A.str() < B.str())));
    }),
    _callback(callback),
    _on_select_callback(on_select_callback)
{
    _default_tab.extension = extension;
    set_tab("");
    
    _base.set_open_files_fn([this](const std::vector<file::Path>& paths) -> bool{
        if(paths.size() != 1)
            return false;
        
        auto path = paths.front();
        if(!_validity || _validity(path)) //path.exists() || path.str() == "/" || path.add_extension("pv").exists())
        {
            file_selected(0, path.str());
            return true;
        } else {
            Error("Path '%S' cannot be opened.", &path.str());
        }
        return false;
    });
    
    _columns->set_policy(HorizontalLayout::TOP);
    //_columns->set_background(Transparent, Red);
    _overall->set_policy(VerticalLayout::CENTER);
    //_overall->set_background(Transparent, Blue);
    
    _list = std::make_shared<ScrollableList<FileItem>>(Bounds(0, 0, _graph->width() - 20 - (_current_tab.content ? _current_tab.content->width() + 5 : 0), _graph->height() - 70 - 10 - 70));
    _list->set_stays_toggled(true);
    //if(_extra)
    //    _extra->set_background(Transparent, Green);
    
    //auto overall_width = _list->width() + (_extra ? _extra->width() : 0);
    
    _button = std::make_shared<Button>("Open", Bounds(_list->pos() + Vec2(0, _list->height() + 40), Size2(100, 30)));
    
    _textfield = std::make_shared<Textfield>("", Bounds(0, 0, _list->width(), 30));
    
    _textfield->on_enter([&](){
        auto path = file::Path(_textfield->text());
        
        //if(path.exists() || path.str() == "/" || path.add_extension("pv").exists())
        if(!_validity || _validity(path))
        {
            file_selected(0, path.str());
        } else {
            Error("Path '%S' cannot be opened.", &path.str());
        }
        
    });
    
    _rows = std::make_shared<VerticalLayout>(std::vector<Layout::Ptr>{
        _description, _textfield, _list
    });
    //_rows->set_background(Transparent, Yellow);
    
    _columns->set_name("Columns");
    _rows->set_name("Rows");
    
    if(_current_tab.content)
        _columns->set_children({_rows, _current_tab.content});
    else
        _columns->set_children({_rows});
    
    _overall->set_children({_columns});
    
    update_size();
    
    if(!_path.exists())
        _path = ".";
    
    try {
        auto files = _path.find_files(_current_tab.extension);
        _files.clear();
        _files.insert(files.begin(), files.end());
        _files.insert("..");
        
    } catch(const UtilsException& ex) {
        Error("Cannot list folder '%S'.", &_path);
    }
    
    update_names();
    _textfield->set_text(_path.str());
    
    _graph->set_scale(_base.dpi_scale() * gui::interface_scale());
    _list->on_select([this](auto i, auto&path){ file_selected(i, path.path()); });
    
    _button->set_font(gui::Font(0.6, Align::Center));
    _button->on_click([this](auto){
        _running = false;
        _confirmed_file = _selected_file;
        if(_on_open)
            _on_open(_confirmed_file);
    });
    
    _list->set_font(gui::Font(0.6, gui::Align::Left));
    
    _base.platform()->set_icons({
        "gfx/"+SETTING(app_name).value<std::string>()+"Icon16.png",
        "gfx/"+SETTING(app_name).value<std::string>()+"Icon32.png",
        "gfx/"+SETTING(app_name).value<std::string>()+"Icon64.png"
    });
}

void FileChooser::set_on_update(std::function<void (DrawStructure &)> && fn) {
    _on_update = std::move(fn);
}

void FileChooser::set_tabs(const std::vector<Settings>& tabs) {
    _tabs.clear();
    tabs_elements.clear();
    
    for(auto tab : tabs) {
        if(tab.extension == "")
            tab.extension = _default_tab.extension;
        _tabs[tab.name] = tab;
        
        auto button = new Button(tab.name, Bounds(0, 0, Base::default_text_bounds(tab.name).width + 20, 33));
        button->set_toggleable(true);
        button->on_click([this, button](auto e){
            if(button->toggled()) {
                set_tab(button->txt());
            }
        });
        auto ptr = std::shared_ptr<Drawable>(button);
        tabs_elements.push_back(ptr);
    }
    
    if(_tabs.size() > 1) {
        _tabs_bar = std::make_shared<HorizontalLayout>(tabs_elements);
    } else {
        _tabs_bar = nullptr;
    }
    
    std::vector<Layout::Ptr> childs;
    if(_tabs_bar)
        childs.push_back(_tabs_bar);
    
    childs.push_back(_columns);
    
    if(_selected_text && _button) {
        childs.push_back(_selected_text);
        childs.push_back(_button);
    }
    
    _overall->set_children(childs);
    
    if(!_tabs.empty())
        set_tab(tabs.front().name);
    else
        set_tab("");
}

void FileChooser::set_tab(std::string tab) {
    if(tab != _current_tab.name) {
        
    } else
        return;
    
    if(tab.empty()) {
        _current_tab = _default_tab;
    } else if(!_tabs.count(tab)) {
        auto str = Meta::toStr(_tabs);
        Except("FileChooser %S does not contain tab '%S'.", &str, &tab);
    } else {
        _current_tab = _tabs.at(tab);
    }
    
    for(auto &ptr : tabs_elements) {
        if(static_cast<Button*>(ptr.get())->txt() != tab) {
            static_cast<Button*>(ptr.get())->set_toggle(false);
            static_cast<Button*>(ptr.get())->set_clickable(true);
        } else
            static_cast<Button*>(ptr.get())->set_clickable(false);
    }
    
    change_folder(_path);
    if(!_selected_file.empty())
        file_selected(0, _selected_file);
    
    if(_current_tab.content) {
        _current_tab.content->auto_size(Margin{0,0});
        _current_tab.content->set_name("Extra");
    }
    
    _graph->set_dirty(&_base);
}

void FileChooser::update_names() {
    _names.clear();
    for(auto &f : _files) {
        if(f.str() == ".." || !utils::beginsWith(f.filename().to_string(), '.'))
            _names.push_back(FileItem(f));
    }
    _list->set_items(_names);
}

FileChooser::FileItem::FileItem(const file::Path& path) : _path(path)
{
    
}

FileChooser::FileItem::operator std::string() const {
    return _path.filename().to_string();
}

Color FileChooser::FileItem::base_color() const {
    return _path.is_folder() ? Color(80, 80, 80, 200) : Color(100, 100, 100, 200);
}

Color FileChooser::FileItem::text_color() const {
    return _path.is_folder() ? Color(180, 255, 255, 255) : White;
}

void FileChooser::open() {
    _base.loop();
    if(_callback)
        _callback(_confirmed_file, _current_tab.name);
}

void FileChooser::change_folder(const file::Path& p) {
    auto org = _path;
    auto copy = _files;
    
    if(p.str() == "..") {
        try {
            _path = _path.remove_filename();
            auto files = _path.find_files(_current_tab.extension);
            _files.clear();
            _files.insert(files.begin(), files.end());
            _files.insert("..");
            
            _list->set_scroll_offset(Vec2());
            _textfield->set_text(_path.str());
            
        } catch(const UtilsException&e) {
            _path = org;
            _files = copy;
        }
        update_names();
        
    } else if(p.is_folder()) {
        try {
            _path = p;
            auto files = _path.find_files(_current_tab.extension);
            _files.clear();
            _files.insert(files.begin(), files.end());
            _files.insert("..");
            
            _list->set_scroll_offset(Vec2());
            _textfield->set_text(_path.str());
            
        } catch(const UtilsException&e) {
            _path = org;
            _files = copy;
        }
        update_names();
    }
}

void FileChooser::file_selected(size_t, file::Path p) {
    if(p.str() == ".." || p.is_folder()) {
        change_folder(p);
    } else {
        _selected_file = p;
        if(!_selected_text)
            _selected_text = std::make_shared<StaticText>("Selected: "+_selected_file.str(), Vec2(), Vec2(700, 0), Font(0.6));
        else
            _selected_text->set_txt("Selected: "+_selected_file.str());
        
        if(_tabs_bar)
            _overall->set_children({
                _tabs_bar,
                _columns,
                _selected_text,
                _button
            });
        else
            _overall->set_children({
                _columns,
                _selected_text,
                _button
            });
        _overall->update_layout();
        //update_size();
        
        if(_on_select_callback)
            _on_select_callback(_selected_file, _current_tab.extension);
        update_size();
    }
    
    update_names();
}

void FileChooser::update_size() {
    float left_column_width = _graph->width() - 20 - (_current_tab.content && _current_tab.content->width() > 20 ? _current_tab.content->width() + 10 : 0) - 10;
    if(_selected_text) {
        _selected_text->set_max_size(Size2(_graph->width() - 20));
    }
    
    if(_tabs_bar) _tabs_bar->auto_size(Margin{0,0});
    if(_tabs_bar) _tabs_bar->update_layout();
    
    float left_column_height = _graph->height() - 70 - 10 - (!_selected_file.empty() ? _button->height() + 10 : 0) - (_tabs_bar ? _tabs_bar->height() + 10 : 0) - (_selected_text && !_selected_file.empty() ? _selected_text->height() : 0);
    _list->set_bounds(Bounds(0, 0, left_column_width, left_column_height - 70));
    
    _textfield->set_bounds(Bounds(0, 0, left_column_width, 30));
    _button->set_bounds(Bounds(_list->pos() + Vec2(0, left_column_height), Size2(100, 30)));
    
    if(_rows) _rows->auto_size(Margin{0,0});
    if(_rows) _rows->update_layout();
    
    if(_current_tab.content)
        _columns->set_children({_rows, _current_tab.content});
    else
        _columns->set_children({_rows});
    
    if(_current_tab.content) _current_tab.content->auto_size(Margin{0,0});
    _columns->auto_size(Margin{0,0});
    _columns->update_layout();
    
    _overall->auto_size(Margin{0,0});
    _overall->update_layout();
}

void FileChooser::execute(std::function<void()>&& fn) {
    std::lock_guard<std::mutex> guard(_execute_mutex);
    _execute.push(std::move(fn));
}

}
