#include "LoadingScene.h"

namespace cmn::gui {

void LoadingScene::set_tabs(const std::vector<Settings>& tabs) {
    _tabs.clear();
    tabs_elements.clear();

    for (auto tab : tabs) {
        if (tab.extension == "")
            tab.extension = _default_tab.extension;
        _tabs[tab.name] = tab;

        auto button = new Button(Str{tab.name}, attr::Size(Base::default_text_bounds(tab.name).width + 20, 40));
        button->set_fill_clr(Color(100, 100, 100, 255));
        button->set_toggleable(true);
        button->on_click([this, button](auto) {
            if (button->toggled()) {
                set_tab(button->txt());
            }
            });
        auto ptr = std::shared_ptr<Drawable>(button);
        tabs_elements.push_back(ptr);
    }

    if (_tabs.size() > 1) {
        _tabs_bar = std::make_shared<HorizontalLayout>(tabs_elements);
    }
    else {
        _tabs_bar = nullptr;
    }

    std::vector<Layout::Ptr> childs;
    if (_tabs_bar)
        childs.push_back(_tabs_bar);

    childs.push_back(_columns);

    if (_selected_text && _button) {
        childs.push_back(_selected_text);
        childs.push_back(_button);
    }

    _overall->set_children(childs);

    if (!_tabs.empty())
        set_tab(tabs.front().name);
    else
        set_tab("");
}

void LoadingScene::deselect() {
    file_selected(0, "");
}

void LoadingScene::set_tab(std::string tab) {
    if (tab != _current_tab.name) {
    }
    else
        return;

    if (tab.empty()) {
        _current_tab = _default_tab;

        if (_on_tab_change)
            _on_tab_change(_current_tab.name);
        deselect();

    }
    else if (!_tabs.count(tab)) {
        auto str = Meta::toStr(_tabs);
        FormatExcept("FileChooser ", str, " does not contain tab ", tab, ".");
    }
    else {
        _current_tab = _tabs.at(tab);
        if (_on_tab_change)
            _on_tab_change(_current_tab.name);
        deselect();
    }

    for (auto& ptr : tabs_elements) {
        if (static_cast<Button*>(ptr.get())->txt() != tab) {
            static_cast<Button*>(ptr.get())->set_toggle(false);
            static_cast<Button*>(ptr.get())->set_clickable(true);
        }
        else {
            static_cast<Button*>(ptr.get())->set_toggle(true);
            static_cast<Button*>(ptr.get())->set_clickable(false);
        }
    }

    change_folder(_path);
    if (!_selected_file.empty())
        file_selected(0, _selected_file);

    if (_current_tab.content) {
        _current_tab.content->auto_size(Margin{ 0,0 });
        _current_tab.content->set_name("Extra");
    }

    if (_current_tab.display == Settings::Display::None) {
        _rows->set_children({});

    }
    else {
        _rows->set_children(std::vector<Layout::Ptr>{
            _textfield, _list
        });
    }

    //update_size();
    //_graph->set_dirty(&_base);
}

void LoadingScene::update_names() {
    _names.clear();
    _search_items.clear();
    for (auto& f : _files) {
        if (f.str() == ".." || !utils::beginsWith((std::string)f.filename(), '.')) {
            _names.push_back(FileItem(f));
            _search_items.push_back(Dropdown::TextItem(f.str()));
        }
    }
    _list->set_items(_names);
    _textfield->set_items(_search_items);
}

void LoadingScene::set_tooltip(int ID, const std::shared_ptr<Drawable>& ptr, const std::string& docs)
{
    auto it = _tooltips.find(ID);
    if (!ptr) {
        if (it != _tooltips.end())
            _tooltips.erase(it);

    }
    else {
        if (it == _tooltips.end()) {
            _tooltips[ID] = std::make_shared<Tooltip>(ptr, 400);
            _tooltips[ID]->text().set_default_font(Font(0.5));
            it = _tooltips.find(ID);
        }
        else
            it->second->set_other(ptr);

        it->second->set_text(docs);
    }
}

LoadingScene::FileItem::FileItem(const file::Path& path) : _path(path)
{

}

LoadingScene::FileItem::operator std::string() const {
    return std::string(_path.filename());
}

Color LoadingScene::FileItem::base_color() const {
    return _path.is_folder() ? Color(80, 80, 80, 200) : Color(100, 100, 100, 200);
}

Color LoadingScene::FileItem::color() const {
    return _path.is_folder() ? Color(180, 255, 255, 255) : White;
}

/*void LoadingScene::open() {
    _base.loop();
    if (_callback)
        _callback(_confirmed_file, _current_tab.name);
}*/

void LoadingScene::change_folder(const file::Path& p) {
    auto org = _path;
    auto copy = _files;

    if (p.str() == "..") {
        try {
            _path = _path.remove_filename();
            auto files = _path.find_files(_current_tab.extension);
            _files.clear();
            _files.insert(files.begin(), files.end());
            _files.insert("..");

            _list->set_scroll_offset(Vec2());
            _textfield->textfield()->set_text(_path.str());

        }
        catch (const UtilsException&) {
            _path = org;
            _files = copy;
        }
        update_names();

    }
    else if (p.is_folder()) {
        try {
            _path = p;
            auto files = _path.find_files(_current_tab.extension);
            _files.clear();
            _files.insert(files.begin(), files.end());
            _files.insert("..");

            _list->set_scroll_offset(Vec2());
            _textfield->textfield()->set_text(_path.str() + file::Path::os_sep());

        }
        catch (const UtilsException&) {
            _path = org;
            _files = copy;
        }
        update_names();
    }
}

void LoadingScene::file_selected(size_t, file::Path p) {
    if (!p.empty() && (p.str() == ".." || p.is_folder())) {
        change_folder(p);

    }
    else {
        _selected_file = p;
        if (!_selected_file.empty() && _selected_file.remove_filename() != _path) {
            change_folder(_selected_file.remove_filename());
        }

        if (p.empty()) {
            _selected_file = file::Path();
            _selected_text = nullptr;
            if (_tabs_bar)
                _overall->set_children({
                    _tabs_bar,
                    _columns
                    });
            else
                _overall->set_children({
                    _columns
                    });

        }
        else {
            if (!_selected_text)
                _selected_text = std::make_shared<StaticText>(Str("Selected: " + _selected_file.str()), SizeLimit(700, 0), Font(0.6f));
            else
                _selected_text->set_txt("Selected: " + _selected_file.str());

            if (_tabs_bar)
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
        }
        _overall->update_layout();

        if (!_selected_file.empty() && _on_select_callback)
            _on_select_callback(_selected_file, _current_tab.extension);
        //update_size();
    }

    update_names();
}

void LoadingScene::update_size(DrawStructure& graph) {
    float s = graph.scale().x / gui::interface_scale();

    if (_selected_text && !_selected_file.empty()) {
        _selected_text->set_max_size(Size2(graph.width() / s, -1));
    }

    //if(_tabs_bar) _tabs_bar->auto_size();
    //if(_tabs_bar) _tabs_bar->update_layout();

    if (_current_tab.display == Settings::Display::None) {
        if (_current_tab.content) {
            _columns->set_children(std::vector<Layout::Ptr>{_current_tab.content});
        }
        else
            _columns->clear_children();

    }
    else if (_current_tab.content && !_selected_file.empty())
        _columns->set_children({ _rows, _current_tab.content });
    else
        _columns->set_children({ _rows });

    //_columns->set_background(Transparent, Purple);
    if (_current_tab.content)
        _current_tab.content->auto_size(Margin{ 0,0 });

    float left_column_height = graph.height() / s - 50 - 10 - (_selected_text && !_selected_file.empty() ? _button->height() + 85 : 0) - (_tabs_bar ? _tabs_bar->height() + 10 : 0);
    _button->set_bounds(Bounds(_list->pos() + Vec2(0, left_column_height), Size2(100, 30)));

    float left_column_width = graph.width() / s - 20
        - (_current_tab.content && _current_tab.content->width() > 20 && !_selected_file.empty() ? _current_tab.content->width() + 30 : 0) - 10;

    _list->set_bounds(Bounds(0, 0, left_column_width, left_column_height));
    _textfield->set_bounds(Bounds(0, 0, left_column_width, 30));

    /*if (_rows) _rows->auto_size();
    if(_rows) _rows->update_layout();

    _columns->auto_size();
    _columns->update_layout();*/

    _overall->auto_size();
    _overall->update_layout();
}

void LoadingScene::execute(std::function<void()>&& fn) {
    std::lock_guard<std::mutex> guard(_execute_mutex);
    _execute.push(std::move(fn));
}

} // namespace gui
