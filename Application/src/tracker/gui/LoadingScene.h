#pragma once
#include <gui/Scene.h>
#include <gui/types/ScrollableList.h>
#include <gui/GuiTypes.h>
#include <gui/types/Dropdown.h>
#include <misc/GlobalSettings.h>
#include <gui/DynamicGUI.h>
#include <gui/IMGUIBase.h>

namespace gui {
using namespace dyn;

class LoadingScene : public Scene {
    class FileItem {
        GETTER(file::Path, path);

    public:
        FileItem(const file::Path& path = "");

        Color base_color() const;
        Color color() const;
        operator std::string() const;
        bool operator!=(const FileItem& other) const {
            return _path != other._path;
        }
    };

public:
    struct Settings {
        enum Display {
            None = 0,
            Browser = 2
        } display;

        std::string name;
        std::string extension;
        derived_ptr<Entangled> content;

        Settings(const std::string& name = "", const std::string& extensions = "", const derived_ptr<Entangled>& content = nullptr, Display d = Display::Browser)
            : display(d), name(name), extension(extensions), content(content)
        {}

        bool is_valid_extension(const file::Path& path) const {
            return file::valid_extension(path, extension);
        }

        std::string toStr() const {
            return name;
        }

        static std::string class_name() {
            return "LoadingScene::Settings";
        }
    };

protected:
    derived_ptr<Text> _description;
    derived_ptr<StaticText> _selected_text;
    derived_ptr<ScrollableList<FileItem>> _list;
    derived_ptr<Button> _button;
    derived_ptr<Dropdown> _textfield;
    derived_ptr<VerticalLayout> _rows;
    derived_ptr<HorizontalLayout> _columns;
    derived_ptr<VerticalLayout> _overall;
    derived_ptr<HorizontalLayout> _tabs_bar;
    std::unordered_map<int, derived_ptr<Tooltip>> _tooltips;
    std::vector<Layout::Ptr> tabs_elements;
    std::vector<FileItem> _names;
    std::vector<Dropdown::TextItem> _search_items;

    file::Path _path;
    bool _running;

    std::set<file::Path, std::function<bool(const file::Path&, const file::Path&)>> _files;
    file::Path _selected_file;
    GETTER(file::Path, confirmed_file);
        std::function<void(const file::Path&, std::string)> _callback, _on_select_callback;
    std::function<void(DrawStructure&)> _on_update;
    std::function<bool(file::Path)> _validity;
    std::function<void(file::Path)> _on_open;
    std::function<void(std::string)> _on_tab_change;
    std::queue<std::function<void()>> _execute;
    std::mutex _execute_mutex;
    std::map<std::string, Settings> _tabs;
    GETTER(Settings, current_tab);
        Settings _default_tab;

    // The HorizontalLayout for the two buttons and the image
    HorizontalLayout _main_layout;

    dyn::Context context {
        dyn::VarFunc("global", [](const VarProps&) -> sprite::Map& {
            return GlobalSettings::map();
        })
    };
    dyn::State state;
    std::vector<Layout::Ptr> objects;

public:
    LoadingScene(Base& window, const file::Path& start, const std::string& extension,
        std::function<void(const file::Path&, std::string)> callback,
        std::function<void(const file::Path&, std::string)> on_select_callback)
        : Scene(window, "loading-scene", [this](auto&, DrawStructure& graph) { _draw(graph); }),
        _description(std::make_shared<Text>(Str("Please choose a file in order to continue."), Loc(10, 10), Font(0.75))),
        _columns(std::make_shared<HorizontalLayout>()),
        _overall(std::make_shared<VerticalLayout>()),
        _path(start),
        _running(true),
        _files([](const file::Path& A, const file::Path& B) -> bool {
        return (A.is_folder() && !B.is_folder()) || (A.is_folder() == B.is_folder() && A.str() < B.str()); //A.str() == ".." || (A.str() != ".." && ((A.is_folder() && !B.is_folder()) || (A.is_folder() == B.is_folder() && A.str() < B.str())));
            }),
        _callback(callback),
        _on_select_callback(on_select_callback)
    {
        auto dpi = ((const IMGUIBase*)&window)->dpi_scale();
        print(window.window_dimensions().mul(dpi));
        _default_tab.extension = extension;
        
    }

    void activate() override {
        set_tab("");

        ((IMGUIBase*)window())->set_open_files_fn([this](const std::vector<file::Path>& paths) -> bool {
            if (paths.size() != 1)
                return false;

            auto path = paths.front();
            if (!_validity || _validity(path)) //path.exists() || path.str() == "/" || path.add_extension("pv").exists())
            {
                file_selected(0, path.str());
                return true;
            }
            else {
                FormatError("Path ", path.str(), " cannot be opened.");
            }
            return false;
        });

        _columns->set_policy(HorizontalLayout::TOP);
        //_columns->set_background(Transparent, Red);
        _overall->set_policy(VerticalLayout::CENTER);
        //_overall->set_background(Transparent, Blue);

        _list = std::make_shared<ScrollableList<FileItem>>(Box(
            0,
            0,
            //_graph->width() - 
            20 - (_current_tab.content ? _current_tab.content->width() + 5 : 0),
            //_graph->height() - 
            70 - 10 - 100 - 70));

        _list->set_stays_toggled(true);
        //if(_extra)
        //    _extra->set_background(Transparent, Green);

        //auto overall_width = _list->width() + (_extra ? _extra->width() : 0);

        _button = Button::MakePtr(Str{"Open"}, Box(_list->pos() + Vec2(0, _list->height() + 40), attr::Size(100, 30)));

        _textfield = std::make_shared<Dropdown>(Box(0, 0, _list->width(), 30));
        //_textfield = std::make_shared
        _textfield->on_select([this](auto, const Dropdown::TextItem& item) {
            file::Path path;

            if (((std::string)item).empty()) {
                path = _textfield->textfield()->text();
            }
            else
                path = file::Path((std::string)item);

            if (!_validity || _validity(path))
            {
                file_selected(0, path.str());
                if (!path.is_regular())
                    _textfield->select_textfield();
            }
            else
                FormatError("Path ", path.str(), " cannot be opened.");
            });

        _textfield->on_text_changed([this](std::string str) {
            auto path = file::Path(str);
            auto file = (std::string)path.filename();

            if (path.empty() || (path == _path || ((!path.exists() || !path.is_folder()) && path.remove_filename() == _path)))
            {
                // still in the same folder
            }
            else if (utils::endsWith(str, file::Path::os_sep()) && path != _path && path.is_folder()) {
                file_selected(0, path);
            }
            });

        _rows = std::make_shared<VerticalLayout>(std::vector<Layout::Ptr>{
            _textfield, _list
        });
        //_rows->set_background(Transparent, Yellow);

        _columns->set_name("Columns");
        _rows->set_name("Rows");

        if (_current_tab.content && !_selected_file.empty())
            _columns->set_children({ _rows, _current_tab.content });
        else
            _columns->set_children({ _rows });

        _overall->set_children({ _columns });

        //update_size();

        if (!_path.exists())
            _path = ".";

        try {
            auto files = _path.find_files(_current_tab.extension);
            _files.clear();
            _files.insert(files.begin(), files.end());
            _files.insert("..");

        }
        catch (const UtilsException& ex) {
            FormatError("Cannot list folder ", _path, " (", ex.what(), ").");
        }

        update_names();

        _textfield->textfield()->set_text(_path.str());
        //_textfield->set_text(_path.str());

        //_graph->set_scale(_base.dpi_scale() * gui::interface_scale());
        _list->on_select([this](auto i, auto& path) { file_selected(i, path.path()); });

        _button->set_font(gui::Font(0.6f, Align::Center));
        _button->on_click([this](auto) {
            _running = false;
            _confirmed_file = _selected_file;
            if (_on_open)
                _on_open(_confirmed_file);
            });

        _list->set(ItemFont_t(0.6f, gui::Align::Left));

        //update_size();
    }

    void deactivate() override {
        // Logic to clear or save state if needed
    }

    void _draw(DrawStructure& graph) {
        using namespace gui;
        tf::show();

        {
            std::lock_guard<std::mutex> guard(_execute_mutex);
            auto N = _execute.size();
            while (!_execute.empty()) {
                _execute.front()();
                _execute.pop();
            }
            if (N > 0)
                update_size(graph);
        }

        _list->set_bounds(Bounds(
            0,
            0,
            graph.width() - 
            20 - (_current_tab.content ? _current_tab.content->width() + 5 : 0),
            graph.height() - 
            70 - 10 - 100 - 70));
        update_size(graph);

        if (!_list)
            return;

        //_graph->wrap_object(*_textfield);
        //_graph->wrap_object(*_list);
        graph.wrap_object(*_overall);
        if (_on_update)
            _on_update(graph);

        auto scale = graph.scale().reciprocal();
        auto dim = window()->window_dimensions().mul(scale * gui::interface_scale());
        graph.draw_log_messages(Bounds(Vec2(), dim));
        if (!_tooltips.empty()) {
            for (auto&& [ID, obj] : _tooltips)
                graph.wrap_object(*obj);
        }

        if (!_selected_file.empty()) {

        }
        if (SETTING(terminate))
            _running = false;
    }

    void set_tabs(const std::vector<Settings>&);
    void set_tab(std::string);
    void open();
    void execute(std::function<void()>&&);
    virtual void update_size(DrawStructure& graph);
    void on_update(std::function<void(DrawStructure&)>&& fn) { _on_update = std::move(fn); }
    void on_open(std::function<void(file::Path)>&& fn) { _on_open = std::move(fn); }
    void on_tab_change(std::function<void(std::string)>&& fn) { _on_tab_change = std::move(fn); }
    void set_validity_check(std::function<bool(file::Path)>&& fn) { _validity = std::move(fn); }
    void deselect();
    void set_tooltip(int ID, Drawable*, const std::string&);

private:
    void file_selected(size_t i, file::Path path);
    void update_names();
    void update_tabs();
    void change_folder(const file::Path&);
};

}
