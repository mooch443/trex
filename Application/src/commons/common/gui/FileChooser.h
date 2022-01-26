#pragma once

#include <types.h>
#include <gui/IMGUIBase.h>
#include <gui/types/ScrollableList.h>
#include <gui/DrawStructure.h>
#include <gui/types/Button.h>
#include <gui/types/Textfield.h>
#include <gui/types/Layout.h>
#include <gui/types/Tooltip.h>
#include <gui/types/Dropdown.h>

namespace gui {

class FileChooser {
    class FileItem {
        GETTER(file::Path, path)
        
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
            return "FileChooser::Settings";
        }
    };
    
protected:
    GETTER_NCONST(std::unique_ptr<DrawStructure>, graph)
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
    GETTER(IMGUIBase, base)
    std::vector<FileItem> _names;
    std::vector<Dropdown::TextItem> _search_items;
    
    file::Path _path;
    bool _running;
    
    std::set<file::Path, std::function<bool(const file::Path&, const file::Path&)>> _files;
    file::Path _selected_file;
    GETTER(file::Path, confirmed_file)
    std::function<void(const file::Path&, std::string)> _callback, _on_select_callback;
    std::function<void(DrawStructure&)> _on_update;
    std::function<bool(file::Path)> _validity;
    std::function<void(file::Path)> _on_open;
    std::function<void(std::string)> _on_tab_change;
    std::queue<std::function<void()>> _execute;
    std::mutex _execute_mutex;
    std::map<std::string, Settings> _tabs;
    GETTER(Settings, current_tab)
    Settings _default_tab;
    
public:
    FileChooser(const file::Path& start, const std::string& extension,
                std::function<void(const file::Path&, std::string)> callback,
                std::function<void(const file::Path&, std::string)> on_select_callback = nullptr);
    virtual ~FileChooser() {}
    
    void set_tabs(const std::vector<Settings>&);
    void set_tab(std::string);
    void open();
    void execute(std::function<void()>&&);
    virtual void update_size();
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
