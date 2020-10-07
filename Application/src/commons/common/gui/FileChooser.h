#pragma once

#include <types.h>
#include <gui/IMGUIBase.h>
#include <gui/types/ScrollableList.h>
#include <gui/DrawStructure.h>
#include <gui/types/Button.h>
#include <gui/types/Textfield.h>
#include <gui/types/Layout.h>

namespace gui {

class FileChooser {
    class FileItem : public CustomItem {
        GETTER(file::Path, path)
        
    public:
        FileItem(const file::Path& path = "");
        
        Color base_color() const override;
        Color text_color() const override;
        operator std::string() const override;
    };
    
    DrawStructure _graph;
    derived_ptr<Text> _description;
    derived_ptr<StaticText> _selected_text;
    derived_ptr<Entangled> _extra;
    derived_ptr<ScrollableList<FileItem>> _list;
    derived_ptr <Button> _button;
    derived_ptr <Textfield> _textfield;
    derived_ptr<VerticalLayout> _rows;
    derived_ptr<HorizontalLayout> _columns;
    derived_ptr<VerticalLayout> _overall;
    IMGUIBase _base;
    std::vector<FileItem> _names;
    
    file::Path _path;
    std::string _filter;
    bool _running;
    
    std::set<file::Path, std::function<bool(const file::Path&, const file::Path&)>> _files;
    file::Path _selected_file, _confirmed_file;
    std::function<void(const file::Path&)> _callback, _on_select_callback;
    std::queue<std::function<void()>> _execute;
    std::mutex _execute_mutex;
    
public:
    FileChooser(const file::Path& start, const std::string& filter_extension, std::function<void(const file::Path&)> callback, std::function<void(const file::Path&)> on_select_callback= [](auto&){}, derived_ptr<Entangled> extra = nullptr);
    
    void open();
    void execute(std::function<void()>&&);
    void update_size();
    
private:
    void file_selected(size_t i, file::Path path);
    void update_names();
};

}
