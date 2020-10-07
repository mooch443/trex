#pragma once

#include <commons/common/commons.pc.h>
#include <gui/FileChooser.h>
#include <gui/types/Layout.h>
#include <gui/types/Checkbox.h>
#include <file/Path.h>

namespace gui {

class VideoOpener {
public:
    struct Result {
        std::string extra_command_lines;
        std::string load_results_from;
        bool load_results;
        
        Result() : load_results(false) {}
        
    } _result;
    
    std::shared_ptr<FileChooser> _file_chooser;
    std::map<std::string, gui::Drawable*> pointers;
    std::map<std::string, std::string> start_values;
    
    gui::derived_ptr<gui::VerticalLayout> _extra, _infos;
    gui::derived_ptr<gui::HorizontalLayout> _horizontal;
    gui::Checkbox *_load_results_checkbox = nullptr;
    std::string _output_prefix;
    std::vector<std::string> _settings_to_show;
    file::Path _selected;
    
public:
    VideoOpener();
    
private:
    void select_file(const file::Path& path);
};

}

