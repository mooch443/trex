#pragma once

#include <commons/common/commons.pc.h>
#include <gui/FileChooser.h>
#include <gui/types/Layout.h>
#include <gui/types/Checkbox.h>
#include <file/Path.h>
#include <video/VideoSource.h>
#include <gui/types/Dropdown.h>
#include <gui/types/Tooltip.h>

namespace gui {

class VideoOpener {
public:
    struct Result {
        std::string extra_command_lines;
        std::string load_results_from;
        std::string cmd;
        FileChooser::Settings tab;
        file::Path selected_file;
        bool load_results;
        
        Result() : load_results(false) {}
        
    } _result;
    
    struct BufferedVideo {
        file::Path _path;
        std::unique_ptr<VideoSource> _video;
        std::unique_ptr<VideoSource> _background_video;
        std::unique_ptr<Image> _background_copy;
        std::unique_ptr<AveragingAccumulator<>> _accumulator;
        //uint64_t _background_samples = 0;
        uint64_t _background_video_index = 0;
        
        std::atomic<bool> _terminated_background_task = true;
        std::atomic<size_t> _number_samples = 0;
        
        std::mutex _frame_mutex;
        std::mutex _video_mutex;
        
        std::unique_ptr<Image> _cached_frame;
        std::atomic<bool> _terminate = false, _terminate_background = false;
        
        std::atomic<uint32_t> _threshold = 0;
        
        std::unique_ptr<std::thread> _update_thread;
        std::unique_ptr<std::thread> _background_thread;
        
        BufferedVideo() {}
        BufferedVideo(const file::Path& path);
        ~BufferedVideo();
        
        std::unique_ptr<Image> next();
        void open(std::function<void(const bool)>&& callback);
        Size2 size();
        
        void restart_background();
    };
    
    std::mutex _video_mutex;
    std::unique_ptr<BufferedVideo> _buffer;
    std::queue<std::unique_ptr<BufferedVideo>> _stale_buffers;
    
    std::shared_ptr<FileChooser> _file_chooser;
    std::map<std::string, gui::Drawable*> pointers;
    std::map<std::string, std::string> start_values;
    
    gui::derived_ptr<gui::VerticalLayout> _extra, _infos, _raw_info, _raw_settings;
    gui::derived_ptr<gui::HorizontalLayout> _horizontal, _horizontal_raw;
    gui::derived_ptr<gui::ExternalImage> _screenshot;
    gui::derived_ptr<gui::Text> _loading_text;
    gui::derived_ptr<gui::StaticText> _raw_description;
    std::unique_ptr<std::thread> _stale_thread;
    std::condition_variable _stale_variable;
    
    double _screenshot_previous_size;
    
    struct LabeledField {
        gui::derived_ptr<gui::Text> _text;
        std::string _docs;
        //gui::derived_ptr<gui::HorizontalLayout> _joint;
        
        LabeledField(const std::string& name = "")
            : _text(std::make_shared<gui::Text>(name))
              //_joint(std::make_shared<gui::HorizontalLayout>(std::vector<Layout::Ptr>{_text, _text_field}))
        {
            _text->set_font(Font(0.6));
            _text->set_color(White);
        }
        
        virtual ~LabeledField() {}
        
        virtual void add_to(std::vector<Layout::Ptr>& v) {
            v.push_back(_text);
        }
        virtual void update() {}
        virtual Drawable* representative() { return _text.get(); }
    };
    struct LabeledTextField : public LabeledField {
        gui::derived_ptr<gui::Textfield> _text_field;
        sprite::Reference _ref;
        LabeledTextField(const std::string& name = "");
        void add_to(std::vector<Layout::Ptr>& v) override {
            LabeledField::add_to(v);
            v.push_back(_text_field);
        }
        void update() override;
        Drawable* representative() override { return _text_field.get(); }
    };
    struct LabeledDropDown : public LabeledField {
        gui::derived_ptr<gui::Dropdown> _dropdown;
        sprite::Reference _ref;
        LabeledDropDown(const std::string& name = "");
        void add_to(std::vector<Layout::Ptr>& v) override {
            LabeledField::add_to(v);
            v.push_back(_dropdown);
        }
        void update() override;
        Drawable* representative() override { return _dropdown.get(); }
    };
    struct LabeledCheckbox : public LabeledField {
        gui::derived_ptr<gui::Checkbox> _checkbox;
        sprite::Reference _ref;
        LabeledCheckbox(const std::string& name = "");
        void add_to(std::vector<Layout::Ptr>& v) override {
            LabeledField::add_to(v);
            v.push_back(_checkbox);
        }
        void update() override;
        Drawable* representative() override { return _checkbox.get(); }
    };
    std::map<std::string, std::unique_ptr<LabeledField>> _text_fields;
    
    gui::Checkbox *_load_results_checkbox = nullptr;
    std::string _output_prefix;
    std::vector<std::string> _settings_to_show;
    file::Path _selected;
    
public:
    VideoOpener();
    ~VideoOpener();
    
private:
    void select_file(const file::Path& path);
    
    std::mutex _stale_mutex;
    void move_to_stale(std::unique_ptr<BufferedVideo>&&);
};

}

