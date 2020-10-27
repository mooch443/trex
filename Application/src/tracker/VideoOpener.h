#pragma once

#include <commons/common/commons.pc.h>
#include <gui/FileChooser.h>
#include <gui/types/Layout.h>
#include <gui/types/Checkbox.h>
#include <file/Path.h>
#include <video/VideoSource.h>

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
        gpuMat _background_image;
        cv::Mat _local;
        gpuMat _flt, _img, _mask, _diff, _alpha, _output;
        cv::Mat _accumulator, _background_copy;
        bool _set_copy_background = false;
        uint64_t _background_samples = 0;
        uint64_t _background_video_index = 0;
        
        std::mutex _frame_mutex;
        std::mutex _video_mutex;
        
        std::unique_ptr<Image> _cached_frame;
        std::atomic<bool> _terminate = false, _terminate_background = false;
        std::atomic<double> _playback_index = 0;
        Timer _video_timer;
        double _seconds_between_frames = 0;
        
        std::atomic<uint32_t> _threshold = 0;
        
        std::unique_ptr<std::thread> _update_thread;
        std::unique_ptr<std::thread> _background_thread;
        
        BufferedVideo() {}
        BufferedVideo(const file::Path& path);
        ~BufferedVideo();
        
        std::unique_ptr<Image> next();
        void open();
        Size2 size();
        
        void restart_background();
        void update_loop();
    };
    
    std::mutex _video_mutex;
    std::unique_ptr<BufferedVideo> _buffer;
    
    std::shared_ptr<FileChooser> _file_chooser;
    std::map<std::string, gui::Drawable*> pointers;
    std::map<std::string, std::string> start_values;
    
    gui::derived_ptr<gui::VerticalLayout> _extra, _infos, _raw_info, _raw_settings;
    gui::derived_ptr<gui::HorizontalLayout> _horizontal, _horizontal_raw;
    gui::derived_ptr<gui::ExternalImage> _screenshot;
    gui::derived_ptr<gui::StaticText> _raw_description;
    double _screenshot_previous_size;
    
    struct LabeledField {
        gui::derived_ptr<gui::Text> _text;
        gui::derived_ptr<gui::Textfield> _text_field;
        //gui::derived_ptr<gui::HorizontalLayout> _joint;
        
        LabeledField(const std::string& name = "")
            : _text(std::make_shared<gui::Text>(name)),
              _text_field(std::make_shared<gui::Textfield>("", Bounds(0, 0, 400, 33)))
              //_joint(std::make_shared<gui::HorizontalLayout>(std::vector<Layout::Ptr>{_text, _text_field}))
        {
            _text->set_font(Font(0.75, Style::Bold));
            _text->set_color(White);
            _text_field->set_placeholder(name);
        }
    };
    std::map<std::string, LabeledField> _text_fields;
    
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

