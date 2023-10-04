#include "VideoOpener.h"
#include <tracker/misc/default_config.h>
#include <misc/GlobalSettings.h>
#include <gui/types/Dropdown.h>
#include <pv.h>
#include <file/DataLocation.h>
#include <tracker/misc/Output.h>
#include <misc/default_settings.h>
#include <gui/types/StaticText.h>
#include <processing/RawProcessing.h>
#include <grabber/misc/default_config.h>
#include <opencv2/core/ocl.hpp>
#include <video/AveragingAccumulator.h>

#define TEMP_SETTING(NAME) (gui::temp_settings[#NAME])

namespace gui {
GlobalSettings::docs_map_t temp_docs;
sprite::Map temp_settings;
constexpr double video_chooser_column_width = 300;

VideoOpener::CustomFileChooser::CustomFileChooser(
        const file::Path& start,
        const std::string& extension,
        std::function<void (const file::Path &, std::string)> callback,
        std::function<void(const file::Path&, std::string)> on_select_callback)
    : FileChooser(start, extension, callback, on_select_callback)
{ }

void VideoOpener::CustomFileChooser::update_size() {
    FileChooser::update_size();
    
    float s = _graph->scale().x / gui::interface_scale();
    auto column = Size2(
        _graph->width() * 0.9 - 50,
        _graph->height() * 0.7 - (_selected_text ? _selected_text->height() + _button->height() + 10 : 0))
       .div(s);
    
    _update(column.width, column.height);
    
    FileChooser::update_size();
}

void VideoOpener::CustomFileChooser::set_update(std::function<void(float,float)> fn) {
    _update = fn;
}

VideoOpener::LabeledCheckbox::LabeledCheckbox(const std::string& name)
    : LabeledField(name),
      _checkbox(std::make_shared<gui::Checkbox>(Str(name))),
      _ref(gui::temp_settings[name])
{
    _docs = gui::temp_docs[name];
    
    _checkbox->set_checked(_ref.value<bool>());
    _checkbox->set_font(Font(0.7f));

    _checkbox->on_change([this](){
        try {
            _ref.get() = _checkbox->checked();

        } catch(...) {}
    });
}

void VideoOpener::LabeledCheckbox::update() {
    _checkbox->set_checked(_ref.value<bool>());
}

VideoOpener::LabeledTextField::LabeledTextField(const std::string& name)
    : LabeledField(name),
      _text_field(std::make_shared<gui::Textfield>(Box(0, 0, video_chooser_column_width, 28))),
      _ref(gui::temp_settings[name])
{
    _text_field->set_placeholder(name);
    _text_field->set_font(Font(0.7f));
    
    _docs = gui::temp_docs[name];

    update();
    _text_field->on_text_changed([this](){
        try {
            _ref.get().set_value_from_string(_text_field->text());

        } catch(...) {}
    });
}

void VideoOpener::LabeledTextField::update() {
    auto str = _ref.get().valueString();
    if(str.length() >= 2 && str.front() == '"' && str.back() == '"') {
        str = str.substr(1,str.length()-2);
    }
    _text_field->set_text(str);
}

VideoOpener::LabeledDropDown::LabeledDropDown(const std::string& name)
    : LabeledField(name),
      _dropdown(std::make_shared<gui::Dropdown>(Box(0, 0, video_chooser_column_width, 28))),
      _ref(gui::temp_settings[name])
{
    _docs = gui::temp_docs[name];

    _dropdown->textfield()->set_font(Font(0.7f));
    assert(_ref.get().is_enum());
    std::vector<Dropdown::TextItem> items;
    int index = 0;
    for(auto &name : _ref.get().enum_values()()) {
        items.push_back(Dropdown::TextItem(name, index++));
    }
    _dropdown->set_items(items);
    _dropdown->select_item(Dropdown::RawIndex{narrow_cast<long>(_ref.get().enum_index()())});
    _dropdown->textfield()->set_text(_ref.get().valueString());
    
    _dropdown->on_select([this](auto index, auto) {
        if(not index.valid())
            return;
        
        try {
            _ref.get().set_value_from_string(_ref.get().enum_values()().at((size_t)index.value));
        } catch(...) {}
        
        _dropdown->set_opened(false);
    });
}

void VideoOpener::LabeledDropDown::update() {
    _dropdown->select_item(Dropdown::RawIndex{narrow_cast<long>(_ref.get().enum_index()())});
}

VideoOpener::VideoOpener()
    : _accumulate_video_frames_thread(nullptr),
      _accumulate_frames_done(true),
      _end_frames_thread(true)
{
    grab::default_config::get(temp_settings, temp_docs, nullptr);
    //::default_config::get(GlobalSettings::map(), temp_docs, nullptr);
    
    _stale_thread = std::make_unique<std::thread>([this](){
        set_thread_name("VideoOpener::stale_thread");
        std::unique_lock guard(_stale_mutex);
        bool quit = false;
        
        while(!quit) {
            _stale_variable.wait(guard);
            
            size_t i=0;
            while(!_stale_buffers.empty()) {
                auto ptr = std::move(_stale_buffers.front());
                _stale_buffers.pop();
                
                if(ptr == nullptr) {
                    quit = true;
                    continue;
                }
                
                guard.unlock();
                try {
                    auto path = ptr->_path;
#ifndef NDEBUG
                    print("Removing stale buffer ",path.str(),"...");
#endif
                    ptr = nullptr;
#ifndef NDEBUG
                    print("Removed stale buffer ", path.str(),".");
#endif
                } catch(const std::exception& e) {
                    FormatExcept("Exception while freeing stale buffer '", e.what(),"'.");
                }
                guard.lock();
                
                ++i;
            }
      
#ifndef NDEBUG
            if(i)
                print("Removed ", i," stale buffers");
#endif
        }
        
#ifndef NDEBUG
        print("Quit stale thread.");
#endif
    });
    
    _horizontal = std::make_shared<gui::HorizontalLayout>();
    _extra = std::make_shared<gui::VerticalLayout>();
    _infos = std::make_shared<gui::VerticalLayout>(Box(10, 10, 10, 10));
    _infos->set_background(DarkCyan.alpha(25), DarkCyan.alpha(125));
    _horizontal->set_policy(gui::HorizontalLayout::TOP);
    _extra->set_policy(gui::VerticalLayout::LEFT);
    _infos->set_policy(gui::VerticalLayout::LEFT);
    
    _horizontal->set_children({_infos, _extra});
    
    TEMP_SETTING(output_name) = file::Path("video");
    TEMP_SETTING(output_dir) = SETTING(output_dir).value<file::Path>();
    gui::temp_docs["output_name"] = "Basename of the converted video in PV-format.";
    TEMP_SETTING(cmd_parameters) = std::string("-reset_average");
    gui::temp_docs["cmd_parameters"] = "Additional command-line parameters for TGrabs.";
    
    _horizontal_raw = std::make_shared<gui::HorizontalLayout>();
    _horizontal_raw->set_clickable(true);
    _recording_panel = std::make_shared<gui::HorizontalLayout>();
    _recording_panel->set_clickable(true);
    _camera = std::make_shared<gui::ExternalImage>(Image::Make(32, 32, 4));
    _raw_settings = std::make_shared<gui::VerticalLayout>();
    _raw_info = std::make_shared<gui::VerticalLayout>(Box(10,10,10,10));
    _raw_info->set_policy(gui::VerticalLayout::LEFT);
    _screenshot = std::make_shared<gui::ExternalImage>();
    _text_fields.clear();
    
    _name = "VideoOpener"+Meta::toStr(uint64_t(this));
    _callback = _name.c_str();
    gui::temp_settings.register_callback(_callback, [this](sprite::Map::Signal signal, auto&map, auto&key, auto&value){
        if(signal == sprite::Map::Signal::EXIT) {
            map.unregister_callback(_callback);
            _callback = nullptr;
            return;
        }
        
        if(key == "threshold") {
            if(_buffer)
                _buffer->_threshold = value.template value<int>();
            
        } else if(is_in(key, "average_samples", "averaging_method")) {
            if(_buffer)
                _buffer->restart_background();
        }
    });
    
    _recording_panel->set_children(std::vector<Layout::Ptr>{
        Layout::Ptr(std::make_shared<Text>(Str("Camera"), gui::Font(0.9f, Style::Bold))),
        _camera
    });
    _recording_panel->set_name("RecordingPanel");

    _text_fields["output_name"] = std::make_unique<LabeledTextField>("output_name");
    _text_fields["threshold"] = std::make_unique<LabeledTextField>("threshold");
    _text_fields["average_samples"] = std::make_unique<LabeledTextField>("average_samples");
    _text_fields["averaging_method"] = std::make_unique<LabeledDropDown>("averaging_method");
    _text_fields["meta_real_width"] = std::make_unique<LabeledTextField>("meta_real_width");
    _text_fields["cmd_parameters"] = std::make_unique<LabeledTextField>("cmd_parameters");
    
    std::vector<Layout::Ptr> objects{
        Layout::Ptr(std::make_shared<Text>(Str("Settings"), White, gui::Font(0.9f, Style::Bold)))
    };
    for(auto &[key, ptr] : _text_fields) {
        ptr->add_to(objects);
    }
    
    _raw_settings->set_children(objects);
    
    _loading_text = std::make_shared<gui::Text>("generating average", Loc(100,0), Cyan, gui::Font(0.6f));
    
    _raw_description = std::make_shared<gui::StaticText>("Info", SizeLimit(video_chooser_column_width, -1), Font(0.5f));
    _raw_description->set_background(Transparent, Transparent);
    _raw_info->set_children({
        Layout::Ptr(std::make_shared<Text>(Str("Preview"),TextClr( White), gui::Font(0.9f, Style::Bold))),
        _screenshot,
        _raw_description
    });
    _raw_info->set_background(DarkCyan.alpha(25), DarkCyan.alpha(125));
    _horizontal_raw->set_children({_raw_settings, _raw_info});
    _horizontal_raw->set_policy(gui::HorizontalLayout::TOP);
    
    _settings_to_show = {
        "track_max_individuals",
        "blob_size_ranges",
        "track_threshold",
        "calculate_posture",
        "auto_train",
        "auto_quit",
        "output_prefix",
        "manual_matches",
        "manual_splits"
    };
    
    _output_prefix = SETTING(output_prefix).value<std::string>();
    
    _file_chooser = std::make_shared<CustomFileChooser>(
        SETTING(output_dir).value<file::Path>(),
        "pv",
        [this](const file::Path& path, std::string) mutable
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
                                print("Selected ",key," = ",name);
                                val = name;
                            } else
                                val = drop->text();
                            
                        } else {
                            print("Unknown type for field ",key);
                        }
                    }
                    
                } else {
                    val = textfield->text();
                }
                
                if(start_values[key] != val) {
                    print(key," = ",val);
                    
                    if(!first)
                        str += "\n";
                    str += key + "=" + val;
                    first = false;
                }
            }
            str += "\n";
            
            if(!first)
                _result.extra_command_lines = str;
            _result.tab = _file_chooser->current_tab();
            _result.tab.content = nullptr;
            _result.selected_file = _file_chooser->confirmed_file();
            
            if(_result.tab.extension == "pv") {
                // PV file, no need to add cmd
            } else if(!_result.selected_file.empty()) {
                auto add = TEMP_SETTING(cmd_parameters).value<std::string>();
                _result.cmd = std::string()
                    + "-d \""+TEMP_SETTING(output_dir).value<file::Path>().str()+"\""
                    +" -i \"" + path.str() + "\""
                    +" -o \""+TEMP_SETTING(output_name).value<file::Path>().str()+"\""
                    +" -threshold "+TEMP_SETTING(threshold).get().valueString()
                    +" -average_samples "+TEMP_SETTING(average_samples).get().valueString()
                    +" -averaging_method "+TEMP_SETTING(averaging_method).get().valueString()
                    +" -meta_real_width "+TEMP_SETTING(meta_real_width).get().valueString()
                    +(add.empty() ? "" : " ")+add;
            }
            
            if(_load_results_checkbox && _load_results_checkbox->checked()) {
                _result.load_results = true;
                _result.load_results_from = "";
            }
            
        }
        
    }, [this](auto& path, std::string) {
        select_file(path);
    });
    
    _file_chooser->set_update([this](float w, float h) {
        if(w < 50)
            w = 50;
        
        if(_file_chooser->current_tab().extension == "pv") {
            if(_info_description) {
                _info_description->set_max_size(Size2(w * 0.5, h));
                h -= _info_description->height();
                if(h <= 0)
                    h = 1;
            }
            
        } else {
            if(_raw_description) {
                _raw_description->set_max_size(Size2(w, -1));
            }
            
            if (_loading_text)
                h -= _loading_text->height() + 20;
        }
        
        _screenshot_max_size = Size2(w * 0.5, h);
        
        if(_background && _background->source()) {
            auto scale = _screenshot_max_size.div(_background->source()->bounds().size());
            if(_mini_bowl) {
                if(scale.width < 1 || scale.height < 1)
                    _mini_bowl->set_scale(Vec2(scale.min()));
                else
                    _mini_bowl->set_scale(Vec2(scale.max()));
            }
        }
        _screenshot_previous_size = Size2(0);
        
        if(_horizontal) {
            _horizontal->auto_size();
            _horizontal->update_layout();
        }
        
        //select_file(_selected);
    });
    
    _file_chooser->set_tabs({
        FileChooser::Settings{std::string("Pre-processed (PV)"), std::string("pv"), _horizontal},
        FileChooser::Settings{std::string("Convert (RAW)"), std::string("mp4;avi;mov;flv;m4v;webm;mkv;mpg"), _horizontal_raw},
        //FileChooser::Settings("Camera (record)", "", _recording_panel, FileChooser::Settings::Display::None)
    });
    
    _file_chooser->on_update([this](auto&) mutable {
        if(_blob_timer.elapsed() >= 0.15) {
            ++_blob_image_index;
            
            std::lock_guard guard(_blob_mutex);
            if(_blob_image_index >= _blob_images.size()) {
                _blob_image_index = 0;
            }
            
            if(!_blob_images.empty()) {
                _mini_bowl->update([this](Entangled& e) {
                    e.advance_wrap(*_background);
                    for(auto& i : _blob_images.at(_blob_image_index))
                        e.advance_wrap(*i);

                });
                //_mini_bowl->set_background(Transparent, Yellow);
                _mini_bowl->auto_size(Margin{0, 0});
            }
            
            _blob_timer.reset();
        }
        
        Drawable* found = nullptr;
        std::string name;
        std::unique_ptr<sprite::Reference> ref;
        
        if(_file_chooser->current_tab().extension == "pv") {
            for(auto& [key, ptr] : pointers) {
                if(ptr->hovered()) {
                    found = ptr;
                    name = key;
                    ref = std::make_unique<sprite::Reference>(GlobalSettings::get(name));
                }
            }
            
        } else {
            for(auto & [key, ptr] : _text_fields) {
                ptr->_text->set_clickable(true);
                
                if(ptr->representative()->hovered()) {
                    found = ptr->representative();
                    name = key;
                }
            }
            
            if(found) {
                ref = std::make_unique<sprite::Reference>(temp_settings[name]);
            }
        }
        
        if(found && ref) {
            _settings_tooltip.set_parameter(name);
            _settings_tooltip.set_other(found);
            _file_chooser->graph()->wrap_object(_settings_tooltip);
        } else
            _settings_tooltip.set_other(nullptr);

        
        std::lock_guard guard(_video_mutex);
        if(_buffer) {
            auto image = _buffer->next();
            if(image) {
                _screenshot->set_source(std::move(image));
                auto max_scale = 1;//_file_chooser->graph()->scale();
                auto max_size = _screenshot_max_size.div(max_scale);
                auto scree_size = _screenshot->source()->bounds().size();
                
                if(_raw_description->max_size() != max_size) {
                    _raw_description->set_max_size(max_size);
                    _screenshot_previous_size = Size2(0);
                }
                
                Vec2 scale;
                
                // width is more too big than height:
                if(scree_size.width > scree_size.height) 
                {
                    scale = Vec2(max_size.min() / scree_size.width );
                } else {
                    scale = Vec2(max_size.min() / scree_size.height);
                }
                
                if(scale != _screenshot_previous_size) {
                    _screenshot->set_scale(scale);
                    
                    _raw_info->auto_size();
                    _raw_settings->auto_size();
                    _horizontal_raw->auto_size();
                    _recording_panel->auto_size();
                    
                    if(_screenshot_previous_size.empty()) {
                        {
                            std::string info_text = "";//<h3>Info</h3>\n";
                            info_text += "<key>resolution</key>: <ref><nr>"+Meta::toStr(_buffer->_video->size().width)+"</nr>x<nr>"+Meta::toStr(_buffer->_video->size().height)+"</nr></ref>\n";
                            
                            DurationUS us{ uint64_t( _buffer->_video->length().get() / double(_buffer->_video->framerate()) * 1000.0 * 1000.0 ) };
                            auto len = us.to_html();
                            info_text += "<key>length</key>: <ref>"+len+"</ref>";
                            
                            _raw_description->set_txt(info_text);
                        }
                        
                        _file_chooser->update_size();
                    }
                    
                    _screenshot_previous_size = scale;
                }
            }
            
            if(!_buffer->_terminated_background_task) {
                if(!contains(_raw_info->children(), (Drawable*)_loading_text.get()))
                    _raw_info->add_child(2, _loading_text);
                //_loading_text->set_pos(_screenshot->pos());
                _loading_text->set_txt("generating average ("+Meta::toStr(min(TEMP_SETTING(average_samples).value<uint32_t>(), _buffer->_number_samples.load()))+"/"+TEMP_SETTING(average_samples).get().valueString()+")");
                
            } else if(contains(_raw_info->children(), (Drawable*)_loading_text.get())) {
                _raw_info->remove_child(_loading_text);
            }

            //_raw_info->set_background(Transparent, Green);
        }
    });
    
    _file_chooser->set_validity_check([this](file::Path path) {
        if((path.exists() && path.is_folder())
           || _file_chooser->current_tab().is_valid_extension(path))
            return true;
        return false;
    });
    
    _file_chooser->on_open([this](auto){
        move_to_stale(std::move(_buffer));
    });
    
    _file_chooser->on_tab_change([this](auto){
        move_to_stale(std::move(_buffer));
    });
    
    _file_chooser->open();
}

VideoOpener::~VideoOpener() {
    if(_infos) {
        _infos->remove_event_handler(EventType::HOVER, nullptr);
    }
    
    if(_callback != nullptr) {
        temp_settings.unregister_callback(_callback);
        _callback = nullptr;
    }
    
    {
        std::lock_guard guard(_stale_mutex);
        if(_buffer)
            _stale_buffers.push(std::move(_buffer));
        _stale_buffers.push(nullptr);
    }
    
    _stale_variable.notify_all();
    _stale_thread->join();
    
    _file_chooser = nullptr;
    
    {
        _end_frames_thread = true;
        if(_accumulate_video_frames_thread != nullptr) {
            Timer timer;
            while(!_accumulate_frames_done) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                if(timer.elapsed() >= 10) {
                    FormatWarning("Have been waiting for a long time on my accumulate_video_frames_thread. Terminating anyway now.");
                    return;
                }
            }
            
            _accumulate_video_frames_thread->join();
            _accumulate_video_frames_thread = nullptr;
        }
    }
}

void VideoOpener::move_to_stale(std::unique_ptr<BufferedVideo>&& ptr) {
    if(!ptr)
        return;
    
    {
        std::lock_guard guard(_stale_mutex);
        _stale_buffers.push(std::move(ptr));
    }
    
    _stale_variable.notify_one();
}

VideoOpener::BufferedVideo::BufferedVideo(const file::Path& path) : _path(path) {
}

VideoOpener::BufferedVideo::~BufferedVideo() {
    _terminate = true;
    
    if(_update_thread)
        _update_thread->join();
    
    {
        std::lock_guard guard(_background_mutex);
        _terminate_background = true;
        
        if(_background_thread)
            _background_thread->join();
    }
}

void VideoOpener::BufferedVideo::restart_background() {
    {
        std::lock_guard guard(_background_mutex);
        if(_background_thread)
            _previous_background_thread = std::move(_background_thread);
    }
    
    _background_thread = std::make_unique<std::thread>([this](){
        set_thread_name("BufferedVideo::background_thread");
        
        { // close old background task, if present
            std::lock_guard guard(_background_mutex);
            if(_previous_background_thread) {
                _terminate_background = true;
                _previous_background_thread->join();
                _previous_background_thread = nullptr;
            }
            _terminate_background = false;
        }
        
        std::unique_ptr<VideoSource> background_video;
        Frame_t background_video_index{0_f};
        std::unique_ptr<AveragingAccumulator> accumulator;
        
        { // open video and load first frame
            background_video = std::make_unique<VideoSource>(_path.str());
            
            std::lock_guard guard(_frame_mutex);
            cv::Mat img;
            background_video->frame(0_f, img);
            if(max(img.cols, img.rows) > video_chooser_column_width)
                resize_image(img, video_chooser_column_width / double(max(img.cols, img.rows)));
            
            background_video_index = 0_f;
            accumulator = std::make_unique<AveragingAccumulator>(TEMP_SETTING(averaging_method).value<averaging_method_t::Class>());
            accumulator->add(img);
        }
        
        _terminated_background_task = false;
        
        auto step = Frame_t(max(1u, uint(background_video->length().get() / max(2u, uint(TEMP_SETTING(average_samples).value<uint32_t>())))));
        cv::Mat flt, img;
        _number_samples = 0;
        
        while(!_terminate_background && background_video_index+1_f+step < background_video->length())
        {
            background_video_index += step;
            
            try {
                background_video->frame(background_video_index, img);
                
                _number_samples += 1;
                if(max(img.cols, img.rows) > video_chooser_column_width)
                    resize_image(img, video_chooser_column_width / double(max(img.cols, img.rows)));
                
                accumulator->add(img);
                
                auto image = accumulator->finalize();
                
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                
                std::lock_guard guard(_frame_mutex);
                _background_copy = std::move(image);
            } catch(...) {
                FormatWarning("Exception while trying to read video frame.");
            }
        }
        
        _terminated_background_task = true;
    });
}

void flush_ocl_queue() {
    cv::ocl::finish();
    //volatile auto const cleanupQueueFlusher = gpuMat::zeros(1, 1, CV_8UC1);
    /*cv::BufferPoolController* c = cv::ocl::getOpenCLAllocator()->getBufferPoolController();
    if (c)
    {
        c->setMaxReservedSize(0);
    }*/
}

void VideoOpener::BufferedVideo::open(std::function<void(const bool)>&& callback) {
    std::lock_guard guard(_video_mutex);
    _update_thread = std::make_unique<std::thread>([this, cb = std::move(callback)]() mutable {
        set_thread_name("BufferedVideo::update_thread");
        
        Frame_t playback_index;
        Timer video_timer;
        double seconds_between_frames = 0;
        static std::mutex _gpu_mutex; // in case multiple videos are still "open"
        
        cv::Mat local;
        gpuMat background_image;
        gpuMat flt, img, mask, diff, alpha, output;
        
        try {
            {
                std::lock_guard guard(_video_mutex);
                _video = std::make_unique<VideoSource>(_path.str());
                _video->frame(0_f, local);
                
                {
                    std::lock_guard gpu_guard(_gpu_mutex);
                    flush_ocl_queue();
                    local.copyTo(img);
                    flush_ocl_queue();
                }
                
                playback_index = 0_f;
                video_timer.reset();
                
                restart_background();
                
                // playback at 2x speed
                seconds_between_frames = 1 / double(_video->framerate());
            }
            
            {
                std::lock_guard gaurd(_frame_mutex);
                _cached_frame = Image::Make(local);
            }
            
            cb(true);
        
            while(!_terminate) {
                std::lock_guard guard(_video_mutex);
                auto dt = video_timer.elapsed();
                if(dt < seconds_between_frames)
                    continue;
                
                ++playback_index;
                video_timer.reset();
                
                if(playback_index+1_f >= _video->length())
                    playback_index = 0_f;
                
                try {
                    _video->frame(playback_index, local);
                    
                    if(_number_samples.load() > 1) {
                        std::lock_guard gpu_guard(_gpu_mutex);
                        flush_ocl_queue();
                        
                        local.copyTo(img);
                        if(max(img.cols, img.rows) > video_chooser_column_width)
                            resize_image(img, video_chooser_column_width / double(max(img.cols, img.rows)));
                        img.convertTo(flt, CV_32FC1);

                        if(alpha.empty()) {
                            alpha = gpuMat(img.rows, img.cols, CV_8UC1);
                            alpha.setTo(cv::Scalar(255));
                        }
                        
                        {
                            std::lock_guard frame_guard(_frame_mutex);
                            if(_background_copy) {
                                _background_copy->get().convertTo(background_image, CV_32FC1);
                                _background_copy = nullptr;
                            }
                            cv::absdiff(background_image, flt, diff);
                        }
                        
                        cv::inRange(diff, _threshold.load(), 255, mask);
                        cv::merge(std::vector<gpuMat>{mask, img, img, alpha}, output);
                        output.copyTo(local);
                        
                        flush_ocl_queue();
                    }
                    
                    std::lock_guard frame_guard(_frame_mutex);
                    _cached_frame = Image::Make(local);
                    
                } catch(const std::exception& e) {
                    FormatExcept("Caught exception while updating: ", e.what());
                }
                
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            
        } catch(const UtilsException& ex) {
            // pass
            cb(false);
        } catch(...) {
            cb(false);
        }
        
        std::lock_guard gpu_guard(_gpu_mutex);
        flush_ocl_queue();
        background_image.release();
        flt.release();
        img.release();
        mask.release();
        diff.release();
        alpha.release();
        output.release();
        flush_ocl_queue();
    });
}

Size2 VideoOpener::BufferedVideo::size() {
    std::lock_guard guard(_video_mutex);
    return Size2(_video->size());
}

Image::Ptr VideoOpener::BufferedVideo::next() {
    std::lock_guard guard(_frame_mutex);
    return std::move(_cached_frame);
}

void VideoOpener::select_file(const file::Path &p) {
    const double max_width = _file_chooser->graph()->width() * 0.25;
    std::lock_guard guard(_file_chooser->graph()->lock());
    _end_frames_thread = true;
    
    if(_file_chooser->current_tab().extension != "pv") {
        auto callback = [this, p, max_width](const bool success){
            if(!success) {
                // immediately move to stale
                std::lock_guard gui_lock(_file_chooser->graph()->lock());
                std::lock_guard guard(_video_mutex);
                FormatExcept("Could not open file ",p.str(),".");
                
                cv::Mat img = cv::Mat::zeros((int)max_width, (int)max_width, CV_8UC1);
                cv::putText(img, "Cannot open video.", Vec2(50, 220), cv::FONT_HERSHEY_PLAIN, 1, White);
                _screenshot->set_source(Image::Make(img));
                _screenshot->set_scale(Vec2(1));
                _file_chooser->deselect();
                move_to_stale(std::move(_buffer));
            }
        };
        
        try {
            if(p.empty())
                throw U_EXCEPTION("No file selected.");
            print("Opening ",p.str());
            
            std::lock_guard guard(_video_mutex);
            {
                TEMP_SETTING(output_name) = file::Path("video");
                auto filename = p;
                
                if(p.has_extension() && p.extension() == "pv")
                    filename = filename.remove_extension();
                
                if(utils::contains((std::string)p.filename(), '%')) {
                    filename = filename.remove_filename();
                }
                
                filename = filename.filename();
                
                TEMP_SETTING(output_name) = filename;
                _text_fields["output_name"]->update();
            }
            
            move_to_stale(std::move(_buffer));
            
            _screenshot_previous_size = Size2(0);
            _buffer = std::make_unique<BufferedVideo>(p);
            try {
                _buffer->_threshold = TEMP_SETTING(threshold).value<int>();
                
            } catch(const std::exception &e) {
                FormatExcept("Converting number: '", e.what(),"'");
            }
            
            _buffer->open(callback);
            
        } catch(const std::exception& e) {
            FormatExcept("Cannot open file ",p.str()," (", e.what(),")");
            callback(false);
        }
        return;
        
    } else {
        std::lock_guard guard(_video_mutex);
        move_to_stale(std::move(_buffer));
    }
    
    using namespace gui;
    using namespace file;
    
    auto ext = _file_chooser->current_tab().extension == "pv" ? "pv" : "";
    GlobalSettings::map().dont_print("filename");
    _selected = p.remove_extension(ext);
    SETTING(filename) = p.remove_extension(ext);
    
    Path settings_file = file::DataLocation::parse("settings");
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
        } 
        catch(const cmn::illegal_syntax& e) {
            FormatWarning("File ", _selected.str()," has illegal syntax: ",e.what());
        }
        catch (const UtilsException& e) {
            FormatWarning("File ", _selected.str(), " cannot load a property value: ", e.what());
        }
    }
    
    std::vector<Layout::Ptr> children {
        Layout::Ptr(std::make_shared<Text>(Str("Settings"),TextClr( White), gui::Font(0.9f, Style::Bold)))
    };
    
    constexpr double settings_width = 240;
    
    for(auto &name : _settings_to_show) {
        std::string start;
        if(tmp[name].is_type<std::string>())
            start = tmp[name].value<std::string>();
        else
            start = tmp[name].get().valueString();
        
        if(tmp[name].is_type<bool>()) {
            children.push_back( Layout::Ptr(std::make_shared<Checkbox>(Str(name), attr::Checked{tmp[name].get().value<bool>()}, gui::Font(0.7f))) );
        } else if(name == "output_prefix") {
            std::vector<std::string> folders;
            for(auto &p : _selected.remove_filename().find_files()) {
                try {
                    if(p.is_folder() && p.filename() != "data" && p.filename() != "..") {
                        if(!p.find_files().empty()) {
                            folders.push_back((std::string)p.filename());
                        }
                    }
                } catch(const UtilsException& ex) {
                    continue; // cannot read folder
                }
            }
            
            children.push_back( Layout::Ptr(std::make_shared<Text>(Str{name}, TextClr{White}, gui::Font(0.7f))) );
            children.push_back( Layout::Ptr(std::make_shared<Dropdown>(Box(0, 0, settings_width, 28), folders)) );
            ((Dropdown*)children.back().get())->textfield()->set_font(Font(0.7f));
            
        } else {
            children.push_back( Layout::Ptr(std::make_shared<Text>(Str{name}, TextClr{White}, gui::Font(0.7f))) );
            children.push_back( Layout::Ptr(std::make_shared<Textfield>(Str(start), Box(0, 0, settings_width, 28))));
            ((Textfield*)children.back().get())->set_font(Font(0.7f));
        }
        
        if(name == "output_prefix") {
            ((Dropdown*)children.back().get())->on_select([dropdown = ((Dropdown*)children.back().get()), this](auto, const Dropdown::TextItem & item)
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
                ((Dropdown*)children.back().get())->select_item(Dropdown::RawIndex());
            else {
                auto items = ((Dropdown*)children.back().get())->items();
                auto N = items.size();
                for(size_t i=0; i<N; ++i) {
                    if(items.at(i).search_name() == _output_prefix) {
                        ((Dropdown*)children.back().get())->select_item(Dropdown::RawIndex{narrow_cast<long>(i)});
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
        children.push_back( Layout::Ptr(std::make_shared<Checkbox>(Str("load results"), attr::Checked{false}, gui::Font(0.7f))) );
        _load_results_checkbox = dynamic_cast<Checkbox*>(children.back().get());
    } else
        children.push_back( Layout::Ptr(std::make_shared<Text>(Str("No loadable results found."),TextClr( Gray), gui::Font(0.7f, Style::Bold))) );
    
    _extra->set_children(children);
    _extra->auto_size();
    _extra->update_layout();
    
    try {
        auto video = std::make_unique<pv::File>(SETTING(filename).value<file::Path>(), pv::FileMode::READ);
        auto text = video->get_info(false);
        
        _mini_bowl = std::make_shared<Entangled>();
        auto scale = _screenshot_max_size.div(Size2(video->average()));
        _mini_bowl->set_scale(Vec2(scale.min()));
        
        _mini_bowl->update([&](Entangled& b){
            std::shared_ptr<ExternalImage> image = std::make_shared<ExternalImage>(Image::Make(video->average()));
            _background = image;
            b.advance_wrap(*_background);
            
            std::lock_guard guard(_blob_mutex);
            _blob_images.clear();
            
#ifndef NDEBUG
            print("Mini bowl update (", _mini_bowl->scale().x," scale):");
#endif
            
            _blob_image_index = 0;
            _blob_timer.reset();
#ifndef NDEBUG
            print("Done.");
#endif
        });
        
        _mini_bowl->auto_size(Margin{0, 0});
        
        gui::derived_ptr<gui::Text> info_text = std::make_shared<gui::Text>(Str("Selected"), TextClr{White}, gui::Font(0.9f, gui::Style::Bold));
        _info_description = std::make_shared<gui::StaticText>(Str(settings::htmlify(text)),  SizeLimit(_screenshot_max_size.div(_file_chooser->graph()->scale()).width * 0.25, _screenshot_max_size.div(_file_chooser->graph()->scale()).height), gui::Font(0.7f));
        //gui::derived_ptr<gui::Text> info_2 = std::make_shared<gui::Text>("Preview", Vec2(), gui::White, gui::Font(0.9f, gui::Style::Bold));
        
        _infos->set_children({
            info_text,
            _info_description,
            //info_2,
            _mini_bowl
        });
        
        _infos->auto_size();
        _infos->update_layout();
        
        _infos->remove_event_handler(EventType::HOVER, NULL);
        _infos->on_hover([this, meta = "<h2>Metadata</h2>"+ settings::htmlify(video->header().metadata)](Event e){
            if(e.hover.hovered) {
                _file_chooser->set_tooltip(1, _background.get(), meta);
            } else
                _file_chooser->set_tooltip(1, nullptr, "");
        });
        
        if(_accumulate_video_frames_thread) {
            _end_frames_thread = true;
            if(!_accumulate_frames_done) {
#ifndef NDEBUG
                FormatWarning("Have to wait for accumulate video frames thread...");
#endif
                while(!_accumulate_frames_done) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
            
            _accumulate_video_frames_thread->join();
            _accumulate_video_frames_thread = nullptr;
        }
        
        _accumulate_frames_done = false;
        _end_frames_thread = false;
        
        _accumulate_video_frames_thread = std::make_unique<std::thread>([this, video = std::move(video)]()
        {
            cmn::set_thread_name("_accumulate_video_frames_thread");
            
            Background bg(Image::Make(video->average()), nullptr);
            
            auto step = Frame_t(max(1ul, min(video->length().get() / 100ul, (ushort)video->framerate())));
            pv::Frame frame;
            std::vector<Drawable*> children;
            
            for(Frame_t i = 0_f; not _end_frames_thread && i<video->length() && i < step * 10_f; i += step) {
                video->read_frame(frame, i);
                
                std::vector<std::unique_ptr<ExternalImage>> images;
                for(auto &blob : frame.get_blobs()) {
                    auto&& [pos, image] = blob->alpha_image(bg, 1);
                    images.push_back(std::make_unique<ExternalImage>(std::move(image), pos));
                }
                
                std::lock_guard guard(_blob_mutex);
                _blob_images.emplace_back(std::move(images));
            }
            
            _accumulate_frames_done = true;
#ifndef NDEBUG
            print("accumulate done");
#endif
        });
        
    } catch(...) {
        FormatExcept{ "Caught an exception while reading info from ",SETTING(filename).value<file::Path>().str(),"." };
    }
    
    _horizontal->auto_size();
    _horizontal->update_layout();
    
    SETTING(filename) = file::Path();
}

}
