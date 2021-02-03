#include "VideoOpener.h"
#include <tracker/misc/default_config.h>
#include <misc/GlobalSettings.h>
#include <gui/types/Dropdown.h>
#include <pv.h>
#include <tracker/misc/Output.h>
#include <misc/default_settings.h>
#include <gui/types/StaticText.h>
#include <processing/RawProcessing.h>
#include <grabber/default_config.h>
#include <opencv2/core/ocl.hpp>

#define TEMP_SETTING(NAME) (gui::temp_settings[#NAME])

namespace gui {
GlobalSettings::docs_map_t temp_docs;
sprite::Map temp_settings;

VideoOpener::LabeledCheckbox::LabeledCheckbox(const std::string& name)
    : LabeledField(name),
      _checkbox(std::make_shared<gui::Checkbox>(Vec2(), name)),
      _ref(gui::temp_settings[name])
{
    _docs = gui::temp_docs[name];
    
    _checkbox->set_checked(_ref.value<bool>());
    _checkbox->set_font(Font(0.6f));

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
      _text_field(std::make_shared<gui::Textfield>("", Bounds(0, 0, 300, 28))),
      _ref(gui::temp_settings[name])
{
    _text_field->set_placeholder(name);
    _text_field->set_font(Font(0.6f));
    
    _docs = gui::temp_docs[name];

    _text_field->set_text(_ref.get().valueString());
    _text_field->on_text_changed([this](){
        try {
            _ref.get().set_value_from_string(_text_field->text());

        } catch(...) {}
    });
}

void VideoOpener::LabeledTextField::update() {
    _text_field->set_text(_ref.get().valueString());
}

VideoOpener::LabeledDropDown::LabeledDropDown(const std::string& name)
    : LabeledField(name),
      _dropdown(std::make_shared<gui::Dropdown>(Bounds(0, 0, 300, 28))),
      _ref(gui::temp_settings[name])
{
    _docs = gui::temp_docs[name];
    
    _dropdown->textfield()->set_font(Font(0.6f));
    assert(_ref.get().is_enum());
    std::vector<Dropdown::TextItem> items;
    int index = 0;
    for(auto &name : _ref.get().enum_values()()) {
        items.push_back(Dropdown::TextItem(name, index++));
    }
    _dropdown->set_items(items);
    _dropdown->select_item(narrow_cast<long>(_ref.get().enum_index()()));
    _dropdown->textfield()->set_text(_ref.get().valueString());
    
    _dropdown->on_select([this](auto index, auto) {
        if(index < 0)
            return;
        
        try {
            _ref.get().set_value_from_string(_ref.get().enum_values()().at((size_t)index));
        } catch(...) {}
        
        _dropdown->set_opened(false);
    });
}

void VideoOpener::LabeledDropDown::update() {
    _dropdown->select_item(narrow_cast<long>(_ref.get().enum_index()()));
}

VideoOpener::VideoOpener() {
    grab::default_config::get(temp_settings, temp_docs, nullptr);
    //::default_config::get(GlobalSettings::map(), temp_docs, nullptr);
    
    _stale_thread = std::make_unique<std::thread>([this](){
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
                    Debug("Removing stale buffer '%S'...", &path.str());
#endif
                    ptr = nullptr;
#ifndef NDEBUG
                    Debug("Removed stale buffer '%S'.", &path.str());
#endif
                } catch(const std::exception& e) {
                    Except("Exception while freeing stale buffer '%s'.", e.what());
                }
                guard.lock();
                
                ++i;
            }
      
#ifndef NDEBUG
            if(i)
                Debug("Removed %d stale buffers", i);
#endif
        }
        
#ifndef NDEBUG
        Debug("Quit stale thread.");
#endif
    });
    
    _horizontal = std::make_shared<gui::HorizontalLayout>();
    _extra = std::make_shared<gui::VerticalLayout>();
    _infos = std::make_shared<gui::VerticalLayout>();
    
    _horizontal->set_policy(gui::HorizontalLayout::TOP);
    _extra->set_policy(gui::VerticalLayout::LEFT);
    _infos->set_policy(gui::VerticalLayout::LEFT);
    
    _horizontal->set_children({_infos, _extra});
    
    TEMP_SETTING(output_name) = file::Path("video");
    gui::temp_docs["output_name"] = "Basename of the converted video in PV-format.";
    TEMP_SETTING(cmd_parameters) = std::string("-reset_average");
    gui::temp_docs["cmd_parameters"] = "Additional command-line parameters for TGrabs.";
    
    _horizontal_raw = std::make_shared<gui::HorizontalLayout>();
    _horizontal_raw->set_clickable(true);
    _raw_settings = std::make_shared<gui::VerticalLayout>();
    _raw_info = std::make_shared<gui::VerticalLayout>();
    _raw_info->set_policy(gui::VerticalLayout::LEFT);
    _screenshot = std::make_shared<gui::ExternalImage>();
    _text_fields.clear();
    
    gui::temp_settings.register_callback((void*)this, [this](auto&map, auto&key, auto&value){
        if(key == "threshold") {
            if(_buffer)
                _buffer->_threshold = value.template value<int>();
            
        } else if(key == "average_samples" || key == "averaging_method") {
            if(_buffer)
                _buffer->restart_background();
        }
    });

    _text_fields["output_name"] = std::make_unique<LabeledTextField>("output_name");
    _text_fields["threshold"] = std::make_unique<LabeledTextField>("threshold");
    _text_fields["average_samples"] = std::make_unique<LabeledTextField>("average_samples");
    _text_fields["averaging_method"] = std::make_unique<LabeledDropDown>("averaging_method");
    _text_fields["cmd_parameters"] = std::make_unique<LabeledTextField>("cmd_parameters");
    
    std::vector<Layout::Ptr> objects{
        Layout::Ptr(std::make_shared<Text>("Settings", Vec2(), White, gui::Font(0.8f, Style::Bold)))
    };
    for(auto &[key, ptr] : _text_fields)
        ptr->add_to(objects);
    
    _raw_settings->set_children(objects);
    
    _loading_text = std::make_shared<gui::Text>("generating average", Vec2(100,0), Cyan, gui::Font(0.5f));
    
    _raw_description = std::make_shared<gui::StaticText>("Info", Vec2(), Size2(500, -1), Font(0.6f));
    _raw_info->set_children({
        Layout::Ptr(std::make_shared<Text>("Preview", Vec2(), White, gui::Font(0.8f, Style::Bold))),
        _screenshot,
        _raw_description
    });
    _horizontal_raw->set_children({_raw_settings, _raw_info});
    _horizontal_raw->set_policy(gui::HorizontalLayout::TOP);
    
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
    
    _file_chooser = std::make_shared<gui::FileChooser>(
        SETTING(output_dir).value<file::Path>(),
        "pv",
        [this](const file::Path& path, std::string tab) mutable
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
            _result.tab = _file_chooser->current_tab();
            _result.tab.content = nullptr;
            _result.selected_file = _file_chooser->confirmed_file();
            
            if(_result.tab.extension == "pv") {
                // PV file, no need to add cmd
            } else if(!_result.selected_file.empty()) {
                auto add = TEMP_SETTING(cmd_parameters).value<std::string>();
                _result.cmd = "-i \"" + path.str() + "\" " + "-o \""+TEMP_SETTING(output_name).value<file::Path>().str()+"\" -threshold "+TEMP_SETTING(threshold).get().valueString()+" -average_samples "+TEMP_SETTING(average_samples).get().valueString()
                    +" -averaging_method "+TEMP_SETTING(averaging_method).get().valueString()
                    +(add.empty() ? "" : " ")+add;
            }
            
            if(_load_results_checkbox && _load_results_checkbox->checked()) {
                _result.load_results = true;
                _result.load_results_from = "";
            }
            
        }
        
    }, [this](auto& path, std::string tab) {
        select_file(path);
    });
    
    _file_chooser->set_tabs({
        FileChooser::Settings{std::string("Pre-processed (PV)"), std::string("pv"), _horizontal},
        FileChooser::Settings{std::string("Convert (RAW)"), std::string("mp4;avi;mov;flv;m4v;webm;mkv;mpg"), _horizontal_raw}
    });
    
    _file_chooser->on_update([this](auto&) mutable {
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
            auto str = "<h3>"+name+"</h3>";
            
            str += "type: " +settings::htmlify(ref->get().type_name()) + "\n";
            if(GlobalSettings::defaults().has(name)) {
                auto ref = GlobalSettings::defaults().operator[](name);
                str += "default: " +settings::htmlify(ref.get().valueString()) + "\n";
            }
            if(gui::temp_docs.find(name) != gui::temp_docs.end())
                str += "\n" + settings::htmlify(gui::temp_docs[name]);
            
            _file_chooser->set_tooltip(0, found, str);
        } else
            _file_chooser->set_tooltip(0, nullptr, "");
        
        std::lock_guard guard(_video_mutex);
        if(_buffer) {
            auto image = _buffer->next();
            if(image) {
                _screenshot->set_source(std::move(image));
                
                const auto mw = _file_chooser->graph()->width() * 0.3f;
                if(_raw_description->max_size().x != mw) {
                    _raw_description->set_max_size(Size2(mw, -1.f));
                    _screenshot_previous_size = 0;
                }
                
                auto size = _screenshot->size().max();
                if(size != _screenshot_previous_size) {
                    auto ratio = mw / size;
                    _screenshot->set_scale(Vec2(ratio));
                    
                    _raw_info->auto_size(Margin{0, 0});
                    _raw_settings->auto_size(Margin{0, 0});
                    _horizontal_raw->auto_size(Margin{0, 0});
                    
                    if(_screenshot_previous_size == 0) {
                        {
                            std::string info_text = "<h3>Info</h3>\n";
                            info_text += "<key>resolution</key>: <ref><nr>"+Meta::toStr(_buffer->_video->size().width)+"</nr>x<nr>"+Meta::toStr(_buffer->_video->size().height)+"</nr></ref>\n";
                            
                            DurationUS us{ uint64_t( _buffer->_video->length() / double(_buffer->_video->framerate()) * 1000.0 * 1000.0 ) };
                            auto len = us.to_html();
                            info_text += "<key>length</key>: <ref>"+len+"</ref>";
                            
                            _raw_description->set_txt(info_text);
                        }
                        
                        _file_chooser->update_size();
                    }
                    
                    _screenshot_previous_size = size;
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
    _file_chooser = nullptr;
}

VideoOpener::~VideoOpener() {
    {
        std::lock_guard guard(_stale_mutex);
        if(_buffer)
            _stale_buffers.push(std::move(_buffer));
        _stale_buffers.push(nullptr);
    }
    
    _stale_variable.notify_all();
    _stale_thread->join();
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
        uint64_t background_video_index = 0;
        std::unique_ptr<AveragingAccumulator> accumulator;
        
        { // open video and load first frame
            background_video = std::make_unique<VideoSource>(_path.str());
            
            std::lock_guard guard(_frame_mutex);
            cv::Mat img;
            background_video->frame(0, img);
            if(max(img.cols, img.rows) > 500)
                resize_image(img, 500 / double(max(img.cols, img.rows)));
            
            background_video_index = 0;
            accumulator = std::make_unique<AveragingAccumulator>(TEMP_SETTING(averaging_method).value<averaging_method_t::Class>());
            accumulator->add(img);
        }
        
        _terminated_background_task = false;
        
        uint step = max(1u, uint(background_video->length() / max(2u, uint(TEMP_SETTING(average_samples).value<uint32_t>()))));
        cv::Mat flt, img;
        _number_samples = 0;
        
        while(!_terminate_background && background_video_index+1+step < background_video->length())
        {
            background_video_index += step;
            _number_samples += 1;
            
            background_video->frame(background_video_index, img);
            if(max(img.cols, img.rows) > 500)
                resize_image(img, 500 / double(max(img.cols, img.rows)));
            
            accumulator->add(img);
            
            auto image = accumulator->finalize();
            
            std::lock_guard guard(_frame_mutex);
            _background_copy = std::move(image);
        }
        
        _terminated_background_task = true;
    });
}

void flush_ocl_queue() {
    cv::ocl::finish();
    volatile auto const cleanupQueueFlusher = gpuMat::zeros(1, 1, CV_8UC1);
    /*cv::BufferPoolController* c = cv::ocl::getOpenCLAllocator()->getBufferPoolController();
    if (c)
    {
        c->setMaxReservedSize(0);
    }*/
}

void VideoOpener::BufferedVideo::open(std::function<void(const bool)>&& callback) {
    std::lock_guard guard(_video_mutex);
    _update_thread = std::make_unique<std::thread>([this, cb = std::move(callback)]() mutable {
        int64_t playback_index = 0;
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
                _video->frame(0, local);
                
                {
                    std::lock_guard gpu_guard(_gpu_mutex);
                    flush_ocl_queue();
                    local.copyTo(img);
                    flush_ocl_queue();
                }
                
                playback_index = 0;
                video_timer.reset();
                
                restart_background();
                
                // playback at 2x speed
                seconds_between_frames = 1 / double(_video->framerate());
            }
            
            {
                std::lock_guard gaurd(_frame_mutex);
                _cached_frame = std::make_unique<Image>(local);
            }
            
            cb(true);
        
            while(!_terminate) {
                std::lock_guard guard(_video_mutex);
                auto dt = video_timer.elapsed();
                if(dt < seconds_between_frames)
                    continue;
                
                ++playback_index;
                video_timer.reset();
                
                if((uint64_t)playback_index+1 >= _video->length())
                    playback_index = 0;
                
                try {
                    _video->frame((size_t)playback_index, local);
                    
                    if(_number_samples.load() > 1) {
                        std::lock_guard gpu_guard(_gpu_mutex);
                        flush_ocl_queue();
                        
                        local.copyTo(img);
                        if(max(img.cols, img.rows) > 500)
                            resize_image(img, 500 / double(max(img.cols, img.rows)));
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
                    _cached_frame = std::make_unique<Image>(local);
                    
                } catch(const std::exception& e) {
                    Except("Caught exception while updating '%s'", e.what());
                }
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

std::unique_ptr<Image> VideoOpener::BufferedVideo::next() {
    std::lock_guard guard(_frame_mutex);
    return std::move(_cached_frame);
}

void VideoOpener::select_file(const file::Path &p) {
    const double max_width = _file_chooser->graph()->width() * 0.3;
    std::lock_guard guard(_file_chooser->graph()->lock());
    
    if(_file_chooser->current_tab().extension != "pv") {
        auto callback = [this, p, max_width](const bool success){
            if(!success) {
                // immediately move to stale
                std::lock_guard gui_lock(_file_chooser->graph()->lock());
                std::lock_guard guard(_video_mutex);
                Except("Could not open file '%S'.", &p.str());
                
                cv::Mat img = cv::Mat::zeros((int)max_width, (int)max_width, CV_8UC1);
                cv::putText(img, "Cannot open video.", Vec2(50, 220), cv::FONT_HERSHEY_PLAIN, 1, White);
                _screenshot->set_source(std::make_unique<Image>(img));
                _screenshot->set_scale(Vec2(1));
                _file_chooser->deselect();
                move_to_stale(std::move(_buffer));
            }
        };
        
        try {
            if(p.empty())
                U_EXCEPTION("No file selected.");
            Debug("Opening '%S'", &p.str());
            
            std::lock_guard guard(_video_mutex);
            {
                TEMP_SETTING(output_name) = file::Path("video");
                auto filename = p;
                
                if(p.has_extension())
                    filename = filename.remove_extension();
                
                if(utils::contains((std::string)p.filename(), '%')) {
                    filename = filename.remove_filename();
                }
                
                filename = filename.filename();
                
                TEMP_SETTING(output_name) = filename;
                _text_fields["output_name"]->update();
            }
            
            move_to_stale(std::move(_buffer));
            
            _screenshot_previous_size = 0;
            _buffer = std::make_unique<BufferedVideo>(p);
            try {
                _buffer->_threshold = TEMP_SETTING(threshold).value<int>();
                
            } catch(const std::exception &e) {
                Except("Converting number: '%s'", e.what());
            }
            
            _buffer->open(callback);
            
        } catch(const std::exception& e) {
            Except("Cannot open file '%S' (%s)", &p.str(), e.what());
            callback(false);
        }
        return;
        
    } else {
        std::lock_guard guard(_video_mutex);
        move_to_stale(std::move(_buffer));
    }
    
    using namespace gui;
    using namespace file;
    
    GlobalSettings::map().dont_print("filename");
    _selected = p.remove_extension();
    SETTING(filename) = p.remove_extension();
    
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
        Layout::Ptr(std::make_shared<Text>("Settings", Vec2(), White, gui::Font(0.8f, Style::Bold)))
    };
    
    for(auto &name : _settings_to_show) {
        std::string start;
        if(tmp[name].is_type<std::string>())
            start = tmp[name].value<std::string>();
        else
            start = tmp[name].get().valueString();
        
        if(tmp[name].is_type<bool>()) {
            children.push_back( Layout::Ptr(std::make_shared<Checkbox>(Vec2(), name, tmp[name].get().value<bool>(), gui::Font(0.6f))) );
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
            
            children.push_back( Layout::Ptr(std::make_shared<Text>(name, Vec2(), White, gui::Font(0.6f))) );
            children.push_back( Layout::Ptr(std::make_shared<Dropdown>(Bounds(0, 0, 300, 28), folders)) );
            ((Dropdown*)children.back().get())->textfield()->set_font(Font(0.6f));
            
        } else {
            children.push_back( Layout::Ptr(std::make_shared<Text>(name, Vec2(), White, gui::Font(0.6f))) );
            children.push_back( Layout::Ptr(std::make_shared<Textfield>(start, Bounds(0, 0, 300, 28))));
            ((Textfield*)children.back().get())->set_font(Font(0.6f));
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
                auto N = items.size();
                for(size_t i=0; i<N; ++i) {
                    if(items.at(i).search_name() == _output_prefix) {
                        ((Dropdown*)children.back().get())->select_item(narrow_cast<long>(i));
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
        children.push_back( Layout::Ptr(std::make_shared<Checkbox>(Vec2(), "load results", false, gui::Font(0.6f))) );
        _load_results_checkbox = dynamic_cast<Checkbox*>(children.back().get());
    } else
        children.push_back( Layout::Ptr(std::make_shared<Text>("No loadable results found.", Vec2(), Gray, gui::Font(0.7f, Style::Bold))) );
    
    _extra->set_children(children);
    _extra->auto_size(Margin{0,0});
    _extra->update_layout();
    
    try {
        pv::File video(SETTING(filename).value<file::Path>());
        video.start_reading();
        auto text = video.get_info(false);
        
        _background = std::make_shared<ExternalImage>(std::move(std::make_unique<Image>(video.average())));
        _background->set_scale(Vec2(300 / float(video.average().cols)));

        gui::derived_ptr<gui::Text> info_text = std::make_shared<gui::Text>("Selected", Vec2(), gui::White, gui::Font(0.8f, gui::Style::Bold));
        gui::derived_ptr<gui::StaticText> info_description = std::make_shared<gui::StaticText>(settings::htmlify(text), Vec2(), Size2(300, 600), gui::Font(0.5));
        gui::derived_ptr<gui::Text> info_2 = std::make_shared<gui::Text>("Background", Vec2(), gui::White, gui::Font(0.8f, gui::Style::Bold));
        
        _infos->set_children({
            info_text,
            info_description,
            info_2,
            _background
        });
        
        _infos->auto_size(Margin{0, 0});
        _infos->update_layout();
        
        _infos->on_hover([this, meta = "<h2>Metadata</h2>"+ settings::htmlify(video.header().metadata)](Event e){
            if(e.hover.hovered) {
                _file_chooser->set_tooltip(1, _background.get(), meta);
            } else
                _file_chooser->set_tooltip(1, nullptr, "");
        });
        
        
        
    } catch(...) {
        Except("Caught an exception while reading info from '%S'.", &SETTING(filename).value<file::Path>().str());
    }
    
    _horizontal->auto_size(Margin{0, 0});
    _horizontal->update_layout();
    
    SETTING(filename) = file::Path();
}

}
