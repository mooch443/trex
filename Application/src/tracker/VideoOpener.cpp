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

#define TEMP_SETTING(NAME) (gui::temp_settings[#NAME])

namespace gui {
GlobalSettings::docs_map_t temp_docs;
sprite::Map temp_settings;

VideoOpener::LabeledCheckbox::LabeledCheckbox(const std::string& name)
    : LabeledField(name),
      _checkbox(std::make_shared<gui::Checkbox>(Vec2(), name)),
      _ref(gui::temp_settings[name])
{
    _checkbox->set_checked(_ref.value<bool>());
    _checkbox->set_font(Font(0.6));

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
    _text_field->set_font(Font(0.6));

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
    _dropdown->textfield()->set_font(Font(0.6));
    assert(_ref.get().is_enum());
    std::vector<Dropdown::TextItem> items;
    int index = 0;
    for(auto &name : _ref.get().enum_values()()) {
        items.push_back(Dropdown::TextItem(name, index++));
    }
    _dropdown->set_items(items);
    _dropdown->select_item(_ref.get().enum_index()());
    _dropdown->textfield()->set_text(_ref.get().valueString());
    
    _dropdown->on_select([this](auto index, auto) {
        try {
            _ref.get().set_value_from_string(_ref.get().enum_values()().at(index));
        } catch(...) {}
        _dropdown->set_opened(false);
    });
}

void VideoOpener::LabeledDropDown::update() {
    _dropdown->select_item(_ref.get().enum_index()());
}

VideoOpener::VideoOpener() {
    grab::default_config::get(temp_settings, temp_docs, nullptr);
    //::default_config::get(GlobalSettings::map(), temp_docs, nullptr);
    
    _horizontal = std::make_shared<gui::HorizontalLayout>();
    _extra = std::make_shared<gui::VerticalLayout>();
    _infos = std::make_shared<gui::VerticalLayout>();
    
    _horizontal->set_policy(gui::HorizontalLayout::TOP);
    _extra->set_policy(gui::VerticalLayout::LEFT);
    _infos->set_policy(gui::VerticalLayout::LEFT);
    
    _horizontal->set_children({_infos, _extra});
    
    TEMP_SETTING(output_name) = file::Path("video");
    
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
    
    std::vector<Layout::Ptr> objects{
        Layout::Ptr(std::make_shared<Text>("Settings", Vec2(), White, gui::Font(0.8, Style::Bold)))
    };
    for(auto &[key, ptr] : _text_fields)
        ptr->add_to(objects);
    
    _raw_settings->set_children(objects);
    
    _raw_description = std::make_shared<gui::StaticText>("Info", Vec2(), Size2(400, -1), Font(0.6));
    _raw_info->set_children({Layout::Ptr(std::make_shared<Text>("Preview", Vec2(), White, gui::Font(0.8, Style::Bold))), _screenshot, _raw_description});
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
                _result.cmd = "-i '" + path.str() + "' " + "-o '"+TEMP_SETTING(output_name).value<file::Path>().str()+"' -threshold "+TEMP_SETTING(threshold).get().valueString()+" -average_samples "+TEMP_SETTING(average_samples).get().valueString()+ " -reset_average"
                    +" -averaging_method "+TEMP_SETTING(averaging_method).get().valueString();
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
        FileChooser::Settings{std::string("Convert (RAW)"), std::string("mp4;avi;mov;flv;m4v;webm"), _horizontal_raw}
    });
    
    _file_chooser->on_update([this](auto&) mutable {
        std::lock_guard guard(_video_mutex);
        if(_buffer) {
            auto image = _buffer->next();
            if(image) {
                _screenshot->set_source(std::move(image));
                
                if(_screenshot->size().max() != _screenshot_previous_size) {
                    _screenshot_previous_size = _screenshot->size().max();
                    
                    const double max_width = 500;
                    auto ratio = max_width / _screenshot_previous_size;
                    Debug("%f (%f / %f)", ratio, max_width, _screenshot_previous_size);
                    _screenshot->set_scale(Vec2(ratio));
                    
                    _raw_info->auto_size(Margin{0, 0});
                    _raw_settings->auto_size(Margin{0, 0});
                    _horizontal_raw->auto_size(Margin{0, 0});
                }
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
        _buffer = nullptr;
    });
    
    _file_chooser->on_tab_change([this](auto){
        _buffer = nullptr;
    });
    
    _file_chooser->open();
}

VideoOpener::BufferedVideo::BufferedVideo(const file::Path& path) : _path(path) {
}

VideoOpener::BufferedVideo::~BufferedVideo() {
    _terminate = true;
    _terminate_background = true;
    
    if(_update_thread)
        _update_thread->join();
    if(_background_thread)
        _background_thread->join();
    
    _background_video = nullptr;
}

void VideoOpener::BufferedVideo::restart_background() {
    _terminate_background = true;
    if(_background_thread)
        _background_thread->join();
    
    _terminate_background = false;
    
    std::lock_guard guard(_frame_mutex);
    cv::Mat img;
    _background_video->frame(0, img);
    if(max(img.cols, img.rows) > 500)
        resize_image(img, 500 / double(max(img.cols, img.rows)));
    
    img.convertTo(_background_image, CV_32FC1);
    _accumulator = std::make_unique<AveragingAccumulator<>>(TEMP_SETTING(averaging_method).value<averaging_method_t::Class>());
    _accumulator->add(img);
    //_background_image.copyTo(_accumulator);
    
    //_background_samples = 1;
    _background_video_index = 0;
    //_accumulator = cv::Mat::zeros(img.rows, img.cols, CV_32FC1);
    
    _background_thread = std::make_unique<std::thread>([this](){
        int step = max(1, int(_background_video->length() / max(2.0, double(TEMP_SETTING(average_samples).value<int>()))));
        Debug("Start calculating background in %d steps", step);
        cv::Mat flt, img;
        
        while(!_terminate_background && _background_video_index+1+step < _background_video->length()) {
            _background_video_index += step;
            
            _background_video->frame(_background_video_index, img);
            if(max(img.cols, img.rows) > 500)
                resize_image(img, 500 / double(max(img.cols, img.rows)));
            
            /*img.convertTo(flt, CV_32FC1);
            if(!_accumulator.empty())
                cv::add(_accumulator, flt, _accumulator);
            else
                flt.copyTo(_accumulator);
            ++_background_samples;*/
            
            _accumulator->add(img);
            
            Debug("%d/%d (%d)", _background_video_index, _background_video->length(), step);
            auto image = _accumulator->finalize();
            
            std::lock_guard guard(_frame_mutex);
            _background_copy = std::move(image);
        }
        
        Debug("Done calculating background");
    });
}

void VideoOpener::BufferedVideo::open() {
    std::lock_guard guard(_video_mutex);
    _video = std::make_unique<VideoSource>(_path.str());
    
    _video->frame(0, _local);
    _local.copyTo(_img);
    
    _background_video = std::make_unique<VideoSource>(_path.str());
    _cached_frame = std::make_unique<Image>(_local);
    
    _playback_index = 0;
    _video_timer.reset();
    
    restart_background();
    
    // playback at 2x speed
    _seconds_between_frames = 1 / double(_video->framerate());

    _update_thread = std::make_unique<std::thread>([this](){
        while(!_terminate) {
            std::lock_guard guard(_video_mutex);
            auto dt = _video_timer.elapsed();
            if(dt < _seconds_between_frames)
                continue;
            
            _playback_index = _playback_index + 1; // loading is too slow...
            
            if((uint32_t)_playback_index.load() % 100 == 0)
                Debug("Playback %.2fms / %.2fms ...", dt * 1000, _seconds_between_frames * 1000);
            
            if(dt > _seconds_between_frames) {
                
            } //else
                //_playback_index = _playback_index + dt / _seconds_between_frames;
            _video_timer.reset();
            
            if(_playback_index+1 >= _video->length())
                _playback_index = 0;
            
            update_loop();
        }
    });
}

void VideoOpener::BufferedVideo::update_loop() {
    try {
        _video->frame((size_t)_playback_index, _local);
        _local.copyTo(_img);
        if(max(_img.cols, _img.rows) > 500)
            resize_image(_img, 500 / double(max(_img.cols, _img.rows)));
        _img.convertTo(_flt, CV_32FC1);

        if(_alpha.empty()) {
            _alpha = gpuMat(_img.rows, _img.cols, CV_8UC1);
            _alpha.setTo(cv::Scalar(255));
        }
        
        {
            std::lock_guard frame_guard(_frame_mutex);
            if(_background_copy) {
                _background_copy->get().convertTo(_background_image, CV_32FC1);
                _background_copy = nullptr;
            }
            cv::absdiff(_background_image, _flt, _diff);
        }
        
        cv::inRange(_diff, _threshold.load(), 255, _mask);
        cv::merge(std::vector<gpuMat>{_mask, _img, _img, _alpha}, _output);
        _output.copyTo(_local);
        
        std::lock_guard frame_guard(_frame_mutex);
        _cached_frame = std::make_unique<Image>(_local);
        
    } catch(const std::exception& e) {
        Except("Caught exception while updating '%s'", e.what());
    }
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
    const double max_width = 500;
    std::lock_guard guard(_file_chooser->graph()->lock());
    
    if(_file_chooser->current_tab().extension != "pv") {
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
                
                if(utils::contains(p.filename().to_string(), '%')) {
                    filename = filename.remove_filename();
                }
                
                filename = filename.filename();
                
                TEMP_SETTING(output_name) = filename;
                if(TEMP_SETTING(output_name).value<file::Path>().empty()) {
                    Warning("No output filename given. Defaulting to 'video'.");
                } else
                    Warning("Given empty filename, the program will default to using input basename '%S'.", &filename.str());
                
                _text_fields["output_name"]->update();
            }
            
            _buffer = std::make_unique<BufferedVideo>(p);
            _buffer->open();
            _screenshot->set_source(std::move(_buffer->next()));
            _screenshot_previous_size = 0;
            
            try {
                _buffer->_threshold = TEMP_SETTING(threshold).value<int>();
                
            } catch(const std::exception &e) {
                Except("Converting number: '%s'", e.what());
            }
            
            {
                std::string info_text = "<h3>Info</h3>\n";
                info_text += "<key>resolution</key>: <ref><nr>"+Meta::toStr(_buffer->_video->size().width)+"</nr>x<nr>"+Meta::toStr(_buffer->_video->size().height)+"</nr></ref>\n";
                
                DurationUS us{ uint64_t( _buffer->_video->length() / double(_buffer->_video->framerate()) * 1000.0 * 1000.0 ) };
                auto len = us.to_html();
                info_text += "<key>length</key>: <ref>"+len+"</ref>";
                
                _raw_description->set_txt(info_text);
            }
            
            auto ratio = max_width / _screenshot->size().max();
            Debug("%f (%f / %f)", ratio, max_width, _screenshot->size().max());
            _screenshot->set_scale(Vec2(ratio));
            
            _raw_info->auto_size(Margin{0, 0});
            _raw_settings->auto_size(Margin{0, 0});
            _horizontal_raw->auto_size(Margin{0, 0});
            
        } catch(const std::exception& e) {
            std::lock_guard guard(_video_mutex);
            Except("Cannot open file '%S' (%s)", &p.str(), e.what());
            
            cv::Mat img = cv::Mat::zeros(max_width, max_width, CV_8UC1);
            cv::putText(img, "Cannot open video.", Vec2(50, 220), cv::FONT_HERSHEY_PLAIN, 1, White);
            _screenshot->set_source(std::make_unique<Image>(img));
            _screenshot->set_scale(Vec2(1));
            _buffer = nullptr;
        }
        return;
        
    } else {
        std::lock_guard guard(_video_mutex);
        if(_buffer) {
            _buffer = nullptr;
        }
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
        Layout::Ptr(std::make_shared<Text>("Settings", Vec2(), White, gui::Font(0.8, Style::Bold)))
    };
    
    for(auto &name : _settings_to_show) {
        std::string start;
        if(tmp[name].is_type<std::string>())
            start = tmp[name].value<std::string>();
        else
            start = tmp[name].get().valueString();
        
        if(tmp[name].is_type<bool>()) {
            children.push_back( Layout::Ptr(std::make_shared<Checkbox>(Vec2(), name, tmp[name].get().value<bool>(), gui::Font(0.6))) );
        } else if(name == "output_prefix") {
            std::vector<std::string> folders;
            for(auto &p : _selected.remove_filename().find_files()) {
                try {
                    if(p.is_folder() && p.filename() != "data" && p.filename() != "..") {
                        if(!p.find_files().empty()) {
                            folders.push_back(p.filename().to_string());
                        }
                    }
                } catch(const UtilsException& ex) {
                    continue; // cannot read folder
                }
            }
            
            children.push_back( Layout::Ptr(std::make_shared<Text>(name, Vec2(), White, gui::Font(0.6))) );
            children.push_back( Layout::Ptr(std::make_shared<Dropdown>(Bounds(0, 0, 300, 28), folders)) );
            ((Dropdown*)children.back().get())->textfield()->set_font(Font(0.6));
            
        } else {
            children.push_back( Layout::Ptr(std::make_shared<Text>(name, Vec2(), White, gui::Font(0.6))) );
            children.push_back( Layout::Ptr(std::make_shared<Textfield>(start, Bounds(0, 0, 300, 28))));
            ((Textfield*)children.back().get())->set_font(Font(0.6));
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
                for(size_t i=0; i<items.size(); ++i) {
                    if(items.at(i).search_name() == _output_prefix) {
                        ((Dropdown*)children.back().get())->select_item(i);
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
        children.push_back( Layout::Ptr(std::make_shared<Checkbox>(Vec2(), "load results", false, gui::Font(0.7, Style::Bold))) );
        _load_results_checkbox = dynamic_cast<Checkbox*>(children.back().get());
    } else
        children.push_back( Layout::Ptr(std::make_shared<Text>("No loadable results found.", Vec2(), Gray, gui::Font(0.7, Style::Bold))) );
    
    _extra->set_children(children);
    _extra->auto_size(Margin{0,0});
    _extra->update_layout();
    
    try {
        pv::File video(SETTING(filename).value<file::Path>());
        video.start_reading();
        auto text = video.get_info();
        

        gui::derived_ptr<gui::Text> info_text = std::make_shared<gui::Text>("Selected", Vec2(), gui::White, gui::Font(0.8, gui::Style::Bold));
        gui::derived_ptr<gui::StaticText> info_description = std::make_shared<gui::StaticText>(settings::htmlify(text), Vec2(), Size2(300, 600), gui::Font(0.5));
        
        _infos->set_children({
            info_text,
            info_description
        });
        
        _infos->auto_size(Margin{0, 0});
        _infos->update_layout();
        
    } catch(...) {
        Except("Caught an exception while reading info from '%S'.", &SETTING(filename).value<file::Path>().str());
    }
    
    _horizontal->auto_size(Margin{0, 0});
    _horizontal->update_layout();
    
    SETTING(filename) = file::Path();
}

}
