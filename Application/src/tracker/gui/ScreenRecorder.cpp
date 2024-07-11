#include "ScreenRecorder.h"

#include <commons.pc.h>
#include <file/Path.h>
#include <misc/Timer.h>
#include <misc/default_config.h>
#include <gui/DrawBase.h>
#include <gui/DrawStructure.h>
#include <gui/WorkProgress.h>
#include <file/DataLocation.h>
#include <gui/IMGUIBase.h>

namespace cmn::gui {



struct ScreenRecorder::Data {
    cv::VideoWriter* _recording_capture = nullptr;
    cv::Size _recording_size;
    file::Path _recording_path;
    default_config::gui_recording_format_t::Class _recording_format;

    Frame_t _recording_start;
    Frame_t _recording_frame;
    Frame_t _last_recording_frame;
    std::atomic_bool _recording = false;
    
    void do_record(Image::Ptr&& image, Base* , Frame_t , Frame_t max_frame) {
        if(!_recording /*|| !_base || (_last_recording_frame.valid() && _recording_frame == _last_recording_frame)*/)
            return;
        
        //assert(_base->frame_recording());
        static Timing timing("recording_timing");
        TakeTiming take(timing);
        
        if(not _recording_frame.valid())
            _recording_frame = 0_f;
        else
            _recording_frame += 1_f;
        
        //if(!_last_recording_frame.valid()) {
        //    _last_recording_frame = _recording_frame;
        //    return; // skip first frame
        //}
        
        _last_recording_frame = _recording_frame;
        
        if(!image || image->empty() || !image->cols || !image->rows) {
            FormatWarning("Expected image, but there is none.");
            return;
        }
        
        auto mat = image->get();
        
        if(_recording_capture) {
            static cv::Mat output;
            auto bounds = Bounds(0, 0, _recording_size.width, _recording_size.height);
            if(output.size() != _recording_size) {
                output = cv::Mat::zeros(_recording_size.height, _recording_size.width, CV_8UC3);
            }
            
            auto input_bounds = bounds;
            input_bounds.restrict_to(Bounds(mat));
            auto output_bounds = input_bounds;
            output_bounds.restrict_to(Bounds(output));
            input_bounds << output_bounds.size();
            
            if(output_bounds.size() != Size2(output))
                output.mul(cv::Scalar(0));
            
            cv::cvtColor(mat(input_bounds), output(output_bounds), cv::COLOR_RGBA2RGB);
            _recording_capture->write(output);
            
        } else {
            std::stringstream ss;
            ss << std::setw(6) << std::setfill('0') << _recording_frame.toStr() << "." << _recording_format.name();
            auto filename = _recording_path / ss.str();
            
            if(_recording_format == "jpg") {
                cv::Mat output;
                cv::cvtColor(mat, output, cv::COLOR_RGBA2RGB);
                if(!cv::imwrite(filename.str(), output, { cv::IMWRITE_JPEG_QUALITY, 100 })) {
                    FormatExcept("Cannot save to ",filename.str(),". Stopping recording.");
                    SETTING(gui_is_recording) = false;
                    _recording = false;
                }
                
            } else if(_recording_format == "png") {
                static std::vector<uchar> binary;
                static Image image;
                if(image.cols != (uint)mat.cols || image.rows != (uint)mat.rows)
                    image.create(mat.rows, mat.cols, 4);
                
                cv::Mat output = image.get();
                cv::cvtColor(mat, output, cv::COLOR_BGRA2RGBA);
                
                to_png(image, binary);
                
                FILE *f = fopen(filename.str().c_str(), "wb");
                if(f) {
                    fwrite(binary.data(), sizeof(char), binary.size(), f);
                    fclose(f);
                } else {
                    FormatExcept("Cannot write to ",filename.str(),". Stopping recording.");
                    SETTING(gui_is_recording) = false;
                    _recording = false;
                }
            }
        }
        
        static Timer last_print;
        if(last_print.elapsed() > 2) {
            DurationUS duration{static_cast<uint64_t>((_recording_frame.try_sub(_recording_start.valid() ? _recording_start : 0_f)).get() / float(SETTING(frame_rate).value<uint32_t>()) * 1000) * 1000};
            auto str = ("frame "+Meta::toStr(_recording_frame)+"/"+Meta::toStr(max_frame)+" length: "+Meta::toStr(duration));
            auto playback_speed = SETTING(gui_playback_speed).value<float>();
            if(playback_speed > 1) {
                duration.timestamp = timestamp_t(double(duration.timestamp) / double(playback_speed));
                str += " (real: "+Meta::toStr(duration)+")";
            }
            Print("[rec] ", str.c_str());
            last_print.reset();
        }
    }
    
    void stop_recording(Base* base, DrawStructure* graph) {
        if(!_recording)
            return;
            
        if(base)
            base->set_frame_recording(false);
        
        if(_recording_capture) {
            delete _recording_capture;
            _recording_capture = NULL;
            
            file::Path ffmpeg = SETTING(ffmpeg_path);
            if(!ffmpeg.empty() && graph) {
                file::Path save_path = _recording_path.replace_extension("mov");
                std::string cmd = ffmpeg.str()+" -i \""+_recording_path.str()+"\" -vcodec h264 -pix_fmt yuv420p -crf 15 -y \""+save_path.str()+"\"";
                
                graph->dialog([save_path, cmd](Dialog::Result result){
                    if(result == Dialog::OKAY) {
                        WorkProgress::add_queue("converting video...", [cmd=cmd, save_path=save_path](){
                            Print("Running ",cmd,"..");
                            if(system(cmd.c_str()) == 0)
                                Print("Saved video at ", save_path.str(),".");
                            else
                                FormatError("Cannot save video at ",save_path.str(),".");
                        });
                    }
                    
                }, "Do you want to convert it, using <str>"+cmd+"</str>?", "Recording finished", "Yes", "No");
            }
            
        } else {
            auto clip_name = std::string(_recording_path.filename());
            printf("ffmpeg -start_number %d -i \"%s/%%06d.%s\" -vcodec h264 -crf 13 -vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' -profile:v main -pix_fmt yuv420p \"%s.mp4\"\n", _recording_start.get(), _recording_path.str().c_str(), _recording_format.name(), clip_name.c_str());
        }
        
        _recording = false;
        SETTING(gui_is_recording) = false;
        _last_recording_frame.invalidate();
        
        file::Path p = _recording_path;
        DebugCallback("Stopped recording to ", p.absolute(), ".");
    }
    
    file::Path frame_output_dir() {
        auto filename = SETTING(filename).value<file::Path>();
        if(filename.has_extension("pv"))
            filename = filename.remove_extension();
        return file::DataLocation::parse("output", file::Path("frames") / (std::string)filename.filename());
    }
    
    void start_recording(Base* base, Frame_t frame) {
        if(!base)
            return;
        
        _recording_start = frame;
        _last_recording_frame = {};
        _recording = true;
        
        file::Path frames = frame_output_dir();
        if(!frames.exists()) {
            if(!frames.create_folder()) {
                FormatError("Cannot create folder ",frames.str(),". Cannot record.");
                _recording = false;
                return;
            }
        }
        
        std::string clip_prefix = (std::string)SETTING(filename).value<file::Path>().filename()+"_";
        
        size_t max_number = 0;
        try {
            for(auto &file : frames.find_files()) {
                auto name = std::string(file.filename());
                if(utils::beginsWith(name, clip_prefix)) {
                    try {
                        if(utils::endsWith(name, ".avi"))
                            name = name.substr(0, name.length() - 4);
                        auto number = Meta::fromStr<size_t>(name.substr(clip_prefix.length()));
                        if(number > max_number)
                            max_number = number;
                        
                    } catch(const std::exception& e) {
                        FormatExcept(name," not a number ('",e.what(),"').");
                    }
                }
            }
            
            ++max_number;
            
        } catch(const UtilsException& ex) {
            Print("Cannot iterate on folder ",frames.str(),". Defaulting to index 0.");
        }
        
        Print("Clip index is ", max_number,". Starting at frame ",frame,".");
        
        frames = frames / (clip_prefix + Meta::toStr(max_number));
        cv::Size size(base
                    && dynamic_cast<IMGUIBase*>(base)
                    ? static_cast<IMGUIBase*>(base)->real_dimensions()
                    : base->window_dimensions());
        
        using namespace default_config;
        auto format = SETTING(gui_recording_format).value<gui_recording_format_t::Class>();
        
        if(is_in(format, gui_recording_format_t::avi, gui_recording_format_t::mp4))
        {
            auto original_dims = size;
            if(size.width % 2 > 0)
                size.width -= size.width % 2;
            if(size.height % 2 > 0)
                size.height -= size.height % 2;
            Print("Trying to record with size ",size.width,"x",size.height," instead of ",original_dims.width,"x",original_dims.height," @ ",SETTING(frame_rate).value<uint32_t>());
            
            frames = frames.add_extension(format.toStr()).str();
            _recording_capture = new cv::VideoWriter{
                frames.str(),
                format == gui_recording_format_t::mp4
#ifdef __APPLE__
                    ? cv::VideoWriter::fourcc('a','v','c','1')
                    : cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
#else
                    ? cv::VideoWriter::fourcc('m', 'p', '4', 'v')
                    : cv::VideoWriter::fourcc('F', 'F', 'V', '1'),
#endif
                (double)SETTING(frame_rate).value<uint32_t>(), size, true};
            
            if(!_recording_capture->isOpened()) {
                FormatExcept("Cannot open video writer for path ",frames,". Please check file permissions, or try another format (`gui_recording_format`).");
                _recording = false;
                delete _recording_capture;
                _recording_capture = NULL;
                
                return;
            }
            
            _recording_capture->set(cv::VIDEOWRITER_PROP_QUALITY, 100);
            
        } else if(is_in(format, gui_recording_format_t::jpg, gui_recording_format_t::png))
        {
            if(!frames.exists()) {
                if(!frames.create_folder()) {
                    FormatError("Cannot create folder ",frames.str(),". Cannot record.");
                    _recording = false;
                    return;
                } else
                    Print("Created folder ", frames.str(),".");
            }
        }
        
        base->set_frame_recording(true);
        Print("Recording to ", frames,"... (",format.name(),")");
        
        _recording_size = size;
        _recording_path = frames;
        _recording_format = format;
        SETTING(gui_is_recording) = true;
    }
};

bool ScreenRecorder::recording() const {
    return _data->_recording.load();
}

ScreenRecorder::ScreenRecorder()
    : _data(new Data)
{
    
}

ScreenRecorder::~ScreenRecorder() {
    delete _data;
}

void ScreenRecorder::update_recording(Image::Ptr&& image, Base *base, Frame_t frame, Frame_t max_frame) {
    _data->do_record(std::move(image), base, frame, max_frame);
}

void ScreenRecorder::start_recording(Base*base, Frame_t frame) {
    _data->start_recording(base, frame);
    
    ((IMGUIBase*)base)->platform()->set_frame_buffer_receiver([this, base](Image::Ptr&& image){
        update_recording(std::move(image), base, Frame_t{}, Frame_t{});
    });
}

void ScreenRecorder::stop_recording(Base *base, DrawStructure *graph) {
    _data->stop_recording(base, graph);
}

void ScreenRecorder::set_frame(cmn::Frame_t frame) {
    _data->_recording_frame = frame;
}


}
