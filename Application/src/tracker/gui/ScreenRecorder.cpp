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

    struct CodecSpec {
        const char* tag;   // for logging / file suffix
        int fourcc;
    };

    inline std::vector<CodecSpec> codec_candidates(default_config::gui_recording_format_t::Class format) {
        using namespace default_config;

#ifdef __APPLE__
        if (format == gui_recording_format_t::mp4) {
            return {
                {"avc1", cv::VideoWriter::fourcc('a','v','c','1')}, // hardware H.264
                {"hvc1", cv::VideoWriter::fourcc('h','v','c','1')}, // HEVC (10.13+)
                {"mp4v", cv::VideoWriter::fourcc('m','p','4','v')}, // MPEG‑4 Visual
                {"MJPG", cv::VideoWriter::fourcc('M','J','P','G')}   // huge but safe
            };
        }
        // .avi or other intra‑frame container on macOS
        return {
            {"MJPG", cv::VideoWriter::fourcc('M','J','P','G')},
            {"FFV1", cv::VideoWriter::fourcc('F','F','V','1')}
        };
#else  // Linux / Windows (FFMPEG, MF, V4L2)
        if (format == gui_recording_format_t::mp4) {
            return {
                {"FFV1", cv::VideoWriter::fourcc('F','F','V','1')},
                {"avc1", cv::VideoWriter::fourcc('a','v','c','1')},
                {"H264", cv::VideoWriter::fourcc('H','2','6','4')},
                {"mp4v", cv::VideoWriter::fourcc('m','p','4','v')},
                {"MJPG", cv::VideoWriter::fourcc('M','J','P','G')},
                {"XVID", cv::VideoWriter::fourcc('X','V','I','D')}
            };
        }
        // .avi branch: intra‑frame or lossless first
        return {
            {"FFV1", cv::VideoWriter::fourcc('F','F','V','1')},
            {"avc1", cv::VideoWriter::fourcc('a','v','c','1')},
            {"H264", cv::VideoWriter::fourcc('H','2','6','4')},
            {"mp4v", cv::VideoWriter::fourcc('m','p','4','v')},
            {"MJPG", cv::VideoWriter::fourcc('M','J','P','G')},
            {"XVID", cv::VideoWriter::fourcc('X','V','I','D')}
        };
#endif
    }

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
            DurationUS duration{static_cast<uint64_t>((_recording_frame.try_sub(_recording_start.valid() ? _recording_start : 0_f)).get() / float(READ_SETTING(frame_rate, uint32_t)) * 1000) * 1000};
            auto str = ("frame "+Meta::toStr(_recording_frame)+"/"+Meta::toStr(max_frame)+" length: "+Meta::toStr(duration));
            auto playback_speed = READ_SETTING(gui_playback_speed, float);
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
        auto filename = READ_SETTING(filename, file::Path);
        if(filename.has_extension("pv"))
            filename = filename.remove_extension();
        return file::DataLocation::parse("output", file::Path("clips") / (std::string)filename.filename());
    }
    
    void start_recording(Base* base, Frame_t frame) {
        _recording_start = frame;
        _last_recording_frame = {};
        
        if (!base)
            return;

        _recording = true;

        file::Path frames = frame_output_dir();
        if (!frames.exists()) {
            if (!frames.create_folder()) {
                FormatError("Cannot create folder ", frames.str(), ". Cannot record.");
                _recording = false;
                return;
            }
        }

        // ---------------------------------------------------------------------
        // pick clip index
        // ---------------------------------------------------------------------
        std::string clip_prefix = (std::string)READ_SETTING(filename, file::Path).filename() + "_";
        size_t max_number = 0;
        try {
            for (auto& file : frames.find_files()) {
                auto name = std::string(file.filename());
                if (utils::beginsWith(name, clip_prefix)) {
                    try {
                        if (utils::endsWith(name, ".avi") || utils::endsWith(name, ".mp4"))
                            name = name.substr(0, name.length() - 4);
                        auto number = Meta::fromStr<size_t>(name.substr(clip_prefix.length()));
                        max_number = std::max(max_number, number);
                    }
                    catch (const std::exception& e) {
                        FormatExcept(name, " not a number ('", e.what(), "').");
                    }
                }
            }
            ++max_number;
        }
        catch (const UtilsException&) {
            Print("Cannot iterate on folder ", frames.str(), ". Defaulting to index 0.");
        }

        Print("Clip index is ", max_number, ". Starting at frame ", frame, ".");

        // base path without extension yet
        frames = frames / (clip_prefix + Meta::toStr(max_number));

        cv::Size size(base && dynamic_cast<IMGUIBase*>(base)
            ? static_cast<IMGUIBase*>(base)->real_dimensions()
            : base->window_dimensions());

        // ---------------------------------------------------------------------
        // enforce even dimensions (required by many codecs)
        // ---------------------------------------------------------------------
        const auto original_dims = size;
        size.width &= ~1;   // drop LSB if odd
        size.height &= ~1;
        if (size != original_dims) {
            Print("Trying to record with size ", size.width, "x", size.height, " instead of ",
                original_dims.width, "x", original_dims.height, " @ ",
                READ_SETTING(frame_rate, uint32_t));
        }

        using namespace default_config;
        auto format = READ_SETTING(gui_recording_format, gui_recording_format_t::Class);

        // ---------------------------------------------------------------------
        // VIDEO branch (mp4 / avi) with codec fallback loop
        // ---------------------------------------------------------------------
        if (is_in(format, gui_recording_format_t::avi, gui_recording_format_t::mp4)) {

            bool opened = false;
            std::string final_file_path;
            const uint32_t fps = READ_SETTING(frame_rate, uint32_t);

            for (const auto& c : codec_candidates(format)) {
                std::ostringstream out_name;
                out_name << frames.str() << '.' << format.toStr();
                const std::string out_path = out_name.str();

                Print("Trying codec ", c.tag, " -> ", out_path);

                delete _recording_capture;                 // clear any prior attempt
                _recording_capture = new cv::VideoWriter{
                    out_path,
                    c.fourcc,
                    static_cast<double>(fps),
                    size,
                    true
                };

                if (_recording_capture->isOpened()) {
                    _recording_capture->set(cv::VIDEOWRITER_PROP_QUALITY, 100);
                    opened = true;
                    final_file_path = out_path;
                    break;
                }
            }

            if (!opened) {
                FormatExcept("Cannot open a VideoWriter for any tested codec in ", frames.str(),
                    ". Please check your codec install or choose another container/format.");
                _recording = false;
                delete _recording_capture;
                _recording_capture = nullptr;
                return;
            }

            // convert back into file::Path for the rest of the pipeline
            frames = final_file_path;

        }
        else if (is_in(format, gui_recording_format_t::jpg, gui_recording_format_t::png)) {
            // -----------------------------------------------------------------
            // IMAGE‑SEQUENCE branch (creates a folder of numbered stills)
            // -----------------------------------------------------------------
            if (!frames.exists()) {
                if (!frames.create_folder()) {
                    FormatError("Cannot create folder ", frames.str(), ". Cannot record.");
                    _recording = false;
                    return;
                }
                else {
                    Print("Created folder ", frames.str(), ".");
                }
            }
        }

        // ---------------------------------------------------------------------
        // Success — flip flag & remember parameters
        // ---------------------------------------------------------------------
        base->set_frame_recording(true);
        Print("Recording to ", frames, "... (", format.name(), ")");

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
    
    Frame_t video_length;
    if(GlobalSettings::is_type<uint64_t>("video_length"))
    {
        video_length = Frame_t(READ_SETTING(video_length, uint64_t));
    }
    
    std::function<Frame_t()> current_frame;
    
    if(GlobalSettings::is_type<Frame_t>("gui_displayed_frame"))
    {
        current_frame = [](){
            return READ_SETTING(gui_displayed_frame, Frame_t);
        };
    } else {
        current_frame = [](){
            return Frame_t{};
        };
    }
    
    ((IMGUIBase*)base)->platform()->set_frame_buffer_receiver([this, base, video_length, current_frame](Image::Ptr&& image){
        update_recording(std::move(image), base, current_frame(), video_length);
    });
}

void ScreenRecorder::stop_recording(Base *base, DrawStructure *graph) {
    _data->stop_recording(base, graph);
}

void ScreenRecorder::set_frame(cmn::Frame_t frame) {
    _data->_recording_frame = frame;
}


}
