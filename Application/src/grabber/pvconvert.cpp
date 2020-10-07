#include <pv.h>
#include <gif.h>
#include <iomanip>
#include <file/Path.h>
#include <misc/GlobalSettings.h>

using namespace file;

ENUM_CLASS(Arguments,
           i,input,o,output,d,dir, settings, start, end, as_gif, step, scale, disable_background)

int main(int argc, char**argv) {
    pv::DataLocation::register_path("settings", [](file::Path path) -> file::Path {
        using namespace file;
        auto settings_file = path.str().empty() ? SETTING(settings_file).value<Path>() : path;
        if(settings_file.empty())
            U_EXCEPTION("settings_file is an empty string.");
        
        if(!settings_file.is_absolute()) {
            settings_file = SETTING(output_dir).value<file::Path>() / settings_file;
        }
        
        if(!settings_file.has_extension() || settings_file.extension().to_string() != "settings")
            settings_file = settings_file.add_extension("settings");
        
        return settings_file;
    });
    
    SETTING(crop_offsets) = CropOffsets();
    SETTING(print_framenr) = true;
    SETTING(filename) = Path("guy_video");
    SETTING(settings_file) = Path();
    SETTING(output_dir) = Path("");
    SETTING(start_frame) = long_t(0);
    SETTING(end_frame) = long_t(-1);
    SETTING(disable_background) = false;
    SETTING(as_gif) = false;
    SETTING(step) = int(1);
    SETTING(scale) = float(1.0);
    SETTING(use_differences) = false;
    SETTING(crop) = cv::Rect2f(0, 0, 1, 1);
    
    GlobalSettings::map().set_do_print(true);
    
    const char *argptr = argv[0];
    
    for (int i=1; i<argc; i++) {
        
        if (argv[i][0] == '-') {
            argptr = argv[i]+1;
            
        } else if(Arguments::has(argptr)) {
            switch (Arguments::get(argptr)) {
                case Arguments::i:
                case Arguments::input:
                    SETTING(filename) = Path(argv[i]);
                    break;
                    
                case Arguments::settings:
                    SETTING(settings_file) = Path(argv[i]);
                    break;
                    
                case Arguments::disable_background:
                    SETTING(disable_background) = std::string(argv[i]) == "true" ? true : false;
                    break;
                    
                case Arguments::o:
                case Arguments::output:
                    SETTING(output_dir) = Path(argv[i]);
                    break;
                    
                case Arguments::start:
                    SETTING(start_frame) = (long_t)std::stol(std::string(argv[i]));
                    break;
                    
                case Arguments::end:
                    SETTING(end_frame) = (long_t)std::stol(std::string(argv[i]));
                    break;
                    
                case Arguments::as_gif:
                    SETTING(as_gif) = std::string(argv[i]) == "true" ? true : false;
                    break;
                    
                case Arguments::step:
                    SETTING(step) = std::stoi(std::string(argv[i]));
                    break;
                    
                case Arguments::scale:
                    SETTING(scale) = std::stof(std::string(argv[i]));
                    break;
                    
                case Arguments::d:
                case Arguments::dir:
                    SETTING(output_dir) = Path(argv[i]);
                    break;
                    
                default:
                    Warning("Unknown option '%s' with value '%s'", argptr, argv[i]);
                    break;
            }
        }
    }
    
    if(!SETTING(settings_file).value<Path>().empty()) {
        if(SETTING(settings_file).value<Path>().exists()) {
            GlobalSettings::load_from_file({}, SETTING(settings_file), AccessLevelType::STARTUP);
        } else
            U_EXCEPTION("Cannot find settings file '%S'", &SETTING(settings_file).value<Path>().str());
    }
    
    Path output_dir = SETTING(output_dir);
    if(output_dir.empty())
        output_dir = ".";
    
    Path input = SETTING(filename);
    if(input.remove_filename().empty())
        input = output_dir/input;
    
    Debug("Input: '%S'", &input);
    Debug("Output to: '%S'", &output_dir);
    
    pv::File video(input);
    video.start_reading();
    
    /**
     * Try to load Settings from the command-line that have been
     * ignored previously.
     */
    argptr = argv[0];
    for (int i=1; i<argc; i++) {
        if(argv[i][0] == '-') {
            argptr = argv[i]+1;
        } else if(!Arguments::has(argptr)) {
            auto keys = GlobalSettings::map().keys();
            if(contains(keys, std::string(argptr))) {
                Debug("Setting option '%s' to value '%s'", argptr, argv[i]);
                
                sprite::parse_values(GlobalSettings::map(), "{'"+std::string(argptr)+"':"+std::string(argv[i])+"}");
            }
        }
    }
    
    if(SETTING(end_frame).value<long_t>() == -1) {
        SETTING(end_frame).value<long_t>() = video.length() - 1;
    }
    
    long_t start_frame = SETTING(start_frame),
         end_frame = SETTING(end_frame);
    
    gpuMat average;
    video.average().copyTo(average);
    if(average.cols == video.size().width && average.rows == video.size().height)
        video.processImage(average, average);
    
    //cv::imshow("average", average);
    //cv::waitKey(1);
    
    long_t frame_index = start_frame;
    const long_t step = SETTING(step).value<int>();
    
    GifWriter *writer = NULL;
    pv::Frame current_frame;
    video.read_frame(current_frame, frame_index);
    
    auto prev_time = current_frame.timestamp();
    
    float framerate;
    {
        video.read_frame(current_frame, frame_index+1);
        framerate = 1000.f / ((current_frame.timestamp() - prev_time) / 1000.f);
    }
    
    const bool as_gif = SETTING(as_gif);
    if(as_gif)
        Debug("Will export as gif from %ld to %ld (step %ld).", start_frame, end_frame, step);
    
    Debug("Press ENTER to continue...");
    getc(stdin);
    
    cv::Rect2f tmp = SETTING(crop);
    cv::Rect2i crop_rect(tmp.x * video.size().width, tmp.y * video.size().height,
                         tmp.width * video.size().width, tmp.height * video.size().height);
    
    if(SETTING(as_gif)) {
        std::stringstream ss;
        ss << output_dir / "animated_frames.gif";
        
        writer = new GifWriter();
        video.read_frame(current_frame,
                         frame_index+step);
        GifBegin(writer, ss.str().c_str(), crop_rect.width * SETTING(scale).value<float>(), crop_rect.height * SETTING(scale).value<float>(), 0);//(current_frame.timestamp()-prev_time)/1000.0);
    }
    
    bool with_background
    = !SETTING(disable_background);
    while(frame_index < (long_t)video.header().num_frames && frame_index < end_frame) {
        cv::Mat image;
        video.read_frame(current_frame, frame_index);
        video.frame_optional_background(frame_index, image, with_background);
        
        if(SETTING(print_framenr)) {
            std::stringstream ss;
            ss << "frame " << frame_index;
            cv::putText(image, ss.str(), cv::Point(10, 10), cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(0));
        }
        
        image = image(crop_rect);
        
        if(SETTING(scale).value<float>() != 1.0) {
            resize_image(image, SETTING(scale).value<float>());
        }
        
        if(SETTING(as_gif)) {
            cv::cvtColor(image, image, cv::COLOR_GRAY2RGBA);
            //Debug("Delta %f", (current_frame.timestamp()-prev_time)/1000.0);
            GifWriteFrame(writer, image.data, image.cols, image.rows, 0);//(current_frame.timestamp()-prev_time)/1000.0);//;
            
        } else {
            std::stringstream ss;
            ss << "frame" << std::setw(7) << std::setfill('0') << frame_index << ".jpg";
            
            if(!output_dir.exists()) {
                if(output_dir.create_folder())
                    Debug("Created folder '%S'.", &output_dir.str());
                else
                    U_EXCEPTION("Cannot create folder '%S'. No write permissions?", &output_dir.str());
            }

            
            file::Path path = output_dir / ss.str();
            if(!cv::imwrite(path.str(), image, { cv::IMWRITE_JPEG_QUALITY, 100 } ))
                U_EXCEPTION("Cannot write to '%S'. No write permissions?", &path.str());
        }
        
        if(frame_index%50 == 0) {
            //cv::imshow("preview", image);
            //cv::waitKey(1);
            Debug("Frame %d/%d", frame_index, end_frame);
        }
        
        prev_time = current_frame.timestamp();
        frame_index += step;
    }
    
    if(writer) {
        GifEnd(writer);
        delete writer;
    } else {
        std::stringstream ss;
        ss << output_dir << "frame";
        std::string file = ss.str();
        
        Debug("For conversion using ffmpeg try this command:");
        printf("\tffmpeg -framerate %d -start_number %d -i %s/frame%%07d.jpg -vcodec libx264 -vf \"fps=60,format=yuv420p\" output.mp4\n", (int)framerate, start_frame, output_dir.str().c_str());
    }
}
