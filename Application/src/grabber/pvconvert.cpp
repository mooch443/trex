#include <pv.h>
#include <gif.h>
#include <iomanip>
#include <file/Path.h>
#include <misc/GlobalSettings.h>
#include <misc/CommandLine.h>

using namespace file;

ENUM_CLASS(Arguments,
           h,
           i,input,o,output,d,dir, s,settings, start, end, as_gif, step, scale, disable_background)

int main(int argc, char**argv) {
    pv::DataLocation::register_path("settings", [](file::Path path) -> file::Path {
        using namespace file;
        auto settings_file = path.str().empty() ? SETTING(settings_file).value<Path>() : path;
        if(settings_file.empty())
            U_EXCEPTION("settings_file is an empty string.");
        
        if(!settings_file.is_absolute()) {
            settings_file = SETTING(output_dir).value<file::Path>() / settings_file;
        }
        
        if(!settings_file.has_extension() || settings_file.extension() != "settings")
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
    SETTING(crop) = CropOffsets();
    
    GlobalSettings::map().set_do_print(true);
    
    CommandLine cmd(argc, argv, true);
    
    /**
     * Try to load Settings from the command-line that have been
     * ignored previously.
     */
    cmd.load_settings();
    
    for(auto &option : cmd) {
        if(Arguments::has(option.name)) {
            switch (Arguments::get(option.name)) {
                case Arguments::h:
                    printf("pvconvert\n");
                    printf("   -i /path/to/pv/file\t\t\tfull path to a .pv file\n");
                    printf("   -o /path/to/output/folder\tthis is where images will be saved\n");
                    printf("   -s /path/to/settings\t\t\tadditional settings in .settings format\n");
                    printf("   -disable_background\t\t\texport objects on black background\n");
                    printf("   -start_frame index\t\t\tstart conversion from here\n");
                    printf("   -end_frame index\t\t\t\tend conversion here\n");
                    printf("   -step N\t\t\t\t\t\tonly save every Nth frame\n");
                    printf("   -as_gif\t\t\t\t\t\tdon't save as images, save as a single gif\n");
                    printf("   -scale S\t\t\t\t\t\tscale results by a factor of S (0-1)\n");
                    printf("   -crop [l,t,r,b]\t\t\t\tcrop percentages from left/top/right/bottom\n");
                    return 0;
                    
                case Arguments::i:
                case Arguments::input:
                    SETTING(filename) = Path(option.value);
                    break;
                    
                case Arguments::s:
                case Arguments::settings:
                    SETTING(settings_file) = Path(option.value);
                    break;
                    
                case Arguments::o:
                case Arguments::output:
                    SETTING(output_dir) = Path(option.value);
                    break;
                    
                case Arguments::d:
                case Arguments::dir:
                    SETTING(output_dir) = Path(option.value);
                    break;
                    
                default:
                    Warning("Unknown option '%S' with value '%S'", &option.name, &option.value);
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
    
    if(SETTING(end_frame).value<long_t>() == -1) {
        SETTING(end_frame).value<long_t>() = video.length() - 1;
    }
    
    long_t start_frame = SETTING(start_frame),
         end_frame = SETTING(end_frame);
    
    gpuMat average;
    video.average().copyTo(average);
    if(average.cols == video.size().width && average.rows == video.size().height)
        video.processImage(average, average);
    
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
    
    print("Press ENTER to continue...");
    getc(stdin);
    
    CropOffsets tmp = SETTING(crop);
    auto crop_rect = tmp.toPixels(video.size());
    
    if(SETTING(as_gif)) {
        std::stringstream ss;
        ss << output_dir / "animated_frames.gif";
        
        writer = new GifWriter();
        video.read_frame(current_frame,
                         frame_index+step);
        GifBegin(writer, ss.str().c_str(), crop_rect.width * SETTING(scale).value<float>(), crop_rect.height * SETTING(scale).value<float>(), 1);//(current_frame.timestamp()-prev_time)/1000.0);
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
            GifWriteFrame(writer, image.data, image.cols, image.rows, 0);//(current_frame.timestamp()-prev_time)/1000.0);//;
            
        } else {
            std::stringstream ss;
            ss << "frame" << std::setw(7) << std::setfill('0') << frame_index << ".jpg";
            
            if(!output_dir.exists()) {
                if(output_dir.create_folder())
                    print("Created folder ", output_dir.str(),".");
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
        
        print("For conversion using ffmpeg try this command:");
        printf("\tffmpeg -framerate %d -start_number %d -i %s/frame%%07d.jpg -vcodec h264 -vf \"fps=60,format=yuv420p\" output.mp4\n", (int)framerate, start_frame, output_dir.str().c_str());
    }
}
