#include <pv.h>
#include <gif.h>
#include <iomanip>
#include <file/Path.h>
#include <misc/GlobalSettings.h>
#include <misc/CommandLine.h>
#include <file/DataLocation.h>
#include <tracker/misc/default_config.h>

using namespace cmn;
using namespace cmn::file;

ENUM_CLASS(Arguments,
           h,
           i,input,o,output,d,dir, s,settings, start, end, as_gif, step, scale, disable_background)

struct bid {
    static constexpr uint32_t invalid = std::numeric_limits<uint32_t>::max();
    uint32_t _id;
    bid() = default;
    constexpr bid(uint32_t v) : _id(v) {}
    
    explicit constexpr operator uint32_t() const {
        return _id;
    }
    explicit constexpr operator int64_t() const { return static_cast<int64_t>(_id); }
    explicit constexpr operator uint64_t() const { return static_cast<uint64_t>(_id); }
    constexpr bool operator==(const bid& other) const {
        return other._id == _id;
    }
    constexpr bool operator!=(const bid& other) const {
        return other._id != _id;
    }
    //constexpr bid(uint32_t b) : _id(b) {}
    constexpr bool valid() const { return _id != invalid; }
    
    constexpr bool operator<(const bid& other) const {
        return _id < other._id;
    }
    constexpr bool operator>(const bid& other) const {
        return _id > other._id;
    }
    
    constexpr bool operator<=(const bid& other) const {
        return _id <= other._id;
    }
    constexpr bool operator>=(const bid& other) const {
        return _id >= other._id;
    }
    
    std::string toStr() const {
        return cmn::Meta::toStr<uint32_t>(_id);
    }
    static std::string class_name() { return "pv::bid"; }
    static bid fromStr(const std::string& str) {
        return bid(cmn::Meta::fromStr<uint32_t>(str));
    }

    static constexpr uint32_t from_data(ushort x0, ushort x1, ushort y0, uint8_t N) {
        assert((uint32_t)x0 < (uint32_t)4096u);
        assert((uint32_t)x1 < (uint32_t)4096u);
        assert((uint32_t)y0 < (uint32_t)4096u);
        
        return (uint32_t(x0 + (x1 - x0) / 2) << 20)
                | ((uint32_t(y0) & 0x00000FFF) << 8)
                |  (uint32_t(N)  & 0x000000FF);
    }
    
    constexpr cmn::Vec2 calc_position() const {
        auto x = (_id >> 20) & 0x00000FFF;
        auto y = (_id >> 8) & 0x00000FFF;
        //auto N = id & 0x000000FF;
        return cmn::Vec2(x, y);
    }
    //static uint32_t id_from_position(const cmn::Vec2&);
    static bid from_blob(const pv::Blob& blob);
    static bid from_blob(const pv::CompressedBlob& blob);
};
                         
 bid bid::from_blob(const pv::Blob& blob) {
     if(!blob.lines() || blob.lines()->empty())
         return bid::invalid;
     
     return from_data(blob.lines()->front().x0,
                      blob.lines()->front().x1,
                      blob.lines()->front().y,
                      blob.lines()->size());
     //return bid::invalid;
 }
 bid bid::from_blob(const pv::CompressedBlob& blob) {
     if(blob.lines().empty())
         return bid::invalid;
     
     return from_data(blob.lines().front().x0(),
                      blob.lines().front().x1(),
                      blob.start_y,
                      blob.lines().size());
     //return bid::invalid;
 }

int main(int argc, char**argv) {
    static_assert(std::is_trivial<bid>::value, "pv::bid has to be trivial.");
    //static_assert(std::is_trivial<pv::bid>::value, "pv::bid has to be trivial.");
    static_assert(std::is_standard_layout<pv::bid>::value, "pv::bid has to be standard layout.");
    
    const char* locale = "C";
    std::locale::global(std::locale(locale));
    
    file::DataLocation::register_path("settings", [](const sprite::Map& map, file::Path path) -> file::Path {
        using namespace file;
        auto settings_file = path.str().empty() ? map.at("settings_file").value<Path>() : path;
        if(settings_file.empty())
            throw U_EXCEPTION("settings_file is an empty string.");
        
        if(!settings_file.is_absolute()) {
            settings_file = map.at("output_dir").value<file::Path>() / settings_file;
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
    
    GlobalSettings::map().set_print_by_default(true);
    
    CommandLine::init(argc, argv, true);
    
    /**
     * Try to load Settings from the command-line that have been
     * ignored previously.
     */
    auto &cmd = CommandLine::instance();
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
                    if(option.value)
                        SETTING(filename) = Path(*option.value);
                    break;
                    
                case Arguments::s:
                case Arguments::settings:
                    if(option.value)
                        SETTING(settings_file) = Path(*option.value);
                    break;
                    
                case Arguments::o:
                case Arguments::output:
                    if(option.value)
                        SETTING(output_dir) = Path(*option.value);
                    break;
                    
                case Arguments::d:
                case Arguments::dir:
                    if(option.value)
                        SETTING(output_dir) = Path(*option.value);
                    break;
                    
                default:
                    FormatWarning("Unknown option ", option.name," with value ",option.value);
                    break;
            }
        }
    }
    
    if(!SETTING(settings_file).value<Path>().empty()) {
        if(SETTING(settings_file).value<Path>().exists()) {
            GlobalSettings::load_from_file(SETTING(settings_file).value<file::Path>().str(), {
                .deprecations = default_config::deprecations(),
                .access = AccessLevelType::STARTUP
            });
            //GlobalSettings::load_from_file({}, SETTING(settings_file), AccessLevelType::STARTUP);
        } else
            throw U_EXCEPTION("Cannot find settings file ",SETTING(settings_file).value<Path>().str());
    }
    
    Path output_dir = SETTING(output_dir);
    if(output_dir.empty())
        output_dir = ".";
    
    Path input = SETTING(filename);
    if(input.remove_filename().empty())
        input = output_dir/input;
    
    Print("Input: ",input);
    Print("Output to: ",output_dir);
    
    auto video = pv::File::Read(input);

    if(SETTING(end_frame).value<long_t>() == -1) {
        SETTING(end_frame) = long_t(video.length().get() - 1);
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
    video.read_frame(current_frame, Frame_t(frame_index));
    
    auto prev_time = current_frame.timestamp();
    
    float framerate;
    {
        video.read_frame(current_frame, Frame_t(frame_index+1));
        framerate = 1000.f / (double(current_frame.timestamp() - prev_time) / 1000.f);
    }
    
    const bool as_gif = SETTING(as_gif);
    if(as_gif)
        Print("Will export as gif from ", start_frame," to ", end_frame," (step ",step,").");
    
    Print("Press ENTER to continue...");
    getc(stdin);
    
    CropOffsets tmp = SETTING(crop);
    auto crop_rect = tmp.toPixels(video.size());
    
    if(SETTING(as_gif)) {
        std::stringstream ss;
        ss << output_dir / "animated_frames.gif";
        
        writer = new GifWriter();
        video.read_frame(current_frame,
                         Frame_t(frame_index+step));
        GifBegin(writer, ss.str().c_str(), crop_rect.width * SETTING(scale).value<float>(), crop_rect.height * SETTING(scale).value<float>(), 1);//(current_frame.timestamp()-prev_time)/1000.0);
    }
    
    bool with_background
    = !SETTING(disable_background);
    while(frame_index < (long_t)video.header().num_frames && frame_index < end_frame) {
        cv::Mat image;
        video.read_frame(current_frame, Frame_t(frame_index));
        video.frame_optional_background(Frame_t(frame_index), image, with_background);
        
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
                    Print("Created folder ", output_dir.str(),".");
                else
                    throw U_EXCEPTION("Cannot create folder ",output_dir.str(),". No write permissions?");
            }

            
            file::Path path = output_dir / ss.str();
            if(!cv::imwrite(path.str(), image, { cv::IMWRITE_JPEG_QUALITY, 100 } ))
                throw U_EXCEPTION("Cannot write to ",path.str(),". No write permissions?");
        }
        
        if(frame_index%50 == 0) {
            //cv::imshow("preview", image);
            //cv::waitKey(1);
            Print("Frame ", frame_index,"/",end_frame);
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
        
        Print("For conversion using ffmpeg try this command:");
        printf("\tffmpeg -framerate %d -start_number %d -i %s/frame%%07d.jpg -vcodec h264 -vf \"fps=60,format=yuv420p\" output.mp4\n", (int)framerate, start_frame, output_dir.str().c_str());
    }
}
