#include <commons.pc.h>
#include <gui/DrawStructure.h>
#include <gui/IMGUIBase.h>
#include <gui/SFLoop.h>
#include <gui/types/Button.h>
#include <video/VideoSource.h>
#include <opencv2/dnn.hpp>
#include <pv.h>
#include <python/GPURecognition.h>
#include <tracking/PythonWrapper.h>
#include <misc/CommandLine.h>
#include <file/DataLocation.h>
#include <misc/default_config.h>
#include <tracking/Tracker.h>
#include <tracking/IndividualManager.h>
#include <misc/PixelTree.h>
#include <gui/Timeline.h>
#include <gui/GUICache.h>
#include <gui/types/Dropdown.h>
#include <gui/types/Textfield.h>
#include <gui/types/List.h>

using namespace cmn;

struct TileImage;

template<typename T>
concept overlay_function = requires {
    requires std::invocable<T, TileImage&&>;
    //{ std::invoke_result<T, const Image&>::type } -> std::convertible_to<Image::UPtr>;
};

static Size2 expected_size(1024, 1024);
using namespace gui;

struct SegmentationData {
    Image::UPtr image;
    pv::Frame frame;
    std::vector<Bounds> tiles;
    
    struct Assignment {
        size_t clid;
        float p;
    };
    
    std::map<pv::bid, Assignment> predictions;
    std::vector<std::vector<Vec2>> outlines;
    
    operator bool() const {
        return image != nullptr;
    }
};

struct TileImage {
    Size2 tile_size;
    Image::UPtr original;
    std::vector<Image::UPtr> images;
    inline static gpuMat resized, converted, thresholded;
    inline static cv::Mat download_buffer;
    std::vector<Vec2> _offsets;
    Size2 source_size, original_size;
    
    TileImage() = default;
    TileImage(TileImage&&) = default;
    TileImage(const TileImage&) = delete;
    
    TileImage& operator=(TileImage&&) = default;
    TileImage& operator=(const TileImage&) = delete;
    
    TileImage(const gpuMat& source, Image::UPtr&& original, Size2 tile_size, Size2 original_size)
        : tile_size(tile_size),
          original(std::move(original)),
          source_size(source.cols, source.rows),
          original_size(original_size)
    {
        
        if(tile_size.width == source.cols
           && tile_size.height == source.rows)
        {
            source_size = tile_size;
            images.emplace_back(Image::Make(source));
            _offsets = {Vec2()};
        }
        else if(tile_size.width > source.cols
             || tile_size.height > source.rows)
        {
            source_size = tile_size;
            cv::resize(source, resized, tile_size);
            images.emplace_back(Image::Make(resized));
            _offsets = {Vec2()};
            
        } else {
            gpuMat tile = gpuMat::zeros(tile_size.height, tile_size.width, CV_8UC4);
            for(int y = 0; y < source.rows; y += tile_size.height) {
                for(int x = 0; x < source.cols; x += tile_size.width) {
                    Bounds bds = Bounds(x, y, tile_size.width, tile_size.height);
                    _offsets.push_back(Vec2(x, y));
                    bds.restrict_to(Bounds(0, 0, source.cols, source.rows));
                    
                    source(bds).copyTo(tile(Bounds{bds.size()}));
                    images.emplace_back(Image::Make(tile));
                    tile.setTo(0);
                }
            }
        }
        
        //print("Tiling image originally ", this->original->dimensions(), " to ", tile_size, " producing: ", offsets(), " (original_size=", original_size,")");
    }
    
    operator bool() const {
        return not images.empty();
    }
    
    std::vector<Vec2> offsets() const {
        return _offsets;
    }
};

ENUM_CLASS(ObjectDetectionType, yolo7, yolo7seg);

template<typename T>
concept ObjectDetection = requires (TileImage tiled, SegmentationData data) {
    { T::apply(std::move(tiled)) } -> std::convertible_to<tl::expected<SegmentationData, const char*>>;
    //{ T::receive(data, Vec2{}, {}) };
};

struct Yolo7ObjectDetection {
    Yolo7ObjectDetection() = delete;
    
    static void init() {
        Python::schedule([](){
            using py = track::PythonIntegration;
            py::ModuleProxy proxy{"bbx_saved_model"};
            proxy.set_variable("model_type", "yolo7");
            proxy.set_variable("model_path", SETTING(model).value<file::Path>().str());
            proxy.set_variable("image_size", int(expected_size.width));
            proxy.run("load_model");
            
        }).get();
    }
    
    static void receive(SegmentationData& data, Vec2 scale_factor, const std::vector<float>& vector) {
        for(size_t i=0; i<vector.size(); i+=4+2) {
            float conf = vector.at(i);
            float cls = vector.at(i+1);
            
            if (SETTING(filter_class).value<bool>() && cls != 1 && cls != 0)
                continue;
            
            Vec2 pos = Vec2(vector.at(i+2), vector.at(i+3));
            Size2 dim = Size2(vector.at(i+4) - pos.x, vector.at(i+5) - pos.y).mul(scale_factor);
            pos = pos.mul(scale_factor);
            
            std::vector<HorizontalLine> lines;
            std::vector<uchar> pixels;
            for(int y = pos.y; y < pos.y + dim.height; ++y) {
                if(y < 0 || y >= data.image->rows)
                    continue;
                
                HorizontalLine line{
                    (coord_t)saturate(int(y), int(0), int(y + dim.height - 1)),
                    (coord_t)saturate(int(pos.x), int(0), int(pos.x + dim.width - 1)),
                    (coord_t)saturate(int(pos.x + dim.width), int(0), int(pos.x + dim.width - 1))
                };
                for(int x = line.x0; x <= line.x1; ++x) {
                    pixels.emplace_back(data.image->get().at<cv::Vec3b>(y, x)[0]);
                }
                //pixels.insert(pixels.end(), (uchar*)mat.ptr(y, line.x0),
                //              (uchar*)mat.ptr(y, line.x1));
                lines.emplace_back(std::move(line));
            }
            
            if(not lines.empty()) {
                pv::Blob blob(lines, 0);
                data.predictions[blob.blob_id()] = { .clid = size_t(cls), .p = float(conf) };
                data.frame.add_object(lines, pixels, 0);
            }
        }
    }
    
    static tl::expected<SegmentationData, const char*> apply(TileImage&& tiled) {
        namespace py = Python;
        
        SegmentationData data{
            .image = std::move(tiled.original)
        };
        
        static Frame_t running_id = 0_f;
        auto fake = double(running_id.get()) / double(FAST_SETTING(frame_rate)) * 1000.0 * 1000.0;
        data.frame.set_timestamp(uint64_t(fake));
        data.frame.set_index(running_id++);
        
        Vec2 scale = SETTING(output_size).value<Size2>().div(tiled.source_size);
        //print("Image scale: ", scale, " with tile source=", tiled.source_size, " image=", data.image->dimensions()," output_size=", SETTING(output_size).value<Size2>(), " original=", tiled.original_size);
        
        py::schedule([&data, scale, offsets = tiled.offsets(), images = std::move(tiled.images)]() mutable {
            using py = track::PythonIntegration;
            py::ModuleProxy bbx("bbx_saved_model");
            bbx.set_variable("offsets", std::move(offsets));
            bbx.set_variable("image", std::move(images));
            
            bbx.set_function("receive", [&](std::vector<float> vector) {
                receive(data, scale, vector);
            });

            try {
                bbx.run("apply");
            }
            catch (...) {
                FormatWarning("Continue after exception...");
                throw;
            }
            
            bbx.unset_function("receive");
            
        }).get();
        
        //tf::imshow("test", ret);
        return data;//Image::Make(ret);
    }
};

struct Yolo7InstanceSegmentation {
    Yolo7InstanceSegmentation() = delete;
    
    static void init() {
        Python::schedule([](){
            using py = track::PythonIntegration;
            py::ModuleProxy proxy{"bbx_saved_model"};
            proxy.set_variable("model_type", "yolo7-seg");
            proxy.set_variable("model_path", SETTING(model).value<file::Path>().str());
            proxy.set_variable("image_size", int(expected_size.width));
            proxy.run("load_model");
            
        }).get();
    }
    
    static void receive(std::vector<Vec2> offsets, SegmentationData& data, Vec2 scale_factor, std::vector<float>& masks, const std::vector<float>& vector, const std::vector<int>& meta) {
        //print(vector);
        size_t N = vector.size() / 6u;
        
        cv::Mat full_image;
        cv::cvtColor(data.image->get(), full_image, cv::COLOR_RGB2GRAY);
        
        for (size_t i = 0; i < N; ++i) {
            Vec2 pos(vector.at(i * 6 + 0), vector.at(i * 6 + 1));
            Size2 dim(vector.at(i * 6 + 2) - pos.x, vector.at(i * 6 + 3) - pos.y);
            
            float conf = vector.at(i * 6 + 4);
            float cls = vector.at(i * 6 + 5);
            
            pos += offsets.at(meta.at(i));
            
            pos = pos.mul(scale_factor);
            dim = dim.mul(scale_factor);
            
            print(i, vector.at(i*6 + 0), " ", vector.at(i*6 + 1), " ",vector.at(i*6 + 2), " ", vector.at(i*6 + 3));
            print("\t->", conf, " ", cls, " ",pos, " ", dim);
            print("\tmeta of object = ", meta.at(i), " offset=", offsets.at(meta.at(i)));
            cls = meta.at(i);
            
            if (SETTING(filter_class).value<bool>() && cls != 14)
                continue;
            if (dim.min() < 1)
                continue;
            
            //if(dim.height + pos.y > full_image.rows
            //   || dim.width + pos.x > full_image.cols)
            //    continue;
            
            cv::Mat m(56, 56, CV_32FC1, masks.data() + i * 56 * 56);
            
            cv::Mat tmp;
            cv::resize(m, tmp, dim);
            
            //cv::Mat dani;
            //tmp.convertTo(dani, CV_8UC1, 255.0);
            //tf::imshow("dani", dani);
            
            cv::threshold(tmp, tmp, 0.6, 1.0, cv::THRESH_BINARY);
            //cv::threshold(tmp, t, 150, 255, cv::THRESH_BINARY);
            //print(Bounds(pos, dim), " and image ", Size2(full_image), " and t ", Size2(t));
            //print("using bounds: ", Size2(full_image(Bounds(pos, dim))), " and ", Size2(t));
            //print("channels: ", full_image.channels(), " and ", t.channels(), " and types ", getImgType(full_image.type()), " ", getImgType(t.type()));
            cv::Mat d;// = full_image(Bounds(pos, dim));
            auto restricted = Bounds(pos, dim);
            restricted.restrict_to(Bounds(full_image));
            if(restricted.width <= 0 || restricted.height <= 0)
                continue;
            
            full_image(restricted).convertTo(d, CV_32FC1);
            
            //tf::imshow("ref", d);
            //tf::imshow("tmp", tmp);
            //tf::imshow("t", t);
            cv::multiply(d, tmp(Bounds(restricted.size())), d);
            d.convertTo(tmp, CV_8UC1);
            //cv::bitwise_and(d, t, tmp);
            
            //cv::subtract(255, tmp, tmp);
            //tf::imshow("tmp", tmp);
            //tf::imshow("image"+Meta::toStr(i), image.get());
            
            auto blobs = CPULabeling::run(tmp);
            if (not blobs.empty()) {
                size_t msize = 0, midx = 0;
                for (size_t j = 0; j < blobs.size(); ++j) {
                    if (blobs.at(j).pixels->size() > msize) {
                        msize = blobs.at(j).pixels->size();
                        midx = j;
                    }
                }
                
                auto&& pair = blobs.at(midx);
                for (auto& line : *pair.lines) {
                    line.x1 += pos.x;
                    line.x0 += pos.x;
                    line.y += pos.y;
                }
                
                pv::Blob blob(*pair.lines, *pair.pixels, pair.extra_flags);
                auto points = pixel::find_outer_points(&blob, 0);
                if (not points.empty()) {
                    data.outlines.emplace_back(std::move(*points.front()));
                    //for (auto& pt : outline_points.back())
                    //    pt = (pt + blob.bounds().pos())/*.mul(dim.div(image.dimensions())) + pos*/;
                }
                data.predictions[blob.blob_id()] = { .clid = size_t(cls), .p = float(conf) };
                data.frame.add_object(std::move(pair));
                //auto big = pixel::threshold_get_biggest_blob(&blob, 1, nullptr);
                //auto [pos, img] = big->image();
                
                /*if (i % 2 && data.frame.index().get() % 10 == 0) {
                    auto [pos, img] = blob.image();
                    cv::Mat vir = cv::Mat::zeros(img->rows, img->cols, CV_8UC3);
                    auto vit = vir.ptr<cv::Vec3b>();
                    for (auto it = img->data(); it != img->data() + img->size(); ++it, ++vit)
                        *vit = Viridis::value(*it / 255.0);
                    tf::imshow("big", vir);
                }*/
            }
        }
    }
    
    static tl::expected<SegmentationData, const char*> apply(TileImage&& tiled) {
        namespace py = Python;
        
        SegmentationData data{
            .image = std::move(tiled.original),
        };
        
        static Frame_t running_id = 0_f;
        auto fake = double(running_id.get()) / double(FAST_SETTING(frame_rate)) * 1000.0 * 1000.0;
        data.frame.set_timestamp(uint64_t(fake));
        data.frame.set_index(running_id++);
        
        Vec2 scale = SETTING(output_size).value<Size2>().div(tiled.source_size);
        print("Image scale: ", scale, " with tile source=", tiled.source_size, " image=", data.image->dimensions()," output_size=", SETTING(output_size).value<Size2>(), " original=", tiled.original_size);
        
        
        /*for(size_t i=0; i<tiled.images.size(); ++i) {
            tf::imshow(Meta::toStr(i)+ " " +Meta::toStr(tiled.offsets().at(i)), tiled.images.at(i)->get());
        }*/
        
        for(auto p : tiled.offsets()) {
            data.tiles.push_back(Bounds(p.x, p.y, tiled.tile_size.width, tiled.tile_size.height).mul(scale));
        }
        
        py::schedule([&data, scale, offsets = tiled.offsets(), images = std::move(tiled.images)]() mutable {
            using py = track::PythonIntegration;
            py::ModuleProxy bbx("bbx_saved_model");
            bbx.set_variable("offsets", std::move(offsets));
            bbx.set_variable("image", std::move(images));
            
            bbx.set_function("receive", [&](std::vector<float> masks, std::vector<float> meta, std::vector<int> indexes) {
                receive(offsets, data, scale, masks, meta, indexes);
            });

            try {
                bbx.run("apply");
            }
            catch (...) {
                FormatWarning("Continue after exception...");
                throw;
            }
            
            bbx.unset_function("receive");
            
        }).get();
        
        //tf::imshow("test", ret);
        return data;//Image::Make(ret);
    }
};

static_assert(ObjectDetection<Yolo7ObjectDetection>);
static_assert(ObjectDetection<Yolo7InstanceSegmentation>);

namespace cmn {
namespace package {

template<typename R, typename... Args>
struct packaged_func {
    using signature = R(Args...);
    package::F<signature> package;
    
    template<typename K>
    packaged_func(K&& fn) : package(package::F<signature>{std::move(fn)}) { }
    
    packaged_func& operator=(packaged_func&& other) = default;
    packaged_func& operator=(const packaged_func& other) = delete;
    
    packaged_func(packaged_func&& other) = default;
    packaged_func(const packaged_func& other) = delete;
    
    template<typename... _Args>
    R operator()(_Args&&... args) {
        return package(std::forward<_Args>(args)...);
    }
};

}
}

template<typename SourceType>
class AbstractVideoSource {
    mutable std::mutex mutex;
    Frame_t i{0_f};
    gpuMat buffer;
    
    VideoSource source;
    mutable package::packaged_func<bool, gpuMat*> retrieve;
    mutable package::packaged_func<Size2> _size_func;
    
public:
    AbstractVideoSource(SourceType&& source)
        :   retrieve([this](gpuMat* buffer) -> bool{
                std::unique_lock guard(mutex);
                try {
                    if(i >= this->source.length())
                        i = 0_f;
                    this->source.frame(i++, *buffer);
                } catch(const std::exception& e) {
                    FormatExcept("Unable to load frame ", i, " from video source ", this->source, " because: ", e.what());
                    return false;
                }
                return true;
            }),
            _size_func([this]() -> Size2 {
                return this->source.size();
            }),
            source(std::move(source))
    {
        this->source.set_colors(VideoSource::ImageMode::RGB);
    }

    AbstractVideoSource(AbstractVideoSource&& other) = default;
    AbstractVideoSource(const AbstractVideoSource& other) = delete;
    AbstractVideoSource() = default;
    
    AbstractVideoSource& operator=(AbstractVideoSource&&) = delete;
    AbstractVideoSource& operator=(const AbstractVideoSource&) = delete;
    
    bool next(gpuMat& output) {
        return retrieve(&output);
    }
    
    bool is_finite() const {
        return true;
    }
    
    Frame_t length() const {
        return 0_f;
    }
    
    short framerate() const {
        return 25;
    }
    
    Size2 size() const {
        return _size_func();
    }
    
    file::Path base() const {
        return file::Path();
    }
    
    std::string toStr() const {
        return "AbstractVideoSource<"+Meta::toStr(source)+">";
    }
};


template<typename F>
    requires ObjectDetection<F>
struct OverlayedVideo {
    AbstractVideoSource<VideoSource> source;
    //VideoSource source;
    F overlay;
    
    mutable std::mutex index_mutex;
    Frame_t i{0};
    Image::UPtr original_image;
    gpuMat buffer, resized, converted;
    cv::Mat downloader;
    
    std::future<tl::expected<SegmentationData, const char*>> next_image;

    static inline std::atomic<float> _fps{0}, _samples{0};
    bool eof() const noexcept {
        if(not source.is_finite())
            return false;
        return i <= source.length();
    }
    
    OverlayedVideo() = delete;
    OverlayedVideo(const OverlayedVideo&) = delete;
    OverlayedVideo& operator=(const OverlayedVideo&) = delete;
    OverlayedVideo(OverlayedVideo&&) = delete;
    OverlayedVideo& operator=(OverlayedVideo&&) = delete;
    
    template<typename SourceType>
    OverlayedVideo(F&& fn, SourceType&& s)
        : source(AbstractVideoSource<SourceType>(std::move(s))), overlay(std::move(fn))
    {
        
    }
    
    ~OverlayedVideo() {
        if(next_image.valid())
            next_image.get();
    }
    
    void reset(Frame_t frame) {
        std::scoped_lock guard(index_mutex);
        i = frame;
    }
    
    //! generates the next frame
    tl::expected<SegmentationData, const char*> generate() noexcept {
        if(eof())
            return tl::unexpected("End of file.");
        
        auto retrieve_next = [this]()
            -> tl::expected<SegmentationData, const char*>
        {
            std::scoped_lock guard(index_mutex);
            TileImage tiled;
            
            try {
                if(not source.next(buffer))
                    return tl::unexpected("Cannot retrieve frame from video source.");
                
                if(SETTING(video_scale).value<float>() != 1) {
                    Size2 new_size = Size2(buffer.cols, buffer.rows) * SETTING(video_scale).value<float>();
                    FormatWarning("Resize ", Size2(buffer.cols, buffer.rows), " -> ", new_size);
                    cv::resize(buffer, buffer, new_size);
                }

                cv::cvtColor(buffer, buffer, cv::COLOR_BGR2RGB);
                original_image = Image::Make(buffer);
                
                Size2 original_size(buffer.cols, buffer.rows);
                
                Size2 new_size(expected_size);
                if(SETTING(tile_image).value<size_t>() > 1) {
                    size_t tiles = SETTING(tile_image).value<size_t>();
                    float ratio = buffer.rows / float(buffer.cols);
                    new_size = Size2(expected_size.width * tiles, expected_size.width * tiles * ratio).map(roundf);
                    while(buffer.cols < new_size.width
                          && buffer.rows < new_size.height
                          && tiles > 0)
                    {
                        new_size = Size2(expected_size.width * tiles, expected_size.width * tiles * ratio).map(roundf);
                        //print("tiles=",tiles," new_size = ", new_size);
                        tiles--;
                    }
                }
                
                if (buffer.cols != new_size.width || buffer.rows != new_size.height) {
                    FormatWarning("Resize ", Size2(buffer.cols, buffer.rows), " -> ", new_size);
                    cv::resize(buffer, buffer, new_size);
                }
                
                ++i;

                TileImage tiled(buffer, std::move(original_image), expected_size, original_size);

                Timer timer;
                auto result = this->overlay.apply(std::move(tiled));
                if (_samples.load() > 100) {
                    _samples = _fps = 0;
                    print("Reset indexes: ", timer.elapsed());
                }
                _fps = _fps.load() + 1.0 / timer.elapsed();
                _samples = _samples.load() + 1;
                return result;
                
            } catch(const std::exception& e) {
                FormatExcept("Error loading frame ", i, " from video ", source, ": ", e.what());
                return tl::unexpected("Error loading frame.");
            }
        };
        
        tl::expected<SegmentationData, const char*> result;
        if(next_image.valid()) {
            result = next_image.get();
            
        } else {
            result = retrieve_next();
        }
        
        if (not result) {
            return tl::unexpected(result.error());
        }

        next_image = std::async(std::launch::async | std::launch::deferred, retrieve_next);
        return result;
    }
};

template<typename OverlayT>
struct Menu {
    Button::Ptr
        hi = Button::MakePtr("Quit", attr::Size(50, 35)),
        bro = Button::MakePtr("Filter", attr::Size(50, 35)),
        reset = Button::MakePtr("Reset", attr::Size(50, 35));
    std::shared_ptr<Text> text = std::make_shared<Text>();
    Image::UPtr next;
    
    std::shared_ptr<HorizontalLayout> buttons = std::make_shared<HorizontalLayout>(
        std::vector<Layout::Ptr>{
            Layout::Ptr(hi),
            Layout::Ptr(bro),
            Layout::Ptr(reset),
            Layout::Ptr(text)
        }
    );
    
    Menu() = delete;
    Menu(Menu&&) = delete;
    Menu(const Menu&) = delete;
    
    template<typename F>
    Menu(F&& reset_func) {
        buttons->set_policy(HorizontalLayout::Policy::TOP);
        buttons->set_pos(Vec2(10, 10));
        bro->set_toggleable(true);
        bro->on_click([this](Event) {
            SETTING(filter_class) = bro->toggled();
        });
        hi->on_click([](Event) {
            SETTING(terminate) = true;
        });
        reset->on_click([fn = std::move(reset_func)](Event){
            fn();
        });
    }
    
    ~Menu() {
        hi = bro = nullptr;
        buttons = nullptr;
    }
    
    void draw(DrawStructure& g) {
        //g.wrap_object(overlay);
        text->set_txt(Meta::toStr(OverlayT::_fps.load() / OverlayT::_samples.load())+"fps");
        
        g.wrap_object(*buttons);
    }
};

struct SettingsDropdown {
    Dropdown _settings_dropdown = Dropdown(Bounds(0, 0, 200, 33), GlobalSettings::map().keys());
    Textfield _value_input = Textfield(Bounds(0, 0, 300, 33));
    std::shared_ptr<gui::List> _settings_choice;
    bool should_select{false};
    
    SettingsDropdown(auto&& on_enter) {
        _settings_dropdown.set_origin(Vec2(0, 1));
        _value_input.set_origin(Vec2(0, 1));
        
        _settings_dropdown.on_select([&](long_t index, const std::string& name) {
            this->selected_setting(index, name, _value_input);
        });
        _value_input.on_enter([this, on_enter = std::move(on_enter)](){
            try {
                auto key = _settings_dropdown.items().at(_settings_dropdown.selected_id()).name();
                if(GlobalSettings::access_level(key) == AccessLevelType::PUBLIC) {
                    GlobalSettings::get(key).get().set_value_from_string(_value_input.text());
                    if(GlobalSettings::get(key).is_type<Color>())
                        this->selected_setting(_settings_dropdown.selected_id(), key, _value_input);
                    if((std::string)key == "auto_apply" || (std::string)key == "auto_train")
                    {
                        SETTING(auto_train_on_startup) = false;
                    }
                    if(key == "auto_tags") {
                        SETTING(auto_tags_on_startup) = false;
                    }
                    
                    on_enter(key);
                    
                } else
                   FormatError("User cannot write setting ", key," (",GlobalSettings::access_level(key).name(),").");
            } catch(const std::logic_error&) {
                //FormatExcept("Cannot set ",settings_dropdown.items().at(settings_dropdown.selected_id())," to value ",textfield.text()," (invalid).");
            } catch(const UtilsException&) {
                //FormatExcept("Cannot set ",settings_dropdown.items().at(settings_dropdown.selected_id())," to value ",textfield.text()," (invalid).");
            }
        });
    }
    
    void selected_setting(long_t index, const std::string& name, Textfield& textfield) {
        print("choosing ",name);
        if(index != -1) {
            //auto name = settings_dropdown.items().at(index);
            auto val = GlobalSettings::get(name);
            if(val.get().is_enum() || val.is_type<bool>()) {
                auto options = val.get().is_enum() ? val.get().enum_values()() : std::vector<std::string>{ "true", "false" };
                auto index = val.get().is_enum() ? val.get().enum_index()() : (val ? 0 : 1);
                
                std::vector<std::shared_ptr<List::Item>> items;
                std::map<std::string, bool> selected_option;
                for(size_t i=0; i<options.size(); ++i) {
                    selected_option[options[i]] = i == index;
                    items.push_back(std::make_shared<TextItem>(options[i]));
                    items.back()->set_selected(i == index);
                }
                
                print("options: ", selected_option);
                
                _settings_choice = std::make_shared<List>(Bounds(0, 0, 150, textfield.height()), "", items, [&textfield, this](List*, const List::Item& item){
                    print("Clicked on item ", item.ID());
                    textfield.set_text(item);
                    textfield.enter();
                    _settings_choice->set_folded(true);
                });
                
                _settings_choice->set_display_selection(true);
                _settings_choice->set_selected(index, true);
                _settings_choice->set_folded(false);
                _settings_choice->set_foldable(true);
                _settings_choice->set_toggle(false);
                _settings_choice->set_accent_color(Color(80, 80, 80, 200));
                _settings_choice->set_origin(Vec2(0, 1));
                
            } else {
                _settings_choice = nullptr;
                
                if(val.is_type<std::string>()) {
                    textfield.set_text(val.value<std::string>());
                } else if(val.is_type<file::Path>()) {
                    textfield.set_text(val.value<file::Path>().str());
                } else
                    textfield.set_text(val.get().valueString());
            }
            
            if(!_settings_choice)
                textfield.set_read_only(GlobalSettings::access_level(name) > AccessLevelType::PUBLIC);
            else
                _settings_choice->set_pos(textfield.pos());
            
            should_select = true;
        }
    }
    
    void draw(IMGUIBase& base, DrawStructure& g) {
        auto stretch_w = g.width() - 10 - _value_input.global_bounds().pos().x;
        if(_value_input.selected())
            _value_input.set_size(Size2(max(300, stretch_w / 1.0), _value_input.height()));
        else
            _value_input.set_size(Size2(300, _value_input.height()));
        
        _settings_dropdown.set_pos(Vec2(10, g.height() - 10).div(base.dpi_scale() * g.scale().y));
        _value_input.set_pos(_settings_dropdown.pos() + Vec2(_settings_dropdown.width(), 0));
        g.wrap_object(_settings_dropdown);
        
        if(_settings_choice) {
            g.wrap_object(*_settings_choice);
            _settings_choice->set_pos(_settings_dropdown.pos() + Vec2(_settings_dropdown.width(), 0));
            
            if(should_select) {
                g.select(_settings_choice.get());
                should_select = false;
            }
            
        } else {
            g.wrap_object(_value_input);
            if(should_select) {
                g.select(&_value_input);
                should_select = false;
            }
        }
    }
};

struct Detection {
    Detection() {
        if(type() == ObjectDetectionType::yolo7) {
            Yolo7ObjectDetection::init();
            
        } else if(type() == ObjectDetectionType::yolo7seg) {
            Yolo7InstanceSegmentation::init();
            
        } else
            throw U_EXCEPTION("Unknown detection type: ", type());
    }
    
    static ObjectDetectionType::Class type() {
        return SETTING(detection_type).value<ObjectDetectionType::Class>();
    }
    
    static tl::expected<SegmentationData, const char*> apply(TileImage&& tiled) {
        if(type() == ObjectDetectionType::yolo7) {
            return Yolo7ObjectDetection::apply(std::move(tiled));
            
        } else if(type() == ObjectDetectionType::yolo7seg) {
            return Yolo7InstanceSegmentation::apply(std::move(tiled));
        }
        
        throw U_EXCEPTION("Unknown detection type: ", type());
    }
    
    static void receive(std::vector<Vec2> offsets, SegmentationData& data, Vec2 scale, std::vector<float>& masks, const std::vector<float>& vector, const std::vector<int>& indexes)
    {
        if(type() == ObjectDetectionType::yolo7seg) {
            Yolo7InstanceSegmentation::receive(offsets, data, scale, masks, vector, indexes);
            return;
        }
        
        throw U_EXCEPTION("Unknown detection type: ", type());
    }
    
    static void receive(SegmentationData& data, Vec2 scale, const std::vector<float>& vector) {
        if(type() == ObjectDetectionType::yolo7) {
            Yolo7ObjectDetection::receive(data, scale, vector);
            return;
        }
        
        throw U_EXCEPTION("Unknown detection type: ", type());
    }
};


int main(int argc, char**argv) {
    using namespace gui;
    
    default_config::register_default_locations();
    default_config::get(GlobalSettings::map(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    file::cd(file::DataLocation::parse("app"));
    
    SETTING(video_scale) = float(1);
    SETTING(source) = std::string("/Users/tristan/goats/DJI_0160.MOV");
    SETTING(model) = file::Path("/Users/tristan/Downloads/tfmodel_goats1024");
    SETTING(image_width) = int(1024);
    SETTING(filename) = file::Path("");
    SETTING(classes) = std::vector<std::string>{
        "goat", "sheep", "human"
    };
    SETTING(detection_type) = ObjectDetectionType::yolo7;
    SETTING(tile_image) = size_t(2);
    
    using namespace cmn;
    namespace py = Python;
    CommandLine cmd(argc, argv);
    for(auto a : cmd) {
        if(a.name == "i") {
            SETTING(source) = std::string(a.value);
        }
        if(a.name == "m") {
            SETTING(model) = file::Path(a.value);
        }
        if(a.name == "dim") {
            SETTING(image_width) = Meta::fromStr<int>(a.value);
        }
        if(a.name == "o") {
            SETTING(filename) = file::Path(a.value);
        }
    }
    
    DebugHeader("Starting tracking of");
    print("model: ",SETTING(model).value<file::Path>());
    print("video: ", SETTING(source).value<std::string>());
    print("model resolution: ", SETTING(image_width).value<int>());
    print("output size: ", SETTING(video_source).value<Size2>());
    
    expected_size = Size2(SETTING(image_width).value<int>(), SETTING(image_width).value<int>());
    
    py::init();
    py::schedule([](){
        track::PythonIntegration::set_settings(GlobalSettings::instance());
        track::PythonIntegration::set_display_function([](auto& name, auto& mat) { tf::imshow(name, mat); });
    });
    
    static_assert(ObjectDetection<Detection>);
    OverlayedVideo video(
        Detection{},
        VideoSource(SETTING(source).value<std::string>())
        //VideoSource("/Users/tristan/goats/DJI_0160.MOV")
        //VideoSource("/Users/tristan/trex/videos/test_frames/frame_%3d.jpg")
    );
    
    
    /*GlobalSettings::map().register_callback("global", [](auto, auto&, auto& name, auto& value) {
        if(name == "gui_frame") {
            
        }
    });*/
    
    std::mutex mutex;
    std::condition_variable messages;
    bool _terminate{false};
    SegmentationData next, dbuffer;
    
    auto f = std::async(std::launch::async, [&](){
        std::unique_lock guard(mutex);
        while(not _terminate) {
            if(dbuffer)
                messages.wait(guard);
            
            if(dbuffer) {
                if(next) {
                    continue;
                }
                
                next = std::move(dbuffer);
            }
            
            guard.unlock();
            
            tl::expected<decltype(next), const char*> result;
            try {
                result = video.generate();
                guard.lock();
                
            } catch(...) {
                guard.lock();
                throw;
            }
            
            if(not result) {
                // end of video
                video.reset(0_f);
                continue;
            }
            
            if(not next) {
                auto ts = result.value().frame.timestamp();
                next = std::move(result.value());
                assert(next.frame.timestamp() == ts);
            } else {
                dbuffer = std::move(result.value());
            }
        }
    });
    
    ::Menu<decltype(video)> menu([&](){
        video.i = 1500_f;
    });
    
    using namespace track;
    
    GlobalSettings::map().set_do_print(true);
    GlobalSettings::map().dont_print("gui_frame");
    SETTING(track_do_history_split) = false;
    SETTING(cm_per_pixel) = Settings::cm_per_pixel_t(0.1);
    SETTING(meta_real_width) = float(expected_size.width * 10);
    SETTING(track_max_speed) = Settings::track_max_speed_t(300);
    SETTING(track_threshold) = Settings::track_threshold_t(0);
    SETTING(track_posture_threshold) = Settings::track_posture_threshold_t(0);
    SETTING(blob_size_ranges) = Settings::blob_size_ranges_t({
        Rangef(10,300)
    });
    SETTING(frame_rate) = Settings::frame_rate_t(video.source.framerate());
    SETTING(track_speed_decay) = Settings::track_speed_decay_t(1);
    SETTING(track_max_reassign_time) = Settings::track_max_reassign_time_t(1);
    SETTING(terminate) = false;
    SETTING(calculate_posture) = false;
    SETTING(gui_interface_scale) = float(1);

    cmd.load_settings();
    
    if(SETTING(filename).value<file::Path>().empty()) {
        SETTING(filename) = file::Path((std::string)file::Path(video.source.base()).filename());
    }
    
    SETTING(filter_class) = false;
    Size2 output_size = (Size2(video.source.size()) * SETTING(video_scale).value<float>()).map(roundf);
    SETTING(output_size) = output_size;
    
    Tracker tracker;
    //cv::Mat bg = cv::Mat::zeros(video.source.size().height, video.source.size().width, CV_8UC1);
    //cv::Mat bg = cv::Mat::zeros(expected_size.width, expected_size.height, CV_8UC1);
    cv::Mat bg = cv::Mat::zeros(output_size.height, output_size.width, CV_8UC1);
    bg.setTo(255);
    tracker.set_average(Image::Make(bg));
    //FrameInfo frameinfo;
    //Timer timer;
    //Timeline timeline(nullptr, [](bool) {}, []() {}, frameinfo);
    print("&video = ", (uint64_t)&video);
    SettingsDropdown settings([&](const std::string& name) {
        if(name == "gui_frame") {
            print("&video[settings] = ", (uint64_t)&video);
            video.reset(SETTING(gui_frame).value<Frame_t>());
        }
    });
    
    DrawStructure graph(1024, output_size.height / output_size.width * 1024);
    
    IMGUIBase base("TRexA", graph, [&, ptr = &base]()->bool {
        
        //timeline.set_base(ptr);
        
        //auto dt = timer.elapsed();
        ///cache.set_dt(dt);
        //timer.reset();

        Frame_t index = SETTING(gui_frame).value<Frame_t>();

        //image.set_pos(last_mouse_pos);
        //graph.wrap_object(image);

        auto scale = graph.scale().reciprocal();
        auto dim = ptr->window_dimensions().mul(scale * gui::interface_scale());
        graph.draw_log_messages();//Bounds(Vec2(0), dim));
        
        /*static Timer frame_timer;
        if (frame_timer.elapsed() >= 1.0 / (double)SETTING(frame_rate).value<uint32_t>()
            && index + 1_f < file.length())
        {
            if (SETTING(gui_run)) {
                index += 1_f;
                SETTING(gui_frame) = index;
            }
            frame_timer.reset();
        }*/

        //frameinfo.mx = graph.mouse_position().x;
        //frameinfo.my = graph.mouse_position().y;
        //frameinfo.frameIndex = index;
        
        return true;
    }, [](Event) {
        
    });

    auto start_time = std::chrono::system_clock::now();
    PPFrame pp;
    auto filename = file::DataLocation::parse("output", SETTING(filename).value<file::Path>());
    DebugHeader("Output: ", filename);
    pv::File file(filename, pv::FileMode::OVERWRITE | pv::FileMode::WRITE);
    std::vector<pv::BlobPtr> objects;
    file.set_average(bg);
    
    //GUICache cache(&graph, &file);
    
    SegmentationData current;
    std::shared_ptr<ExternalImage>
        background = std::make_shared<ExternalImage>(),
        overlay = std::make_shared<ExternalImage>();

    auto fetch_files = [&](){
        std::this_thread::sleep_for(std::chrono::milliseconds(30));

        std::unique_lock guard(mutex);
        if (next.image) {
            objects.clear();
            current = std::move(next);

            for (size_t i = 0; i < current.frame.n(); ++i) {
                objects.emplace_back(current.frame.blob_at(i));
            }

            auto rgba = Image::Make(current.image->rows, current.image->cols, 4);
            cv::cvtColor(current.image->get(), rgba->get(), cv::COLOR_BGR2BGRA);
            background->set_source(std::move(rgba));
            
            if (not file.is_open()) {
                file.set_start_time(start_time);
                file.set_resolution(output_size);
            }
            file.add_individual(pv::Frame(current.frame));

            Tracker::preprocess_frame(file, pv::Frame(current.frame), pp, nullptr, PPFrame::NeedGrid::NoNeed, false);
            tracker.add(pp);
            if (pp.index().get() % 100 == 0) {
                print(IndividualManager::num_individuals(), " individuals known in frame ", pp.index());
            }
            messages.notify_one();
        }
    };
    
    //graph.set_scale(1. / base.dpi_scale());
    
    gui::SFLoop loop(graph, &base, [&](gui::SFLoop&, LoopStatus) {
        fetch_files();
        
        {
            //track::LockGuard guard(track::ro_t{}, "update", 10);
            //cache.update_data(current.frame.index());
            //frameinfo.analysis_range = tracker.analysis_range();
            //frameinfo.video_length = file.length().get();
            //frameinfo.consecutive = tracker.consecutive();
            //frameinfo.current_fps = fps;
        }
        
        graph.section("video", [&](auto&, Section* section){
            auto output_size = SETTING(output_size).value<Size2>();
            auto window_size = base.real_dimensions();
            
            auto ratio = output_size.width / output_size.height;
            Size2 wdim;
            
            if(window_size.width * output_size.height < window_size.height * output_size.width)
            {
                wdim = Size2(window_size.width, window_size.width / ratio);
            } else {
                wdim = Size2(window_size.height * ratio, window_size.height);
            }
            
            auto scale = wdim.div(output_size).mul(base.dpi_scale());
            
            //ratio = ratio.T();
            //scale = scale.mul(ratio);
            if(not current.frame.index().valid() || current.frame.index().get()%10 == 0)
                print("gui scale: ", scale, " dpi:",base.dpi_scale(), " graph:", graph.scale(), " window:", base.window_dimensions(), " video:", SETTING(output_size).value<Size2>(), " scale:", Size2(graph.width(), graph.height()).div(SETTING(output_size).value<Size2>()), " ratio:", ratio, " wdim:", wdim);
            section->set_scale(scale);
            SETTING(gui_frame) = current.frame.index();
            
            if(background->source()) {
                graph.wrap_object(*background);
            }
            
            for(auto box : current.tiles)
                graph.rect(box, attr::FillClr{Transparent}, attr::LineClr{Red});
            
            auto classes = SETTING(classes).value<std::vector<std::string>>();
            for(auto& blob : objects) {
                const auto bds = blob->bounds();
                graph.rect(bds, attr::LineClr(Gray), attr::FillClr(Gray.alpha(25)));
                
                SegmentationData::Assignment assign;
                if(Tracker::end_frame().valid()) {
                    auto it = current.predictions.find(blob->blob_id());
                    if(it != current.predictions.end()) {
                        assign = it->second;
                    } else {
                        print("[draw] blob ", blob->blob_id(), " not found...");
                    }
                }
                
                auto cname = classes.size() > assign.clid ? classes.at(assign.clid) :
                "<unknown:"+Meta::toStr(assign.clid)+">";
                
                auto loc = attr::Loc(bds.pos());
                auto text = graph.text(cname, loc, attr::FillClr(Cyan.alpha(100)), attr::Font(0.75, Style::Bold), attr::Scale(scale.mul(graph.scale()).reciprocal()),
                                       attr::Origin(0, 1));
                loc.x += text->local_bounds().width;
                graph.text(": "+Meta::toStr(assign.p) + " - "+ Meta::toStr(blob->num_pixels()) + " " + Meta::toStr(blob->recount(FAST_SETTING(track_threshold), *tracker.background())), loc, attr::FillClr(White.alpha(100)), attr::Font(0.75), attr::Scale(scale.mul(graph.scale()).reciprocal()),
                           attr::Origin(0, 1));
                
            }

            if (not current.outlines.empty()) {
                graph.text(Meta::toStr(current.outlines.size())+" lines", attr::Loc(10,50), attr::Font(0.35));
                
                ColorWheel wheel;
                for (const auto& v : current.outlines) {
                    auto clr = wheel.next();
                    graph.line(v, 1, clr.alpha(150));
                }
            }
            
            using namespace track;
            IndividualManager::transform_all([&](Idx_t , Individual* fish)
            {
                if(not fish->has(tracker.end_frame()))
                    return;
                auto p = fish->iterator_for(tracker.end_frame());
                auto segment = p->get();
                
                auto basic = fish->compressed_blob(tracker.end_frame());
                auto bds = basic->calculate_bounds();//.mul(scale);
                std::vector<Vertex> line;
                fish->iterate_frames(Range(tracker.end_frame().try_sub(50_f), tracker.end_frame()), [&](Frame_t , const std::shared_ptr<SegmentInformation> &ptr, const BasicStuff *basic, const PostureStuff *) -> bool
                {
                    if(ptr.get() != segment) //&& (ptr)->end() != segment->start().try_sub(1_f))
                        return true;
                    auto p = basic->centroid.pos<Units::PX_AND_SECONDS>();//.mul(scale);
                    line.push_back(Vertex(p.x, p.y, fish->identity().color()));
                    return true;
                });
                graph.rect(bds, attr::FillClr(Transparent), attr::LineClr(fish->identity().color()));
                graph.vertices(line);
            });
        });
        
        graph.section("menus", [&](auto&, Section* section){
            section->set_scale(graph.scale().reciprocal());
            menu.draw(graph);
            settings.draw(base, graph);
        });
        
        
        //timeline.draw(graph);
        
        graph.set_dirty(nullptr);

        if (graph.is_key_pressed(Keyboard::Right)) {
            SETTING(filter_class) = true;
        } else if (graph.is_key_pressed(Keyboard::Left)) {
            SETTING(filter_class) = false;
        }
    });
    
    {
        std::unique_lock guard(mutex);
        _terminate = true;
        messages.notify_all();
    }
    f.get();
    graph.root().set_stage(nullptr);
    py::deinit();
    file.close();
    
    {
        pv::File test(file.filename(), pv::FileMode::READ);
        test.print_info();
    }
    return 0;
}

