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
    
    struct Assignment {
        size_t clid;
        float p;
    };
    
    std::map<pv::bid, Assignment> predictions;
    std::vector<std::vector<Vec2>> outlines;
    
    operator bool() const {
        return image != nullptr;
    }
    
    void receive(Vec2 scale_factor, const std::vector<float>& vector) {
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
                if(y < 0 || y >= image->rows)
                    continue;
                
                HorizontalLine line{
                    (coord_t)saturate(int(y), int(0), int(y + dim.height - 1)),
                    (coord_t)saturate(int(pos.x), int(0), int(pos.x + dim.width - 1)),
                    (coord_t)saturate(int(pos.x + dim.width), int(0), int(pos.x + dim.width - 1))
                };
                for(int x = line.x0; x <= line.x1; ++x) {
                    pixels.emplace_back(image->get().at<cv::Vec4b>(y, x)[0]);
                }
                //pixels.insert(pixels.end(), (uchar*)mat.ptr(y, line.x0),
                //              (uchar*)mat.ptr(y, line.x1));
                lines.emplace_back(std::move(line));
            }
            
            if(not lines.empty()) {
                pv::Blob blob(lines, 0);
                predictions[blob.blob_id()] = { .clid = size_t(cls), .p = float(conf) };
                frame.add_object(lines, pixels, 0);
            }
        }
    }
    
    void receive_seg(std::vector<float>& masks, const std::vector<float>& vector) {
        //print(vector);
        size_t N = vector.size() / 6u;
        
        cv::Mat full_image;
        cv::cvtColor(image->get(), full_image, cv::COLOR_RGBA2GRAY);
        
        for (size_t i = 0; i < N; ++i) {
            Vec2 pos(vector.at(i * 6 + 0), vector.at(i * 6 + 1));
            Size2 dim(vector.at(i * 6 + 2) - pos.x, vector.at(i * 6 + 3) - pos.y);
            
            float conf = vector.at(i * 6 + 4);
            float cls = vector.at(i * 6 + 5);
            
            print(conf, " ", cls, " ",pos, " ", dim);
            
            if (SETTING(filter_class).value<bool>() && cls != 14)
                continue;
            if (dim.min() < 1)
                continue;
            
            cv::Mat m(56, 56, CV_32FC1, masks.data() + i * 56 * 56);
            
            cv::Mat tmp;
            cv::resize(m, tmp, dim);
            
            cv::Mat dani;
            tmp.convertTo(dani, CV_8UC1, 255.0);
            tf::imshow("dani", dani);
            
            cv::threshold(tmp, tmp, 0.6, 1.0, cv::THRESH_BINARY);
            //cv::threshold(tmp, t, 150, 255, cv::THRESH_BINARY);
            //print(Bounds(pos, dim), " and image ", Size2(full_image), " and t ", Size2(t));
            //print("using bounds: ", Size2(full_image(Bounds(pos, dim))), " and ", Size2(t));
            //print("channels: ", full_image.channels(), " and ", t.channels(), " and types ", getImgType(full_image.type()), " ", getImgType(t.type()));
            cv::Mat d;// = full_image(Bounds(pos, dim));
            full_image(Bounds(pos, dim)).convertTo(d, CV_32FC1);
            
            //tf::imshow("ref", d);
            //tf::imshow("tmp", tmp);
            //tf::imshow("t", t);
            cv::multiply(d, tmp, d);
            d.convertTo(tmp, CV_8UC1);
            //cv::bitwise_and(d, t, tmp);
            
            //cv::subtract(255, tmp, tmp);
            tf::imshow("tmp", tmp);
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
                    outlines.emplace_back(std::move(*points.front()));
                    //for (auto& pt : outline_points.back())
                    //    pt = (pt + blob.bounds().pos())/*.mul(dim.div(image.dimensions())) + pos*/;
                }
                predictions[blob.blob_id()] = { .clid = size_t(cls), .p = float(conf) };
                frame.add_object(std::move(pair));
                //auto big = pixel::threshold_get_biggest_blob(&blob, 1, nullptr);
                //auto [pos, img] = big->image();
                
                if (i % 2 && frame.index().get() % 10 == 0) {
                    auto [pos, img] = blob.image();
                    cv::Mat vir = cv::Mat::zeros(img->rows, img->cols, CV_8UC3);
                    auto vit = vir.ptr<cv::Vec3b>();
                    for (auto it = img->data(); it != img->data() + img->size(); ++it, ++vit)
                        *vit = Viridis::value(*it / 255.0);
                    tf::imshow("big", vir);
                }
            }
        }
    }
};


struct TileImage {
    Size2 tile_size;
    Image::UPtr original;
    std::vector<Image::UPtr> images;
    inline static gpuMat resized, converted, thresholded;
    inline static cv::Mat download_buffer;
    Size2 source_size, original_size;
    
    TileImage() = default;
    TileImage(TileImage&&) = default;
    TileImage(const TileImage&) = delete;
    
    TileImage& operator=(TileImage&&) = default;
    TileImage& operator=(const TileImage&) = delete;
    
    TileImage(const gpuMat& source, Size2 tile_size, Size2 original_size)
        : tile_size(tile_size),
          source_size(source.cols, source.rows),
          original_size(original_size)
    {
        cv::Mat local;
        cv::cvtColor(source, local, cv::COLOR_BGR2RGBA);
        original = Image::Make(local);
        
        if(tile_size.width >= source.cols
           || tile_size.height >= source.rows)
        {
            cv::resize(source, resized, expected_size);
            cv::cvtColor(resized, converted, cv::COLOR_BGR2RGBA);
            converted.copyTo(download_buffer);
            source_size = expected_size;
            //print("Loaded frame ", i);
            images.emplace_back(Image::Make(download_buffer));
            
        } else {
            cv::Mat local;
            
            for(int y = 0; y < source.rows; y += tile_size.height) {
                for(int x = 0; x < source.cols; x += tile_size.width) {
                    gpuMat tile = gpuMat::zeros(tile_size.height, tile_size.width, CV_8UC3);
                    Bounds bds = Bounds(x, y, tile_size.width, tile_size.height);
                    bds.restrict_to(Bounds(0, 0, source.cols, source.rows));
                    source(bds).copyTo(tile(Bounds(0, 0, bds.width, bds.height)));
                    
                    
                    //cv::threshold(tile, thresholded, 150, 255, cv::THRESH_BINARY_INV);
                    //thresholded.copyTo(local);
                    //tf::imshow("offset"+Meta::toStr(x)+","+Meta::toStr(y), local);
                    cv::cvtColor(tile, local, cv::COLOR_BGR2RGBA);
                    tile.copyTo(local);
                    images.emplace_back(Image::Make(local));
                    //tf::imshow("offset"+Meta::toStr(x)+","+Meta::toStr(y), local);
                }
            }
        }
    }
    
    operator bool() const {
        return not images.empty();
    }
    
    std::vector<Vec2> offsets() const {
        std::vector<Vec2> o;
        for(int y = 0; y < source_size.height; y += tile_size.height) {
            for(int x = 0; x < source_size.width; x += tile_size.width) {
                o.emplace_back(x, y);
            }
        }
        return o;
    }
};

template<typename T>
concept Segmentation = requires (T t, TileImage tiled) {
    { t.apply(std::move(tiled)) } -> std::convertible_to<tl::expected<SegmentationData, const char*>>;
};

struct MLSegmentation {
    MLSegmentation() {
        Python::schedule([](){
            using py = track::PythonIntegration;
            py::ModuleProxy proxy{"bbx_saved_model"};
            py::check_module("bbx_saved_model");
            //py::set_variable("model_path", file::Path("/Users/tristan/Downloads/best_saved_model_octopus/best_saved_model").str(), "bbx_saved_model");
            //py::set_variable("model_path", file::Path("/Users/tristan/Downloads/best_saved_model/best_saved_model/best_saved_model").str(), "bbx_saved_model");
            proxy.set_variable("model_type", "yolo7");
            proxy.set_variable("model_path", SETTING(model).value<file::Path>().str());
            proxy.set_variable("image_size", int(expected_size.width));
            
            //py::set_variable("image_size", 1280, "bbx_saved_model");
            py::run("bbx_saved_model", "load_model");
            
        }).get();
    }
    
    tl::expected<SegmentationData, const char*> apply(TileImage&& tiled) {
        namespace py = Python;
        
        SegmentationData data{
            .image = std::move(tiled.original)
        };
        
        static Frame_t running_id = 0_f;
        auto fake = double(running_id.get()) / double(FAST_SETTING(frame_rate)) * 1000.0 * 1000.0;
        data.frame.set_timestamp(uint64_t(fake));
        data.frame.set_index(running_id++);
        
        //Vec2 scale = Vec2(SETTING(video_scale).value<float>()).reciprocal();//
        Vec2 scale = SETTING(output_size).value<Size2>().div(data.image->dimensions());
        print("Image scale: ", scale, " with tile source=", tiled.source_size, " image=", data.image->dimensions()," output_size=", SETTING(output_size).value<Size2>(), " original=", tiled.original_size);
        if(scale.x != 1 or scale.y != 1) {
            cv::Mat buffer;
            cv::resize(data.image->get(), buffer, data.image->dimensions().mul(scale) + 0.5);
            data.image = Image::Make(buffer);
            print("Resized image to ", data.image->dimensions(), " using scale ", scale);
        }
        
        else if(tiled.source_size != SETTING(output_size).value<Size2>()) {
            scale = SETTING(output_size).value<Size2>().div(tiled.source_size);
        }
        
        py::schedule([&data, scale, offsets = tiled.offsets(), images = std::move(tiled.images)]() mutable {
            using py = track::PythonIntegration;
            py::ModuleProxy bbx("bbx_saved_model");
            py::check_module("bbx_saved_model");
            
            bbx.set_variable("offsets", std::move(offsets));
            bbx.set_variable("image", std::move(images));
            
            bbx.set_function("receive", [&](std::vector<float> vector) {
                data.receive(scale, vector);
            });
            
            bbx.set_function("receive_seg", [&](std::vector<float> masks, std::vector<float> meta) {
                data.receive_seg(masks, meta);
            });

            try {
                py::run("bbx_saved_model", "apply");
            }
            catch (...) {
                FormatWarning("Continue after exception...");
            }
            py::unset_function("receive", "bbx_saved_model");
            py::unset_function("receive_seg", "bbx_saved_model");
            
        }).get();
        
        //tf::imshow("test", ret);
        return data;//Image::Make(ret);
    }
};

static_assert(Segmentation<MLSegmentation>);

template<typename F>
    requires Segmentation<F>
struct OverlayedVideo {
    VideoSource source;
    F overlay;
    
    mutable std::mutex index_mutex;
    Frame_t i{0};
    gpuMat buffer, resized, converted;
    std::future<tl::expected<TileImage, const char*>> next_image;
    static inline std::atomic<float> _fps{0}, _samples{0};
    bool eof() const noexcept {
        std::scoped_lock guard(index_mutex);
        return i >= source.length();
    }
    
    OverlayedVideo(F&& fn, VideoSource&& source)
        : source(std::move(source)), overlay(std::move(fn))
    {
        
    }
    
    ~OverlayedVideo() {
        if(next_image.valid())
            next_image.get();
    }
    
    void reset() {
        std::scoped_lock guard(index_mutex);
        i = 0_f;
    }
    
    //! generates the next frame
    tl::expected<SegmentationData, const char*> generate() noexcept {
        if(eof())
            return tl::unexpected("End of file.");
        
        auto retrieve_next = [this]()
            -> tl::expected<TileImage, const char*>
        {
            std::scoped_lock guard(index_mutex);
            TileImage tiled;
            
            try {
                source.frame(i, buffer);
                
                if(SETTING(video_scale).value<float>() != 1) {
                    Size2 new_size = Size2(buffer.cols, buffer.rows) * SETTING(video_scale).value<float>();
                    cv::resize(buffer, buffer, new_size);
                }
                
                Size2 original_size(buffer.cols, buffer.rows);
                
                /*if(Size2(buffer.cols, buffer.rows).max() > expected_size.width * 4) {
                    Size2 new_size = Size2(expected_size.width * 4, expected_size.width * 4 * buffer.rows / float(buffer.cols));
                    print("image resize ", Size2(buffer.cols, buffer.rows), " to ", new_size);
                    cv::resize(buffer, buffer, new_size);
                }*/
                
                size_t tiles = 2;
                Size2 new_size = Size2(expected_size.width * tiles, expected_size.width * tiles * buffer.rows / float(buffer.cols));
                cv::resize(buffer, buffer, new_size);
                
                ++i;
                return TileImage(buffer, expected_size, original_size);
                
            } catch(...) {
                FormatExcept("Error loading frame ", i, " from video ", source, ".");
                return tl::unexpected("Error loading frame.");
            }
        };
        
        TileImage tiled;
        if(next_image.valid()) {
            auto result = next_image.get();
            if(not result) {
                return tl::unexpected(result.error());
            }
            
            tiled = std::move(result.value());
            
        } else {
            auto result = retrieve_next();
            if(not result) {
                return tl::unexpected(result.error());
            }
            
            tiled = std::move(result.value());
        }
        
        next_image = std::async(std::launch::async | std::launch::deferred, retrieve_next);
        
        Timer timer;
        auto result = this->overlay.apply(std::move(tiled));
        if(_samples.load() > 100) {
            _samples = _fps = 0;
            print("Reset indexes: ", timer.elapsed());
        }
        _fps = _fps.load() + 1.0 / timer.elapsed();
        _samples = _samples.load() + 1;
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
    std::shared_ptr<ExternalImage>
        background = std::make_shared<ExternalImage>(),
        overlay = std::make_shared<ExternalImage>();
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
        background = overlay = nullptr;
        buttons = nullptr;
    }
    
    void draw(DrawStructure& g) {
        g.wrap_object(*background);
        //g.wrap_object(overlay);
        text->set_txt(Meta::toStr(OverlayT::_fps.load() / OverlayT::_samples.load())+"fps");
        
        if(background->source()) {
            //overlay.set_color(Red.alpha(125));
            background->set_scale(Size2(g.width(), g.height()).div( background->source()->dimensions()));
            overlay->set_scale(background->scale());
        }
        
        g.wrap_object(*buttons);
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

    OverlayedVideo video{
        MLSegmentation{},
        VideoSource(SETTING(source).value<std::string>())
        //VideoSource("/Users/tristan/goats/DJI_0160.MOV")
        //VideoSource("/Users/tristan/trex/videos/test_frames/frame_%3d.jpg")
    };
    
    video.source.set_colors(VideoSource::ImageMode::RGB);
    
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
            }
            
            if(not result) {
                // end of video
                video.reset();
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

    cmd.load_settings();
    
    if(SETTING(filename).value<file::Path>().empty()) {
        SETTING(filename) = file::Path((std::string)file::Path(video.source.base()).filename());
    }
    
    SETTING(filter_class) = false;
    Size2 output_size = Size2(video.source.size()) * SETTING(video_scale).value<float>();
    SETTING(output_size) = output_size;
    
    Tracker tracker;
    //cv::Mat bg = cv::Mat::zeros(video.source.size().height, video.source.size().width, CV_8UC1);
    //cv::Mat bg = cv::Mat::zeros(expected_size.width, expected_size.height, CV_8UC1);
    cv::Mat bg = cv::Mat::zeros(output_size.height, output_size.width, CV_8UC1);
    bg.setTo(255);
    tracker.set_average(Image::Make(bg));
    
    DrawStructure graph(1024, 768);
    IMGUIBase base("TRexA", graph, [&]()->bool {
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
    
    SegmentationData current;

    auto fetch_files = [&](){
        std::this_thread::sleep_for(std::chrono::milliseconds(30));

        std::unique_lock guard(mutex);
        if (next.image) {
            objects.clear();

            {
                current = std::move(next);
            }

            for (size_t i = 0; i < current.frame.n(); ++i) {
                auto blob = current.frame.blob_at(i);

                objects.emplace_back(std::move(blob));

                //bds = Bounds(bds.pos().mul(scale), bds.size().mul(scale));
                //print(blob->bounds(), " -> ", bds);
                //graph.rect(bds, attr::FillClr(Red.alpha(25)), attr::LineClr(Red));
                //auto [pos, img] = blob->image();
                //graph.image(pos.mul(scale), std::move(img), scale, White.alpha(25));
            }

            menu.background->set_source(Image::Make(*current.image));
            //menu.overlay.set_source(std::move(next.overlay));
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
    
    gui::SFLoop loop(graph, &base, [&](gui::SFLoop&, LoopStatus) {
        fetch_files();

        graph.set_scale(1. / base.dpi_scale());
        menu.buttons->set_scale(graph.scale().reciprocal());
        menu.draw(graph);
        
        auto scale = Size2(graph.width(), graph.height()).div( menu.background->source()->dimensions());
            
        auto classes = SETTING(classes).value<std::vector<std::string>>();
        for(auto& blob : objects) {
            const auto bds = blob->bounds().mul(scale);
            //bds = Bounds(bds.pos().mul(scale), bds.size().mul(scale));
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
            
            auto cname = classes.size() > assign.clid ? classes.at(assign.clid) : "<unknown:"+Meta::toStr(assign.clid)+">";
            graph.text(cname+":"+Meta::toStr(assign.p) + " - " + Meta::toStr(blob->num_pixels()) + " " + Meta::toStr(blob->recount(FAST_SETTING(track_threshold), *tracker.background())), attr::Loc(bds.pos() - Vec2(0, 10)), attr::FillClr(White.alpha(100)), attr::Font(0.35));
        }

        if (not current.outlines.empty()) {
            graph.text(Meta::toStr(current.outlines.size())+" lines", attr::Loc(10,50), attr::Font(0.35));
            graph.section("track", [&](auto& , Section* s) {
                s->set_scale(scale);

                ColorWheel wheel;
                for (const auto& v : current.outlines) {
                    auto clr = wheel.next();
                    graph.line(v, 1, clr.alpha(150));
                }
            });
            
        }
        
        using namespace track;
        IndividualManager::transform_all([&](Idx_t , Individual* fish)
        {
            if(not fish->has(tracker.end_frame()))
                return;
            auto p = fish->iterator_for(tracker.end_frame());
            auto segment = p->get();
            
            auto basic = fish->compressed_blob(tracker.end_frame());
            auto bds = basic->calculate_bounds().mul(scale);
            std::vector<Vertex> line;
            const SegmentInformation* previous{nullptr};
            fish->iterate_frames(Range(tracker.end_frame().try_sub(50_f), tracker.end_frame()), [&](Frame_t , const std::shared_ptr<SegmentInformation> &ptr, const BasicStuff *basic, const PostureStuff *) -> bool
            {
                if(ptr.get() != segment) //&& (ptr)->end() != segment->start().try_sub(1_f))
                    return true;
                auto p = basic->centroid.pos<Units::PX_AND_SECONDS>().mul(scale);
                line.push_back(Vertex(p.x, p.y, fish->identity().color()));
                previous = ptr.get();
                return true;
            });
            graph.rect(bds, attr::FillClr(Transparent), attr::LineClr(fish->identity().color()));
            graph.vertices(line);
        });

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

