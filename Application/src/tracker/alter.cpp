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

template<typename T>
concept overlay_function = requires {
    requires std::invocable<T, const Image&>;
    //{ std::invoke_result<T, const Image&>::type } -> std::convertible_to<Image::UPtr>;
};

static Size2 expected_size(1024, 1024);

template<typename F>
    requires overlay_function<F>
struct OverlayedVideo {
    VideoSource source;
    F overlay;
    
    struct Overlay {
        Image::UPtr image;
        pv::Frame overlay;
    };
    
    mutable std::mutex index_mutex;
    Frame_t i{0};
    gpuMat buffer;
    cv::Mat download_buffer;
    std::future<tl::expected<Image::UPtr, const char*>> next_image;
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
    tl::expected<Overlay, const char*> generate() noexcept {
        if(eof())
            return tl::unexpected("End of file.");
        
        auto retrieve_next = [this]()
            -> tl::expected<Image::UPtr, const char*>
        {
            std::scoped_lock guard(index_mutex);
            try {
                source.frame(i, buffer);
                
                cv::Mat resized;
                cv::resize(buffer, download_buffer, expected_size);
                cv::cvtColor(download_buffer, download_buffer, cv::COLOR_BGR2RGBA);
                
                
                //print("Loaded frame ", i);
                auto ptr = Image::Make(download_buffer);
                ++i;
                return ptr;
            } catch(...) {
                FormatExcept("Error loading frame ", i, " from video ", source, ".");
                return tl::unexpected("Error loading frame.");
            }
        };
        
        Image::UPtr ptr;
        if(next_image.valid()) {
            auto result = next_image.get();
            if(not result) {
                return tl::unexpected(result.error());
            }
            
            ptr = std::move(result.value());
            
        } else {
            auto result = retrieve_next();
            if(not result) {
                return tl::unexpected(result.error());
            }
            
            ptr = std::move(result.value());
        }
        
        next_image = std::async(std::launch::async | std::launch::deferred, retrieve_next);
        
        Timer timer;
        auto overlay = this->overlay(*ptr);
        overlay.set_timestamp((uint64_t)ptr->timestamp());
        //overlay.set_index(Frame_t(ptr->index()));
        
        if(_samples.load() > 100) {
            _samples = _fps = 0;
            print("Reset indexes: ", timer.elapsed());
        }
        _fps = _fps.load() + 1.0 / timer.elapsed();
        _samples = _samples.load() + 1;
        return Overlay{
            .image = std::move(ptr),
            .overlay = std::move(overlay)
        };
    }
    
};

cv::Mat letterbox(cv::Mat &img, cv::Size new_shape, cv::Scalar color, bool _auto, bool scaleFill, bool scaleup, int stride)
{
    float width = img.cols;
    float height = img.rows;
    float r = min(new_shape.width / width, new_shape.height / height);
    if (!scaleup)
        r = min(r, 1.0f);
    int new_unpadW = int(round(width * r));
    int new_unpadH = int(round(height * r));
    int dw = new_shape.width - new_unpadW;
    int dh = new_shape.height - new_unpadH;
    if (_auto)
    {
        dw %= stride;
        dh %= stride;
    }
    dw /= 2, dh /= 2;
    cv::Mat dst;
    cv::resize(img, dst, Size2(new_unpadW, new_unpadH), 0, 0, cv::INTER_LINEAR);
    int top = int(round(dh - 0.1));
    int bottom = int(round(dh + 0.1));
    int left = int(round(dw - 0.1));
    int right = int(round(dw + 0.1));
    cv::copyMakeBorder(dst, dst, top, bottom, left, right, cv::BORDER_CONSTANT, color);
    return dst;
}

int main(int argc, char**argv) {
    using namespace gui;
    
    default_config::register_default_locations();
    default_config::get(GlobalSettings::map(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    file::cd(file::DataLocation::parse("app"));
    
    SETTING(source) = std::string("/Users/tristan/goats/DJI_0160.MOV");
    SETTING(model) = file::Path("/Users/tristan/Downloads/tfmodel_goats1024");
    SETTING(image_width) = int(1024);
    
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
    }
    
    DebugHeader("Starting tracking of");
    print("model: ",SETTING(model).value<file::Path>());
    print("video: ", SETTING(source).value<std::string>());
    print("model resolution: ", SETTING(image_width).value<int>());
    
    expected_size = Size2(SETTING(image_width).value<int>(), SETTING(image_width).value<int>());
    
    py::init();
    
    py::schedule([](){
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
    
    //auto model = cv::dnn::readNetFromONNX("/Users/tristan/Downloads/best_octopus.onnx");
    //model.setPreferableTarget(CV_DNN_INFERENCE_ENGINE_CPU_TYPE_ARM_COMPUTE);
    std::mutex pred_mutex;
    std::map<Frame_t, std::map<pv::bid, std::tuple<float, int>>> predictions;
    std::vector<std::vector<Vec2>> outline_points;

    //cv::dnn::blobFromImage(mat, blob, 1 / 255.0f, cv::Size(img.cols, img.rows), cv::Scalar(0, 0, 0), true, false);
    OverlayedVideo video{
        [&](const Image& image) -> pv::Frame {
            pv::Frame frame;
            
            static Frame_t running_id = 0_f;
            auto fake = double(running_id.get()) / double(FAST_SETTING(frame_rate)) * 1000.0 * 1000.0;
            frame.set_timestamp(uint64_t(fake));
            frame.set_index(running_id++);

            /*cv::Mat mat;
            auto INPUT_WIDTH = 640;
            auto INPUT_HEIGHT = 640;
            cv::cvtColor(image.get(), mat, cv::COLOR_BGRA2BGR);
            cv::resize(mat, mat, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
            //cv::cvtColor(mat, mat, cv::COLOR_GRAY2BGR);
            auto blob = cv::dnn::blobFromImage(mat, 1.0/255.0, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(0, 0, 0),    true, false);
            model.setInput(blob);
            std::vector<std::string> outLayerNames = model.getUnconnectedOutLayersNames();
            std::vector<cv::Mat> detections;
            model.forward(detections, outLayerNames);
            
            std::vector<cv::Rect> boxes;
            std::vector<int> classIDs;
            std::vector<float> confidences;
            
            cv::Mat out = cv::Mat(detections[0].size[1], detections[0].size[2], CV_32F, detections[0].ptr<float>());
            
            for (int r = 0; r < out.rows; r++)
            {
                float cx = out.at<float>(r, 0);
                float cy = out.at<float>(r, 1);
                float w = out.at<float>(r, 2);
                float h = out.at<float>(r, 3);
                float sc = out.at<float>(r, 4);
                cv::Mat confs = out.row(r).colRange(5,out.cols);
                confs*=sc;
                double minV=0,maxV=0;
                double *minI=&minV;
                double *maxI=&maxV;
                cv::minMaxIdx(confs,minI,maxI);
                confidences.push_back(maxV);
                boxes.push_back(cv::Rect(cx - w / 2, cy - h / 2, w, h));
                classIDs.push_back(r);
            }
            cv::dnn::NMSBoxes(boxes, confidences, 0.25f, 0.45f, classIDs);
            
            cv::Mat ret = cv::Mat::zeros(INPUT_WIDTH, INPUT_HEIGHT, CV_8UC4);//image.rows, image.cols, CV_8UC4);
            cv::cvtColor(mat, ret, cv::COLOR_RGB2RGBA);
            cv::resize(ret, ret, image.dimensions());
            
            Vec2 scale(image.cols / double(INPUT_WIDTH),
                       image.rows / double(INPUT_HEIGHT));
            //Vec2 scale(1, 1);
            for (auto &ind : classIDs)
            {
                std::cout << Vec2((double)boxes[ind].x - (double)boxes[ind].width * 0.5, (double)boxes[ind].y - boxes[ind].height * 0.5).mul(scale).toStr() << " " <<
                Vec2(boxes[ind].width, boxes[ind].height).mul(scale).toStr() << ":" << confidences[ind] << std::endl;
                Size2 dim = Vec2(boxes[ind].width, boxes[ind].height);
                Vec2 pos = Vec2{
                    (float)boxes[ind].x, //- dim.width * 0.5f,
                    (float)boxes[ind].y, //- dim.height * 0.5f
                }.mul(scale);
                
                cv::rectangle(ret, pos, pos + dim.mul(scale), Blue, 1);
                
                std::vector<HorizontalLine> lines;
                std::vector<uchar> pixels;
                for(int y = pos.y; y < pos.y + dim.height; ++y) {
                    HorizontalLine line{
                        (coord_t)saturate(int(y), int(0), int(y + dim.height - 1)),
                        (coord_t)saturate(int(pos.x), int(0), int(pos.x + dim.width - 1)),
                        (coord_t)saturate(int(pos.x + dim.width), int(0), int(pos.x + dim.width - 1))
                    };
                    for(int x = line.x0; x <= line.x1; ++x) {
                        pixels.emplace_back(image.get().at<cv::Vec4b>(y, x)[0]);
                    }
                    //pixels.insert(pixels.end(), (uchar*)mat.ptr(y, line.x0),
                    //              (uchar*)mat.ptr(y, line.x1));
                    lines.emplace_back(std::move(line));
                }
                
                pv::Blob blob(lines, pixels, 0);
                if(blob.bounds().size().min() > 1) {
                    //auto[p, img] = blob.image();
                    //tf::imshow("blob", img->get());
                }
                frame.add_object(lines, pixels, 0);
            }*/
            
            py::schedule([ref = Image::Make(image), image = Image::Make(image), &frame, &predictions, &pred_mutex, &outline_points]() mutable {
                using py = track::PythonIntegration;
                py::check_module("bbx_saved_model");
                std::vector<Image::UPtr> images;
                
                images.emplace_back(std::move(image));
                py::set_variable("image", std::move(images), "bbx_saved_model");
                py::set_function("receive", [&](std::vector<float> vector){
                    //print(vector);
                    std::unique_lock guard(pred_mutex);
                    
                    for(size_t i=0; i<vector.size(); i+=4+2) {
                        float conf = vector.at(i);
                        float cls = vector.at(i+1);
                        Vec2 pos(vector.at(i+2), vector.at(i+3));
                        Size2 dim(vector.at(i+4) - pos.x, vector.at(i+5) - pos.y);
                        
                        std::vector<HorizontalLine> lines;
                        std::vector<uchar> pixels;
                        for(int y = pos.y; y < pos.y + dim.height; ++y) {
                            if(y < 0)
                                continue;
                            
                            HorizontalLine line{
                                (coord_t)saturate(int(y), int(0), int(y + dim.height - 1)),
                                (coord_t)saturate(int(pos.x), int(0), int(pos.x + dim.width - 1)),
                                (coord_t)saturate(int(pos.x + dim.width), int(0), int(pos.x + dim.width - 1))
                            };
                            for(int x = line.x0; x <= line.x1; ++x) {
                                pixels.emplace_back(ref->get().at<cv::Vec4b>(y, x)[0]);
                            }
                            //pixels.insert(pixels.end(), (uchar*)mat.ptr(y, line.x0),
                            //              (uchar*)mat.ptr(y, line.x1));
                            lines.emplace_back(std::move(line));
                        }
                        
                        pv::Blob blob(lines, 0);
                        predictions[frame.index()][blob.blob_id()] = { float(conf), int(cls) };
                        /*if(blob.bounds().size().min() > 1) {
                            auto[p, img] = blob.image();
                            tf::imshow("blob", img->get());
                        }*/
                        //print(blob);
                        frame.add_object(lines, pixels, 0);
                    }
                    
                }, "bbx_saved_model");
                
                py::set_function("receive_seg", [&](std::vector<uchar> masks, std::vector<float> vector){
                    //print(vector);
                    std::unique_lock guard(pred_mutex);
                    size_t N = vector.size() / 6u;
                    outline_points.clear();

                    for (size_t i = 0; i < N; ++i) {
                        float conf = vector.at(i * 6);
                        float cls = vector.at(i * 6 + 1);
                        Vec2 pos(vector.at(i * 6 + 2), vector.at(i * 6 + 3));
                        Size2 dim(vector.at(i * 6 + 4) - pos.x, vector.at(i * 6 + 5) - pos.y);
 
                        if (SETTING(filter_class).value<bool>() && cls != 14)
                            continue;
                        if (dim.min() < 1)
                            continue;

                        /*std::vector<uchar> vec(masks.data() + i * 54 * 54, masks.data() + (i + 1) * 54 * 54);
                        for (auto v : vec)
                            printf("%03u ", v);
                        printf("\n");*/
                        //std::span<uchar, 54 * 54> view(masks.data() + i * 54 * 54, masks.data() + (i+1) * 54 * 54);
                        //Image image(54, 54, 1, view.data());
                        Image image(56, 56, 1, masks.data() + i * 56 * 56);
                        cv::Mat tmp, t;
                        cv::resize(image.get(), tmp, dim);
                        cv::threshold(tmp, t, 150, 255, cv::THRESH_BINARY);
                        cv::bitwise_and(tmp, t, tmp);
                        //cv::subtract(255, tmp, tmp);
                        
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
                                outline_points.emplace_back(std::move(*points.front()));
                                //for (auto& pt : outline_points.back())
                                //    pt = (pt + blob.bounds().pos())/*.mul(dim.div(image.dimensions())) + pos*/;
                            }
                            predictions[frame.index()][blob.blob_id()] = { float(conf), int(cls) };
                            frame.add_object(std::move(pair));
                            //auto big = pixel::threshold_get_biggest_blob(&blob, 1, nullptr);
                            //auto [pos, img] = big->image();
                            
                            if (i == 0 && frame.index().get() % 10 == 0) {
                                auto [pos, img] = blob.image();
                                cv::Mat vir = cv::Mat::zeros(img->rows, img->cols, CV_8UC3);
                                auto vit = vir.ptr<cv::Vec3b>();
                                for (auto it = img->data(); it != img->data() + img->size(); ++it, ++vit)
                                    *vit = Viridis::value(*it / 255.0);
                                //tf::imshow("big", img->get());
                                tf::imshow("big", vir);
                            }
                        }
                    }
                    
                }, "bbx_saved_model");

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
            return frame;//Image::Make(ret);
        },
        VideoSource(SETTING(source).value<std::string>())
        //VideoSource("/Users/tristan/goats/DJI_0160.MOV")
        //VideoSource("/Users/tristan/trex/videos/test_frames/frame_%3d.jpg")
    };
    
    video.source.set_colors(VideoSource::ImageMode::RGB);
    
    std::mutex mutex;
    std::condition_variable messages;
    bool _terminate{false};
    decltype(video)::Overlay next, dbuffer;
    
    auto f = std::async(std::launch::async, [&](){
        std::unique_lock guard(mutex);
        while(not _terminate) {
            if(dbuffer.image)
                messages.wait(guard);
            
            if(dbuffer.image) {
                if(next.image) {
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
            
            if(not next.image) {
                auto ts = result.value().overlay.timestamp();
                next = std::move(result.value());
                assert(next.overlay.timestamp() == ts);
            } else {
                dbuffer = std::move(result.value());
            }
        }
    });
    
    struct Menu {
        Button::Ptr
            hi = Button::MakePtr("Quit", attr::Size(50, 35)),
            bro = Button::MakePtr("Filter", attr::Size(50, 35));
        std::shared_ptr<Text> text = std::make_shared<Text>();
        std::shared_ptr<ExternalImage>
            background = std::make_shared<ExternalImage>(),
            overlay = std::make_shared<ExternalImage>();
        Image::UPtr next;
        
        std::shared_ptr<HorizontalLayout> buttons = std::make_shared<HorizontalLayout>(
            std::vector<Layout::Ptr>{
                Layout::Ptr(hi),
                Layout::Ptr(bro),
                Layout::Ptr(text)
            }
        );
        
        Menu() {
            buttons->set_policy(HorizontalLayout::Policy::TOP);
            buttons->set_pos(Vec2(10, 10));
            bro->set_toggleable(true);
            bro->on_click([this](Event e) {
                SETTING(filter_class) = bro->toggled();
            });
            hi->on_click([](auto e) {
                SETTING(terminate) = true;
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
            text->set_txt(Meta::toStr(decltype(video)::_fps.load() / decltype(video)::_samples.load())+"fps");
            
            if(background->source()) {
                //overlay.set_color(Red.alpha(125));
                background->set_scale(Size2(g.width(), g.height()).div( background->source()->dimensions()));
                overlay->set_scale(background->scale());
            }
            
            g.wrap_object(*buttons);
        }
    } menu;
    
    pv::Frame current;
    using namespace track;
    
    SETTING(track_do_history_split) = false;
    SETTING(cm_per_pixel) = Settings::cm_per_pixel_t(0.1);
    SETTING(meta_real_width) = float(expected_size.width * 10);
    SETTING(track_max_speed) = Settings::track_max_speed_t(300);
    SETTING(track_threshold) = Settings::track_threshold_t(0);
    SETTING(blob_size_ranges) = Settings::blob_size_ranges_t({
        Rangef(10,300)
    });
    SETTING(frame_rate) = Settings::frame_rate_t(video.source.framerate());
    SETTING(track_speed_decay) = Settings::track_speed_decay_t(1);
    SETTING(track_max_reassign_time) = Settings::track_max_reassign_time_t(1);
    SETTING(terminate) = false;
    SETTING(calculate_posture) = false;

    cmd.load_settings();
    
    Tracker tracker;
    //cv::Mat bg = cv::Mat::zeros(video.source.size().height, video.source.size().width, CV_8UC1);
    cv::Mat bg = cv::Mat::zeros(expected_size.width, expected_size.height, CV_8UC1);
    bg.setTo(255);
    tracker.set_average(Image::Make(bg));
    
    DrawStructure graph(1024, 768);
    IMGUIBase base("TRexA", graph, [&]()->bool {
        return true;
    }, [](Event) {
        
    });

    auto start_time = std::chrono::system_clock::now();
    PPFrame pp;
    pv::File file(file::DataLocation::parse("output", "tmp.pv"), pv::FileMode::OVERWRITE | pv::FileMode::WRITE);
    std::vector<pv::BlobPtr> objects;
    file.set_average(bg);
    
    std::vector<std::vector<Vec2>> tmp_outline;
    SETTING(filter_class) = false;

    auto fetch_files = [&](){
        //while(not _terminate) 
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(30));

            std::unique_lock guard(mutex);
            if (next.image) {
                objects.clear();

                {
                    std::unique_lock g(pred_mutex);
                    tmp_outline = std::move(outline_points);

                    if (current.index().valid()
                        && predictions.contains(current.index()))
                    {
                        predictions.erase(current.index());
                    }
                }

                current = std::move(next.overlay);
                
                for (size_t i = 0; i < current.n(); ++i) {
                    auto blob = current.blob_at(i);
                    //print(blob->bounds());
                    //auto bds = blob->bounds();
                    {
                        std::unique_lock g(pred_mutex);
                        if (predictions.contains(current.index())) {
                            //print(blob->blob_id(), " in ", current.index(), ": ", predictions.at(current.index()).contains(blob->blob_id()));
                        }
                        else
                            print("no current frame ", current.index());
                    }

                    objects.emplace_back(std::move(blob));

                    //bds = Bounds(bds.pos().mul(scale), bds.size().mul(scale));
                    //print(blob->bounds(), " -> ", bds);
                    //graph.rect(bds, attr::FillClr(Red.alpha(25)), attr::LineClr(Red));
                    //auto [pos, img] = blob->image();
                    //graph.image(pos.mul(scale), std::move(img), scale, White.alpha(25));
                }

                menu.background->set_source(std::move(next.image));
                //menu.overlay.set_source(std::move(next.overlay));
                if (not file.is_open()) {
                    file.set_start_time(start_time);
                    file.set_resolution(expected_size);
                }
                file.add_individual(pv::Frame(current));

                Tracker::preprocess_frame(file, std::move(current), pp, nullptr, PPFrame::NeedGrid::NoNeed, false);
                tracker.add(pp);
                if (pp.index().get() % 100 == 0) {
                    print(IndividualManager::num_individuals(), " individuals known in frame ", pp.index());
                }
                messages.notify_one();
            }
        }
    };
    
    gui::SFLoop loop(graph, &base, [&](gui::SFLoop&, LoopStatus) {
        fetch_files();

            graph.set_scale(1. / base.dpi_scale());
            menu.buttons->set_scale(graph.scale().reciprocal());
            menu.draw(graph);
        auto scale = Size2(graph.width(), graph.height()).div( menu.background->source()->dimensions());
            
        for(auto& blob : objects) {
            const auto bds = blob->bounds().mul(scale);
            //bds = Bounds(bds.pos().mul(scale), bds.size().mul(scale));
            graph.rect(bds, attr::LineClr(Gray), attr::FillClr(Gray.alpha(25)));
            
            float p{-1};
            int clid{-1};
            {
                std::unique_lock guard(pred_mutex);
                if(Tracker::end_frame().valid()
                   && predictions.contains(Tracker::end_frame()))
                {
                    auto &m = predictions.at(Tracker::end_frame());
                    if(m.contains(blob->blob_id())) {
                        auto r = m.at(blob->blob_id());
                        p = std::get<0>(r);
                        clid = std::get<1>(r);
                    } else {
                        print("[draw] blob ", blob->blob_id(), " not found...");
                    }
                }
            }
            
            graph.text(Meta::toStr(clid)+":"+Meta::toStr(p) + " - " + Meta::toStr(blob->num_pixels()) + " " + Meta::toStr(blob->recount(FAST_SETTING(track_threshold), *tracker.background())), attr::Loc(bds.pos() - Vec2(0, 10)), attr::FillClr(White.alpha(100)), attr::Font(0.75));
        }

        if (not tmp_outline.empty()) {

            graph.section("track", [&](auto& , Section* s) {
                s->set_scale(scale);

                ColorWheel wheel;
                for (const auto& v : tmp_outline) {
                    auto clr = wheel.next();
                    graph.line(v, 1, White);
                }
            });
            
        }
        
        using namespace track;
        IndividualManager::transform_all([&](Idx_t fdx, Individual* fish)
        {
            if(not fish->has(tracker.end_frame()))
                return;
            auto p = fish->iterator_for(tracker.end_frame());
            auto segment = p->get();
            
            auto basic = fish->compressed_blob(tracker.end_frame());
            auto bds = basic->calculate_bounds().mul(scale);
            std::vector<Vertex> line;
            const SegmentInformation* previous{nullptr};
            fish->iterate_frames(Range(tracker.end_frame().try_sub(15_f), tracker.end_frame()), [&](Frame_t i, const std::shared_ptr<SegmentInformation> &ptr, const BasicStuff *basic, const PostureStuff *) -> bool
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
