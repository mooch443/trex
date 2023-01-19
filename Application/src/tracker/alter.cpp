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

using namespace cmn;

template<typename T>
concept overlay_function = requires {
    requires std::invocable<T, const Image&>;
    //{ std::invoke_result<T, const Image&>::type } -> std::convertible_to<Image::UPtr>;
};

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
                cv::resize(buffer, download_buffer, Size2(640, 640));
                cv::cvtColor(download_buffer, download_buffer, cv::COLOR_BGR2RGBA);
                
                
                ++i;
                //print("Loaded frame ", i);
                return Image::Make(download_buffer);
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
    
    using namespace cmn;
    namespace py = Python;
    CommandLine cmd(argc, argv);
    py::init();
    
    py::schedule([](){
        using py = track::PythonIntegration;
        print("Scheduled");
        
        py::ModuleProxy proxy{"bbx_saved_model"};
        py::check_module("bbx_saved_model");
        //py::set_variable("model_path", file::Path("/Users/tristan/Downloads/best_saved_model_octopus/best_saved_model").str(), "bbx_saved_model");
        //py::set_variable("model_path", file::Path("/Users/tristan/Downloads/best_saved_model/best_saved_model/best_saved_model").str(), "bbx_saved_model");
        proxy.set_variable("model_type", "yolo7");
        proxy.set_variable("model_path", "/Users/tristan/Downloads/tfmodel");
        proxy.set_variable("image_size", 640);
        
        //py::set_variable("image_size", 1280, "bbx_saved_model");
        py::run("bbx_saved_model", "load_model");
        
    }).get();
    
    //auto model = cv::dnn::readNetFromONNX("/Users/tristan/Downloads/best_octopus.onnx");
    //model.setPreferableTarget(CV_DNN_INFERENCE_ENGINE_CPU_TYPE_ARM_COMPUTE);
    
    
    //cv::dnn::blobFromImage(mat, blob, 1 / 255.0f, cv::Size(img.cols, img.rows), cv::Scalar(0, 0, 0), true, false);
    OverlayedVideo video{
        [&](const Image& image) -> pv::Frame {
            pv::Frame frame;
            
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
            
            py::schedule([ref = Image::Make(image), image = Image::Make(image), &frame]() mutable {
                using py = track::PythonIntegration;
                py::check_module("bbx_saved_model");
                std::vector<Image::UPtr> images;
                
                images.emplace_back(std::move(image));
                py::set_variable("image", std::move(images), "bbx_saved_model");
                py::set_function("receive", [&](std::vector<int> vector){
                    //print(vector);
                    
                    for(size_t i=0; i<vector.size(); i+=4) {
                        Vec2 pos(vector.at(i), vector.at(i+1));
                        Size2 dim(vector.at(i+2) - pos.x, vector.at(i+3) - pos.y);
                        
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
                        
                        //pv::Blob blob(lines, pixels, 0);
                        //if(blob.bounds().size().min() > 1) {
                            //auto[p, img] = blob.image();
                            //tf::imshow("blob", img->get());
                        //}
                        //print(blob);
                        frame.add_object(lines, pixels, 0);
                    }
                    
                }, "bbx_saved_model");
                py::run("bbx_saved_model", "apply");
                py::unset_function("receive", "bbx_saved_model");
                
            }).get();
            
            //tf::imshow("test", ret);
            static Frame_t running_id = 0_f;
            frame.set_index(running_id++);
            return frame;//Image::Make(ret);
        },
        VideoSource("/Users/tristan/Downloads/MOV_0008.mp4")
        //VideoSource("/Users/tristan/goats/DJI_0160.MOV")
        //VideoSource("/Users/tristan/trex/videos/test_frames/frame_%3d.jpg")
    };
    
    video.source.set_colors(VideoSource::ImageMode::RGB);
    
    std::mutex mutex;
    bool _terminate{false};
    decltype(video)::Overlay next;
    
    auto f = std::async(std::launch::async, [&](){
        Timer timer;
        while(not _terminate) {
            if(timer.elapsed() > 1.0 / double(video.source.framerate())) {
                timer.reset();
                
                auto result = video.generate();
                if(not result) {
                    // end of video
                    video.reset();
                    continue;
                }
                
                
                std::unique_lock guard(mutex);
                next = std::move(result.value());
                
            } else
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
    
    struct Menu {
        Button::Ptr
            hi = Button::MakePtr("Hi", attr::Size(50, 35)),
            bro = Button::MakePtr("bro", attr::Size(50, 35));
        std::shared_ptr<Text> text = std::make_shared<Text>();
        std::shared_ptr<ExternalImage>
            background = std::make_shared<ExternalImage>(),
            overlay = std::make_shared<ExternalImage>();
        Image::UPtr next;
        
        std::shared_ptr<HorizontalLayout> buttons = std::make_shared<HorizontalLayout>(
            std::vector<Layout::Ptr>{
                Layout::Ptr(text),
                Layout::Ptr(hi),
                Layout::Ptr(bro)
            }
        );
        
        Menu() {
            buttons->set_policy(HorizontalLayout::Policy::TOP);
            buttons->set_pos(Vec2(10, 10));
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
    Tracker tracker;
    tracker.set_average(Image::Make(cv::Mat::zeros(video.source.size().height, video.source.size().width, CV_8UC1)));
    
    DrawStructure graph(1024, 768);
    IMGUIBase base("TRexA", graph, [&]()->bool {
        return true;
    }, [](Event) {
        
    });
    
    PPFrame pp;
    pv::File file("tmp.pv", pv::FileMode::OVERWRITE | pv::FileMode::WRITE);
    
    gui::SFLoop loop(graph, &base, [&](gui::SFLoop&, LoopStatus) {
        
            {
                std::unique_lock guard(mutex);
                if(next.image) {
                    
                    current = std::move(next.overlay);
                    menu.background->set_source(std::move(next.image));
                    //menu.overlay.set_source(std::move(next.overlay));
                    
                    //Tracker::preprocess_frame(file, std::move(current), pp, nullptr, PPFrame::NeedGrid::NoNeed, false);
                    //tracker.add(pp);
                }
            }
            
            graph.set_scale(1. / base.dpi_scale());
            menu.buttons->set_scale(graph.scale().reciprocal());
            menu.draw(graph);
        auto scale = Size2(graph.width(), graph.height()).div( menu.background->source()->dimensions());
            for(size_t i=0; i<current.n(); ++i) {
                auto blob = current.blob_at(i);
                //print(blob->bounds());
                auto bds = blob->bounds();
                bds = Bounds(bds.pos().mul(scale), bds.size().mul(scale));
                //print(blob->bounds(), " -> ", bds);
                graph.rect(bds, attr::FillClr(Red.alpha(25)), attr::LineClr(Red));
                auto [pos, img] = blob->image();
                graph.image(pos.mul(scale), std::move(img), scale, White.alpha(25));
            }
        
    });
    
    {
        std::unique_lock guard(mutex);
        _terminate = true;
    }
    f.get();
    graph.root().set_stage(nullptr);
    py::deinit();
    return 0;
}
