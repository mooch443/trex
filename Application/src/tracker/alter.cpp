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
#include <grabber/misc/default_config.h>
#include <gui/DynamicGUI.h>
#include <gui/SettingsDropdown.h>
#include "Alterface.h"
#include <misc/RepeatedDeferral.h>
#include <GitSHA1.h>

#include <misc/TaskPipeline.h>

using namespace cmn;

struct TileImage;
using useMat = gpuMat;

std::string date_time() {
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];
    
    time (&rawtime);
    timeinfo = localtime(&rawtime);
    
    strftime(buffer,sizeof(buffer),"%d-%m-%Y %H:%M:%S",timeinfo);
    std::string str(buffer);
    return str;
}

template<typename T>
concept overlay_function = requires {
    requires std::invocable<T, TileImage&&>;
    //{ std::invoke_result<T, const Image&>::type } -> std::convertible_to<Image::Ptr>;
};

using namespace gui;

class Scene {
    GETTER(std::string, name)
    std::vector<Layout::Ptr> _children;
    const Base* _window{nullptr};
    std::function<void(Scene&, DrawStructure& base)> _draw;
    
    static inline std::mutex _mutex;
    static inline std::unordered_map<const DrawStructure*, std::string> _active_scenes;
    
public:
    Scene(const Base& window, const std::string& name, std::function<void(Scene&, DrawStructure& base)> draw)
        : _name(name), _window(&window), _draw(draw)
    {
        
    }
    
    auto window() const { return _window; }
    virtual ~Scene() {
        deactivate();
    }
    
    virtual void activate() {
        print("Activating scene ", _name);
    }
    virtual void deactivate() {
        print("Deactivating scene ", _name);
    }
    
    void draw(DrawStructure& base) {
        _draw(*this, base);
    }
};

class SceneManager {
    Scene* active_scene{nullptr};
    std::map<std::string, Scene*> _scene_registry;
    std::queue<std::function<void()>> _queue;
    std::mutex _mutex;
    
    // Private constructor to prevent external instantiation
    SceneManager() {}
    
public:
    // Deleted copy constructor and assignment operator
    SceneManager(const SceneManager&) = delete;
    SceneManager& operator=(const SceneManager&) = delete;
    
    static SceneManager& getInstance() {
        static SceneManager instance;  // Singleton instance
        return instance;
    }

    void set_active(Scene* scene) {
        auto fn = [this, scene]() {
            if (active_scene && active_scene != scene) {
                active_scene->deactivate();
            }
            active_scene = scene;
            if (scene)
                scene->activate();
        };
        enqueue(fn);
    }
    
    void register_scene(Scene* scene) {
        std::unique_lock guard{_mutex};
        _scene_registry[scene->name()] = scene;
    }
    
    void set_active(std::string name) {
        if (name.empty()) {
            set_active(nullptr);
            return;
        }
        
        Scene* ptr{nullptr};
        
        if (std::unique_lock guard{_mutex};
            _scene_registry.contains(name))
        {
            ptr = _scene_registry.at(name);
        }
        
        if (ptr) {
            set_active(ptr);
        } else {
            throw std::invalid_argument("Cannot find the given Scene name.");
        }
    }
    
    ~SceneManager() {
        update_queue();
        if (active_scene)
            active_scene->deactivate();
    }
    
    void update(DrawStructure& graph) {
        update_queue();
        if (active_scene)
            active_scene->draw(graph);
    }
    
    void update_queue() {
        std::unique_lock guard{_mutex};
        while (not _queue.empty()) {
            auto f = std::move(_queue.front());
            _queue.pop();
            guard.unlock();
            try {
                f();
            } catch (...) {
                // pass
            }
            guard.lock();
        }
    }
    
private:
    void enqueue(auto&& task) {
        std::unique_lock guard(_mutex);
        _queue.push(std::move(task));
    }
};

static Size2 expected_size(640, 640);
using namespace gui;

struct TileImage {
    Size2 tile_size;
    SegmentationData data;
    std::vector<Image::Ptr> images;
    inline static useMat resized, converted, thresholded;
    inline static cv::Mat download_buffer;
    std::vector<Vec2> _offsets;
    Size2 source_size, original_size;
    std::promise<SegmentationData> promise;
    std::function<void()> callback;

    inline static std::vector<Image::Ptr> buffers;
    inline static std::mutex buffer_mutex;

    static void move_back(Image::Ptr&& ptr) {
        if (std::unique_lock guard{ buffer_mutex }; 
            ptr) 
        {
            buffers.emplace_back(std::move(ptr));
        }
    }
    
    TileImage() = default;
    TileImage(TileImage&&) = default;
    TileImage(const TileImage&) = delete;
    
    TileImage& operator=(TileImage&&) = default;
    TileImage& operator=(const TileImage&) = delete;
    
    TileImage(const useMat& source, Image::Ptr&& original, Size2 tile_size, Size2 original_size)
        : tile_size(tile_size),
          source_size(source.cols, source.rows),
          original_size(original_size)
    {
        data.image = std::move(original);
        
        static const auto get_buffer = []() {
            if (std::unique_lock guard{ buffer_mutex };
                not buffers.empty())
            {
                auto buffer = std::move(buffers.back());
                buffers.pop_back();
                return buffer;
            }
            else {
                return Image::Make();
            }
        };
        
        if(tile_size.width == source.cols
           && tile_size.height == source.rows)
        {
            source_size = tile_size;
            auto buffer = get_buffer();
            buffer->create(source);
            images.emplace_back(std::move(buffer));
            //images.emplace_back(Image::Make(source));
            _offsets = {Vec2()};
        }
        else if(tile_size.width > source.cols
             || tile_size.height > source.rows)
        {
            source_size = tile_size;
            cv::resize(source, resized, tile_size);

            auto buffer = get_buffer();
            buffer->create(resized);
            images.emplace_back(std::move(buffer));
            //images.emplace_back(Image::Make(resized));
            _offsets = {Vec2()};
            
        } else {
            useMat tile = useMat::zeros(tile_size.height, tile_size.width, CV_8UC3);
            for(int y = 0; y < source.rows; y += tile_size.height) {
                for(int x = 0; x < source.cols; x += tile_size.width) {
                    Bounds bds = Bounds(x, y, tile_size.width, tile_size.height);
                    _offsets.push_back(Vec2(x, y));
                    bds.restrict_to(Bounds(0, 0, source.cols, source.rows));
                    
                    source(bds).copyTo(tile(Bounds{bds.size()}));

                    auto buffer = get_buffer();
                    buffer->create(tile);
                    images.emplace_back(std::move(buffer));
                    //images.emplace_back(Image::Make(tile));
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

ENUM_CLASS(ObjectDetectionType, yolo7, yolo7seg, customseg);
static inline std::atomic<float> _fps{0}, _samples{0};
static inline std::atomic<float> _network_fps{0}, _network_samples{0};
static inline std::atomic<float> _video_fps{ 0 }, _video_samples{ 0 };

static ObjectDetectionType::Class detection_type() {
    return SETTING(detection_type).value<ObjectDetectionType::Class>();
}

template<typename T>
concept MultiObjectDetection = requires (std::vector<TileImage> tiles) {
    { T::apply(std::move(tiles)) };
    //{ T::receive(data, Vec2{}, {}) };
};

template<typename T>
concept SingleObjectDetection = requires (TileImage tiles) {
    { T::apply(std::move(tiles)) } -> std::convertible_to<tl::expected<SegmentationData, const char*>>;
    //{ T::receive(data, Vec2{}, {}) };
};

template<typename T>
concept ObjectDetection = MultiObjectDetection<T> || SingleObjectDetection<T>;

struct Yolo7ObjectDetection {
    Yolo7ObjectDetection() = delete;
    
    static void reinit(track::PythonIntegration::ModuleProxy& proxy) {
        proxy.set_variable("model_type", detection_type().toStr());
        proxy.set_variable("model_path", SETTING(model).value<file::Path>().str());
        if(SETTING(segmentation_model).value<file::Path>().exists()) {
            proxy.set_variable("segmentation_path", SETTING(segmentation_model).value<file::Path>().str());
            proxy.set_variable("segmentation_resolution", (uint64_t)SETTING(segmentation_resolution).value<uint16_t>());
        }
        proxy.set_variable("image_size", expected_size);
        proxy.run("load_model");
    }
    
    static void init() {
        Python::schedule([](){
            using py = track::PythonIntegration;
            py::ModuleProxy proxy{
                "bbx_saved_model",
                Yolo7ObjectDetection::reinit
            };
        }).get();
    }
    
    static void receive(SegmentationData& data, Vec2 scale_factor, const std::span<float>& vector) {
        //thread_print("Received seg-data for frame ", data.frame.index());
        static const auto meta_encoding = SETTING(meta_encoding).value<grab::default_config::meta_encoding_t::Class>();
        for(size_t i=0; i<vector.size(); i+=4+2) {
            float conf = vector[i];
            float cls = vector[i+1];
            
            if (SETTING(do_filter).value<bool>() && not contains(SETTING(filter_classes).value<std::vector<uint8_t>>(), cls))
                continue;
            
            Vec2 pos = Vec2(vector[i+2], vector[i+3]);
            Size2 dim = Size2(vector[i+4] - pos.x, vector[i+5] - pos.y).mul(scale_factor);
            pos = pos.mul(scale_factor);
            
            std::vector<HorizontalLine> lines;
            std::vector<uchar> pixels;
            auto conversion = [&]<ImageMode mode>(){
                for(int y = pos.y; y < pos.y + dim.height; ++y) {
                    // integer overflow deals with this, lol
                    if(/*y < 0 || */uint(y) >= data.image->rows)
                        continue;
                    
                    HorizontalLine line{
                        (coord_t)saturate(int(y), int(0), int(y + dim.height - 1)),
                        (coord_t)saturate(int(pos.x), int(0), int(pos.x + dim.width - 1)),
                        (coord_t)saturate(int(pos.x + dim.width), int(0), int(min(data.image->cols-1.f, pos.x + dim.width - 1)))
                    };
                    
                    const auto channel = SETTING(color_channel).value<uint8_t>() % 3;
                    auto mat = data.image->get();
                    for(int x = line.x0; x <= line.x1; ++x) {
                        if constexpr (mode == ImageMode::R3G3B2) {
                            pixels.emplace_back(vec_to_r3g3b2(mat.at<cv::Vec3b>(y, x)));
                        } else {
                            pixels.emplace_back(mat.at<cv::Vec3b>(y, x)[channel]);
                        }
                    }
                    
                    lines.emplace_back(std::move(line));
                }
            };
            
            
            if(meta_encoding == grab::default_config::meta_encoding_t::r3g3b2)
                conversion.operator() <ImageMode::R3G3B2>();
            else
                conversion.operator() <ImageMode::GRAY>();
            //cv::Mat full_image;
            //cv::Mat back;
            //convert_to_r3g3b2(data.image->get(), full_image);
            //convert_from_r3g3b2(full_image, back);
            //cv::cvtColor(back, back, cv::COLOR_BGR2RGB);
            
            //tf::imshow("mat", full_image);
            //tf::imshow("back2", back);
            
            if(not lines.empty()) {
                pv::Blob blob(lines, 0);
                data.predictions[blob.blob_id()] = { .clid = size_t(cls), .p = float(conf) };
                data.frame.add_object(lines, pixels, 0, blob::Prediction{ .clid = uint8_t(cls), .p = uint8_t(float(conf) * 255.f) });
            }
        }
    }
    
    static void apply(std::vector<TileImage>&& tiles) {
        namespace py = Python;
        std::vector<Image::Ptr> images;
        std::vector<Image::Ptr> oimages;
        std::vector<SegmentationData> datas;
        std::vector<Vec2> scales;
        std::vector<Vec2> offsets;
        std::vector<std::promise<SegmentationData>> promises;
        std::vector<std::function<void()>> callbacks;
        
        for(auto&& tiled : tiles) {
            images.insert(images.end(), std::make_move_iterator(tiled.images.begin()), std::make_move_iterator(tiled.images.end()));
            if(tiled.images.size() == 1)
                oimages.emplace_back(Image::Make(*tiled.data.image));
            else
                FormatWarning("Cannot use oimages with tiled.");
            
            promises.push_back(std::move(tiled.promise));
            
            scales.push_back( SETTING(output_size).value<Size2>().div(tiled.source_size));
            //print("Image scale: ", scale, " with tile source=", tiled.source_size, " image=", data.image->dimensions()," output_size=", SETTING(output_size).value<Size2>(), " original=", tiled.original_size);
            
            for(auto p : tiled.offsets()) {
                tiled.data.tiles.push_back(Bounds(p.x, p.y, tiled.tile_size.width, tiled.tile_size.height).mul(scales.back()));
            }
            
            auto o = tiled.offsets();
            offsets.insert(offsets.end(), o.begin(), o.end());
            datas.emplace_back(std::move(tiled.data));
            callbacks.emplace_back(tiled.callback);
        }
        
        py::schedule([datas = std::move(datas),
                      images = std::move(images),
                      oimages = std::move(oimages),
                      scales = std::move(scales),
                      offsets = std::move(offsets),
                      callbacks = std::move(callbacks),
                      promises = std::move(promises)]() mutable
        {
            Timer timer;
            using py = track::PythonIntegration;

            const size_t _N = datas.size();
            py::ModuleProxy bbx("bbx_saved_model", Yolo7ObjectDetection::reinit);
            bbx.set_variable("offsets", std::move(offsets));
            bbx.set_variable("image", images);
            bbx.set_variable("oimages", oimages);
            
            auto recv = [&](std::vector<uint64_t> Ns,
                            std::vector<float> vector)
            {
                size_t elements{0};
                //thread_print("Received a number of results: ", Ns);
                //thread_print("For elements: ", datas);

                if(Ns.empty()) {
                    for(size_t i=0; i<datas.size(); ++i) {
                        try {
                            promises.at(i).set_value(std::move(datas.at(i)));
                        } catch(...) {
                            promises.at(i).set_exception(std::current_exception());
                        }
                        
                        try {
                            callbacks.at(i)();
                        } catch(...) {
                            FormatExcept("Exception in callback of element ", i," in python results.");
                        }
                    }
                    FormatExcept("Empty data for ", datas);
                    return;
                }

                assert(Ns.size() == datas.size());
                for(size_t i=0; i<datas.size(); ++i) {
                    auto& data = datas.at(i);
                    auto& scale = scales.at(i);
                    
                    std::span<float> span(vector.data() + elements * 6u,
                                          vector.data() + (elements + Ns.at(i)) * 6u);
                    elements += Ns.at(i);
                    
                    try {
                        receive(data, scale, span);
                        promises.at(i).set_value(std::move(data));
                    } catch(...) {
                        promises.at(i).set_exception(std::current_exception());
                    }
                    
                    try {
                        callbacks.at(i)();
                    } catch(...) {
                        FormatExcept("Exception in callback of element ", i," in python results.");
                    }
                }
            };
            
            bbx.set_function("receive", recv);
            bbx.set_function("receive_with_seg", [&](std::vector<uint64_t> Ns,
                                        std::vector<float> vector,
                                        std::vector<float> masks,
                                        std::vector<float> meta,
                                        std::vector<int> indexes,
                                        std::vector<int> segNs)
            {
                thread_print("Received masks:", masks.size(), " -> ", double(masks.size()) / 56.0 / 56.0);
                thread_print("Received meta:", meta.size());
                thread_print("Received indexes:", indexes);
                thread_print("Received segNs:", segNs);

                std::unordered_map<size_t, std::unique_ptr<cv::Mat>> converted_images;
                const auto threshold = saturate(float(SETTING(threshold).value<int>()), 0.f, 255.f) / 255.0;
                
                //size_t offset = 0;
                for(size_t offset = 0; offset < indexes.size(); ++offset) {
                    auto idx = indexes.at(offset);
                    //auto N = segNs.at(idx);
                    //if(N == 0)
                    //    continue;
                    
                    auto &data = datas.at(idx);
                    const cv::Mat* full_image;
                    if(not converted_images.contains(idx)) {
                        converted_images[idx] = std::make_unique<cv::Mat>();
                        convert_to_r3g3b2(data.image->get(), *converted_images[idx]);
                        full_image = converted_images[idx].get();
                    } else {
                        full_image = converted_images.at(idx).get();
                    }
                    
                    auto scale_factor = scales.at(idx);
                    
                    assert(meta.size() >= (offset + 1) * 6u);
                    assert(masks.size() >= (offset + 1) * 56u * 56u);
                    std::span<float> m(meta.data() + offset * 6, (offset + 1) * 6);
                    std::span<float> s(masks.data() + offset * 56u * 56u, (offset + 1) * 56u * 56u);
                    
                    thread_print(" * working mask for frame ", data.original_index(), " (", m.size()," and images ",s.size(),")");
                    
                    Vec2 pos(m[0], m[1]);
                    Size2 dim = Size2(m[2] - pos.x, m[3] - pos.y).map(roundf);
                    
                    float conf = m[4];
                    float cls = m[5];
                    
                    //thread_print(" \t - pos: ", pos, " dim: ", dim, " conf: ", conf, " cls: ", cls, " offset: ", offsets.at(idx));
                    
                    pos += offsets.at(idx);
                    
                    //pos = pos.mul(scale_factor);
                    //dim = dim.mul(scale_factor);
                    
                    //thread_print(" \t>> - pos: ", pos, " dim: ", dim, " conf: ", conf, " cls: ", cls, " offset: ", offsets.at(idx));
                    
                    //print(i, vector.at(i*6 + 0), " ", vector.at(i*6 + 1), " ",vector.at(i*6 + 2), " ", vector.at(i*6 + 3));
                    //print("\t->", conf, " ", cls, " ",pos, " ", dim);
                    //print("\tmeta of object = ", m, " offset=", offsets.at(i));
                    
                    if (SETTING(do_filter).value<bool>() && not contains(SETTING(filter_classes).value<std::vector<uint8_t>>(), cls))
                        continue;
                    if (dim.min() < 1)
                        continue;
                    
                    //if(dim.height + pos.y > full_image.rows
                    //   || dim.width + pos.x > full_image.cols)
                    //    continue;
                    {
                        cv::Mat m(56, 56, CV_32FC1, s.data());
                        
                        cv::Mat tmp;
                        cv::resize(m, tmp, dim);
                        
                        //cv::Mat dani;
                        //cv::subtract(cv::Scalar(1.0), tmp, dani);
                        //dani.convertTo(dani, CV_8UC1, 255.0);
                        //tmp.convertTo(dani, CV_8UC1, 255.0);
                        //tf::imshow("dani", dani);
                        
                        cv::threshold(tmp, tmp, threshold, 1.0, cv::THRESH_BINARY);
                        //cv::threshold(tmp, t, 150, 255, cv::THRESH_BINARY);
                        //print(Bounds(pos, dim), " and image ", Size2(full_image), " and t ", Size2(t));
                        //print("using bounds: ", Size2(full_image(Bounds(pos, dim))), " and ", Size2(t));
                        //print("channels: ", full_image.channels(), " and ", t.channels(), " and types ", getImgType(full_image.type()), " ", getImgType(t.type()));
                        cv::Mat d;// = full_image(Bounds(pos, dim));
                        auto restricted = Bounds(pos, dim);
                        restricted.restrict_to(Bounds(*full_image));
                        if(restricted.width <= 0 || restricted.height <= 0)
                            continue;
                        
                        (*full_image)(restricted).convertTo(d, CV_32FC1);
                        
                        //tf::imshow("ref", d);
                        //tf::imshow("tmp", tmp);
                        //tf::imshow("t", t);
                        
                        //print("d(", getImgType(d.type()), ") ",Size2(d)," tmp(", getImgType(tmp.type()), "): ", Size2(tmp));
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
                            
                            pair.pred = blob::Prediction{
                                .clid = static_cast<uint8_t>(cls),
                                .p = uint8_t(float(conf) * 255.f)
                            };
                            
                            pv::Blob blob(*pair.lines, *pair.pixels, pair.extra_flags, pair.pred);
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
                    
                    // move further in all sub arrays on to the next original image
                    //offset += N;
                }
                
                //print("Passing on to recv: Ns=", Ns, " vector=", vector);
                recv(Ns, vector);
                //print("Done.");
            });

            try {
                bbx.run("apply");
            }
            catch (...) {
                FormatWarning("Continue after exception...");
                throw;
            }
            
            bbx.unset_function("receive");
            bbx.unset_function("receive_with_seg");

            for (auto&& img : images) {
                TileImage::move_back(std::move(img));
            }

            if (_network_samples.load() > 10) {
                _network_samples = _network_fps = 0;
            }
            _network_fps = _network_fps.load() + (double(_N) / timer.elapsed());
            _network_samples = _network_samples.load() + 1;
            
        }).get();
    }
};

struct Yolo7InstanceSegmentation {
    Yolo7InstanceSegmentation() = delete;
    
    static void reinit(track::PythonIntegration::ModuleProxy& proxy) {
        proxy.set_variable("model_type", detection_type().toStr());
        proxy.set_variable("model_path", SETTING(segmentation_model).value<file::Path>().str());
        proxy.set_variable("image_size", expected_size);
        proxy.run("load_model");
    }
    
    static void init() {
        Python::schedule([](){
            using py = track::PythonIntegration;
            py::ModuleProxy proxy{"bbx_saved_model", reinit};
            
        }).get();
    }
    
    static void receive(std::vector<Vec2> offsets, SegmentationData& data, Vec2 scale_factor, std::vector<float>& masks, const std::vector<float>& vector, const std::vector<int>& meta) {
        //print(vector);
        size_t N = vector.size() / 6u;
        
        cv::Mat full_image;
        //cv::Mat back;
        convert_to_r3g3b2(data.image->get(), full_image);
        //convert_from_r3g3b2(full_image, back);
        //cv::cvtColor(back, back, cv::COLOR_BGR2RGB);
        
        //tf::imshow("mat", full_image);
        //tf::imshow("back2", back);
        //cv::cvtColor(data.image->get(), full_image, cv::COLOR_RGB2GRAY);
        
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
            
            if (SETTING(do_filter).value<bool>() && not contains(SETTING(filter_classes).value<std::vector<uint8_t>>(), cls))
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
            //cv::subtract(cv::Scalar(1.0), tmp, dani);
            //dani.convertTo(dani, CV_8UC1, 255.0);
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
                
                pair.pred = blob::Prediction{
                    .clid = static_cast<uint8_t>(cls),
                    .p = uint8_t(float(conf) * 255.f)
                };
                
                pv::Blob blob(*pair.lines, *pair.pixels, pair.extra_flags, pair.pred);
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
        
        Vec2 scale = SETTING(output_size).value<Size2>().div(tiled.source_size);
        print("Image scale: ", scale, " with tile source=", tiled.source_size, " image=", tiled.data.image->dimensions()," output_size=", SETTING(output_size).value<Size2>(), " original=", tiled.original_size);
        
        for(auto p : tiled.offsets()) {
            tiled.data.tiles.push_back(Bounds(p.x, p.y, tiled.tile_size.width, tiled.tile_size.height).mul(scale));
        }
        
        py::schedule([&tiled, scale, offsets = tiled.offsets()]() mutable {
            using py = track::PythonIntegration;
            py::ModuleProxy bbx("bbx_saved_model", reinit);
            bbx.set_variable("offsets", std::move(offsets));
            bbx.set_variable("image", tiled.images);
            
            bbx.set_function("receive", [&](std::vector<float> masks, std::vector<float> meta, std::vector<int> indexes) {
                receive(offsets, tiled.data, scale, masks, meta, indexes);
            });

            Timer timer;
            try {
                bbx.run("apply");
            }
            catch (...) {
                FormatWarning("Continue after exception...");
                throw;
            }
            
            bbx.unset_function("receive");
            if (_network_samples.load() > 100) {
                _network_samples = _network_fps = 0;
            }
            _network_fps = _network_fps.load() + 1.0 / timer.elapsed();
            _network_samples = _network_samples.load() + 1;
            
        }).get();
        
        return std::move(tiled.data);
    }
};

static_assert(ObjectDetection<Yolo7ObjectDetection>);
static_assert(ObjectDetection<Yolo7InstanceSegmentation>);

template<typename SourceType>
class AbstractVideoSource {
    //mutable std::mutex mutex;
    Frame_t i{0_f};
    
    using gpuMatPtr = std::unique_ptr<useMat>;
    std::mutex buffer_mutex;
    std::vector<gpuMatPtr> buffers;
    
    SourceType source;
    GETTER(file::Path, base)
    GETTER(Size2, size)
    GETTER(short, framerate)
    
    const bool _finite;
    const Frame_t _length;
    
    RepeatedDeferral<std::function<tl::expected<std::tuple<Frame_t, gpuMatPtr>, const char*>()>> _source_frame;
    //mutable package::packaged_func<tl::expected<std::tuple<Frame_t, gpuMatPtr, Image::Ptr>, const char*>> _post_process;
    //mutable package::packaged_func<void, Frame_t> _set_frame;
    RepeatedDeferral<std::function<tl::expected<std::tuple<Frame_t, gpuMatPtr, Image::Ptr>, const char*>()>> _resize_cvt;
    
    
public:
    template<typename T = SourceType>
        requires _clean_same<SourceType, VideoSource>
    AbstractVideoSource(SourceType&& source)
        :
            source(std::move(source)),
            _base(this->source.base()),
            _source_frame(75u, 25u,
              std::string("source.frame"),
                          [this]() {
                return fetch_next();
              }
            ),
            _size(this->source.size()),
            _framerate(this->source.framerate()),
            _finite(true),
            _length(this->source.length()),
            _resize_cvt(50u, 10u,
                std::string("resize+cvtColor"),
                [this]() -> tl::expected<std::tuple<Frame_t, gpuMatPtr, Image::Ptr>, const char*> {
                    return this->fetch_next_process();
            })
    { }

    AbstractVideoSource(AbstractVideoSource&& other) = default;
    AbstractVideoSource(const AbstractVideoSource& other) = delete;
    AbstractVideoSource() = delete;
    
    AbstractVideoSource& operator=(AbstractVideoSource&&) = delete;
    AbstractVideoSource& operator=(const AbstractVideoSource&) = delete;
    
    void move_back(gpuMatPtr&& ptr) {
        std::unique_lock guard(buffer_mutex);
        buffers.push_back(std::move(ptr));
    }
    
    std::tuple<Frame_t, gpuMatPtr, Image::Ptr> next() {
        auto result = _resize_cvt.next();
        if(not result)
            return std::make_tuple(Frame_t{}, nullptr, nullptr);
        
        return std::move(result.value());
    }
    
    tl::expected<std::tuple<Frame_t, gpuMatPtr>, const char*> fetch_next() {
        if (i >= this->source.length()) {
            //i = 0_f;
            SETTING(terminate) = true;
            return tl::unexpected("EOF");
        }

        try {
            if (not i.valid() or i >= this->source.length()) {
                i = 0_f;
            }

            auto index = i++;

            gpuMatPtr buffer;

            if (std::unique_lock guard{ buffer_mutex };
                not buffers.empty())
            {
                buffer = std::move(buffers.back());
                buffers.pop_back();
            }
            else {
                buffer = std::make_unique<useMat>();
            }

            static gpuMatPtr tmp = std::make_unique<useMat>();
            static Timing timing("Source(frame)", 1);
            TakeTiming take(timing);

            //static cv::Mat tmp;
            //static auto source_size = this->size();
            //if(image->rows != source_size.height || source_size.width != image->cols)
            //    image->create(source_size.height, source_size.width, 3);

            //auto mat = image->get();

            static cv::Mat cpuBuffer;
            this->source.frame(index, cpuBuffer);
            cpuBuffer.copyTo(*buffer);

            cv::cvtColor(*buffer, *tmp, cv::COLOR_BGR2RGB);
            std::swap(buffer, tmp);
            return std::make_tuple(index, std::move(buffer));
        }
        catch (const std::exception& e) {
            return tl::unexpected(e.what());
        }
    }
    
    tl::expected<std::tuple<Frame_t, gpuMatPtr, Image::Ptr>, const char*> fetch_next_process() {
        try {
            Timer timer;
            auto result = _source_frame.next();
            if(result) {
                auto& [index, buffer] = result.value();
                static gpuMatPtr tmp = std::make_unique<useMat>();
                if (not index.valid())
                    throw U_EXCEPTION("Invalid index");
                
                //! resize according to settings
                //! (e.g. multiple tiled image size)
                if (SETTING(meta_video_scale).value<float>() != 1) {
                    Size2 new_size = Size2(buffer->cols, buffer->rows) * SETTING(meta_video_scale).value<float>();
                    //FormatWarning("Resize ", Size2(buffer.cols, buffer.rows), " -> ", new_size);
                    cv::resize(*buffer, *tmp, new_size);
                    std::swap(buffer, tmp);
                }

                //! throws bad optional access if the returned frame is not valid
                assert(index.valid());

                auto image = OverlayBuffers::get_buffer();
                //image->set_index(index.get());
                image->create(*buffer, index.get());
            
                if (_video_samples.load() > 1000) {
                    _video_samples = _video_fps = 0;
                }
                _video_fps = _video_fps.load() + (1.0 / timer.elapsed());
                _video_samples = _video_samples.load() + 1;

                return std::make_tuple(index, std::move(buffer), std::move(image));
                
            } else
                return tl::unexpected(result.error());
                //throw U_EXCEPTION("Unable to load frame: ", result.error());
            
        } catch(const std::exception& e) {
            FormatExcept("Unable to load frame ", i, " from video source ", this->source, " because: ", e.what());
            return tl::unexpected(e.what());
        }
    }
    
    bool is_finite() const {
        return true;
    }
    
    void set_frame(Frame_t frame) {
        if(not is_finite())
            throw std::invalid_argument("Cannot skip on infinite source.");
        i = frame;
    }
    
    Frame_t length() const {
        if(not is_finite())
            throw std::invalid_argument("Cannot return length of infinite source.");
        return _length;
    }
    
    std::string toStr() const {
        return "AbstractVideoSource<"+Meta::toStr(source)+">";
    }
};

template<typename F>
    requires ObjectDetection<F>
struct OverlayedVideo {
    AbstractVideoSource<VideoSource> source;

    F overlay;
    
    mutable std::mutex index_mutex;
    Frame_t i{0};
    //Image::Ptr original_image;
    //cv::Mat downloader;
    useMat resized;
    
    using return_t = tl::expected<std::tuple<Frame_t, std::future<SegmentationData>>, const char*>;
    RepeatedDeferral<std::function<return_t()>> apply_net;
    
    bool eof() const noexcept {
        if(not source.is_finite())
            return false;
        return i >= source.length();
    }
    
    OverlayedVideo() = delete;
    OverlayedVideo(const OverlayedVideo&) = delete;
    OverlayedVideo& operator=(const OverlayedVideo&) = delete;
    OverlayedVideo(OverlayedVideo&&) = delete;
    OverlayedVideo& operator=(OverlayedVideo&&) = delete;
    
    template<typename SourceType, typename Callback>
    OverlayedVideo(F&& fn, SourceType&& s, Callback&& callback)
        : source(AbstractVideoSource<SourceType>(std::move(s))), overlay(std::move(fn)),
            apply_net(50u,
                10u,
                "ApplyNet",
                [this, callback = std::move(callback)](){
                    return retrieve_next(callback);
                })
    {
        
    }
    
    ~OverlayedVideo() {
    }
    
    tl::expected<std::tuple<Frame_t, std::future<SegmentationData>>, const char*> retrieve_next(const std::function<void()>& callback)
    {
        static Timing timing("retrieve_next");
        TakeTiming take(timing);
        
        std::scoped_lock guard(index_mutex);
        TileImage tiled;
        auto loaded = i;
        
        try {
            Timer _timer;
            auto&& [nix, buffer, image] = source.next();
            if(not nix.valid())
                return tl::unexpected("Cannot retrieve frame from video source.");
            
            static double _average = 0, _samples = 0;
            _average += _timer.elapsed() * 1000;
            ++_samples;
            if ((size_t)_samples % 100 == 0) {
                print("Waited for source frame for ", _average / _samples,"ms");
                _samples = 0;
                _average = 0;
            }

            useMat *use { buffer.get() };
            image->set_index(nix.get());
            
            Size2 original_size(use->cols, use->rows);
            
            Size2 new_size(expected_size);
            if(SETTING(tile_image).value<size_t>() > 1) {
                size_t tiles = SETTING(tile_image).value<size_t>();
                float ratio = use->rows / float(use->cols);
                new_size = Size2(expected_size.width * tiles, expected_size.width * tiles * ratio).map(roundf);
                while(use->cols < new_size.width
                      && use->rows < new_size.height
                      && tiles > 0)
                {
                    new_size = Size2(expected_size.width * tiles, expected_size.width * tiles * ratio).map(roundf);
                    tiles--;
                }
            }
            
            if (use->cols != new_size.width || use->rows != new_size.height) {
                cv::resize(*use, resized, new_size);
                use = &resized;
            }
            
            i = nix + 1_f;

            //! tile image to make it ready for processing in the network
            TileImage tiled(*use, std::move(image), expected_size, original_size);
            tiled.callback = callback;
            source.move_back(std::move(buffer));
            
            //thread_print("Queueing image ", nix);
            //! network processing, and record network fps
            return std::make_tuple(nix, this->overlay.apply(std::move(tiled)));
            
        } catch(const std::exception& e) {
            FormatExcept("Error loading frame ", loaded, " from video ", source, ": ", e.what());
            return tl::unexpected("Error loading frame.");
        }
    }
    
    void reset(Frame_t frame) {
        std::scoped_lock guard(index_mutex);
        i = frame;
        if(source.is_finite())
            source.set_frame(i);
    }
    
    //! generates the next frame
    tl::expected<std::tuple<Frame_t, std::future<SegmentationData>>, const char*> generate() noexcept
    {
        if(eof())
            return tl::unexpected("End of file.");
        return apply_net.next();
    }
};

using namespace dyn;
inline static sprite::Map fish =  [](){
    sprite::Map fish;
    fish.set_do_print(false);
    fish["name"] = std::string("fish0");
    fish["color"] = Red;
    fish["pos"] = Vec2(100, 150);
    return fish;
}();

inline static sprite::Map _video_info = [](){
    sprite::Map fish;
    fish.set_do_print(false);
    fish["frame"] = Frame_t();
    fish["resolution"] = Size2();
    return fish;
}();

struct Detection {
    
    Detection() {
        if(type() == ObjectDetectionType::yolo7) {
            Yolo7ObjectDetection::init();
            
        } else if(type() == ObjectDetectionType::yolo7seg || type() == ObjectDetectionType::customseg) {
            Yolo7InstanceSegmentation::init();
            
        } else
            throw U_EXCEPTION("Unknown detection type: ", type());
    }
    
    static ObjectDetectionType::Class type() {
        return SETTING(detection_type).value<ObjectDetectionType::Class>();
    }
    
    static std::future<SegmentationData> apply(TileImage&& tiled) {
        if(type() == ObjectDetectionType::yolo7) {
            auto f = tiled.promise.get_future();
            manager.enqueue(std::move(tiled));
            return f;
        } else if(type() == ObjectDetectionType::yolo7seg || type() == ObjectDetectionType::customseg) {
            std::promise<SegmentationData> p;
            auto e = Yolo7InstanceSegmentation::apply(std::move(tiled));
            try {
                p.set_value(std::move(e.value()));
            } catch(...) {
                p.set_exception(std::current_exception());
            }
            return p.get_future();
        }
        
        throw U_EXCEPTION("Unknown detection type: ", type());
    }
    
    static void apply(std::vector<TileImage>&& tiled) {
        if(type() == ObjectDetectionType::yolo7) {
            Yolo7ObjectDetection::apply(std::move(tiled));
            tiled.clear();
            return;
            
        } else if(type() == ObjectDetectionType::yolo7seg) {
            //return Yolo7InstanceSegmentation::apply(std::move(tiled));
        }
        
        throw U_EXCEPTION("Unknown detection type: ", type());
    }
    
    inline static auto manager = PipelineManager<TileImage>(10.0, [](std::vector<TileImage>&& images) {
        // do what has to be done when the queue is full
        // i.e. py::execute()
        Detection::apply(std::move(images));
    });
};

file::Path average_name() {
    auto path = file::DataLocation::parse("output", "average_" + (std::string)SETTING(filename).value<file::Path>().filename() + ".png");
    return path;
}

std::string window_title() {
    auto output_prefix = SETTING(output_prefix).value<std::string>();
    return SETTING(app_name).value<std::string>()
        + (SETTING(version).value<std::string>().empty() ? "" : (" " + SETTING(version).value<std::string>()))
        + " (" + (std::string)SETTING(filename).value<file::Path>().filename() + ")"
        + (output_prefix.empty() ? "" : (" ["+output_prefix+"]"));
}

class ConvertScene : public Scene {
    // condition variables and mutexes for thread synchronization
    std::condition_variable _cv_messages, _cv_ready_for_tracking;
    std::mutex _mutex_general, _mutex_current;
    std::atomic<bool> _should_terminate{false};

    // Segmentation data for the next frame
    SegmentationData _next_frame_data;

    // Progress and current data for tracking
    SegmentationData _progress_data, _current_data, _transferred_current_data;

    // External images for background and overlay
    std::shared_ptr<ExternalImage> _background_image = std::make_shared<ExternalImage>(),
                                   _overlay_image = std::make_shared<ExternalImage>();

    // Vectors for object blobs and GUI objects
    std::vector<pv::BlobPtr> _object_blobs, _progress_blobs, _transferred_blobs;
    std::vector<std::shared_ptr<VarBase_t>> _gui_objects;

    // Individual properties for each object
    std::vector<sprite::Map> _individual_properties;

    // Overlayed video with detections and tracker for object tracking
    std::unique_ptr<OverlayedVideo<Detection>> _overlayed_video;
    std::unique_ptr<Tracker> _tracker;

    // File for output
    std::unique_ptr<pv::File> _output_file;

    // Threads for tracking and generation
    std::thread _tracking_thread, _generator_thread;

    // Frame data
    Frame_t _actual_frame, _video_frame;

    // Size of output and start time for timing operations
    GETTER(Size2, output_size)
    std::chrono::time_point<std::chrono::system_clock> _start_time;
    
    Alterface menu{
        dyn::Context{
            .actions = {
                {
                    "QUIT", [](auto) {
                        auto& manager = SceneManager::getInstance();
                        manager.set_active("");
                    }
                },
                {
                    "FILTER", [](auto) {
                        static bool filter { false };
                        filter = not filter;
                        SETTING(do_filter) = filter;
                    }
                },
                {
                    "RESET", [this](auto){
                        _overlayed_video->reset(1500_f);
                    }
                }
            },
            
            .variables = {
                {
                    "fps", std::unique_ptr<VarBase_t>(new Variable([](std::string) {
                        return ::_fps.load() / ::_samples.load();
                    }))
                },
                {
                    "net_fps", std::unique_ptr<VarBase_t>(new Variable([](std::string) {
                        return ::_network_fps.load() / ::_network_samples.load();
                    }))
                },
                {
                    "vid_fps", std::unique_ptr<VarBase_t>(new Variable([](std::string) {
                        return ::_video_fps.load() / ::_video_samples.load();
                    }))
                },
                {
                    "fish",
                    std::unique_ptr<VarBase_t>(new Variable([](std::string) -> sprite::Map& {
                        return fish;
                    }))
                },
                {
                    "global",
                    std::unique_ptr<VarBase_t>(new Variable([](std::string) -> sprite::Map& {
                        return GlobalSettings::map();
                    }))
                },
                {
                    "actual_frame", std::unique_ptr<VarBase_t>(new Variable([this](std::string) {
                        return _actual_frame;
                    }))
                },
                {
                    "video", std::unique_ptr<VarBase_t>(new Variable([](std::string) -> sprite::Map& {
                        return _video_info;
                    }))
                }
            }
        },
        [&](const std::string& name) {
            if(name == "gui_frame") {
                _overlayed_video->reset(SETTING(gui_frame).value<Frame_t>());
            }
        }
        
    };
    
public:
    ConvertScene(const Base& window) : Scene(window, "converting", [this](Scene&, DrawStructure& graph){
        _draw(graph);
    }) {
        menu.context.variables.emplace("fishes", new Variable([this](std::string) -> std::vector<std::shared_ptr<VarBase_t>>& {
            return _gui_objects;
        }));
    }
    
    ~ConvertScene() {
        if(not _should_terminate)
            deactivate();
    }
    
private:
    void deactivate() override {
        _should_terminate = true;
        
        std::unique_lock guard(_mutex_general);
        _cv_ready_for_tracking.notify_all();
        _cv_messages.notify_all();
        
        if (_tracking_thread.joinable()) {
            _tracking_thread.join();
        }
        
        Detection::manager.clean_up();
        
        if (_generator_thread.joinable()) {
            _generator_thread.join();
        }
        
        if (_output_file) {
            _output_file->close();
        }
        
        _overlayed_video = nullptr;
        _tracker = nullptr;
        
        pv::File test(_output_file->filename(), pv::FileMode::READ);
        test.print_info();
    }

    void activate() override {
        VideoSource video_base(SETTING(source).value<std::string>());
        video_base.set_colors(ImageMode::RGB);
        
        SETTING(frame_rate) = Settings::frame_rate_t(video_base.framerate() != short(-1) ? video_base.framerate() : 25);
        
        if(SETTING(filename).value<file::Path>().empty()) {
            SETTING(filename) = file::Path(file::Path(video_base.base()).filename());
        }
        
        setDefaultSettings();
        _output_size = (Size2(video_base.size()) * SETTING(meta_video_scale).value<float>()).map(roundf);
        SETTING(output_size) = _output_size;
        _video_info["resolution"] = _output_size;
        
        _overlayed_video = std::make_unique<OverlayedVideo<Detection>>(
            Detection{},
            std::move(video_base),
            [this](){
                _cv_messages.notify_one();
            }
        );
        
        printDebugInformation();
        
        cv::Mat bg = cv::Mat::zeros(_output_size.height, _output_size.width, CV_8UC1);
        bg.setTo(255);
        
        VideoSource tmp(SETTING(source).value<std::string>());
        if(not average_name().exists()) {
            tmp.generate_average(bg, 0);
            cv::imwrite(average_name().str(), bg);
        } else {
            print("Loading from file...");
            bg = cv::imread(average_name().str());
            cv::cvtColor(bg, bg, cv::COLOR_BGR2GRAY);
        }
        
        _tracker = std::make_unique<Tracker>(Image::Make(bg), float(expected_size.width * 10));
        static_assert(ObjectDetection<Detection>);
        
        _start_time = std::chrono::system_clock::now();
        auto filename = file::DataLocation::parse("output", SETTING(filename).value<file::Path>());
        DebugHeader("Output: ", filename);
        
        auto path = filename.remove_filename();
        if(not path.exists()) {
            path.create_folder();
        }
        
        _output_file = std::make_unique<pv::File>(filename, pv::FileMode::OVERWRITE | pv::FileMode::WRITE);
        _output_file->set_average(bg);
        
        _generator_thread = std::thread([this](){
            generator_thread();
        });
        _tracking_thread = std::thread([this](){
            tracking_thread();
        });
    }

    void setDefaultSettings() {
        SETTING(do_filter) = false;
        SETTING(filter_classes) = std::vector<uint8_t>{};
        SETTING(is_writing) = true;
    }

    void printDebugInformation() {
        DebugHeader("Starting tracking of");
        print("average at: ", average_name());
        print("model: ",SETTING(model).value<file::Path>());
        print("video: ", SETTING(source).value<std::string>());
        print("model resolution: ", SETTING(image_width).value<uint16_t>());
        print("output size: ", SETTING(output_size).value<Size2>());
        print("output path: ", SETTING(filename).value<file::Path>());
        print("color encoding: ", SETTING(meta_encoding).value<grab::default_config::meta_encoding_t::Class>());
    }

    void fetch_new_data() {
        static std::once_flag flag;
        std::call_once(flag, [](){
            set_thread_name("GUI");
        });
        
        {
            std::unique_lock guard(_mutex_current);
            if (_transferred_current_data.image) {
                _current_data = std::move(_transferred_current_data);
                _object_blobs = std::move(_transferred_blobs);
            }
        }

        if (_current_data.image) {
            if(_background_image->source()
               && _background_image->source()->rows == _current_data.image->rows
               && _background_image->source()->cols == _current_data.image->cols
               && _background_image->source()->dims == 4)
            {
                cv::cvtColor(_current_data.image->get(), _background_image->unsafe_get_source().get(), cv::COLOR_BGR2BGRA);
                OverlayBuffers::put_back(std::move(_current_data.image));
                _background_image->updated_source();
            } else {
                auto rgba = Image::Make(_current_data.image->rows,
                                        _current_data.image->cols, 4);
                cv::cvtColor(_current_data.image->get(), rgba->get(), cv::COLOR_BGR2BGRA);
                OverlayBuffers::put_back(std::move(_current_data.image));
                _background_image->set_source(std::move(rgba));
            }
            
            _current_data.image = nullptr;
        }
    }
    
    // Helper function to calculate window dimensions
    Size2 calculateWindowSize(const Size2& output_size, const Size2& window_size) {
        auto ratio = output_size.width / output_size.height;
        Size2 wdim;

        if(window_size.width * output_size.height < window_size.height * output_size.width) {
            wdim = Size2(window_size.width, window_size.width / ratio);
        } else {
            wdim = Size2(window_size.height * ratio, window_size.height);
        }

        return wdim;
    }

    // Helper function to draw outlines
    void drawOutlines(DrawStructure& graph, const Size2& scale) {
        if (not _current_data.outlines.empty()) {
            graph.text(Meta::toStr(_current_data.outlines.size())+" lines", attr::Loc(10,50), attr::Font(0.35), attr::Scale(scale.mul(graph.scale()).reciprocal()));
                    
            ColorWheel wheel;
            for (const auto& v : _current_data.outlines) {
                auto clr = wheel.next();
                graph.line(v, 1, clr.alpha(150));
            }
        }
    }
    
    void drawBlobs(const std::vector<std::string>& meta_classes, const Vec2& scale, const std::unordered_map<pv::bid, Identity>& visible_bdx) {
        for(auto &blob : _object_blobs) {
            const auto bds = blob->bounds();
            //graph.rect(bds, attr::LineClr(Gray), attr::FillClr(Gray.alpha(25)));
            auto [pos, image] = blob->image();
            
            SegmentationData::Assignment assign{
                .clid = size_t(-1)
            };
            if(_current_data.frame.index().valid()) {
                if(blob->parent_id().valid()) {
                    if(auto it = _current_data.predictions.find(blob->parent_id());
                       it != _current_data.predictions.end())
                    {
                        assign = it->second;
                        
                    } else if((it = _current_data.predictions.find(blob->blob_id())) != _current_data.predictions.end())
                    {
                        assign = it->second;
                        
                    } else
                        print("[draw]3 blob ", blob->blob_id(), " not found...");
                    
                } else if(auto it = _current_data.predictions.find(blob->blob_id()); it != _current_data.predictions.end())
                {
                    assign = it->second;
                    
                } else
                    print("[draw]4 blob ", blob->blob_id(), " not found...");
            }
            
            auto cname = meta_classes.size() > assign.clid
                        ? meta_classes.at(assign.clid)
                        : "<unknown:"+Meta::toStr(assign.clid)+">";
            
            sprite::Map tmp;
            tmp.set_do_print(false);
            tmp["pos"] = bds.pos().mul(scale);
            tmp["size"] = Size2(bds.size().mul(scale));
            tmp["type"] = std::string(cname);

            if (contains(visible_bdx, blob->blob_id())) {
                auto id = visible_bdx.at(blob->blob_id());
                tmp["color"] = id.color();
                tmp["id"] = id.ID();
                tmp["tracked"] = true;
                
            } else if(blob->parent_id().valid() && contains(visible_bdx, blob->parent_id()))
            {
                auto id = visible_bdx.at(blob->parent_id());
                tmp["color"] = id.color();
                tmp["id"] = id.ID();
                tmp["tracked"] = true;
                
            } else {
                tmp["tracked"] = false;
                tmp["color"] = Gray;
                tmp["id"] = Idx_t();
            }
            tmp["p"] = Meta::toStr(assign.p);
            _individual_properties.push_back(std::move(tmp));
            _gui_objects.emplace_back(new Variable([&, i = _individual_properties.size() - 1](std::string) -> sprite::Map& {
                return _individual_properties.at(i);
            }));
        }
    }

    // Main _draw function
    void _draw(DrawStructure& graph) {
        fetch_new_data();
        
        const auto meta_classes = SETTING(meta_classes).value<std::vector<std::string>>();
        graph.section("video", [&](auto&, Section* section){
            auto output_size = SETTING(output_size).value<Size2>();
            auto window_size = window()->window_dimensions();

            // Calculate window dimensions
            Size2 wdim = calculateWindowSize(output_size, window_size);
                
            auto scale = wdim.div(output_size);
            section->set_scale(scale);

            LockGuard lguard(ro_t{}, "drawing", 10);
            if (not lguard.locked()) {
                section->reuse_objects();
                return;
            }

            SETTING(gui_frame) = _current_data.frame.index();
            
            if(_background_image->source()) {
                graph.wrap_object(*_background_image);
            }
            
            for(auto box : _current_data.tiles)
                graph.rect(box, attr::FillClr{Transparent}, attr::LineClr{Red});
            
            static Frame_t last_frame;
            bool dirty{false};
            if(last_frame != _current_data.frame.index()) {
                last_frame = _current_data.frame.index();
                _gui_objects.clear();
                _individual_properties.clear();
                dirty = true;
            }

            // Draw outlines
            drawOutlines(graph, scale);
            
            using namespace track;
            std::unordered_map<pv::bid, Identity> visible_bdx;
            
            IndividualManager::transform_all([&](Idx_t , Individual* fish)
            {
                if(not fish->has(_current_data.frame.index()))
                    return;
                auto p = fish->iterator_for(_current_data.frame.index());
                auto segment = p->get();
                
                auto basic = fish->compressed_blob(_current_data.frame.index());
                //auto bds = basic->calculate_bounds();//.mul(scale);
                
                if(dirty) {
                    if(basic->parent_id.valid())
                        visible_bdx[basic->parent_id] = fish->identity();
                    visible_bdx[basic->blob_id()] = fish->identity();
                }
                
                std::vector<Vertex> line;
                fish->iterate_frames(Range(_current_data.frame.index().try_sub(50_f), _current_data.frame.index()), [&](Frame_t , const std::shared_ptr<SegmentInformation> &ptr, const BasicStuff *basic, const PostureStuff *) -> bool
                {
                    if(ptr.get() != segment) //&& (ptr)->end() != segment->start().try_sub(1_f))
                        return true;
                    auto p = basic->centroid.pos<Units::PX_AND_SECONDS>();//.mul(scale);
                    line.push_back(Vertex(p.x, p.y, fish->identity().color()));
                    return true;
                });
                
                graph.vertices(line);
            });
            
            //! do not need to continue further if the view isnt dirty
            if(not dirty)
                return;
            
            drawBlobs(meta_classes, scale, visible_bdx);
        });

        graph.section("menus", [&](auto&, Section* section){
            section->set_scale(graph.scale().reciprocal());
            _video_info["frame"] = _current_data.frame.index();
            _actual_frame = _current_data.frame.source_index();
            _video_frame = _current_data.frame.index();
            
            menu.draw(*(IMGUIBase*)window(), graph);
        });
    }
    
    void generator_thread() {
        set_thread_name("GeneratorT");
        std::vector<std::tuple<Frame_t, std::future<SegmentationData>>> items;
        
        std::unique_lock guard(_mutex_general);
        while(not _should_terminate) {
            try {
                if(not _next_frame_data and not items.empty()) {
                    if(std::get<1>(items.front()).wait_for(std::chrono::milliseconds(1)) == std::future_status::ready) {
                        auto data = std::get<1>(items.front()).get();
                        //thread_print("Got data for item ", data.frame.index());
                        
                        _next_frame_data = std::move(data);
                        _cv_ready_for_tracking.notify_one();
                        
                        items.erase(items.begin());
                    }
                }
                
                auto result = _overlayed_video->generate();
                if(not result) {
                    _overlayed_video->reset(0_f);
                } else {
                    items.push_back(std::move(result.value()));
                }
                
            } catch(...) {
                // pass
            }
            
            if(items.size() >= 10 && _next_frame_data) {
                //thread_print("Entering wait with ", items.size(), " items queued up.");
                _cv_messages.wait(guard, [&](){
                    return not _next_frame_data or _should_terminate;
                });
                //thread_print("Received notification: next(", (bool)next, ") and ", items.size()," items in queue");
            }
        }
        
        thread_print("ended.");
    };
    
    void perform_tracking() {
        static Frame_t running_id = 0_f;
        auto fake = double(running_id.get()) / double(FAST_SETTING(frame_rate)) * 1000.0 * 1000.0;
        _progress_data.frame.set_timestamp(uint64_t(fake));
        _progress_data.frame.set_index(running_id++);
        _progress_data.frame.set_source_index(Frame_t(_progress_data.image->index()));
        assert(_progress_data.frame.source_index() == Frame_t(_progress_data.image->index()));
        
        _progress_blobs.clear();
        for (size_t i = 0; i < _progress_data.frame.n(); ++i) {
            _progress_blobs.emplace_back(_progress_data.frame.blob_at(i));
        }
        
        if(SETTING(is_writing)) {
            if (not _output_file->is_open()) {
                _output_file->set_start_time(_start_time);
                _output_file->set_resolution(_output_size);
            }
            _output_file->add_individual(pv::Frame(_progress_data.frame));
        }
        
        {
            PPFrame pp;
            Tracker::preprocess_frame(pv::Frame(_progress_data.frame), pp, nullptr, PPFrame::NeedGrid::Need, false);
            _tracker->add(pp);
            if (pp.index().get() % 100 == 0) {
                print(IndividualManager::num_individuals(), " individuals known in frame ", pp.index());
            }
        }

        {
            std::unique_lock guard(_mutex_current);
            //thread_print("Replacing GUI current ", current.frame.index()," => ", progress.frame.index());
            _transferred_current_data = std::move(_progress_data);
            _transferred_blobs = std::move(_progress_blobs);
        }
        
        static Timer last_add;
        static double average{0}, samples{0};
        auto c = last_add.elapsed();
        average += c;
        ++samples;
        

        static Timer frame_counter;
        static size_t num_frames{0};
        static std::mutex mFPS;
        static double FPS{0};
        
        {
            std::unique_lock g(mFPS);
            num_frames++;
            
            if(frame_counter.elapsed() > 1) {
                FPS = num_frames / frame_counter.elapsed();
                num_frames = 0;
                _fps = FPS;
                _samples = 1;
                frame_counter.reset();
                print("FPS: ", FPS);
            }
            
        }
        
        if(samples > 100) {
            print("Average time since last frame: ", average / samples * 1000.0,"ms (",c * 1000,"ms)");
            
            average /= samples;
            samples = 1;
        }
        last_add.reset();
    };
    
    void tracking_thread() {
        set_thread_name("Tracking thread");
        std::unique_lock guard(_mutex_general);
        while(not _should_terminate) {
            if(_next_frame_data) {
                try {
                    _progress_data = std::move(_next_frame_data);
                    assert(not _next_frame_data);
                    //thread_print("Got next: ", progress.frame.index());
                } catch(...) {
                    FormatExcept("Exception while moving to progress");
                    continue;
                }
                //guard.unlock();
                try {
                    perform_tracking();
                    //guard.lock();
                } catch(...) {
                    FormatExcept("Exception while tracking");
                    throw;
                }
            }
            
            //thread_print("Waiting for next...");
            _cv_messages.notify_one();
            if(not _should_terminate)
                _cv_ready_for_tracking.wait(guard);
            //thread_print("Received notification: next(", (bool)next,")");
        }
        thread_print("Tracking ended.");
    };
};

int main(int argc, char**argv) {
    using namespace gui;
    
    default_config::register_default_locations();
    
    ::default_config::get(GlobalSettings::map(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    
    grab::default_config::get(GlobalSettings::map(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    grab::default_config::get(GlobalSettings::set_defaults(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    
    SETTING(meta_video_scale) = float(1);
    SETTING(source) = std::string("");
    SETTING(model) = file::Path("");
    SETTING(segmentation_resolution) = uint16_t(128);
    SETTING(segmentation_model) = file::Path("");
    SETTING(image_width) = uint16_t(640);
    SETTING(filename) = file::Path("");
    SETTING(meta_classes) = std::vector<std::string>{ };
    SETTING(detection_type) = ObjectDetectionType::yolo7;
    SETTING(tile_image) = size_t(0);
    
    using namespace cmn;
    namespace py = Python;
    print("CWD: ", file::cwd());
    DebugHeader("LOADING COMMANDLINE");
    CommandLine cmd(argc, argv, true);
    file::cd(file::DataLocation::parse("app").absolute());
    print("CWD: ", file::cwd());
    
    for(auto a : cmd) {
        if(a.name == "i") {
            SETTING(source) = std::string(a.value);
        }
        if(a.name == "m") {
            SETTING(model) = file::Path(a.value);
        }
        if(a.name == "sm") {
            SETTING(segmentation_model) = file::Path(a.value);
        }
        if(a.name == "d") {
            SETTING(output_dir) = file::Path(a.value);
        }
        if(a.name == "dim") {
            SETTING(image_width) = Meta::fromStr<uint16_t>(a.value);
        }
        if(a.name == "o") {
            SETTING(filename) = file::Path(a.value);
        }
    }
    
    _video_info.set_do_print(false);
    fish.set_do_print(false);
    
    expected_size = Size2(SETTING(image_width).value<uint16_t>());
    
    py::init();
    py::schedule([](){
        track::PythonIntegration::set_settings(GlobalSettings::instance());
        track::PythonIntegration::set_display_function([](auto& name, auto& mat) { tf::imshow(name, mat); });
    });
    
    
    using namespace track;
    
    GlobalSettings::map().set_do_print(true);
    GlobalSettings::map().dont_print("gui_frame");
    SETTING(app_name) = std::string("TRexA");
    SETTING(threshold) = int(100);
    SETTING(track_do_history_split) = false;
    SETTING(cm_per_pixel) = Settings::cm_per_pixel_t(0.1);
    SETTING(meta_real_width) = float(expected_size.width * 10);
    SETTING(track_max_speed) = Settings::track_max_speed_t(300);
    SETTING(track_threshold) = Settings::track_threshold_t(0);
    SETTING(track_posture_threshold) = Settings::track_posture_threshold_t(0);
    SETTING(blob_size_ranges) = Settings::blob_size_ranges_t({
        Rangef(10,300)
    });
    SETTING(track_speed_decay) = Settings::track_speed_decay_t(1);
    SETTING(track_max_reassign_time) = Settings::track_max_reassign_time_t(1);
    SETTING(terminate) = false;
    SETTING(calculate_posture) = false;
    SETTING(gui_interface_scale) = float(1);
    SETTING(meta_source_path) = SETTING(source).value<std::string>();
    
    std::stringstream ss;
    for(int i=0; i<argc; ++i) {
        if(i > 0)
            ss << " ";
        if(argv[i][0] == '-')
            ss << argv[i];
        else
            ss << "'" << argv[i] << "'";
    }
    SETTING(meta_cmd) = ss.str();
#if WITH_GITSHA1
    SETTING(meta_build) = std::string(g_GIT_SHA1);
#else
    SETTING(meta_build) = std::string("<undefined>");
#endif
    SETTING(meta_conversion_time) = std::string(date_time());
    SETTING(meta_encoding) = grab::default_config::meta_encoding_t::r3g3b2;

    cmd.load_settings();
    
    DrawStructure graph(1024, 768);
    IMGUIBase base(window_title(), graph, [&, ptr = &base]()->bool {
        UNUSED(ptr);
        graph.draw_log_messages();
        
        return true;
    }, [](Event) {
        
    });
    
    auto& manager = SceneManager::getInstance();
    ConvertScene converting(base);
    manager.register_scene(&converting);
    manager.set_active(&converting);
    
    graph.set_size(Size2(1024, converting.output_size().height / converting.output_size().width * 1024));
    
    base.platform()->set_icons({
        file::DataLocation::parse("app", "gfx/"+SETTING(app_name).value<std::string>()+"_16.png"),
        file::DataLocation::parse("app", "gfx/"+SETTING(app_name).value<std::string>()+"_32.png"),
        file::DataLocation::parse("app", "gfx/"+SETTING(app_name).value<std::string>()+"_64.png")
    });
    
    file::cd(file::DataLocation::parse("app"));
    
    gui::SFLoop loop(graph, &base, [&](gui::SFLoop&, LoopStatus) {
        manager.update(graph);
        
        if (graph.is_key_pressed(Keyboard::Right)) {
            SETTING(do_filter) = true;
        } else if (graph.is_key_pressed(Keyboard::Left)) {
            SETTING(do_filter) = false;
        }
    });
    
    manager.set_active(nullptr);
    manager.update_queue();
    graph.root().set_stage(nullptr);
    py::deinit();
    return 0;
}

