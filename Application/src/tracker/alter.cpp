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
#include <GitSHA1.h>

using namespace cmn;

struct TileImage;

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
    //{ std::invoke_result<T, const Image&>::type } -> std::convertible_to<Image::UPtr>;
};

static Size2 expected_size(640, 640);
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

    SegmentationData() = default;
    SegmentationData(SegmentationData&& other) = default;
    SegmentationData(Image::UPtr&& original) : image(std::move(original)) {}
    SegmentationData& operator=(SegmentationData&&);
    ~SegmentationData();
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
            gpuMat tile = gpuMat::zeros(tile_size.height, tile_size.width, CV_8UC3);
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
static inline std::atomic<float> _fps{0}, _samples{0};
static inline std::atomic<float> _network_fps{0}, _network_samples{0};

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
                data.frame.add_object(lines, pixels, 0, blob::Prediction{ .clid = uint8_t(cls), .p = uint8_t(float(conf) * 255.f) });
            }
        }
    }
    
    static tl::expected<SegmentationData, const char*> apply(TileImage&& tiled) {
        namespace py = Python;
        
        SegmentationData data{
            std::move(tiled.original)
        };
        
        static Frame_t running_id = 0_f;
        auto fake = double(running_id.get()) / double(FAST_SETTING(frame_rate)) * 1000.0 * 1000.0;
        data.frame.set_timestamp(uint64_t(fake));
        data.frame.set_index(running_id++);
        
        Vec2 scale = SETTING(output_size).value<Size2>().div(tiled.source_size);
        //print("Image scale: ", scale, " with tile source=", tiled.source_size, " image=", data.image->dimensions()," output_size=", SETTING(output_size).value<Size2>(), " original=", tiled.original_size);
        
        for(auto p : tiled.offsets()) {
            data.tiles.push_back(Bounds(p.x, p.y, tiled.tile_size.width, tiled.tile_size.height).mul(scale));
        }
        
        py::schedule([&data, scale, offsets = tiled.offsets(), images = std::move(tiled.images)]() mutable {
            Timer timer;
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
            if (_network_samples.load() > 100) {
                _network_samples = _network_fps = 0;
            }
            _network_fps = _network_fps.load() + 1.0 / timer.elapsed();
            _network_samples = _network_samples.load() + 1;
            
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
        
        SegmentationData data{
            std::move(tiled.original),
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
    R operator()(_Args... args) const {
        return package(std::forward<_Args>(args)...);
    }
};

}
}

GenericThreadPool pool(32, "all_pool");

template<typename F, typename R = typename cmn::detail::return_type<F>::type>
struct RepeatedDeferral {
    std::future<R> _next_image;
    F _fn;
    std::string name;
    size_t predicted{0}, unpredicted{0};
    std::mutex mtiming;
    double _waiting{0}, _samples{0};
    double _since_last{0}, _ssamples{0};
    
    double _runtime{0}, _rsamples{0};
    Timer timer, since_last;
    
    RepeatedDeferral(std::string name, F fn) : _fn(std::forward<F>(fn)), name(name) {
        
    }
    
    ~RepeatedDeferral() {
        if(_next_image.valid())
            _next_image.get();
    }
    
    template<typename... Args>
    R next(Args... args) {
        auto e = since_last.elapsed();
        {
            std::unique_lock guard(mtiming);
            _since_last += e;
            _ssamples++;
            if(_ssamples > 1000) {
                _since_last = _since_last / _ssamples;
                _ssamples = 1;
            }
        }
        
        R f;
        
        if(_next_image.valid()) {
            ++predicted;
            timer.reset();
            f = _next_image.get();
            
            std::unique_lock guard(mtiming);
            _waiting += timer.elapsed();
            _samples += 1;
            
            if(_samples > 1000) {
                _waiting /= _samples;
                _samples = 1;
            }
            
        } else {
            ++unpredicted;
            f = _fn(std::forward<Args>(args)...);
        }
        
        if(predicted % 50 == 0 || unpredicted % 50 == 0) {
            std::unique_lock guard(mtiming);
            auto total = (_waiting / _samples) * 1000;
            print(name.c_str(),": ", total,"ms with runtime ",(_runtime / _rsamples) * 1000,"ms (", (_since_last/_ssamples)*1000,"ms > ", (_waiting / _samples) * 1000, "ms)");
        }
        
        _next_image = pool.enqueue([this](Args... args) {
            Timer runtime;
            auto r = _fn(std::forward<Args>(args)...);
            auto e = runtime.elapsed();
            
            std::unique_lock guard(mtiming);
            _runtime += e;
            _rsamples++;
            
            if(_rsamples > 1000) {
                _runtime = _runtime / _rsamples;
                _rsamples = 1;
            }
            
            return r;
            
        }, std::forward<Args>(args)...);
        
        since_last.reset();
        return f;
    }
};

namespace OverlayBuffers {

inline static std::mutex buffer_mutex;
inline static std::vector<Image::UPtr> buffers;


Image::UPtr get_buffer() {
    if (std::unique_lock guard(OverlayBuffers::buffer_mutex);
        not OverlayBuffers::buffers.empty())
    {
        auto ptr = std::move(OverlayBuffers::buffers.back());
        OverlayBuffers::buffers.pop_back();
        //print("Received from buffers ", ptr->bounds());
        return ptr;
    }

    return Image::Make();
}

void put_back(Image::UPtr&& ptr) {
    if (not ptr)
        return;
    std::unique_lock guard(OverlayBuffers::buffer_mutex);
    //print("Pushed back buffer ", ptr->bounds());
    OverlayBuffers::buffers.push_back(std::move(ptr));
}

}

template<typename SourceType>
class AbstractVideoSource {
    //mutable std::mutex mutex;
    Frame_t i{0_f};
    
    using gpuMatPtr = std::unique_ptr<gpuMat>;
    std::mutex buffer_mutex;
    std::vector<std::tuple<gpuMatPtr, Image::UPtr>> buffers;
    
    SourceType source;
    GETTER(file::Path, base)
    GETTER(Size2, size)
    GETTER(short, framerate)
    
    const bool _finite;
    const Frame_t _length;
    
    mutable package::packaged_func<std::tuple<Frame_t, gpuMatPtr, Image::UPtr>> _retrieve;
    mutable package::packaged_func<void, Frame_t> _set_frame;
    
public:
    template<typename T = SourceType>
        requires _clean_same<SourceType, VideoSource>
    AbstractVideoSource(SourceType&& source)
        :
            source(std::move(source)),
            _base(this->source.base()),
            _retrieve([this]() -> std::tuple<Frame_t, gpuMatPtr, Image::UPtr> {
                try {
                    static RepeatedDeferral def{
                        std::string("source.frame"),
                        [this]() -> tl::expected<std::tuple<Frame_t, gpuMatPtr, Image::UPtr>, const char*>  {
                            //std::unique_lock guard(mutex);
                            if(i >= this->source.length())
                                i = 0_f;
                            
                            gpuMatPtr buffer;
                            Image::UPtr image;
                            
                            if(std::unique_lock guard{buffer_mutex};
                               not buffers.empty())
                            {
                                auto&&[buf, img] = std::move(buffers.back());
                                buffer = std::move(buf);
                                image = std::move(img);
                                buffers.pop_back();
                            } else {
                                buffer = std::make_unique<gpuMat>();
                                image = Image::Make();
                            }
                            
                            this->source.frame(i++, *buffer);
                            
                            return std::make_tuple(i - 1_f, std::move(buffer), std::move(image));
                        }
                    };
                    auto result = def.next();
                    if(result) {
                        auto& [index, buffer, image] = result.value();
                        static gpuMatPtr tmp = std::make_unique<gpuMat>();

                        //! resize according to settings
                        //! (e.g. multiple tiled image size)
                        if (SETTING(meta_video_scale).value<float>() != 1) {
                            Size2 new_size = Size2(buffer->cols, buffer->rows) * SETTING(meta_video_scale).value<float>();
                            //FormatWarning("Resize ", Size2(buffer.cols, buffer.rows), " -> ", new_size);
                            cv::resize(*buffer, *buffer, new_size);
                        }

                        image->create(*buffer, index.get());

                        cv::cvtColor(*buffer, *tmp, cv::COLOR_BGR2RGB);
                        std::swap(buffer, tmp);
                        
                        return std::make_tuple(index, std::move(buffer), std::move(image));
                        
                    } else
                        throw U_EXCEPTION("Unable to load frame: ", result.error());
                    
                } catch(const std::exception& e) {
                    FormatExcept("Unable to load frame ", i, " from video source ", this->source, " because: ", e.what());
                    return {Frame_t{}, nullptr, nullptr};
                }
            }),
            _set_frame([this](Frame_t frame){
                //std::unique_lock guard(mutex);
                i = frame;
            }),
            _size(this->source.size()),
            _framerate(this->source.framerate()),
            _finite(true),
            _length(this->source.length())
    {
        this->source.set_colors(VideoSource::ImageMode::RGB);
    }

    AbstractVideoSource(AbstractVideoSource&& other) = default;
    AbstractVideoSource(const AbstractVideoSource& other) = delete;
    AbstractVideoSource() = delete;
    
    AbstractVideoSource& operator=(AbstractVideoSource&&) = delete;
    AbstractVideoSource& operator=(const AbstractVideoSource&) = delete;
    
    void move_back(gpuMatPtr&& ptr) {
        std::unique_lock guard(buffer_mutex);
        buffers.push_back(std::make_tuple(std::move(ptr), OverlayBuffers::get_buffer()));
    }
    
    std::tuple<Frame_t, gpuMatPtr, Image::UPtr> next() {
        static RepeatedDeferral _def{
            std::string(".next()"),
            [this]() -> tl::expected<std::tuple<Frame_t, gpuMatPtr, Image::UPtr>, const char*> {
                return _retrieve();
            }
        };
        
        auto result = _def.next();
        if(not result)
            return std::make_tuple(Frame_t{}, nullptr, nullptr);
        
        return std::move(result.value());
    }
    
    bool is_finite() const {
        return true;
    }
    
    void set_frame(Frame_t frame) {
        if(not is_finite())
            throw std::invalid_argument("Cannot skip on infinite source.");
        _set_frame(frame);
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
    //Image::UPtr original_image;
    //cv::Mat downloader;
    gpuMat resized;
    
    std::future<tl::expected<SegmentationData, const char*>> next_image;
    
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
        if(source.is_finite())
            source.set_frame(i);
    }
    
    //! generates the next frame
    tl::expected<SegmentationData, const char*> generate() noexcept {
        if(eof())
            return tl::unexpected("End of file.");
        
        auto retrieve_next = [this]()
            -> tl::expected<SegmentationData, const char*>
        {
            static Timing timing("retrieve_next");
            TakeTiming take(timing);
            
            std::scoped_lock guard(index_mutex);
            TileImage tiled;
            auto loaded = i;
            
            try {
                auto&& [nix, buffer, image] = source.next();
                if(not nix.valid())
                    return tl::unexpected("Cannot retrieve frame from video source.");
                
                gpuMat *use { buffer.get() };
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
                source.move_back(std::move(buffer));
                
                //! network processing, and record network fps
                return this->overlay.apply(std::move(tiled));
                
            } catch(const std::exception& e) {
                FormatExcept("Error loading frame ", loaded, " from video ", source, ": ", e.what());
                return tl::unexpected("Error loading frame.");
            }
        };
        
        static RepeatedDeferral def{
            "Tile+ApplyNet",
            retrieve_next
        };
        
        return def.next();
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

template<typename OverlayT>
struct Menu {
    
    //Button::Ptr
    /*    hi = Button::MakePtr("Quit", attr::Size(50, 35)),
        bro = Button::MakePtr("Filter", attr::Size(50, 35)),
        reset = Button::MakePtr("Reset", attr::Size(50, 35));*/
    //std::shared_ptr<Text> text = std::make_shared<Text>();
    Image::UPtr next;
    dyn::Context context;
    dyn::State state;
    std::vector<Layout::Ptr> objects;
    Frame_t _actual_frame, _video_frame;
    
    /*std::shared_ptr<HorizontalLayout> buttons = std::make_shared<HorizontalLayout>(
        std::vector<Layout::Ptr>{
            Layout::Ptr(hi),
            Layout::Ptr(bro),
            Layout::Ptr(reset),
            Layout::Ptr(text)
        }
    );*/
    
    Menu() = delete;
    Menu(Menu&&) = delete;
    Menu(const Menu&) = delete;
    
    template<typename F>
    Menu(F&& reset_func) : context(dyn::Context{
        .actions = {
            {
                "QUIT", [](auto) {
                    SETTING(terminate) = true;
                }
            },
            {
                "FILTER", [](auto) {
                    static bool filter { false };
                    filter = not filter;
                    SETTING(filter_class) = filter;
                }
            },
            {
                "RESET", [reset_func = std::move(reset_func)](auto) {
                    reset_func();
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
                "video", std::unique_ptr<VarBase_t>(new Variable([this](std::string) -> sprite::Map& {
                    return _video_info;
                }))
            }
        }
    }) { }
    
    ~Menu() {
        context = {};
        state = {};
        
        objects.clear();
    }
    
    void draw(DrawStructure& g, Frame_t actual_frame, Frame_t video_frame) {
        _actual_frame = actual_frame;
        _video_frame = video_frame;
        
        dyn::update_layout("alter_layout.json", context, state, objects);
        
        g.section("buttons", [&](auto&, Section* section) {
            section->set_scale(g.scale().reciprocal());
            for(auto &obj : objects) {
                dyn::update_objects(g, obj, context, state);
                g.wrap_object(*obj);
            }
        });
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
        
        _settings_dropdown.set_pos(Vec2(10, base.window_dimensions().height - 10));
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

SegmentationData::~SegmentationData() {
    if (image) {
        OverlayBuffers::put_back(std::move(image));
    }
}

SegmentationData& SegmentationData::operator=(SegmentationData&& other)
{
    frame = std::move(other.frame);
    tiles = std::move(other.tiles);
    predictions = std::move(other.predictions);
    outlines = std::move(other.outlines);

    if (image) {
        OverlayBuffers::put_back(std::move(image));
    }
    image = std::move(other.image);
    return *this;
}

int main(int argc, char**argv) {
    using namespace gui;
    
    default_config::register_default_locations();
    
    ::default_config::get(GlobalSettings::map(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    
    grab::default_config::get(GlobalSettings::map(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    grab::default_config::get(GlobalSettings::set_defaults(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    
    file::cd(file::DataLocation::parse("app"));
    
    SETTING(meta_video_scale) = float(1);
    SETTING(source) = std::string("/Users/tristan/goats/DJI_0160.MOV");
    SETTING(model) = file::Path("/Users/tristan/Downloads/tfmodel_goats1024");
    SETTING(image_width) = int(1024);
    SETTING(filename) = file::Path("");
    SETTING(meta_classes) = std::vector<std::string>{
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
    print("output path: ", SETTING(filename).value<file::Path>());
    
    _video_info.set_do_print(false);
    fish.set_do_print(false);
    
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
    );
    
    std::mutex mutex, current_mutex;
    std::condition_variable messages;
    bool _terminate{false};
    SegmentationData next;
    
    ::Menu<decltype(video)> menu([&](){
        video.reset(1500_f);
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

    cmd.load_settings();
    
    if(SETTING(filename).value<file::Path>().empty()) {
        SETTING(filename) = file::Path((std::string)file::Path(video.source.base()).filename());
    }
    
    SETTING(filter_class) = false;
    Size2 output_size = (Size2(video.source.size()) * SETTING(meta_video_scale).value<float>()).map(roundf);
    SETTING(output_size) = output_size;
    
    _video_info["resolution"] = output_size;
    
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
        UNUSED(ptr);
        
        //timeline.set_base(ptr);
        
        //auto dt = timer.elapsed();
        ///cache.set_dt(dt);
        //timer.reset();

        //Frame_t index = SETTING(gui_frame).value<Frame_t>();

        //image.set_pos(last_mouse_pos);
        //graph.wrap_object(image);

        //auto scale = graph.scale().reciprocal();
        //auto dim = ptr->window_dimensions().mul(scale * gui::interface_scale());
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
    static std::vector<std::shared_ptr<VarBase_t>> gui_objects;
    static std::vector<sprite::Map> individual_properties;
    
    const auto meta_classes = SETTING(meta_classes).value<std::vector<std::string>>();
    
    menu.context.variables.emplace("fishes", new Variable([](std::string) -> std::vector<std::shared_ptr<VarBase_t>>& {
        return gui_objects;
    }));
    
    SegmentationData progress, current;
    std::shared_ptr<ExternalImage>
        background = std::make_shared<ExternalImage>(),
        overlay = std::make_shared<ExternalImage>();
    
    std::thread thread([&](){
        set_thread_name("video.generate() thread");
        
        RepeatedDeferral f{
            std::string("video.generate()"),
            [&]() -> tl::expected<SegmentationData, const char*> {
                static Timing timing("video.generate()");
                TakeTiming take(timing);
                Timer timer;
                
                auto result = video.generate();
                if(not result) {
                    video.reset(0_f);
                }
                
                if (_samples.load() > 100) {
                    _samples = _fps = 0;
                }
                _fps = _fps.load() + 1.0 / timer.elapsed();
                _samples = _samples.load() + 1;
                
                return result;
            }
        };
        
        std::unique_lock guard(mutex);
        tl::expected<SegmentationData, const char*> result;
        while(not _terminate) {
            try {
                if (not next) {
                    guard.unlock();
                    result = f.next();
                    guard.lock();
                    if (result)
                        next = std::move(result.value());
                }
            } catch(...) {
                // pass
            }
            messages.wait(guard);
        }
        
        print("thread ended.");
    });

    auto fetch_files = [&](){
        //std::this_thread::sleep_for(std::chrono::milliseconds(30));
        //static Timing timing("fetch_files");
        //TakeTiming take(timing);
        
        //static Timing timing2("fetch_files#2");
        //TakeTiming take2(timing2);
        std::unique_lock guard(mutex);
        if (next) {
            objects.clear();
            progress = std::move(next);
            messages.notify_one();

            progress.frame.set_source_index(Frame_t(progress.image->index()));

            for (size_t i = 0; i < progress.frame.n(); ++i) {
                objects.emplace_back(progress.frame.blob_at(i));
            }

            if(background->source()
               && background->source()->rows == progress.image->rows
               && background->source()->cols == progress.image->cols
               && background->source()->dims == 4)
            {
                cv::cvtColor(progress.image->get(), background->unsafe_get_source().get(), cv::COLOR_BGR2BGRA);
                background->updated_source();
            } else {
                auto rgba = Image::Make(progress.image->rows, progress.image->cols, 4);
                cv::cvtColor(progress.image->get(), rgba->get(), cv::COLOR_BGR2BGRA);
                background->set_source(std::move(rgba));
            }
            
            if (not file.is_open()) {
                file.set_start_time(start_time);
                file.set_resolution(output_size);
            }
            file.add_individual(pv::Frame(progress.frame));

            {
                Tracker::preprocess_frame(file, pv::Frame(progress.frame), pp, nullptr, PPFrame::NeedGrid::NoNeed, false);
                tracker.add(pp);
                if (pp.index().get() % 100 == 0) {
                    print(IndividualManager::num_individuals(), " individuals known in frame ", pp.index());
                }
            }

            {
                std::unique_lock guard(current_mutex);
                current = std::move(progress);
            }
            
            static Timer last_add;
            static double average{0}, samples{0};
            auto current = last_add.elapsed();
            average += current;
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
                    frame_counter.reset();
                    print("FPS: ", FPS);
                }
                
            }
            
            if(samples > 100) {
                print("Average time since last frame: ", average / samples * 1000.0,"ms (",current * 1000,"ms)");
                
                average /= samples;
                samples = 1;
            }
            last_add.reset();
        }
    };

    std::thread testing([&]() {
        while (not _terminate) {
            fetch_files();
        }
    });
    
    //graph.set_scale(1. / base.dpi_scale());
    
    gui::SFLoop loop(graph, &base, [&](gui::SFLoop&, LoopStatus) {
        
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
            auto window_size = base.window_dimensions();
            
            auto ratio = output_size.width / output_size.height;
            Size2 wdim;
            
            if(window_size.width * output_size.height < window_size.height * output_size.width)
            {
                wdim = Size2(window_size.width, window_size.width / ratio);
            } else {
                wdim = Size2(window_size.height * ratio, window_size.height);
            }
            
            auto scale = wdim.div(output_size);
            
            //ratio = ratio.T();
            //scale = scale.mul(ratio);
            //if(not current.frame.index().valid() || current.frame.index().get()%10 == 0)
            //    print("gui scale: ", scale, " dpi:",base.dpi_scale(), " graph:", graph.scale(), " window:", base.window_dimensions(), " video:", SETTING(output_size).value<Size2>(), " scale:", Size2(graph.width(), graph.height()).div(SETTING(output_size).value<Size2>()), " ratio:", ratio, " wdim:", wdim);
            section->set_scale(scale);

            std::unique_lock guard(current_mutex);
            LockGuard lguard(ro_t{}, "drawing", 10);
            if (not lguard.locked()) {
                section->reuse_objects();
                return;
            }

            SETTING(gui_frame) = current.frame.index();
            
            if(background->source()) {
                graph.wrap_object(*background);
            }
            
            for(auto box : current.tiles)
                graph.rect(box, attr::FillClr{Transparent}, attr::LineClr{Red});
            
            static Frame_t last_frame;
            bool dirty{false};
            if(last_frame != current.frame.index()) {
                last_frame = current.frame.index();
                gui_objects.clear();
                individual_properties.clear();
                dirty = true;
            }
            
            if (not current.outlines.empty()) {
                graph.text(Meta::toStr(current.outlines.size())+" lines", attr::Loc(10,50), attr::Font(0.35), attr::Scale(scale.mul(graph.scale()).reciprocal()));
                
                ColorWheel wheel;
                for (const auto& v : current.outlines) {
                    auto clr = wheel.next();
                    graph.line(v, 1, clr.alpha(150));
                }
            }
            
            using namespace track;
            std::unordered_set<pv::bid> visible_bdx;
            
            IndividualManager::transform_all([&](Idx_t , Individual* fish)
            {
                if(not fish->has(current.frame.index()))
                    return;
                auto p = fish->iterator_for(current.frame.index());
                auto segment = p->get();
                
                auto basic = fish->compressed_blob(current.frame.index());
                auto bds = basic->calculate_bounds();//.mul(scale);
                
                if(dirty) {
                    SegmentationData::Assignment assign;
                    if(current.frame.index().valid()) {
                        if(basic->parent_id.valid()) {
                            if(auto it = current.predictions.find(basic->parent_id);
                               it != current.predictions.end())
                            {
                                assign = it->second;
                                
                            } else if((it = current.predictions.find(basic->blob_id())) != current.predictions.end())
                            {
                                assign = it->second;
                                
                            } else
                                print("[draw]1 blob ", basic->blob_id(), " not found...");
                            
                        } else if(auto it = current.predictions.find(basic->blob_id()); it != current.predictions.end())
                        {
                            assign = it->second;
                            
                        } else
                            print("[draw]2 blob ", basic->blob_id(), " not found...");
                    }
                    
                    auto cname = meta_classes.size() > assign.clid
                                ? meta_classes.at(assign.clid)
                                : "<unknown:"+Meta::toStr(assign.clid)+">";
                    
                    sprite::Map tmp;
                    tmp.set_do_print(false);
                    tmp["pos"] = bds.pos().mul(scale);
                    tmp["size"] = Size2(bds.size().mul(scale));
                    tmp["type"] = std::string(cname);
                    tmp["tracked"] = true;
                    tmp["color"] = fish->identity().color();
                    tmp["id"] = fish->identity().ID();
                    tmp["p"] = Meta::toStr(assign.p);
                    individual_properties.push_back(std::move(tmp));
                    gui_objects.emplace_back(new Variable([&, i = individual_properties.size() - 1](std::string) -> sprite::Map& {
                        return individual_properties.at(i);
                    }));
                    
                    visible_bdx.insert(basic->blob_id());
                }
                
                std::vector<Vertex> line;
                fish->iterate_frames(Range(current.frame.index().try_sub(50_f), current.frame.index()), [&](Frame_t , const std::shared_ptr<SegmentInformation> &ptr, const BasicStuff *basic, const PostureStuff *) -> bool
                {
                    if(ptr.get() != segment) //&& (ptr)->end() != segment->start().try_sub(1_f))
                        return true;
                    auto p = basic->centroid.pos<Units::PX_AND_SECONDS>();//.mul(scale);
                    line.push_back(Vertex(p.x, p.y, fish->identity().color()));
                    return true;
                });
                
                //graph.rect(bds, attr::FillClr(Transparent), attr::LineClr(fish->identity().color()));
                graph.vertices(line);
            });
            
            if(dirty && objects.size() != visible_bdx.size()) {
                for(auto &blob : objects) {
                    if(contains(visible_bdx, blob->blob_id()))
                        continue;
                    
                    const auto bds = blob->bounds();
                    //graph.rect(bds, attr::LineClr(Gray), attr::FillClr(Gray.alpha(25)));
                    
                    SegmentationData::Assignment assign;
                    if(current.frame.index().valid()) {
                        if(blob->parent_id().valid()) {
                            if(auto it = current.predictions.find(blob->parent_id());
                               it != current.predictions.end())
                            {
                                assign = it->second;
                                
                            } else if((it = current.predictions.find(blob->blob_id())) != current.predictions.end())
                            {
                                assign = it->second;
                                
                            } else
                                print("[draw]3 blob ", blob->blob_id(), " not found...");
                            
                        } else if(auto it = current.predictions.find(blob->blob_id()); it != current.predictions.end())
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
                    tmp["tracked"] = false;
                    tmp["color"] = Gray;
                    tmp["id"] = Idx_t();
                    tmp["p"] = Meta::toStr(assign.p);
                    individual_properties.push_back(std::move(tmp));
                    gui_objects.emplace_back(new Variable([&, i = individual_properties.size() - 1](std::string) -> sprite::Map& {
                        return individual_properties.at(i);
                    }));
                }
            }
        });
        
        graph.section("menus", [&](auto&, Section* section){
            section->set_scale(graph.scale().reciprocal());
            _video_info["frame"] = current.frame.index();
            menu.draw(graph, current.frame.source_index(), current.frame.index());
            
            settings.draw(base, graph);
        });
        
        //graph.set_dirty(nullptr);

        if (graph.is_key_pressed(Keyboard::Right)) {
            SETTING(filter_class) = true;
        } else if (graph.is_key_pressed(Keyboard::Left)) {
            SETTING(filter_class) = false;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    });
    
    {
        std::unique_lock guard(mutex);
        _terminate = true;
        messages.notify_all();
    }
    
    thread.join();
    graph.root().set_stage(nullptr);
    py::deinit();
    file.close();
    
    {
        pv::File test(file.filename(), pv::FileMode::READ);
        test.print_info();
    }
    return 0;
}

