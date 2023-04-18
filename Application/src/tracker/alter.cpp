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
        proxy.set_variable("image_size", int(expected_size.width));
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
        std::vector<SegmentationData> datas;
        std::vector<Vec2> scales;
        std::vector<Vec2> offsets;
        std::vector<std::promise<SegmentationData>> promises;
        std::vector<std::function<void()>> callbacks;
        
        for(auto&& tiled : tiles) {
            images.insert(images.end(), std::make_move_iterator(tiled.images.begin()), std::make_move_iterator(tiled.images.end()));
            
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
            
            bbx.set_function("receive", [&](std::vector<uint64_t> Ns,
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
            });

            try {
                bbx.run("apply");
            }
            catch (...) {
                FormatWarning("Continue after exception...");
                throw;
            }
            
            bbx.unset_function("receive");

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
        proxy.set_variable("model_path", SETTING(model).value<file::Path>().str());
        proxy.set_variable("image_size", int(expected_size.width));
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
        
        /*SegmentationData data{
            std::move(tiled.original),
        };
        
        static Frame_t running_id = 0_f;
        auto fake = double(running_id.get()) / double(FAST_SETTING(frame_rate)) * 1000.0 * 1000.0;
        data.frame.set_timestamp(uint64_t(fake));
        data.frame.set_index(running_id++);*/
        
        Vec2 scale = SETTING(output_size).value<Size2>().div(tiled.source_size);
        print("Image scale: ", scale, " with tile source=", tiled.source_size, " image=", tiled.data.image->dimensions()," output_size=", SETTING(output_size).value<Size2>(), " original=", tiled.original_size);
        
        
        /*for(size_t i=0; i<tiled.images.size(); ++i) {
            tf::imshow(Meta::toStr(i)+ " " +Meta::toStr(tiled.offsets().at(i)), tiled.images.at(i)->get());
        }*/
        
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
        
        //tf::imshow("test", ret);
        return std::move(tiled.data);//Image::Make(ret);
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

        auto index = i++;
        
        gpuMatPtr buffer;
        
        if(std::unique_lock guard{buffer_mutex};
           not buffers.empty())
        {
            buffer = std::move(buffers.back());
            buffers.pop_back();
        } else {
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
    
    tl::expected<std::tuple<Frame_t, gpuMatPtr, Image::Ptr>, const char*> fetch_next_process() {
        try {
            Timer timer;
            auto result = _source_frame.next();
            if(result) {
                auto& [index, buffer] = result.value();
                static gpuMatPtr tmp = std::make_unique<useMat>();
                
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

int main(int argc, char**argv) {
    using namespace gui;
    
    default_config::register_default_locations();
    
    ::default_config::get(GlobalSettings::map(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    
    grab::default_config::get(GlobalSettings::map(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    grab::default_config::get(GlobalSettings::set_defaults(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    
    file::cd(file::DataLocation::parse("app"));
    
    SETTING(meta_video_scale) = float(1);
    SETTING(source) = std::string("");
    SETTING(model) = file::Path("");
    SETTING(image_width) = int(640);
    SETTING(filename) = file::Path("");
    SETTING(meta_classes) = std::vector<std::string>{ };
    SETTING(detection_type) = ObjectDetectionType::yolo7;
    SETTING(tile_image) = size_t(0);
    
    using namespace cmn;
    namespace py = Python;
    print("CWD: ", file::cwd());
    DebugHeader("LOADING COMMANDLINE");
    CommandLine cmd(argc, argv, true);
    file::cd(file::DataLocation::parse("app"));
    print("CWD: ", file::cwd());
    
    for(auto a : cmd) {
        if(a.name == "i") {
            SETTING(source) = std::string(a.value);
        }
        if(a.name == "m") {
            SETTING(model) = file::Path(a.value);
        }
        if(a.name == "d") {
            SETTING(output_dir) = file::Path(a.value);
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
    
    std::condition_variable messages;
    std::mutex mutex, current_mutex;
    std::atomic<bool> _terminate{false};
    SegmentationData next;
    
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
    SETTING(track_speed_decay) = Settings::track_speed_decay_t(1);
    SETTING(track_max_reassign_time) = Settings::track_max_reassign_time_t(1);
    SETTING(terminate) = false;
    SETTING(calculate_posture) = false;
    SETTING(gui_interface_scale) = float(1);
    SETTING(meta_source_path) = SETTING(source).value<std::string>();
    
    VideoSource video_base(SETTING(source).value<std::string>());
    video_base.set_colors(ImageMode::RGB);
    
    //! TODO: Major thing with framerate not being set for VideoSources based on single images.
    if(video_base.framerate() != short(-1))
        SETTING(frame_rate) = Settings::frame_rate_t(video_base.framerate());
    else
        SETTING(frame_rate) = Settings::frame_rate_t(25);
    
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
    
    if(SETTING(filename).value<file::Path>().empty()) {
        SETTING(filename) = file::Path((std::string)file::Path(video_base.base()).filename());
    }
    
    SETTING(do_filter) = true;
    SETTING(filter_classes) = std::vector<uint8_t>{};
    Size2 output_size = (Size2(video_base.size()) * SETTING(meta_video_scale).value<float>()).map(roundf);
    SETTING(output_size) = output_size;
    SETTING(is_writing) = true;
    
    _video_info["resolution"] = output_size;
    
    //cv::Mat bg = cv::Mat::zeros(video.source.size().height, video.source.size().width, CV_8UC1);
    //cv::Mat bg = cv::Mat::zeros(expected_size.width, expected_size.height, CV_8UC1);
    cv::Mat bg = cv::Mat::zeros(output_size.height, output_size.width, CV_8UC1);
    //bg.setTo(255);
    
    {
        VideoSource tmp(SETTING(source).value<std::string>());
        /*tmp.set_colors(ImageMode::RGB);
        Timer timer;
        useMat m;
        double average = 0, samples = 0;
        for (int i = 0; i < 1000; ++i) {
            tmp.frame(Frame_t(i), m);
            average += timer.elapsed() * 1000;
            timer.reset();
            samples++;
        }
        print("Average time / frame: ", average / samples, "ms");*/
        
        tmp.generate_average(bg, 0);
    }
    
    Tracker tracker(Image::Make(bg), float(expected_size.width * 10));
    //FrameInfo frameinfo;
    //Timer timer;
    //Timeline timeline(nullptr, [](bool) {}, []() {}, frameinfo);
    
    static_assert(ObjectDetection<Detection>);
    
    OverlayedVideo video(
        Detection{},
        std::move(video_base),
         [&messages](){
             messages.notify_one();
         }
    );
    
    Frame_t _actual_frame, _video_frame;
    
    Alterface menu(dyn::Context{
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
                    SETTING(do_filter) = filter;
                }
            },
            {
                "RESET", [&](auto){
                    video.reset(1500_f);
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
                "actual_frame", std::unique_ptr<VarBase_t>(new Variable([&_actual_frame](std::string) {
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
            video.reset(SETTING(gui_frame).value<Frame_t>());
        }
    });
    
    DrawStructure graph(1024, output_size.height / output_size.width * 1024);
    
    IMGUIBase base("TRexA", graph, [&, ptr = &base]()->bool {
        UNUSED(ptr);
        graph.draw_log_messages();
        
        return true;
    }, [](Event) {
        
    });

    auto start_time = std::chrono::system_clock::now();
    auto filename = file::DataLocation::parse("output", SETTING(filename).value<file::Path>());
    DebugHeader("Output: ", filename);
    
    auto path = filename.remove_filename();
    if(not path.exists()) {
        path.create_folder();
    }
    
    pv::File file(filename, pv::FileMode::OVERWRITE | pv::FileMode::WRITE);
    std::vector<pv::BlobPtr> objects, progress_objects, _trans_objects;
    file.set_average(bg);
    
    static std::vector<std::shared_ptr<VarBase_t>> gui_objects;
    static std::vector<sprite::Map> individual_properties;
    
    const auto meta_classes = SETTING(meta_classes).value<std::vector<std::string>>();
    
    menu.context.variables.emplace("fishes", new Variable([](std::string) -> std::vector<std::shared_ptr<VarBase_t>>& {
        return gui_objects;
    }));
    
    std::condition_variable ready_for_tracking;
    SegmentationData progress, current, _trans_current;
    std::shared_ptr<ExternalImage>
        background = std::make_shared<ExternalImage>(),
        overlay = std::make_shared<ExternalImage>();
    
    std::thread thread([&](){
        set_thread_name("GeneratorT");
        std::vector<std::tuple<Frame_t, std::future<SegmentationData>>> items;
        
        std::unique_lock guard(mutex);
        while(not _terminate) {
            try {
                if(not next and not items.empty()) {
                    if(std::get<1>(items.front()).wait_for(std::chrono::milliseconds(1)) == std::future_status::ready) {
                        auto data = std::get<1>(items.front()).get();
                        //thread_print("Got data for item ", data.frame.index());
                        
                        next = std::move(data);
                        ready_for_tracking.notify_one();
                        
                        items.erase(items.begin());
                    }
                }
                
                auto result = video.generate();
                
                if(not result) {
                    video.reset(0_f);
                } else {
                    items.push_back(std::move(result.value()));
                }
                
            } catch(...) {
                // pass
            }
            
            if(items.size() >= 10 && next) {
                //thread_print("Entering wait with ", items.size(), " items queued up.");
                messages.wait(guard, [&](){
                    return not next or _terminate;
                });
                //thread_print("Received notification: next(", (bool)next, ") and ", items.size()," items in queue");
            }
        }
        
        thread_print("ended.");
    });
    
    auto perform_tracking = [&](){
        static Frame_t running_id = 0_f;
        auto fake = double(running_id.get()) / double(FAST_SETTING(frame_rate)) * 1000.0 * 1000.0;
        progress.frame.set_timestamp(uint64_t(fake));
        progress.frame.set_index(running_id++);
        progress.frame.set_source_index(Frame_t(progress.image->index()));
        assert(progress.frame.source_index() == Frame_t(progress.image->index()));
        
        progress_objects.clear();
        for (size_t i = 0; i < progress.frame.n(); ++i) {
            progress_objects.emplace_back(progress.frame.blob_at(i));
        }
        
        if(SETTING(is_writing)) {
            if (not file.is_open()) {
                file.set_start_time(start_time);
                file.set_resolution(output_size);
            }
            file.add_individual(pv::Frame(progress.frame));
        }
        
        {
            PPFrame pp;
            Tracker::preprocess_frame(pv::Frame(progress.frame), pp, nullptr, PPFrame::NeedGrid::Need, false);
            tracker.add(pp);
            if (pp.index().get() % 100 == 0) {
                print(IndividualManager::num_individuals(), " individuals known in frame ", pp.index());
            }
        }

        {
            std::unique_lock guard(current_mutex);
            //thread_print("Replacing GUI current ", current.frame.index()," => ", progress.frame.index());
            _trans_current = std::move(progress);
            _trans_objects = std::move(progress_objects);
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
    
    std::thread tracking([&](){
        set_thread_name("Tracking thread");
        std::unique_lock guard(mutex);
        while(not _terminate) {
            if(next) {
                try {
                    progress = std::move(next);
                    assert(not next);
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
            messages.notify_one();
            if(not _terminate)
                ready_for_tracking.wait(guard);
            //thread_print("Received notification: next(", (bool)next,")");
        }
        thread_print("Tracking ended.");
    });

    auto fetch_files = [&](){
        static std::once_flag flag;
        std::call_once(flag, [](){
            set_thread_name("GUI");
        });
        
        {
            std::unique_lock guard(current_mutex);
            if (_trans_current.image) {
                current = std::move(_trans_current);
                objects = std::move(_trans_objects);
            }
        }

        if (current.image) {
            if(background->source()
               && background->source()->rows == current.image->rows
               && background->source()->cols == current.image->cols
               && background->source()->dims == 4)
            {
                cv::cvtColor(current.image->get(), background->unsafe_get_source().get(), cv::COLOR_BGR2BGRA);
                OverlayBuffers::put_back(std::move(current.image));
                background->updated_source();
            } else {
                auto rgba = Image::Make(current.image->rows, current.image->cols, 4);
                cv::cvtColor(current.image->get(), rgba->get(), cv::COLOR_BGR2BGRA);
                OverlayBuffers::put_back(std::move(current.image));
                background->set_source(std::move(rgba));
            }
            
            current.image = nullptr;
        }
    };

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
            section->set_scale(scale);

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
            std::unordered_map<pv::bid, Identity> visible_bdx;
            
            IndividualManager::transform_all([&](Idx_t , Individual* fish)
            {
                if(not fish->has(current.frame.index()))
                    return;
                auto p = fish->iterator_for(current.frame.index());
                auto segment = p->get();
                
                auto basic = fish->compressed_blob(current.frame.index());
                auto bds = basic->calculate_bounds();//.mul(scale);
                
                if(dirty) {
                    if(basic->parent_id.valid())
                        visible_bdx[basic->parent_id] = fish->identity();
                    visible_bdx[basic->blob_id()] = fish->identity();
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
                
                graph.vertices(line);
            });
            
            //! do not need to continue further if the view isnt dirty
            if(not dirty)
                return;
            
            for(auto &blob : objects) {
                
                const auto bds = blob->bounds();
                //graph.rect(bds, attr::LineClr(Gray), attr::FillClr(Gray.alpha(25)));
                auto [pos, image] = blob->image();
                
                SegmentationData::Assignment assign{
                    .clid = size_t(-1)
                };
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
                individual_properties.push_back(std::move(tmp));
                gui_objects.emplace_back(new Variable([&, i = individual_properties.size() - 1](std::string) -> sprite::Map& {
                    return individual_properties.at(i);
                }));
            }
        });
        
        graph.section("menus", [&](auto&, Section* section){
            section->set_scale(graph.scale().reciprocal());
            _video_info["frame"] = current.frame.index();
            _actual_frame = current.frame.source_index();
            _video_frame = current.frame.index();
            
            menu.draw(base, graph);
        });
        
        //graph.set_dirty(nullptr);

        if (graph.is_key_pressed(Keyboard::Right)) {
            SETTING(do_filter) = true;
        } else if (graph.is_key_pressed(Keyboard::Left)) {
            SETTING(do_filter) = false;
        }
    });
    
    {
        //
        {
            std::unique_lock guard(mutex);
            _terminate = true;
            ready_for_tracking.notify_all();
        }
        tracking.join();
        
        std::unique_lock guard(mutex);
        messages.notify_all();
    }
    
    Detection::manager.clean_up();
    
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

