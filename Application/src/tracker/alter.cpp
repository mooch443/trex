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
#include <grabber/misc/Webcam.h>

#include <misc/TaskPipeline.h>
#include <Scene.h>

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

ENUM_CLASS(ObjectDetectionType, yolo7, yolo7seg, yolo8seg, customseg);
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
        
        if(SETTING(model).value<file::Path>().empty())
            throw U_EXCEPTION("When using yolov7 object detection, please set model using command-line argument -m <path> to set a model (tensorflow saved model).");
        else if(not SETTING(model).value<file::Path>().exists())
            throw U_EXCEPTION("Cannot find model file ",SETTING(model).value<file::Path>(),".");
        
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
                data.predictions.push_back({ .clid = size_t(cls), .p = float(conf) });
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
                            pair.extra_flags |= pv::Blob::flag(pv::Blob::Flags::is_instance_segmentation);
                            
                            pv::Blob blob(*pair.lines, *pair.pixels, pair.extra_flags, pair.pred);
                            auto points = pixel::find_outer_points(&blob, 0);
                            if (not points.empty()) {
                                data.outlines.emplace_back(std::move(*points.front()));
                                //for (auto& pt : outline_points.back())
                                //    pt = (pt + blob.bounds().pos())/*.mul(dim.div(image.dimensions())) + pos*/;
                            }
                            
                            data.predictions.push_back({ .clid = size_t(cls), .p = float(conf) });
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

/*struct Yolo8InstanceSegmentation {
    Yolo8InstanceSegmentation() = delete;
    
    static void reinit(track::PythonIntegration::ModuleProxy& proxy) {
        proxy.set_variable("model_type", detection_type().toStr());
        
        if(SETTING(segmentation_model).value<file::Path>().empty()) {
            if(SETTING(model).value<file::Path>().empty())
                throw U_EXCEPTION("When using yolov8, please set model using command-line argument -sm <path> to set an instance segmentation model (pytorch), or -m <path> to set an object detection model, or both to use the segmentation model only for segmentation of cropped object detections.");
        } else if(not SETTING(segmentation_model).value<file::Path>().exists())
            FormatWarning("Cannot find segmentation instance model file ",SETTING(segmentation_model).value<file::Path>(),".");
        
        if(SETTING(model).value<file::Path>().exists())
            proxy.set_variable("model_path", SETTING(model).value<file::Path>().str());
        else
            proxy.set_variable("model_path", "");
        
        if(SETTING(segmentation_model).value<file::Path>().exists())
            proxy.set_variable("segmentation_path", SETTING(segmentation_model).value<file::Path>().str());
        else
            proxy.set_variable("segmentation_path", "");
        
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
                
                pair.extra_flags |= pv::Blob::flag(pv::Blob::Flags::is_instance_segmentation);
                
                pv::Blob blob(*pair.lines, *pair.pixels, pair.extra_flags, pair.pred);
                auto points = pixel::find_outer_points(&blob, 0);
                if (not points.empty()) {
                    data.outlines.emplace_back(std::move(*points.front()));
                    //for (auto& pt : outline_points.back())
                    //    pt = (pt + blob.bounds().pos());
                }
                data.predictions[blob.blob_id()] = { .clid = size_t(cls), .p = float(conf) };
                data.frame.add_object(std::move(pair));
                //auto big = pixel::threshold_get_biggest_blob(&blob, 1, nullptr);
                //auto [pos, img] = big->image();
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
};*/
struct Yolo8InstanceSegmentation {
    Yolo8InstanceSegmentation() = delete;
    
    static void reinit(track::PythonIntegration::ModuleProxy& proxy) {
        proxy.set_variable("model_type", detection_type().toStr());
        
        if(SETTING(segmentation_model).value<file::Path>().empty()) {
            if(SETTING(model).value<file::Path>().empty())
                throw U_EXCEPTION("When using yolov8, please set model using command-line argument -sm <path> to set an instance segmentation model (pytorch), or -m <path> to set an object detection model, or both to use the segmentation model only for segmentation of cropped object detections.");
        } else if(not SETTING(segmentation_model).value<file::Path>().exists())
            FormatWarning("Cannot find segmentation instance model file ",SETTING(segmentation_model).value<file::Path>(),".");
        
        if(SETTING(model).value<file::Path>().exists())
            proxy.set_variable("model_path", SETTING(model).value<file::Path>().str());
        else
            proxy.set_variable("model_path", "");
        
        if(SETTING(segmentation_model).value<file::Path>().exists())
            proxy.set_variable("segmentation_path", SETTING(segmentation_model).value<file::Path>().str());
        else
            proxy.set_variable("segmentation_path", "");
        
        proxy.set_variable("image_size", expected_size);
        proxy.run("load_model");
    }
    
    static void init() {
        Python::schedule([](){
            using py = track::PythonIntegration;
            py::ModuleProxy proxy{
                "bbx_saved_model",
                Yolo8InstanceSegmentation::reinit
            };
        }).get();
    }
    
    static void receive(SegmentationData& data, Vec2 scale_factor, const std::span<float>& vector, 
        const std::span<float>& mask_points, const std::span<uint64_t>& mask_Ns) 
    {
        static const auto meta_encoding = SETTING(meta_encoding).value<grab::default_config::meta_encoding_t::Class>();
        static const auto mode = meta_encoding == grab::default_config::meta_encoding_t::r3g3b2 ? ImageMode::R3G3B2 : ImageMode::GRAY;

        const Vec2* ptr = (const Vec2*)mask_points.data();
        const size_t N = mask_points.size() / 2u;
        assert(mask_points.size() % 2u == 0);

        // convert list of points to integer coordinates
        std::vector<cv::Point> integer;
        integer.reserve(N);
        for (auto it = ptr, end = ptr + N; it != end; ++it) {
            integer.emplace_back(roundf(saturate(it->x, 0.f, 1.f) * data.image->cols),
                roundf(saturate(it->y, 0.f, 1.f) * data.image->rows));
        }

        size_t mask_index = 0;
        cv::Mat r3;
        if  (mode == ImageMode::R3G3B2)
            convert_to_r3g3b2(data.image->get(), r3);
        else if  (mode == ImageMode::GRAY)
            cv::cvtColor(data.image->get(), r3, cv::COLOR_BGR2GRAY);

        //thread_print("Received seg-data for frame ", data.frame.index());
        for(size_t i=0, m = 0; i<vector.size(); i+=4+2, m++) {
            float conf = vector[i+4];
            float cls = vector[i+5];
            
            Vec2 pos = Vec2(vector[i+0], vector[i+1]);
            Size2 dim = Size2(vector[i+2] - pos.x, vector[i+3] - pos.y).mul(scale_factor);
            pos = pos.mul(scale_factor);
            
            if (mask_Ns[m] == 0)
                continue;

            //std::vector<HorizontalLine> lines;
            //std::vector<uchar> pixels;

            thread_print("** ", i, " ", pos, " ", dim, " ", conf, " ", cls, " ", mask_Ns[m], " **");
            thread_print("  getting integers from ", mask_index, " to ", mask_index + mask_Ns[m], " (", integer.size(), "/",N,")");
            assert(mask_index + mask_Ns[m] <= integer.size());
            std::vector<cv::Point> points{ integer.data() + mask_index, integer.data() + mask_index + mask_Ns[m] };
            mask_index += mask_Ns[m];

            if (SETTING(do_filter).value<bool>() && not contains(SETTING(filter_classes).value<std::vector<uint8_t>>(), cls))
                continue;

            Bounds boundaries(FLT_MAX, FLT_MAX, 0, 0);
            for (auto& pt : points) {
                boundaries.insert_point(pt);
            }

            boundaries.width -= boundaries.x;
            boundaries.height -= boundaries.y;

            boundaries.restrict_to(data.image->bounds());

            // subtract boundary xy from all points
            for (auto& p : points) {
                p.x -= boundaries.x;
                p.y -= boundaries.y;
            }

            print("Boundaries: ", boundaries);

            cv::Mat mask = cv::Mat::zeros(boundaries.height + 1, boundaries.width + 1, CV_8UC1);
            cv::fillPoly(mask, points, 255);
            assert(mask.cols > 0 && mask.rows > 0);
            //tf::imshow("current mask " + Meta::toStr(i), mask);

            //cv::bitwise_and(r3, mask, r3);

            auto blobs = CPULabeling::run(mask);
            if (not blobs.empty()) {
                size_t msize = 0, midx = 0;
                for (size_t j = 0; j < blobs.size(); ++j) {
                    if (blobs.at(j).pixels->size() > msize) {
                        msize = blobs.at(j).pixels->size();
                        midx = j;
                    }
                }

                auto&& pair = blobs.at(midx);
                size_t num_pixels{ 0u };
                for (auto& line : *pair.lines) {
                    line.x0 = saturate(coord_t(line.x0 + boundaries.x), coord_t(0), coord_t(r3.cols - 1));
                    line.x1 = saturate(coord_t(line.x1 + boundaries.x), line.x0, coord_t(r3.cols - 1));
                    line.y = saturate(coord_t(line.y + boundaries.y), coord_t(0), coord_t(r3.rows - 1));
                    num_pixels += line.x1 - line.x0 + 1;
                }

                pair.pred = blob::Prediction{
                    .clid = static_cast<uint8_t>(cls),
                    .p = uint8_t(float(conf) * 255.f)
                };
                pair.extra_flags |= pv::Blob::flag(pv::Blob::Flags::is_instance_segmentation);

                for (auto& line : *pair.lines) {
                    if (line.x0 >= r3.cols
                        || line.x1 >= r3.cols
                        || line.y >= r3.rows)
                        throw U_EXCEPTION("Coordinates of line ", line, " are invalid for image ", r3.cols, "x", r3.rows);
                }
                pv::Blob blob(*pair.lines, *pair.pixels, pair.extra_flags, pair.pred);
                //pair.pixels = (blob.calculate_pixels(r3));
                pair.pixels = std::make_unique<std::vector<uchar>>(num_pixels);
                std::fill(pair.pixels->begin(), pair.pixels->end(), 255);

                auto points = pixel::find_outer_points(&blob, 0);
                if (not points.empty()) {
                    data.outlines.emplace_back(std::move(*points.front()));
                }

                data.predictions.push_back({ .clid = size_t(cls), .p = float(conf) });
                data.frame.add_object(std::move(pair));
            }
            //cv::cvtColor(mask, mask, cv::COLOR_GRAY2BGR);
            //cv::rectangle(mask, boundaries, cv::Scalar(0, 0, 255), 1);


            auto conversion = [&]<ImageMode mode>(){


                /*for (int y = pos.y; y < pos.y + dim.height; ++y) {
                    // integer overflow deals with this, lol
                    if(uint(y) >= data.image->rows)
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
                }*/
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
            
            /*if (not lines.empty()) {
                pv::Blob blob(lines, 0);
                data.predictions.push_back({ .clid = size_t(cls), .p = float(conf) });
                data.frame.add_object(lines, pixels, 0, blob::Prediction{ .clid = uint8_t(cls), .p = uint8_t(float(conf) * 255.f) });
            }*/
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
            py::ModuleProxy bbx("bbx_saved_model", Yolo8InstanceSegmentation::reinit);
            bbx.set_variable("offsets", std::move(offsets));
            bbx.set_variable("image", images);
            bbx.set_variable("oimages", oimages);
            
            std::vector<uint64_t> mask_Ns;
            std::vector<float> mask_points;
            
            auto recv = [&](std::vector<uint64_t> Ns,
                            std::vector<float> vector)
            {
                size_t elements{0};
                size_t outline_elements{0};
                //thread_print("Received a number of results: ", Ns);
                //thread_print("For elements: ", datas);
                
                if(datas.empty() and not Ns.empty()) {
                    FormatExcept("Empty datas with Ns being set: ", datas.size(), " / ", Ns.size());
                    return;
                }

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
                size_t mask_Ns_index{0};
                for(size_t i=0; i<datas.size(); ++i) {
                    auto& data = datas.at(i);
                    auto& scale = scales.at(i);
                    
                    std::span<float> span(vector.data() + elements * 6u,
                                          vector.data() + (elements + Ns.at(i)) * 6u);
                    std::span<uint64_t> Ns_span;
                    elements += Ns.at(i);
                    
                    std::span<float> mask_span;
                    if(not mask_Ns.empty()) {
                        size_t e{0};
                        for (size_t m = 0; m < Ns.at(i); ++m) {
                            e += mask_Ns.at(mask_Ns_index + m);
                        }

                        Ns_span = {
                            mask_Ns.data() + mask_Ns_index,
                            mask_Ns.data() + mask_Ns_index + Ns.at(i)
                        };
                        mask_span = {
                            mask_points.data() + outline_elements * 2u,
                            mask_points.data() + (outline_elements + e) * 2u
                        };
                        thread_print("Data[", i, "] -> ", mask_span.size(), " elements.");
                        outline_elements += e;
                        mask_Ns_index += Ns.at(i);
                    }
                    
                    try {
                        receive(data, scale, span, mask_span, Ns_span);
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
                                                     std::vector<float> m_points)
            {
                thread_print("Received seg masks with:", Ns);
                thread_print("and ", m_points.size() / 2, " mask points.");
                
                mask_Ns = std::move(Ns);
                mask_points = std::move(m_points);
            });
            /*bbx.set_function("receive_with_seg", [&](std::vector<uint64_t> Ns,
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
                    
                    pos += offsets.at(idx);
                    
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
                        
                        cv::threshold(tmp, tmp, threshold, 1.0, cv::THRESH_BINARY);
                        
                        cv::Mat d;// = full_image(Bounds(pos, dim));
                        auto restricted = Bounds(pos, dim);
                        restricted.restrict_to(Bounds(*full_image));
                        if(restricted.width <= 0 || restricted.height <= 0)
                            continue;
                        
                        (*full_image)(restricted).convertTo(d, CV_32FC1);
                        
                        cv::multiply(d, tmp(Bounds(restricted.size())), d);
                        d.convertTo(tmp, CV_8UC1);
                        
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
                            pair.extra_flags |= pv::Blob::flag(pv::Blob::Flags::is_instance_segmentation);
                            
                            pv::Blob blob(*pair.lines, *pair.pixels, pair.extra_flags, pair.pred);
                            auto points = pixel::find_outer_points(&blob, 0);
                            if (not points.empty()) {
                                data.outlines.emplace_back(std::move(*points.front()));
                            }
                            
                            data.predictions.push_back({ .clid = size_t(cls), .p = float(conf) });
                            data.frame.add_object(std::move(pair));
                        }
                    }
                }
                
                recv(Ns, vector);
            });*/

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
        
        if(SETTING(segmentation_model).value<file::Path>().empty())
            throw U_EXCEPTION("When using yolov7 instance segmentation, please set model using command-line argument -sm <path> to set a model (pytorch model).");
        else if(not SETTING(segmentation_model).value<file::Path>().exists())
            throw U_EXCEPTION("Cannot find segmentation instance model file ",SETTING(segmentation_model).value<file::Path>(),".");
        
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
                
                pair.extra_flags |= pv::Blob::flag(pv::Blob::Flags::is_instance_segmentation);
                
                pv::Blob blob(*pair.lines, *pair.pixels, pair.extra_flags, pair.pred);
                auto points = pixel::find_outer_points(&blob, 0);
                if (not points.empty()) {
                    data.outlines.emplace_back(std::move(*points.front()));
                    //for (auto& pt : outline_points.back())
                    //    pt = (pt + blob.bounds().pos())/*.mul(dim.div(image.dimensions())) + pos*/;
                }
                data.predictions.push_back({ .clid = size_t(cls), .p = float(conf) });
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
static_assert(ObjectDetection<Yolo8InstanceSegmentation>);

struct VideoInfo {
    file::Path base;
    Size2 size;
    short framerate;
    bool finite;
    Frame_t length;
};

class AbstractBaseVideoSource {
protected:
    Frame_t i{0_f};
    using gpuMatPtr = std::unique_ptr<useMat>;
    std::mutex buffer_mutex;
    std::vector<gpuMatPtr> buffers;
    
    VideoInfo info;
    
    RepeatedDeferral<std::function<tl::expected<std::tuple<Frame_t, gpuMatPtr>, const char*>()>> _source_frame;
    RepeatedDeferral<std::function<tl::expected<std::tuple<Frame_t, gpuMatPtr, Image::Ptr>, const char*>()>> _resize_cvt;
    
public:
    AbstractBaseVideoSource(VideoInfo info)
    : info(info),
    _source_frame(50u, 15u,
                  std::string("source.frame"),
                  [this]() -> tl::expected<std::tuple<Frame_t, gpuMatPtr>, const char*>
                  {
        return fetch_next();
    }),
    _resize_cvt(50u, 15u,
                std::string("resize+cvtColor"),
                [this]() -> tl::expected<std::tuple<Frame_t, gpuMatPtr, Image::Ptr>, const char*> {
        return this->fetch_next_process();
    })
    {
        notify();
    }
    virtual ~AbstractBaseVideoSource() = default;
    void notify() {
        _source_frame.notify();
        _resize_cvt.notify();
    }
    
    Size2 size() const { return info.size; }
    
    void move_back(gpuMatPtr&& ptr) {
        std::unique_lock guard(buffer_mutex);
        buffers.push_back(std::move(ptr));
    }
    
    std::tuple<Frame_t, gpuMatPtr, Image::Ptr> next() {
        auto result = _resize_cvt.next();
        if(!result)
            return std::make_tuple(Frame_t{}, nullptr, nullptr);
        
        return std::move(result.value());
    }
    
    virtual tl::expected<std::tuple<Frame_t, gpuMatPtr>, const char*> fetch_next() = 0;
    
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
            auto desc = toStr();
            FormatExcept("Unable to load frame ", i, " from video source ", desc.c_str(), " because: ", e.what());
            return tl::unexpected(e.what());
        }
    }
    
    bool is_finite() const {
        return info.finite;
    }
    
    void set_frame(Frame_t frame) {
        if(!is_finite())
            throw std::invalid_argument("Cannot skip on infinite source.");
        i = frame;
    }
    
    Frame_t length() const {
        if(!is_finite()) {
            FormatWarning("Cannot return length of infinite source (", i,").");
            return i;
        }
        return info.length;
    }
    
    virtual std::string toStr() const {return "AbstractBaseVideoSource<>";}
    static std::string class_name() { return "AbstractBaseVideoSource"; }
};


class WebcamVideoSource : public AbstractBaseVideoSource {
    fg::Webcam source;
    
public:
    using SourceType = fg::Webcam;
    
public:
    WebcamVideoSource(fg::Webcam&& source)
        : AbstractBaseVideoSource({.base = "Webcam",
                                   .size = source.size(),
                                   .framerate = short(source.frame_rate()),
                                   .finite = false,
                                   .length = Frame_t{}}),
          source(std::move(source))
    { }

    tl::expected<std::tuple<Frame_t, gpuMatPtr>, const char*> fetch_next() override {
        try {
            if (not i.valid()) {
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
            static Image cpuBuffer(this->source.size().height, this->source.size().width, 3);
            this->source.next(cpuBuffer);
            //this->source.frame(index, cpuBuffer);
            cpuBuffer.get().copyTo(*buffer);

            cv::cvtColor(*buffer, *tmp, cv::COLOR_BGR2RGB);
            std::swap(buffer, tmp);
            return std::make_tuple(index, std::move(buffer));
        }
        catch (const std::exception& e) {
            return tl::unexpected(e.what());
        }
    }

    std::string toStr() const override {
        return "WebcamVideoSource<"+Meta::toStr(source)+">";
    }
    static std::string class_name() { return "WebcamVideoSource"; }
};

class VideoSourceVideoSource : public AbstractBaseVideoSource {
    VideoSource source;
public:
    using SourceType = VideoSource;
    
public:
    VideoSourceVideoSource(VideoSource&& source)
        : AbstractBaseVideoSource({.base = source.base(),
                                   .size = source.size(),
                                   .framerate = source.framerate(),
                                   .finite = true,
                                   .length = source.length()}),
          source(std::move(source))
    { }
    
    tl::expected<std::tuple<Frame_t, gpuMatPtr>, const char*> fetch_next() override {
        if (i >= this->source.length()) {
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

    std::string toStr() const override {
        return "VideoSourceVideoSource<"+Meta::toStr(source)+">";
    }
    static std::string class_name() { return "VideoSourceVideoSource"; }
};


template<typename F>
    requires ObjectDetection<F>
struct OverlayedVideo {
    std::unique_ptr<AbstractBaseVideoSource> source;

    F overlay;
    
    mutable std::mutex index_mutex;
    Frame_t i{0};
    //Image::Ptr original_image;
    //cv::Mat downloader;
    useMat resized;
    
    using return_t = tl::expected<std::tuple<Frame_t, std::future<SegmentationData>>, const char*>;
    RepeatedDeferral<std::function<return_t()>> apply_net;
    
    bool eof() const noexcept {
        assert(source);
        if(not source->is_finite())
            return false;
        return i >= source->length();
    }
    
    OverlayedVideo() = delete;
    OverlayedVideo(const OverlayedVideo&) = delete;
    OverlayedVideo& operator=(const OverlayedVideo&) = delete;
    OverlayedVideo(OverlayedVideo&&) = delete;
    OverlayedVideo& operator=(OverlayedVideo&&) = delete;
    
    template<typename SourceType, typename Callback>
        requires _clean_same<SourceType, VideoSource>
    OverlayedVideo(F&& fn, SourceType&& s, Callback&& callback)
        : source(std::make_unique<VideoSourceVideoSource>(std::move(s))), overlay(std::move(fn)),
            apply_net(10u,
                5u,
                "ApplyNet",
                [this, callback = std::move(callback)](){
                    return retrieve_next(callback);
                })
    {
        apply_net.notify();
    }
    
    template<typename SourceType, typename Callback>
        requires _clean_same<SourceType, fg::Webcam>
    OverlayedVideo(F&& fn, SourceType&& s, Callback&& callback)
        : source(std::make_unique<WebcamVideoSource>(std::move(s))), overlay(std::move(fn)),
            apply_net(10u,
                5u,
                "ApplyNet",
                [this, callback = std::move(callback)](){
                    return retrieve_next(callback);
                })
    {
        apply_net.notify();
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
            assert(source);
            auto&& [nix, buffer, image] = source->next();
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
            source->move_back(std::move(buffer));
            
            //thread_print("Queueing image ", nix);
            //! network processing, and record network fps
            return std::make_tuple(nix, this->overlay.apply(std::move(tiled)));
            
        } catch(const std::exception& e) {
            FormatExcept("Error loading frame ", loaded, " from video ", *source, ": ", e.what());
            return tl::unexpected("Error loading frame.");
        }
    }
    
    void reset(Frame_t frame) {
        std::scoped_lock guard(index_mutex);
        i = frame;
        assert(source);
        if(source->is_finite())
            source->set_frame(i);
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
        switch (type()) {
            case ObjectDetectionType::yolo7:
                Yolo7ObjectDetection::init();
                break;
                
            case ObjectDetectionType::customseg:
            case ObjectDetectionType::yolo7seg:
                Yolo7InstanceSegmentation::init();
                break;
                
            case ObjectDetectionType::yolo8seg:
                Yolo8InstanceSegmentation::init();
                break;
                
            default:
                throw U_EXCEPTION("Unknown detection type: ", type());
        }
    }
    
    static ObjectDetectionType::Class type() {
        return SETTING(detection_type).value<ObjectDetectionType::Class>();
    }
    
    static std::future<SegmentationData> apply(TileImage&& tiled) {
        switch (type()) {
            case ObjectDetectionType::yolo7: {
                auto f = tiled.promise.get_future();
                manager.enqueue(std::move(tiled));
                return f;
            }
                
            case ObjectDetectionType::customseg:
            case ObjectDetectionType::yolo7seg: {
                std::promise<SegmentationData> p;
                auto e = Yolo7InstanceSegmentation::apply(std::move(tiled));
                try {
                    p.set_value(std::move(e.value()));
                } catch(...) {
                    p.set_exception(std::current_exception());
                }
                return p.get_future();
            }
                
            case ObjectDetectionType::yolo8seg: {
                auto f = tiled.promise.get_future();
                manager.enqueue(std::move(tiled));
                return f;
            }
                
            default:
                throw U_EXCEPTION("Unknown detection type: ", type());
        }
    }
    
    static void apply(std::vector<TileImage>&& tiled) {
        if(type() == ObjectDetectionType::yolo7) {
            Yolo7ObjectDetection::apply(std::move(tiled));
            tiled.clear();
            return;
            
        } else if(type() == ObjectDetectionType::yolo8seg) {
            Yolo8InstanceSegmentation::apply(std::move(tiled));
            tiled.clear();
            return;
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
    ConvertScene(Base& window) : Scene(window, "converting", [this](Scene&, DrawStructure& graph){
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
        
        {
            std::unique_lock guard(_mutex_general);
            _cv_ready_for_tracking.notify_all();
            _cv_messages.notify_all();
        }
        
        if (_tracking_thread.joinable()) {
            _tracking_thread.join();
        }
        
        Detection::manager.clean_up();
        
        if (_generator_thread.joinable()) {
            _generator_thread.join();
        }
        
        std::unique_lock guard(_mutex_general);
        if (_output_file) {
            _output_file->close();
        }
        
        _overlayed_video = nullptr;
        _tracker = nullptr;
        
        if(_output_file) {
            pv::File test(_output_file->filename(), pv::FileMode::READ);
            test.print_info();
        }
        
        _output_file = nullptr;
        _next_frame_data = {};
        _progress_data = {};
        _current_data = {};
        _transferred_blobs.clear();
        _object_blobs.clear();
        _progress_blobs.clear();
        _transferred_current_data = {};
    }
    
    void open_video() {
        VideoSource video_base(SETTING(source).value<std::string>());
        video_base.set_colors(ImageMode::RGB);
        
        SETTING(frame_rate) = Settings::frame_rate_t(video_base.framerate() != short(-1) ? video_base.framerate() : 25);
        
        print("filename = ", SETTING(filename).value<file::Path>());
        print("video_base = ", video_base.base());
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
        _video_info["length"] = _overlayed_video->source->length();
        SETTING(video_length) = uint64_t(_overlayed_video->source->length().get());
        SETTING(meta_real_width) = float(expected_size.width * 10);
        
        //SETTING(cm_per_pixel) = float(SETTING(meta_real_width).value<float>() / _overlayed_video->source.size().width);
        
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
            if(bg.cols == tmp.size().width && bg.rows == tmp.size().height)
                cv::cvtColor(bg, bg, cv::COLOR_BGR2GRAY);
            else {
                tmp.generate_average(bg, 0);
                cv::imwrite(average_name().str(), bg);
            }
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

    void open_camera() {
        using namespace grab;
        fg::Webcam camera;
        camera.set_color_mode(ImageMode::RGB);
        
        SETTING(frame_rate) = Settings::frame_rate_t(25);
        if(SETTING(filename).value<file::Path>().empty())
            SETTING(filename) = file::Path("webcam");
        
        setDefaultSettings();
        _output_size = (Size2(camera.size()) * SETTING(meta_video_scale).value<float>()).map(roundf);
        SETTING(output_size) = _output_size;
        _video_info["resolution"] = _output_size;
        
        _overlayed_video = std::make_unique<OverlayedVideo<Detection>>(
               Detection{},
               std::move(camera),
               [this](){
                   _cv_messages.notify_one();
               }
        );
        
        _overlayed_video->source->notify();
        
        _video_info["length"] = _overlayed_video->source->length();
        SETTING(video_length) = uint64_t(_overlayed_video->source->length().get());
        SETTING(meta_real_width) = float(expected_size.width * 10);
        
        //SETTING(cm_per_pixel) = float(SETTING(meta_real_width).value<float>() / _overlayed_video->source.size().width);
        
        printDebugInformation();
        
        cv::Mat bg = cv::Mat::zeros(_output_size.height, _output_size.width, CV_8UC1);
        bg.setTo(255);
        
        /*VideoSource tmp(SETTING(source).value<std::string>());
        if(not average_name().exists()) {
            tmp.generate_average(bg, 0);
            cv::imwrite(average_name().str(), bg);
        } else {
            print("Loading from file...");
            bg = cv::imread(average_name().str());
            cv::cvtColor(bg, bg, cv::COLOR_BGR2GRAY);
        }*/
        
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
    
    void activate() override {
        _should_terminate = false;
        
        try {
            print("Loading source = ", SETTING(source).value<std::string>());
            if(SETTING(source).value<std::string>() == "webcam")
                open_camera();
            else
                open_video();
            
            auto size = _overlayed_video->source->size();
            window()->set_window_size(Size2(1024, size.height / size.width * 1024));
        
        } catch(const std::exception& e) {
            FormatExcept("Exception when switching scenes: ", e.what());
            SceneManager::getInstance().set_active("starting-scene");
        }
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
                if(blob->prediction().valid()) {
                    auto pred = blob->prediction();
                    assign = {
                        .clid = pred.clid,
                        .p = static_cast<float>(pred.p) / 255.f
                    };
                    /*if(auto it = _current_data.predictions.find(blob->parent_id());
                       it != _current_data.predictions.end())
                    {
                        assign = it->second;
                        
                    } else if((it = _current_data.predictions.find(blob->blob_id())) != _current_data.predictions.end())
                    {
                        assign = it->second;
                        
                    } else
                        print("[draw]3 blob ", blob->blob_id(), " not found...");*/
                    
                } else
                    print("[draw]4 blob ", blob->blob_id(), " prediction not found...");
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
                    if(std::get<1>(items.front()).valid()
                       && std::get<1>(items.front()).wait_for(std::chrono::milliseconds(1)) == std::future_status::ready)
                    {
                        auto data = std::get<1>(items.front()).get();
                        //thread_print("Got data for item ", data.frame.index());
                        
                        _next_frame_data = std::move(data);
                        _cv_ready_for_tracking.notify_one();
                        
                        items.erase(items.begin());
                    } else if(not std::get<1>(items.front()).valid()) {
                        FormatExcept("Invalid future ", std::get<0>(items.front()));
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
                //try {
                    perform_tracking();
                    //guard.lock();
                //} catch(...) {
                //    FormatExcept("Exception while tracking");
                //    throw;
                //}
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

class LoadingScene : public Scene {
    class FileItem {
        GETTER(file::Path, path)

    public:
        FileItem(const file::Path& path = "");

        Color base_color() const;
        Color color() const;
        operator std::string() const;
        bool operator!=(const FileItem& other) const {
            return _path != other._path;
        }
    };

public:
    struct Settings {
        enum Display {
            None = 0,
            Browser = 2
        } display;

        std::string name;
        std::string extension;
        derived_ptr<Entangled> content;

        Settings(const std::string& name = "", const std::string& extensions = "", const derived_ptr<Entangled>& content = nullptr, Display d = Display::Browser)
            : display(d), name(name), extension(extensions), content(content)
        {}

        bool is_valid_extension(const file::Path& path) const {
            return file::valid_extension(path, extension);
        }

        std::string toStr() const {
            return name;
        }

        static std::string class_name() {
            return "LoadingScene::Settings";
        }
    };

protected:
    derived_ptr<Text> _description;
    derived_ptr<StaticText> _selected_text;
    derived_ptr<ScrollableList<FileItem>> _list;
    derived_ptr<Button> _button;
    derived_ptr<Dropdown> _textfield;
    derived_ptr<VerticalLayout> _rows;
    derived_ptr<HorizontalLayout> _columns;
    derived_ptr<VerticalLayout> _overall;
    derived_ptr<HorizontalLayout> _tabs_bar;
    std::unordered_map<int, derived_ptr<Tooltip>> _tooltips;
    std::vector<Layout::Ptr> tabs_elements;
    std::vector<FileItem> _names;
    std::vector<Dropdown::TextItem> _search_items;

    file::Path _path;
    bool _running;

    std::set<file::Path, std::function<bool(const file::Path&, const file::Path&)>> _files;
    file::Path _selected_file;
    GETTER(file::Path, confirmed_file)
        std::function<void(const file::Path&, std::string)> _callback, _on_select_callback;
    std::function<void(DrawStructure&)> _on_update;
    std::function<bool(file::Path)> _validity;
    std::function<void(file::Path)> _on_open;
    std::function<void(std::string)> _on_tab_change;
    std::queue<std::function<void()>> _execute;
    std::mutex _execute_mutex;
    std::map<std::string, Settings> _tabs;
    GETTER(Settings, current_tab)
        Settings _default_tab;

    // The HorizontalLayout for the two buttons and the image
    HorizontalLayout _main_layout;

    dyn::Context context {
        .variables = {
            {
                "global",
                std::unique_ptr<VarBase_t>(new Variable([](std::string) -> sprite::Map& {
                    return GlobalSettings::map();
                }))
            }
        }
    };
    dyn::State state;
    std::vector<Layout::Ptr> objects;

public:
    LoadingScene(Base& window, const file::Path& start, const std::string& extension,
        std::function<void(const file::Path&, std::string)> callback,
        std::function<void(const file::Path&, std::string)> on_select_callback)
        : Scene(window, "loading-scene", [this](auto&, DrawStructure& graph) { _draw(graph); }),
        _description(std::make_shared<Text>("Please choose a file in order to continue.", Loc(10, 10), Font(0.75))),
        _columns(std::make_shared<HorizontalLayout>()),
        _overall(std::make_shared<VerticalLayout>()),
        _path(start),
        _running(true),
        _files([](const file::Path& A, const file::Path& B) -> bool {
        return (A.is_folder() && !B.is_folder()) || (A.is_folder() == B.is_folder() && A.str() < B.str()); //A.str() == ".." || (A.str() != ".." && ((A.is_folder() && !B.is_folder()) || (A.is_folder() == B.is_folder() && A.str() < B.str())));
            }),
        _callback(callback),
        _on_select_callback(on_select_callback)
    {
        auto dpi = ((const IMGUIBase*)&window)->dpi_scale();
        print(window.window_dimensions().mul(dpi));
        _default_tab.extension = extension;
        
    }

    void activate() override {
        set_tab("");

        ((IMGUIBase*)window())->set_open_files_fn([this](const std::vector<file::Path>& paths) -> bool {
            if (paths.size() != 1)
                return false;

            auto path = paths.front();
            if (!_validity || _validity(path)) //path.exists() || path.str() == "/" || path.add_extension("pv").exists())
            {
                file_selected(0, path.str());
                return true;
            }
            else {
                FormatError("Path ", path.str(), " cannot be opened.");
            }
            return false;
        });

        _columns->set_policy(HorizontalLayout::TOP);
        //_columns->set_background(Transparent, Red);
        _overall->set_policy(VerticalLayout::CENTER);
        //_overall->set_background(Transparent, Blue);

        _list = std::make_shared<ScrollableList<FileItem>>(Bounds(
            0,
            0,
            //_graph->width() - 
            20 - (_current_tab.content ? _current_tab.content->width() + 5 : 0),
            //_graph->height() - 
            70 - 10 - 100 - 70));

        _list->set_stays_toggled(true);
        //if(_extra)
        //    _extra->set_background(Transparent, Green);

        //auto overall_width = _list->width() + (_extra ? _extra->width() : 0);

        _button = Button::MakePtr("Open", Bounds(_list->pos() + Vec2(0, _list->height() + 40), attr::Size(100, 30)));

        _textfield = std::make_shared<Dropdown>(Bounds(0, 0, _list->width(), 30));
        //_textfield = std::make_shared
        _textfield->on_select([this](long_t, const Dropdown::TextItem& item) {
            file::Path path;

            if (((std::string)item).empty()) {
                path = _textfield->textfield()->text();
            }
            else
                path = file::Path((std::string)item);

            if (!_validity || _validity(path))
            {
                file_selected(0, path.str());
                if (!path.is_regular())
                    _textfield->select_textfield();
            }
            else
                FormatError("Path ", path.str(), " cannot be opened.");
            });

        _textfield->on_text_changed([this](std::string str) {
            auto path = file::Path(str);
            auto file = (std::string)path.filename();

            if (path.empty() || (path == _path || ((!path.exists() || !path.is_folder()) && path.remove_filename() == _path)))
            {
                // still in the same folder
            }
            else if (utils::endsWith(str, file::Path::os_sep()) && path != _path && path.is_folder()) {
                file_selected(0, path);
            }
            });

        _rows = std::make_shared<VerticalLayout>(std::vector<Layout::Ptr>{
            _textfield, _list
        });
        //_rows->set_background(Transparent, Yellow);

        _columns->set_name("Columns");
        _rows->set_name("Rows");

        if (_current_tab.content && !_selected_file.empty())
            _columns->set_children({ _rows, _current_tab.content });
        else
            _columns->set_children({ _rows });

        _overall->set_children({ _columns });

        //update_size();

        if (!_path.exists())
            _path = ".";

        try {
            auto files = _path.find_files(_current_tab.extension);
            _files.clear();
            _files.insert(files.begin(), files.end());
            _files.insert("..");

        }
        catch (const UtilsException& ex) {
            FormatError("Cannot list folder ", _path, " (", ex.what(), ").");
        }

        update_names();

        _textfield->textfield()->set_text(_path.str());
        //_textfield->set_text(_path.str());

        //_graph->set_scale(_base.dpi_scale() * gui::interface_scale());
        _list->on_select([this](auto i, auto& path) { file_selected(i, path.path()); });

        _button->set_font(gui::Font(0.6f, Align::Center));
        _button->on_click([this](auto) {
            _running = false;
            _confirmed_file = _selected_file;
            if (_on_open)
                _on_open(_confirmed_file);
            });

        _list->set_font(gui::Font(0.6f, gui::Align::Left));

        //update_size();
    }

    void deactivate() override {
        // Logic to clear or save state if needed
    }

    void _draw(DrawStructure& graph) {
        using namespace gui;
        tf::show();

        {
            std::lock_guard<std::mutex> guard(_execute_mutex);
            auto N = _execute.size();
            while (!_execute.empty()) {
                _execute.front()();
                _execute.pop();
            }
            if (N > 0)
                update_size(graph);
        }

        _list->set_bounds(Bounds(
            0,
            0,
            graph.width() - 
            20 - (_current_tab.content ? _current_tab.content->width() + 5 : 0),
            graph.height() - 
            70 - 10 - 100 - 70));
        update_size(graph);

        if (!_list)
            return;

        //_graph->wrap_object(*_textfield);
        //_graph->wrap_object(*_list);
        graph.wrap_object(*_overall);
        if (_on_update)
            _on_update(graph);

        auto scale = graph.scale().reciprocal();
        auto dim = window()->window_dimensions().mul(scale * gui::interface_scale());
        graph.draw_log_messages(Bounds(Vec2(), dim));
        if (!_tooltips.empty()) {
            for (auto&& [ID, obj] : _tooltips)
                graph.wrap_object(*obj);
        }

        if (!_selected_file.empty()) {

        }
        if (SETTING(terminate))
            _running = false;
    }

    void set_tabs(const std::vector<Settings>&);
    void set_tab(std::string);
    void open();
    void execute(std::function<void()>&&);
    virtual void update_size(DrawStructure& graph);
    void on_update(std::function<void(DrawStructure&)>&& fn) { _on_update = std::move(fn); }
    void on_open(std::function<void(file::Path)>&& fn) { _on_open = std::move(fn); }
    void on_tab_change(std::function<void(std::string)>&& fn) { _on_tab_change = std::move(fn); }
    void set_validity_check(std::function<bool(file::Path)>&& fn) { _validity = std::move(fn); }
    void deselect();
    void set_tooltip(int ID, Drawable*, const std::string&);

private:
    void file_selected(size_t i, file::Path path);
    void update_names();
    void update_tabs();
    void change_folder(const file::Path&);
};

void LoadingScene::set_tabs(const std::vector<Settings>& tabs) {
    _tabs.clear();
    tabs_elements.clear();

    for (auto tab : tabs) {
        if (tab.extension == "")
            tab.extension = _default_tab.extension;
        _tabs[tab.name] = tab;

        auto button = new Button(tab.name, attr::Size(Base::default_text_bounds(tab.name).width + 20, 40));
        button->set_fill_clr(Color(100, 100, 100, 255));
        button->set_toggleable(true);
        button->on_click([this, button](auto) {
            if (button->toggled()) {
                set_tab(button->txt());
            }
            });
        auto ptr = std::shared_ptr<Drawable>(button);
        tabs_elements.push_back(ptr);
    }

    if (_tabs.size() > 1) {
        _tabs_bar = std::make_shared<HorizontalLayout>(tabs_elements);
    }
    else {
        _tabs_bar = nullptr;
    }

    std::vector<Layout::Ptr> childs;
    if (_tabs_bar)
        childs.push_back(_tabs_bar);

    childs.push_back(_columns);

    if (_selected_text && _button) {
        childs.push_back(_selected_text);
        childs.push_back(_button);
    }

    _overall->set_children(childs);

    if (!_tabs.empty())
        set_tab(tabs.front().name);
    else
        set_tab("");
}

void LoadingScene::deselect() {
    file_selected(0, "");
}

void LoadingScene::set_tab(std::string tab) {
    if (tab != _current_tab.name) {
    }
    else
        return;

    if (tab.empty()) {
        _current_tab = _default_tab;

        if (_on_tab_change)
            _on_tab_change(_current_tab.name);
        deselect();

    }
    else if (!_tabs.count(tab)) {
        auto str = Meta::toStr(_tabs);
        FormatExcept("FileChooser ", str, " does not contain tab ", tab, ".");
    }
    else {
        _current_tab = _tabs.at(tab);
        if (_on_tab_change)
            _on_tab_change(_current_tab.name);
        deselect();
    }

    for (auto& ptr : tabs_elements) {
        if (static_cast<Button*>(ptr.get())->txt() != tab) {
            static_cast<Button*>(ptr.get())->set_toggle(false);
            static_cast<Button*>(ptr.get())->set_clickable(true);
        }
        else {
            static_cast<Button*>(ptr.get())->set_toggle(true);
            static_cast<Button*>(ptr.get())->set_clickable(false);
        }
    }

    change_folder(_path);
    if (!_selected_file.empty())
        file_selected(0, _selected_file);

    if (_current_tab.content) {
        _current_tab.content->auto_size(Margin{ 0,0 });
        _current_tab.content->set_name("Extra");
    }

    if (_current_tab.display == Settings::Display::None) {
        _rows->set_children({});

    }
    else {
        _rows->set_children(std::vector<Layout::Ptr>{
            _textfield, _list
        });
    }

    //update_size();
    //_graph->set_dirty(&_base);
}

void LoadingScene::update_names() {
    _names.clear();
    _search_items.clear();
    for (auto& f : _files) {
        if (f.str() == ".." || !utils::beginsWith((std::string)f.filename(), '.')) {
            _names.push_back(FileItem(f));
            _search_items.push_back(Dropdown::TextItem(f.str()));
        }
    }
    _list->set_items(_names);
    _textfield->set_items(_search_items);
}

void LoadingScene::set_tooltip(int ID, Drawable* ptr, const std::string& docs)
{
    auto it = _tooltips.find(ID);
    if (!ptr) {
        if (it != _tooltips.end())
            _tooltips.erase(it);

    }
    else {
        if (it == _tooltips.end()) {
            _tooltips[ID] = std::make_shared<Tooltip>(ptr, 400);
            _tooltips[ID]->text().set_default_font(Font(0.5));
            it = _tooltips.find(ID);
        }
        else
            it->second->set_other(ptr);

        it->second->set_text(docs);
    }
}

LoadingScene::FileItem::FileItem(const file::Path& path) : _path(path)
{

}

LoadingScene::FileItem::operator std::string() const {
    return std::string(_path.filename());
}

Color LoadingScene::FileItem::base_color() const {
    return _path.is_folder() ? Color(80, 80, 80, 200) : Color(100, 100, 100, 200);
}

Color LoadingScene::FileItem::color() const {
    return _path.is_folder() ? Color(180, 255, 255, 255) : White;
}

/*void LoadingScene::open() {
    _base.loop();
    if (_callback)
        _callback(_confirmed_file, _current_tab.name);
}*/

void LoadingScene::change_folder(const file::Path& p) {
    auto org = _path;
    auto copy = _files;

    if (p.str() == "..") {
        try {
            _path = _path.remove_filename();
            auto files = _path.find_files(_current_tab.extension);
            _files.clear();
            _files.insert(files.begin(), files.end());
            _files.insert("..");

            _list->set_scroll_offset(Vec2());
            _textfield->textfield()->set_text(_path.str());

        }
        catch (const UtilsException&) {
            _path = org;
            _files = copy;
        }
        update_names();

    }
    else if (p.is_folder()) {
        try {
            _path = p;
            auto files = _path.find_files(_current_tab.extension);
            _files.clear();
            _files.insert(files.begin(), files.end());
            _files.insert("..");

            _list->set_scroll_offset(Vec2());
            _textfield->textfield()->set_text(_path.str() + file::Path::os_sep());

        }
        catch (const UtilsException&) {
            _path = org;
            _files = copy;
        }
        update_names();
    }
}

void LoadingScene::file_selected(size_t, file::Path p) {
    if (!p.empty() && (p.str() == ".." || p.is_folder())) {
        change_folder(p);

    }
    else {
        _selected_file = p;
        if (!_selected_file.empty() && _selected_file.remove_filename() != _path) {
            change_folder(_selected_file.remove_filename());
        }

        if (p.empty()) {
            _selected_file = file::Path();
            _selected_text = nullptr;
            if (_tabs_bar)
                _overall->set_children({
                    _tabs_bar,
                    _columns
                    });
            else
                _overall->set_children({
                    _columns
                    });

        }
        else {
            if (!_selected_text)
                _selected_text = std::make_shared<StaticText>("Selected: " + _selected_file.str(), SizeLimit(700, 0), Font(0.6f));
            else
                _selected_text->set_txt("Selected: " + _selected_file.str());

            if (_tabs_bar)
                _overall->set_children({
                    _tabs_bar,
                    _columns,
                    _selected_text,
                    _button
                    });
            else
                _overall->set_children({
                    _columns,
                    _selected_text,
                    _button
                    });
        }
        _overall->update_layout();

        if (!_selected_file.empty() && _on_select_callback)
            _on_select_callback(_selected_file, _current_tab.extension);
        //update_size();
    }

    update_names();
}

void LoadingScene::update_size(DrawStructure& graph) {
    float s = graph.scale().x / gui::interface_scale();

    if (_selected_text && !_selected_file.empty()) {
        _selected_text->set_max_size(Size2(graph.width() / s, -1));
    }

    //if(_tabs_bar) _tabs_bar->auto_size(Margin{0,0});
    //if(_tabs_bar) _tabs_bar->update_layout();

    if (_current_tab.display == Settings::Display::None) {
        if (_current_tab.content) {
            _columns->set_children(std::vector<Layout::Ptr>{_current_tab.content});
        }
        else
            _columns->clear_children();

    }
    else if (_current_tab.content && !_selected_file.empty())
        _columns->set_children({ _rows, _current_tab.content });
    else
        _columns->set_children({ _rows });

    //_columns->set_background(Transparent, Purple);
    if (_current_tab.content)
        _current_tab.content->auto_size(Margin{ 0,0 });

    float left_column_height = graph.height() / s - 50 - 10 - (_selected_text && !_selected_file.empty() ? _button->height() + 85 : 0) - (_tabs_bar ? _tabs_bar->height() + 10 : 0);
    _button->set_bounds(Bounds(_list->pos() + Vec2(0, left_column_height), Size2(100, 30)));

    float left_column_width = graph.width() / s - 20
        - (_current_tab.content && _current_tab.content->width() > 20 && !_selected_file.empty() ? _current_tab.content->width() + 30 : 0) - 10;

    _list->set_bounds(Bounds(0, 0, left_column_width, left_column_height));
    _textfield->set_bounds(Bounds(0, 0, left_column_width, 30));

    /*if (_rows) _rows->auto_size(Margin{0,0});
    if(_rows) _rows->update_layout();

    _columns->auto_size(Margin{0,0});
    _columns->update_layout();*/

    _overall->auto_size(Margin{ 0,0 });
    _overall->update_layout();
}

void LoadingScene::execute(std::function<void()>&& fn) {
    std::lock_guard<std::mutex> guard(_execute_mutex);
    _execute.push(std::move(fn));
}

class StartingScene : public Scene {
    file::Path _image_path = file::DataLocation::parse("app", "gfx/"+SETTING(app_name).value<std::string>()+"_1024.png");

    // The image of the logo
    std::shared_ptr<ExternalImage> _logo_image;
    std::shared_ptr<Entangled> _title = std::make_shared<Entangled>();

    // The list of recent items
    std::shared_ptr<ScrollableList<>> _recent_items;
    std::shared_ptr<VerticalLayout> _buttons_and_items = std::make_shared<VerticalLayout>();
    
    std::shared_ptr<VerticalLayout> _logo_title_layout = std::make_shared<VerticalLayout>();
    std::shared_ptr<HorizontalLayout> _button_layout;
    
    // The two buttons for user interactions, now as Layout::Ptr
    std::shared_ptr<Button> _video_file_button;
    std::shared_ptr<Button> _camera_button;

    // The HorizontalLayout for the two buttons and the image
    HorizontalLayout _main_layout;
    
    dyn::Context context {
        .variables = {
            {
                "global",
                std::unique_ptr<VarBase_t>(new Variable([](std::string) -> sprite::Map& {
                    return GlobalSettings::map();
                }))
            }
        }
    };
    dyn::State state;
    std::vector<Layout::Ptr> objects;

public:
    StartingScene(Base& window)
    : Scene(window, "starting-scene", [this](auto&, DrawStructure& graph){ _draw(graph); }),
      _logo_image(std::make_shared<ExternalImage>(Image::Make(cv::imread(_image_path.str(), cv::IMREAD_UNCHANGED)))),
      _recent_items(std::make_shared<ScrollableList<>>(Bounds(0, 10, 310, 500))),
      _video_file_button(std::make_shared<Button>("Open file", attr::Size(150, 50))),
      _camera_button(std::make_shared<Button>("Camera", attr::Size(150, 50)))
    {
        auto dpi = ((const IMGUIBase*)&window)->dpi_scale();
        print(window.window_dimensions().mul(dpi), " and logo ", _logo_image->size());
        
        // Callback for video file button
        _video_file_button->on_click([](auto){
            // Implement logic to handle the video file
            SceneManager::getInstance().set_active("converting");
        });

        // Callback for camera button
        _camera_button->on_click([](auto){
            // Implement logic to start recording from camera
            SETTING(source).value<std::string>() = "webcam";
            SceneManager::getInstance().set_active("converting");
        });
        
        // Create a new HorizontalLayout for the buttons
        _button_layout = std::make_shared<HorizontalLayout>(std::vector<Layout::Ptr>{
            Layout::Ptr(_video_file_button),
            Layout::Ptr(_camera_button)
        });
        //_button_layout->set_pos(Vec2(1024 - 10, 550));
        //_button_layout->set_origin(Vec2(1, 0));

        _buttons_and_items->set_children({
            Layout::Ptr(_recent_items),
            Layout::Ptr(_button_layout)
        });
        
        _logo_title_layout->set_children({
            Layout::Ptr(_title),
            Layout::Ptr(_logo_image)
        });
        
        // Set the list and button layout to the main layout
        _main_layout.set_children({
            Layout::Ptr(_logo_title_layout),
            Layout::Ptr(_buttons_and_items)
        });
        //_main_layout.set_origin(Vec2(1, 0));
    }

    void activate() override {
        // Fill the recent items list
        _recent_items->set_items({
            TextItem("olivia_momo-carrot_t4t_041822.mp4", 0),
            TextItem("DJI_0268.MOV", 1),
            TextItem("8guppies_20s", 2)
        });
    }

    void deactivate() override {
        // Logic to clear or save state if needed
    }

    void _draw(DrawStructure& graph) {
        dyn::update_layout("welcome_layout.json", context, state, objects);
        
        //auto dpi = ((const IMGUIBase*)window())->dpi_scale();
        auto max_w = window()->window_dimensions().width * 0.65;
        auto max_h = window()->window_dimensions().height - _button_layout->height() - 25;
        auto scale = Vec2(max_w * 0.4 / _logo_image->width());
        _logo_image->set_scale(scale);
        _title->set_size(Size2(max_w, 25));
        _recent_items->set_size(Size2(_recent_items->width(), max_h));
        
        graph.wrap_object(_main_layout);
        
        std::vector<Layout::Ptr> _objs{objects.begin(), objects.end()};
        _objs.insert(_objs.begin(), Layout::Ptr(_title));
        _objs.push_back(Layout::Ptr(_logo_image));
        _logo_title_layout->set_children(_objs);
        _logo_title_layout->set_policy(VerticalLayout::Policy::CENTER);
        
        
        for(auto &obj : objects) {
            dyn::update_objects(graph, obj, context, state);
            //graph.wrap_object(*obj);
        }
        
        _buttons_and_items->auto_size(Margin{0,0});
        _logo_title_layout->auto_size(Margin{0,0});
        _main_layout.auto_size(Margin{0,0});
    }
};

void launch_gui() {
    DrawStructure graph(1024, 768);
    IMGUIBase base(window_title(), graph, [&, ptr = &base]()->bool {
        UNUSED(ptr);
        graph.draw_log_messages();
        
        return true;
    }, [](Event e) {
        if(e.type == EventType::KEY) {
            if(e.key.code == Keyboard::Escape) {
                SETTING(terminate) = true;
            }
        }
    });
    
    auto& manager = SceneManager::getInstance();
    
    StartingScene start(base);
    manager.register_scene(&start);
    manager.set_active(&start);
    
    ConvertScene converting(base);
    manager.register_scene(&converting);
    //manager.set_active(&converting);

    LoadingScene loading(base, file::DataLocation::parse("output"), ".pv", [](const file::Path&, std::string) {
        }, [](const file::Path&, std::string) {

        });
    manager.register_scene(&loading);

    graph.set_size(Size2(1024, converting.output_size().height / converting.output_size().width * 1024));
    
    base.platform()->set_icons({
        //file::DataLocation::parse("app", "gfx/"+SETTING(app_name).value<std::string>()+"_16.png"),
        file::DataLocation::parse("app", "gfx/"+SETTING(app_name).value<std::string>()+"_32.png"),
        file::DataLocation::parse("app", "gfx/"+SETTING(app_name).value<std::string>()+"_48.png"),
        file::DataLocation::parse("app", "gfx/"+SETTING(app_name).value<std::string>()+"_64.png"),
        file::DataLocation::parse("app", "gfx/"+SETTING(app_name).value<std::string>()+"_128.png"),
        file::DataLocation::parse("app", "gfx/"+SETTING(app_name).value<std::string>()+"_256.png")
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
    Detection::manager.~PipelineManager<TileImage>();
}

int main(int argc, char**argv) {
    using namespace gui;
    
    default_config::register_default_locations();
    
    ::default_config::get(GlobalSettings::map(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    
    grab::default_config::get(GlobalSettings::map(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    grab::default_config::get(GlobalSettings::set_defaults(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    
    gui::init_errorlog();
    set_thread_name("main");
    
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
    
    SETTING(do_filter) = false;
    SETTING(filter_classes) = std::vector<uint8_t>{};
    SETTING(is_writing) = false;
    
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
    
    launch_gui();
    
    py::deinit();
    return 0;
}

