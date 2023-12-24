#include "Yolo7ObjectDetection.h"
#include <tracking/PythonWrapper.h>
#include <video/Video.h>
#include <grabber/misc/default_config.h>
#include <misc/AbstractVideoSource.h>
#include <misc/PixelTree.h>

namespace track {

void Yolo7ObjectDetection::reinit(ModuleProxy& proxy) {
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
    proxy.set_variable("image_size", get_model_image_size());
    proxy.run("load_model");
}

void Yolo7ObjectDetection::init() {
    Python::schedule([](){
        using py = track::PythonIntegration;
        ModuleProxy proxy{
            "bbx_saved_model",
            Yolo7ObjectDetection::reinit
        };
    }).get();
}

void Yolo7ObjectDetection::receive(SegmentationData& data, Vec2 scale_factor, const std::span<float>& vector) {
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
            data.frame.add_object(lines, pixels, 0, 
                                  blob::Prediction{ .clid = uint8_t(cls), .p = uint8_t(float(conf) * 255.f) });
        }
    }
}

void Yolo7ObjectDetection::apply(std::vector<TileImage>&& tiles) {
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
        
        if(not tiled.promise)
            throw U_EXCEPTION("Promise was not set.");
        promises.push_back(std::move(*tiled.promise));
        tiled.promise = nullptr;
        
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
        ModuleProxy bbx("bbx_saved_model", Yolo7ObjectDetection::reinit);
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
                    if(data.image->dims == 3)
                        convert_to_r3g3b2<3>(data.image->get(), *converted_images[idx]);
                    else
                        convert_to_r3g3b2<4>(data.image->get(), *converted_images[idx]);
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

        if (AbstractBaseVideoSource::_network_samples.load() > 10) {
            AbstractBaseVideoSource::_network_samples = AbstractBaseVideoSource::_network_fps = 0;
        }
        AbstractBaseVideoSource::_network_fps = AbstractBaseVideoSource::_network_fps.load() + (double(_N) / timer.elapsed());
        AbstractBaseVideoSource::_network_samples = AbstractBaseVideoSource::_network_samples.load() + 1;
        
    }).get();
}

}
