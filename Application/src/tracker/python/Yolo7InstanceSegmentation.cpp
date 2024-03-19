/*#include <python/Yolo7InstanceSegmentation.h>
#include <misc/PythonWrapper.h>
#include <processing/CPULabeling.h>
#include <misc/AbstractVideoSource.h>
#include <misc/PixelTree.h>
#include <python/ModuleProxy.h>

namespace track {

void Yolo7InstanceSegmentation::reinit(ModuleProxy& proxy) {
    proxy.set_variable("model_type", detect::detection_type().toStr());
    
    if(SETTING(segmentation_model).value<file::Path>().empty())
        throw U_EXCEPTION("When using yolov7 instance segmentation, please set model using command-line argument -sm <path> to set a model (pytorch model).");
    else if(not SETTING(segmentation_model).value<file::Path>().exists())
        throw U_EXCEPTION("Cannot find segmentation instance model file ",SETTING(segmentation_model).value<file::Path>(),".");
    
    proxy.set_variable("model_path", SETTING(segmentation_model).value<file::Path>().str());
    proxy.set_variable("image_size", detect::get_model_image_size());
    proxy.run("load_model");
}

void Yolo7InstanceSegmentation::init() {
    Python::schedule([](){
        using py = track::PythonIntegration;
        ModuleProxy proxy{"bbx_saved_model", reinit};
        
    }).get();
}

void Yolo7InstanceSegmentation::receive(std::vector<Vec2> offsets, SegmentationData& data, Vec2 scale_factor, std::vector<float>& masks, const std::vector<float>& vector, const std::vector<int>& meta) {
    //print(vector);
    size_t N = vector.size() / 6u;
    
    cv::Mat full_image;
    //cv::Mat back;
    if(data.image->dims == 3)
        convert_to_r3g3b2<3>(data.image->get(), full_image);
    else
        convert_to_r3g3b2<4>(data.image->get(), full_image);
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
            }
            data.predictions.push_back({ .clid = size_t(cls), .p = float(conf) });
            data.frame.add_object(std::move(pair));
        }
    }
}

tl::expected<SegmentationData, const char*> Yolo7InstanceSegmentation::apply(TileImage&& tiled) {
    namespace py = Python;
    
    Vec2 scale = SETTING(output_size).value<Size2>().div(tiled.source_size);
    print("Image scale: ", scale, " with tile source=", tiled.source_size, " image=", tiled.data.image->dimensions()," output_size=", SETTING(output_size).value<Size2>(), " original=", tiled.original_size);
    
    for(auto p : tiled.offsets()) {
        tiled.data.tiles.push_back(Bounds(p.x, p.y, tiled.tile_size.width, tiled.tile_size.height).mul(scale));
    }
    
    py::schedule([&tiled, scale, offsets = tiled.offsets()]() mutable {
        using py = track::PythonIntegration;
        ModuleProxy bbx("bbx_saved_model", reinit);
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
        if (AbstractBaseVideoSource::_network_samples.load() > 100) {
            AbstractBaseVideoSource::_network_samples = AbstractBaseVideoSource::_network_fps = 0;
        }
        AbstractBaseVideoSource::_network_fps = AbstractBaseVideoSource::_network_fps.load() + 1.0 / timer.elapsed();
        AbstractBaseVideoSource::_network_samples = AbstractBaseVideoSource::_network_samples.load() + 1;
        
    }).get();
    
    return std::move(tiled.data);
}

}
*/
