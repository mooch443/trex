#include "Yolo8.h"
#include <misc/PixelTree.h>
#include <misc/PythonWrapper.h>
#include <grabber/misc/default_config.h>
#include <video/Video.h>
#include <misc/Timer.h>

namespace track {

std::mutex running_mutex;
std::shared_future<void> running_prediction;
std::promise<void> running_promise;

std::mutex init_mutex;
std::future<void> init_future;

std::atomic<bool> yolo8_initialized{false};
std::atomic<double> _network_fps{0.0};
std::atomic<size_t> _network_samples{0u};

std::vector<detect::ModelConfig> _loaded_models;

std::string Yolo8::default_model() {
    return "yolov8n-pose.pt";
}

bool Yolo8::valid_model(const file::Path& path) {
    std::string input_string = path.str();
    if (path.has_extension() && path.extension() == "pt")
        return true;

    return false;
}

bool Yolo8::is_default_model(const file::Path& path) {
    std::string input_string = path.str();
    std::regex regex_pattern("yolov8.*\\.pt");
    std::regex regex_pattern2("yolov8.*$");

    if (std::regex_match(input_string, regex_pattern)) {
        return true;

    } else if(std::regex_match(input_string, regex_pattern2)) {
        return true;
    }
    
    if(path.exists() && path.has_extension() && path.extension() == "pt")
        return true;
    
    return false;
}

void Yolo8::reinit(ModuleProxy& proxy) {
    proxy.set_variable("model_type", detect::detection_type().toStr());
    
    if(SETTING(detect_model).value<file::Path>().empty()) {
        print("You can provide a model for object detection using the command-line argument -m <path>. Otherwise, we will assume YOLOv8n-pose");
        SETTING(detect_model) = file::Path("yolov8n-pose");
    }

    using namespace track::detect;
    _loaded_models.clear();

    // caching here since it can be modified above
    auto path = SETTING(detect_model).value<file::Path>();
    if(is_default_model(path)
       || (valid_model(path) && path.exists()))
    {
        if(not path.has_extension()) {
            path = path.add_extension("pt"); // pytorch model
        }
        
        _loaded_models.emplace_back(
            ModelTaskType::detect,
            SETTING(yolo8_tracking_enabled).value<bool>(),
            path.str(),
            SETTING(detect_resolution).value<uint16_t>()
        );
        
    } else
        throw U_EXCEPTION("This does not seem like a valid model to use: ", path,". When using object detection, please provide a model path using the command-line argument -m <path>.");

    if(SETTING(region_model).value<file::Path>().exists())
        _loaded_models.emplace_back(
            ModelTaskType::region,
            false, // region models dont have tracking
            SETTING(region_model).value<file::Path>().str(),
            SETTING(region_resolution).value<uint16_t>()
        );

    if(_loaded_models.empty()) {
        if(not path.empty())
            throw U_EXCEPTION("Cannot find model ", path);
        
        throw U_EXCEPTION("Please provide at least one model to use for segmentation.");
    }
    
    _loaded_models = PythonIntegration::set_models(_loaded_models, proxy.m);
    
    for(auto &config : _loaded_models) {
        if(config.task == ModelTaskType::detect) {
            SETTING(detect_format) = ObjectDetectionFormat_t(config.output_format);
        }
    }
}

void Yolo8::init() {
    bool expected = false;
    if(yolo8_initialized.compare_exchange_strong(expected, true)) {
        _network_fps = _network_samples = 0;

        std::unique_lock guard(init_mutex);
        if(init_future.valid())
            init_future.get();
        
        init_future = Python::schedule([](){
            ModuleProxy proxy{
                "bbx_saved_model",
                Yolo8::reinit
            };
        });//.get();
    }
}

void Yolo8::deinit() {
    bool expected = true;
    if(yolo8_initialized.compare_exchange_strong(expected, false)) {
        std::unique_lock guard(running_mutex);
        if(running_prediction.valid()) {
            print("Still have an active prediction running, waiting...");
            running_prediction.wait();
            print("Got it.");
        }
        
        if(not Python::python_initialized())
            throw U_EXCEPTION("Please Yolo8::deinit before calling Python::deinit().");
        
        Python::schedule([](){
            track::PythonIntegration::unload_module("bbx_saved_model");
        }).get();
    }
}

void Yolo8::receive(SegmentationData& data, track::detect::Result&& result) {
    const auto mode = Background::image_mode();

    cv::Mat r3;
    if (mode == ImageMode::R3G3B2) {
        if (data.image->dims == 3)
            convert_to_r3g3b2<3>(data.image->get(), r3);
        else if (data.image->dims == 4)
            convert_to_r3g3b2<4>(data.image->get(), r3);
        else
            throw U_EXCEPTION("Invalid number of channels (",data.image->dims,") in input image for the network.");
    }
    else if (mode == ImageMode::GRAY) {
        if(data.image->dims == 3)
            cv::cvtColor(data.image->get(), r3, cv::COLOR_BGR2GRAY);
        else if(data.image->dims == 4)
            cv::cvtColor(data.image->get(), r3, cv::COLOR_BGRA2GRAY);
        else
			throw U_EXCEPTION("Invalid number of channels (",data.image->dims,") in input image for the network.");
    } else
        throw U_EXCEPTION("Invalid image mode ", mode);

    size_t N_rows = result.boxes().num_rows();

    auto& boxes = result.boxes();
    const coord_t w = max(0, r3.cols - 1);
    const coord_t h = max(0, r3.rows - 1);
    const auto detect_classes = SETTING(detect_classes).value<std::vector<uint8_t>>();

    //! decide on whether to use masks (if available), or bounding boxes
    //! if masks are not available. for the boxes we simply copy over all
    //! of the pixels in the bounding box, for the masks we copy over only
    //! the pixels that are inside the mask.
    if (result.masks().empty()) {
        for (size_t i = 0; i < N_rows; ++i) {
            auto& row = boxes[i];
            if (not detect_classes.empty()
                && not contains(detect_classes, (uint8_t)row.clid))
            {
                continue;
            }
            
            Bounds bounds = row.box;
            //bounds = bounds.mul(scale_factor);
            bounds.restrict_to(Bounds(0, 0, w, h));
            
            std::vector<uchar> pixels;
            std::vector<HorizontalLine> lines;

            for (int y = bounds.y; y < bounds.y + bounds.height; ++y) {
                // integer overflow deals with this, lol
                //assert(uint(y) < data.image->rows);

                HorizontalLine line{
                    saturate(coord_t(y), coord_t(0), coord_t(h)),
                    saturate(coord_t(bounds.x), coord_t(0), coord_t(w)),
                    saturate(coord_t(bounds.x + bounds.width), coord_t(0), coord_t(w))
                };
                pixels.insert(pixels.end(), r3.ptr<uchar>(line.y, line.x0), r3.ptr<uchar>(line.y, line.x1 + 1));
                lines.emplace_back(std::move(line));
            }

            if (not lines.empty()) {
                pv::Blob blob(lines, 0);
                data.predictions.push_back({ .clid = size_t(row.clid), .p = float(row.conf) });
                
                blob::Pose pose;
                if(not result.keypoints().empty()) {
                    auto p = result.keypoints()[i];
                    //print("pose ",i, " = ", p);
                    
                    pose = p.toPose();
                    data.keypoints.push_back(std::move(p));
                }
                data.frame.add_object(lines, pixels, 0, blob::Prediction{.clid = uint8_t(row.clid), .p = uint8_t(float(row.conf) * 255.f), .pose = std::move(pose) });
                
            }
        }
        
        return;
    }
    
    for (size_t i = 0; i < N_rows; ++i) {
        auto& row = boxes[i];
        if (not detect_classes.empty()
            && not contains(detect_classes, (uint8_t)row.clid))
        {
            continue;
        }
        Bounds bounds = row.box;
        //bounds = bounds.mul(scale_factor);
        auto& mask = result.masks()[i];

        auto blobs = CPULabeling::run(mask.mat);
        if (not blobs.empty()) {
            size_t msize = 0, midx = 0;
            for (size_t j = 0; j < blobs.size(); ++j) {
                if (blobs.at(j).pixels->size() > msize) {
                    msize = blobs.at(j).pixels->size();
                    midx = j;
                }
            }

            auto&& pair = blobs.at(midx);
            //size_t num_pixels{ 0u };
            for (auto& line : *pair.lines) {
                line.x0 = saturate(coord_t(line.x0 + bounds.x), coord_t(0), w);
                line.x1 = saturate(coord_t(line.x1 + bounds.x), line.x0, w);
                line.y = saturate(coord_t(line.y + bounds.y), coord_t(0), h);
                //num_pixels += ptr_safe_t(line.x1) - ptr_safe_t(line.x0) + ptr_safe_t(1);
                
                if (line.x0 >= r3.cols
                    || line.x1 >= r3.cols
                    || line.y >= r3.rows)
                    throw U_EXCEPTION("Coordinates of line ", line, " are invalid for image ", r3.cols, "x", r3.rows);
            }

            pair.pred = blob::Prediction{
                .clid = static_cast<uint8_t>(row.clid),
                .p = uint8_t(float(row.conf) * 255.f)
            };
            pair.extra_flags |= pv::Blob::flag(pv::Blob::Flags::is_instance_segmentation);

            pv::Blob blob(*pair.lines, *pair.pixels, pair.extra_flags, pair.pred);
            pair.pixels = (blob.calculate_pixels(r3));

            auto points = pixel::find_outer_points(&blob, 0);
            if (not points.empty()) {
                data.outlines.emplace_back(std::move(*points.front()));
            }

            data.predictions.push_back({ .clid = size_t(row.clid), .p = float(row.conf) });
            data.frame.add_object(std::move(pair));
        }
    }
}

void Yolo8::receive(SegmentationData& data, Vec2 scale_factor, const std::span<float>& vector, 
    const std::span<float>& mask_points, const std::span<uint64_t>& mask_Ns) 
{
    const auto mode = Background::image_mode();
    const auto detect_classes = SETTING(detect_classes).value<std::vector<uint8_t>>();

    const Vec2* ptr = (const Vec2*)mask_points.data();
    const size_t N = mask_points.size() / 2u;
    assert(mask_points.size() % 2u == 0);

    // convert list of points to integer coordinates
    std::vector<cv::Point> integer;
    integer.reserve(N);
    for (auto it = ptr, end = ptr + N; it != end; ++it) {
        if(it->x * scale_factor.x > data.image->cols)
            thread_print("Warning: point ", *it, " is outside image bounds ", data.image->cols, "x", data.image->rows);
        if(it->y * scale_factor.y > data.image->rows)
            thread_print("Warning: point ", *it, " is outside image bounds ", data.image->cols, "x", data.image->rows);
        integer.emplace_back(
            roundf(saturate(it->x * scale_factor.x, 0.f, (float)data.image->cols)),//roundf(saturate(it->x, 0.f, 1.f) * data.image->cols),
            roundf(saturate(it->y * scale_factor.y, 0.f, (float)data.image->rows))//roundf(saturate(it->y, 0.f, 1.f) * data.image->rows)
        );
    }

    //const auto channel = SETTING(color_channel).value<uint8_t>() % 3;
    size_t mask_index = 0;
    cv::Mat r3;
    if  (mode == ImageMode::R3G3B2)
        convert_to_r3g3b2<4>(data.image->get(), r3);
    else if  (mode == ImageMode::GRAY)
        cv::cvtColor(data.image->get(), r3, cv::COLOR_BGR2GRAY);

    auto rows = reinterpret_cast<const track::detect::Row*>(vector.data());
    const size_t N_rows = vector.size() * sizeof(float) / sizeof(track::detect::Row);
    auto image_dims = data.image->bounds() - Size2(1, 1);

    //thread_print("Received seg-data for frame ", data.frame.index());
    for(size_t i=0; i< N_rows; ++i) {
        const auto & row = rows[i];
        auto bounds = (Bounds)row.box;
        bounds = bounds.mul(scale_factor);
        bounds.restrict_to(image_dims);
        
        if (mask_Ns.empty()) {
            std::vector<uchar> pixels;
            std::vector<HorizontalLine> lines;

            for (int y = bounds.y; y < bounds.y + bounds.height; ++y) {
                // integer overflow deals with this, lol
                assert(uint(y) < data.image->rows);

                HorizontalLine line{ (coord_t)y, (coord_t)bounds.x, coord_t(bounds.x + bounds.width) };
                pixels.insert(pixels.end(), r3.ptr<uchar>(line.y, line.x0), r3.ptr<uchar>(line.y, line.x1));
                lines.emplace_back(std::move(line));
            }

            if (not lines.empty()) {
                pv::Blob blob(lines, 0);
                data.predictions.push_back({ .clid = size_t(row.clid), .p = float(row.conf) });
                data.frame.add_object(lines, pixels, 0, blob::Prediction{.clid = uint8_t(row.clid), .p = uint8_t(float(row.conf) * 255.f) });
            }

            continue;
        }

        if (not mask_Ns.empty() && mask_Ns[i] == 0)
            continue;

        //thread_print("** ", i, " ", pos, " ", dim, " ", conf, " ", cls, " ", mask_Ns[m], " **");
        //thread_print("  getting integers from ", mask_index, " to ", mask_index + mask_Ns[m], " (", integer.size(), "/",N,")");
        assert(mask_index + mask_Ns[i] <= integer.size());
        std::vector<cv::Point> points{ integer.data() + mask_index, integer.data() + mask_index + mask_Ns[i] };
        mask_index += mask_Ns[i];

        if (not detect_classes.empty()
            && not contains(detect_classes, row.clid))
        {
            continue;
        }

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

        cv::Mat mask = cv::Mat::zeros(boundaries.height + 1, boundaries.width + 1, CV_8UC1);
        cv::fillPoly(mask, points, 255);
        assert(mask.cols > 0 && mask.rows > 0);

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
            for (auto& line : *pair.lines) {
                line.x0 = saturate(coord_t(line.x0 + boundaries.x), coord_t(0), coord_t(r3.cols - 1));
                line.x1 = saturate(coord_t(line.x1 + boundaries.x), line.x0, coord_t(r3.cols - 1));
                line.y = saturate(coord_t(line.y + boundaries.y), coord_t(0), coord_t(r3.rows - 1));
            }

            pair.pred = blob::Prediction{
                .clid = static_cast<uint8_t>(row.clid),
                .p = uint8_t(float(row.conf) * 255.f)
            };
            pair.extra_flags |= pv::Blob::flag(pv::Blob::Flags::is_instance_segmentation);

            for (auto& line : *pair.lines) {
                if (line.x0 >= r3.cols
                    || line.x1 >= r3.cols
                    || line.y >= r3.rows)
                    throw U_EXCEPTION("Coordinates of line ", line, " are invalid for image ", r3.cols, "x", r3.rows);
            }
            pv::Blob blob(*pair.lines, *pair.pixels, pair.extra_flags, pair.pred);
            pair.pixels = (blob.calculate_pixels(r3));
            //pair.pixels = std::make_unique<std::vector<uchar>>(num_pixels);
            //std::fill(pair.pixels->begin(), pair.pixels->end(), 255);

            auto points = pixel::find_outer_points(&blob, 0);
            if (not points.empty()) {
                data.outlines.emplace_back(std::move(*points.front()));
            }

            data.predictions.push_back({ .clid = size_t(row.clid), .p = float(row.conf) });
            data.frame.add_object(std::move(pair));
        }
    }
}

bool Yolo8::is_initializing() {
    std::unique_lock guard(init_mutex);
    return init_future.valid();
}

double Yolo8::fps() {
    if(_network_samples.load() == 0u)
		return 0.0;
    return _network_fps.load() / double(_network_samples.load());
}

struct Yolo8::TransferData {
    std::vector<Image::Ptr> images;
    //std::vector<Image::Ptr> oimages;
    std::vector<SegmentationData> datas;
    std::vector<Vec2> scales;
    std::vector<Vec2> offsets;
    std::vector<size_t> orig_id;
    std::vector<std::promise<SegmentationData>> promises;
    std::vector<std::function<void()>> callbacks;

    TransferData() = default;
    TransferData(TransferData&&) = delete;
    TransferData& operator=(TransferData&&) = delete;

    ~TransferData() {
        for (auto&& img : images) {
            TileImage::move_back(std::move(img));
        }
        //thread_print("** deleting ", (uint64_t)this);
    }
};

void Yolo8::StartPythonProcess(TransferData&& transfer) {
    if (not yolo8_initialized) {
        // probably shutting down at the moment
        for (size_t i = 0; i < transfer.datas.size(); ++i) {
            transfer.promises.at(i).set_exception(nullptr);

            try {
                transfer.callbacks.at(i)();
            }
            catch (...) {
                FormatExcept("Exception in callback of element ", i, " in python results.");
            }
        }
        FormatExcept("System shutting down.");
        return;
    }

    Timer timer;
    using py = track::PythonIntegration;
    //thread_print("** transfer of ", (uint64_t)& transfer);

    const size_t _N = transfer.datas.size();
    ModuleProxy bbx("bbx_saved_model", Yolo8::reinit, true);
    //bbx.set_variable("offsets", std::move(transfer.offsets));
    //bbx.set_variable("image", transfer.images);
    //bbx.set_variable("oimages", transfer.oimages);

    std::vector<uint64_t> mask_Ns;
    std::vector<float> mask_points;

    try {
        track::detect::YoloInput input{ 
            std::move(transfer.images), 
            (transfer.offsets), 
            (transfer.scales), 
            (transfer.orig_id),
            [](std::vector<Image::Ptr>&& images)
            {
                for (auto&& image : images)
                    TileImage::move_back(std::move(image));
            }
        };

        //auto results = py::predict(std::move(input), bbx.m);
        //print("C++ results = ", results);
        auto results = py::predict(std::move(input), bbx.m);
        double elapsed = timer.elapsed();
        timer.reset();
        ReceivePackage(std::move(transfer), std::move(results));
        //bbx.run("apply");
        //double cpp_elapsed = timer.elapsed();

        auto samples = _network_samples.load();
        auto fps = _network_fps.load();
        if (samples > 10u) {
            fps = fps / double(samples);
            samples = 1;
        }
        _network_fps = fps + (double(_N) / elapsed);
        _network_samples = samples + 1;
        //print("[py] network: ", elapsed);
        //print("[cpp] network: ", cpp_elapsed);
    }
    catch (const std::exception& ex) {
        FormatError("Exception: ", ex.what());
        throw SoftException(std::string(ex.what()));
    }
    catch (...) {
        FormatWarning("Continue after exception...");

        throw;
    }
}

void Yolo8::ReceivePackage(TransferData&& transfer, std::vector<track::detect::Result>&& results) {
    //size_t elements{0};
    //size_t outline_elements{0};
    //thread_print("Received a number of results: ", results.size());
    //thread_print("For elements: ", datas);
    //for(auto &t : transfer.oimages)
    //    TileImage::buffers.move_back(std::move(t));

    if (results.empty()) {
        if (not transfer.images.empty())
            tf::imshow("ma", transfer.images.front()->get());
        for (size_t i = 0; i < transfer.datas.size(); ++i) {
            try {
                transfer.promises.at(i).set_value(std::move(transfer.datas.at(i)));
            }
            catch (...) {
                FormatExcept("A promise failed for ", transfer.datas.at(i));
                transfer.promises.at(i).set_exception(std::current_exception());
            }

            try {
                transfer.callbacks.at(i)();
            }
            catch (...) {
                FormatExcept("Exception in callback of element ", i, " in python results.");
            }
        }
        FormatExcept("Empty data for ", transfer.datas, " image=", transfer.orig_id);
        return;
    }

    for (size_t i = 0; i < transfer.datas.size(); ++i) {
        auto&& result = results.at(i);
        auto& data = transfer.datas.at(i);
        //auto& scale = transfer.scales.at(i);

        try {
            receive(data, std::move(result));
            transfer.promises.at(i).set_value(std::move(data));
        }
        catch (...) {
            FormatExcept("A promise failed for ", transfer.datas.at(i));
            transfer.promises.at(i).set_exception(std::current_exception());
        }

        try {
            transfer.callbacks.at(i)();
        }
        catch (...) {
            FormatExcept("Exception in callback of element ", i, " in python results.");
        }
    }
}

void Yolo8::apply(std::vector<TileImage>&& tiles) {
    while(true) {
        if(std::unique_lock guard(init_mutex);
           init_future.valid())
        {
            if(init_future.wait_for(std::chrono::milliseconds(1)) == std::future_status::ready) {
                init_future.get();
                break;
            }
        } else
            break;
    }
    
    namespace py = Python;
    TransferData transfer;

    size_t i = 0;
    for(auto&& tiled : tiles) {
        transfer.images.insert(transfer.images.end(), std::make_move_iterator(tiled.images.begin()), std::make_move_iterator(tiled.images.end()));
        
        if(not tiled.promise)
            throw U_EXCEPTION("Promise was not set.");
        transfer.promises.emplace_back(std::move(*tiled.promise));
        tiled.promise = nullptr;
        
        //print("Image scale: ", scale, " with tile source=", tiled.source_size, " image=", data.image->dimensions()," output_size=", SETTING(output_size).value<Size2>(), " original=", tiled.original_size);
        
        for(auto p : tiled.offsets()) {
            transfer.orig_id.push_back(i);
            transfer.scales.push_back( //SETTING(output_size).value<Size2>()
                                      tiled.original_size.div(tiled.source_size) );
            tiled.data.tiles.push_back(Bounds(p.x, p.y, tiled.tile_size.width, tiled.tile_size.height).mul(transfer.scales.back()));
        }
        
        auto o = tiled.offsets();
        transfer.offsets.insert(transfer.offsets.end(), o.begin(), o.end());
        transfer.datas.emplace_back(std::move(tiled.data));
        transfer.callbacks.emplace_back(tiled.callback);
        
        ++i;
    }

    tiles.clear();
    
    try {
        {
            std::unique_lock guard(running_mutex);
            if(running_prediction.valid())
                running_prediction.get();
            running_promise = {};
            running_prediction = running_promise.get_future().share();
        }

        py::schedule([&transfer]() mutable {
            StartPythonProcess(std::move(transfer));
        }).get();
        
        running_promise.set_value();
        
    } catch(...) {
        running_promise.set_value();
        throw;
    }
}

}
