#include "AbstractVideoSource.h"

using AVS = AbstractBaseVideoSource;

AbstractBaseVideoSource::AbstractBaseVideoSource(VideoInfo info)
  : _info(info),
    mat_buffers("mat_buffers", _info.size),
    image_buffers("image_buffers", _info.size),
    _source_frame(10u, 5u,
                std::string("frame."+_info.base.str()),
                [this]()
    {
        return fetch_next();
    }),
    _resize_cvt(10u, 5u,
                std::string("resize."+_info.base.str()),
                [this]()
    {
        return this->fetch_next_process();
    })
{
    //notify();
}
AbstractBaseVideoSource::~AbstractBaseVideoSource() {
    quit();
}
void AbstractBaseVideoSource::quit() {
    _source_frame.quit();
    _resize_cvt.quit();
}
void AbstractBaseVideoSource::notify() {
    _source_frame.notify();
    _resize_cvt.notify();
}

Size2 AbstractBaseVideoSource::size() const { return _info.size; }

void AbstractBaseVideoSource::move_back(useMatPtr_t&& ptr) {
    /*
       if(not ptr
       || ptr->rows != info.size.height
       || ptr->cols != info.size.width) 
    {
        return;
    }
    */
    mat_buffers.move_back(std::move(ptr));
}

void AbstractBaseVideoSource::move_back(Image::Ptr&& ptr) {
    /*if (not ptr
        || ptr->rows != info.size.height
        || ptr->cols != info.size.width)
    {
        return;
    }*/
    image_buffers.move_back(std::move(ptr));
}

AVS::PreprocessResult_t AbstractBaseVideoSource::next() {
    auto result = _resize_cvt.next();
    if (!result)
        return std::unexpected(result.error());
    
    return std::move(result.value());
}

AVS::PreprocessResult_t AbstractBaseVideoSource::fetch_next_process() {
    try {
        /// Pipeline step:
        ///  1) Pull raw frame from `_source_frame`.
        ///  2) Validate index and run undistortion if maps exist.
        ///  3) Optionally resize according to `_video_scale` (GPU).
        ///  4) Wrap in pooled `Image` and update decode FPS counters.
        Timer timer;
        // get image from 1. step (source.frame) => here (resize+cvtColor)
        auto result = _source_frame.next();
        if(result) {
            auto& [index, buffer] = result.value();
            if (not index.valid())
                throw U_EXCEPTION("Invalid index");
            
            undistort(*buffer, *buffer);
            
            //! resize according to settings
            //! (e.g. multiple tiled image size)
            if (_video_scale != 1) {
                Size2 new_size = Size2(buffer->cols, buffer->rows) * _video_scale.load();
                //FormatWarning("Resize ", Size2(buffer.cols, buffer.rows), " -> ", new_size);
                
                if (not tmp)
                    tmp = MAKE_GPU_MAT;
                cv::resize(*buffer, *tmp, new_size);
                move_back(std::move(buffer));
                
                std::swap(buffer, tmp);
            }
            
            //! throws bad optional access if the returned frame is not valid
            assert(index.valid());
            
            auto image = image_buffers.get(source_location::current());
            image->create(*buffer, index.get());
            
            if (_video_samples.load() > 1000) {
                _video_samples = _video_fps = 0;
            }
            // Accumulate running-average components for GUI `vid_fps` display.
            _video_fps = _video_fps.load() + (1.0 / timer.elapsed());
            _video_samples = _video_samples.load() + 1;
            
            return PreprocessedFrame{
                .index = index,
                .buffer = std::move(buffer),
                .ptr = std::move(image)
            };
            
        } else
            return std::unexpected(result.error());
        //throw U_EXCEPTION("Unable to load frame: ", result.error());
        
    } catch(const std::exception& e) {
        auto desc = toStr();
        FormatExcept("Unable to load frame ", i, " from video source ", desc.c_str(), " because: ", e.what());
        return std::unexpected(std::string(e.what()));
    }
}

void AbstractBaseVideoSource::set_video_scale(float scale) {
    _video_scale = scale;
}

bool AbstractBaseVideoSource::is_finite() const {
    return _info.finite;
}

void AbstractBaseVideoSource::set_frame(Frame_t frame) {
    if(!is_finite())
        throw std::invalid_argument("Cannot skip on infinite source.");
    i = frame;
}

void AbstractBaseVideoSource::set_loop(bool loop) {
    _loop = loop;
}

Frame_t AbstractBaseVideoSource::length() const {
    if(!is_finite()) {
        FormatWarning("Cannot return length of infinite source (", i,").");
        return i;
    }
    return _info.length;
}

std::string AbstractBaseVideoSource::toStr() const {return "AbstractBaseVideoSource<>";}
std::string AbstractBaseVideoSource::class_name() { return "AbstractBaseVideoSource"; }

void AbstractBaseVideoSource::set_undistortion(
       std::optional<std::vector<double>> &&cam_matrix,
       std::optional<std::vector<double>> &&undistort_vector)
{
    if(not cam_matrix
       || not undistort_vector)
    {
        map1 = gpuMat{};
        map2 = gpuMat{};
        assert(map1.empty() && map2.empty());
        return;
    }
    
    GenericVideo::initialize_undistort(size(),
                                       std::move(cam_matrix.value()),
                                       std::move(undistort_vector.value()),
                                       map1, map2);
}

void AbstractBaseVideoSource::undistort(const gpuMat &input, gpuMat &output) {
    if(map1.empty() || map2.empty())
        return; // no undistortion
    
    if(map1.cols == input.cols
       && map1.rows == input.rows
       && map2.cols == input.cols
       && map2.rows == input.rows)
    {
        if(!map1.empty() && !map2.empty()) {
            //Print("Undistorting ", input.cols,"x",input.rows);
            cv::remap(input, output, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
        } else {
            FormatWarning("remap maps are empty.");
        }
    } else {
        FormatError("Undistortion maps are of invalid size (", map1.cols, "x", map1.rows, " vs ", input.cols, "x", input.rows, ").");
    }
}

void AbstractBaseVideoSource::undistort(const cv::Mat &input, cv::Mat &output) {
    if(map1.empty() || map2.empty())
        return; // no undistortion
    
    if(map1.cols == input.cols
       && map1.rows == input.rows
       && map2.cols == input.cols
       && map2.rows == input.rows)
    {
        if(!map1.empty() && !map2.empty()) {
            //Print("Undistorting ", input.cols,"x",input.rows);
            // upload to gpu
            input.copyTo(gpuBuffer);
            cv::remap(gpuBuffer, output, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
        } else {
            FormatWarning("remap maps are empty.");
        }
    } else {
        FormatError("Undistortion maps are of invalid size (", map1.cols, "x", map1.rows, " vs ", input.cols, "x", input.rows, ").");
    }
}
