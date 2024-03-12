#include "VideoVideoSource.h"

VideoSourceVideoSource::VideoSourceVideoSource(VideoSource&& source)
    : AbstractBaseVideoSource({.base = source.base(),
                                .size = source.size(),
                                .framerate = source.framerate(),
                                .finite = true,
                                .length = source.length()}),
        source(std::move(source))
{ }

VideoSourceVideoSource::~VideoSourceVideoSource() {
    quit();
}

tl::expected<std::tuple<Frame_t, useMatPtr_t>, const char*> VideoSourceVideoSource::fetch_next() {
    try {
        if(not i.valid())
            i = 0_f;
        if (i >= this->source.length()) {
            if(_loop.load() && i > 0_f)
                i = 0_f;
            else
                return tl::unexpected("EOF");
        }

        auto index = i++;
        auto buffer = mat_buffers.get(source_location::current());
        if(not buffer)
            throw U_EXCEPTION("Failed to get buffer");

        //if(not tmp)
        //    tmp = MAKE_GPU_MAT;
        //if(not tmp)
        //    throw U_EXCEPTION("Failed to get tmp");

        if(buffer->cols != source.size().width or buffer->rows != source.size().height)
            buffer->create(source.size().height, source.size().width, CV_8UC4);
        //if (tmp->cols != source.size().width or tmp->rows != source.size().height)
        //    tmp->create(source.size().height, source.size().width, CV_8UC4);
        
        try {
            //thread_print("Reading index = ", index);
            source.frame(index, *buffer);
        }
        catch (const std::exception& ex) {
            return tl::unexpected(ex.what());
        }

        //if (detection_type() != ObjectDetectionType::yolo8) 
        /*{
            cv::cvtColor(*buffer, *tmp, cv::COLOR_BGR2RGB);
            std::swap(buffer, tmp);
        }*/
        
        return std::make_tuple(index, std::move(buffer));
    }
    catch (const std::exception& e) {
        return tl::unexpected(e.what());
    }
}

std::string VideoSourceVideoSource::toStr() const {
    return "VideoSourceVideoSource<"+Meta::toStr(source)+">";
}

uint8_t VideoSourceVideoSource::channels() const {
	return required_channels(source.colors());
}

std::set<std::string_view> VideoSourceVideoSource::recovered_errors() const {
    return source.recovered_errors();
}

void VideoSourceVideoSource::undistort(const gpuMat &input, gpuMat &output) {
    if (not GlobalSettings::map().has("cam_undistort")
        || not SETTING(cam_undistort))
    {
        return;
    }
    
    if(map1.empty())
        GlobalSettings::get("cam_undistort1").value<cv::Mat>().copyTo(map1);
    if(map2.empty())
        GlobalSettings::get("cam_undistort2").value<cv::Mat>().copyTo(map2);
    
    if(map1.cols == input.cols
       && map1.rows == input.rows
       && map2.cols == input.cols
       && map2.rows == input.rows)
    {
        if(!map1.empty() && !map2.empty()) {
            //print("Undistorting ", input.cols,"x",input.rows);
            cv::remap(input, output, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
        } else {
            FormatWarning("remap maps are empty.");
        }
    } else {
        FormatError("Undistortion maps are of invalid size (", map1.cols, "x", map1.rows, " vs ", input.cols, "x", input.rows, ").");
    }
}

void VideoSourceVideoSource::undistort(const cv::Mat &input, cv::Mat &output) {
    if (not GlobalSettings::map().has("cam_undistort")
        || not SETTING(cam_undistort))
    {
        return;
    }
    
    if(map1.empty())
        GlobalSettings::get("cam_undistort1").value<cv::Mat>().copyTo(map1);
    if(map2.empty())
        GlobalSettings::get("cam_undistort2").value<cv::Mat>().copyTo(map2);
    
    if(map1.cols == input.cols
       && map1.rows == input.rows
       && map2.cols == input.cols
       && map2.rows == input.rows)
    {
        if(!map1.empty() && !map2.empty()) {
            //print("Undistorting ", input.cols,"x",input.rows);
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
