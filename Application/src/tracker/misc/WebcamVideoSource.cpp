#include "WebcamVideoSource.h"

WebcamVideoSource::WebcamVideoSource(fg::Webcam&& source)
    : AbstractBaseVideoSource({.base = "Webcam",
                                .size = source.size(),
                                .framerate = short(source.frame_rate()),
                                .finite = false,
                                .length = Frame_t{}}),
        source(std::move(source))
{
    notify();
}

WebcamVideoSource::~WebcamVideoSource() {
    quit();
}

tl::expected<std::tuple<Frame_t, useMatPtr_t>, const char*> WebcamVideoSource::fetch_next() {
    try {
        if (not i.valid()) {
            i = 0_f;
        }

        auto index = i++;

        auto buffer = buffers.get(source_location::current());
        //auto tmp = buffers::get(source_location::current());
        
        size_t tries = 0;
        bool result;
        
        if(buffer->cols != source.size().width
           || buffer->rows != source.size().height
           || buffer->channels() != 3) 
        {
            buffer->create(source.size().height, source.size().width, CV_8UC3);
        }
        
        if(not tmp)
            tmp = buffers.get(source_location::current());
        
        if(tmp->cols != source.size().width
           || tmp->rows != source.size().height
           || tmp->channels() != 3)
        {
            tmp->create(source.size().height, source.size().width, CV_8UC3);
        }
        
        while(not (result = this->source.next(*buffer)) && tries++ < 5)
        {
            //if (detection_type() != ObjectDetectionType::yolo8) FormatError("Dropping an erroreous frame from the webcam.");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        
        if(not result) {
            move_back(std::move(buffer));
            return tl::unexpected("Cannot retrieve webcam frame after several tries.");
        }
        
        {
            cv::cvtColor(*buffer, *tmp, cv::COLOR_BGR2RGB);
            std::swap(buffer, tmp);
        }
        
        //move_back(std::move(tmp));
        return std::make_tuple(index, std::move(buffer));
    }
    catch (const std::exception& e) {
        return tl::unexpected(e.what());
    }
    
    return tl::unexpected("Cannot retrieve webcam frame.");
}

std::string WebcamVideoSource::toStr() const {
    return "WebcamVideoSource<"+Meta::toStr(source)+">";
}

uint8_t WebcamVideoSource::channels() const {
    return source.color_mode() == ImageMode::GRAY ? 1 : 3;
}