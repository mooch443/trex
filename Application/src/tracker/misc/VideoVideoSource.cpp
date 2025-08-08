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

std::expected<std::tuple<Frame_t, useMatPtr_t>, UnexpectedError_t> VideoSourceVideoSource::fetch_next() {
    try {
        if(not i.valid())
            i = 0_f;
        if (i >= this->source.length()) {
            if(_loop.load() && i > 0_f)
                i = 0_f;
            else
                return std::unexpected("EOF");
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
            return std::unexpected(ex.what());
        }

        //if (detection_type() != ObjectDetectionType::yolo) 
        /*{
            cv::cvtColor(*buffer, *tmp, cv::COLOR_BGR2RGB);
            std::swap(buffer, tmp);
        }*/

        if (not buffer || buffer->empty())
            throw std::unexpected("Failed to read frame");
        
        return std::make_tuple(index, std::move(buffer));
    }
    catch (const std::exception& e) {
        return std::unexpected(e.what());
    }
}

std::string VideoSourceVideoSource::toStr() const {
    return "VideoSourceVideoSource<"+Meta::toStr(source)+">";
}

uint8_t VideoSourceVideoSource::channels() const {
	return required_storage_channels(source.colors());
}

std::set<std::string_view> VideoSourceVideoSource::recovered_errors() const {
    return source.recovered_errors();
}
