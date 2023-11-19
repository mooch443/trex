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
    if (i >= this->source.length()) {
        //if(not SETTING(terminate))
        //    SETTING(terminate) = true;
        return tl::unexpected("EOF");
    }

    try {
        if (not i.valid() or i >= this->source.length()) {
            i = 0_f;
        }

        auto index = i++;
        auto buffer = buffers.get(source_location::current());
        if(not buffer)
            throw U_EXCEPTION("Failed to get buffer");

        if(not tmp)
            tmp = MAKE_GPU_MAT;
        if(not tmp)
            throw U_EXCEPTION("Failed to get tmp");

        if(buffer->cols != source.size().width or buffer->rows != source.size().height)
            buffer->create(source.size().height, source.size().width, CV_8UC3);
        if (tmp->cols != source.size().width or tmp->rows != source.size().height)
            tmp->create(source.size().height, source.size().width, CV_8UC3);
        
        try {
            if constexpr (are_the_same<useMat_t, cv::Mat>) {
                source.frame(index, *buffer);
            }
            else {
                this->source.frame(index, cpuBuffer);
                cpuBuffer.copyTo(*buffer);
            }
        }
        catch (const std::exception& ex) {
            return tl::unexpected(ex.what());
        }

        //if (detection_type() != ObjectDetectionType::yolo8) 
        {
            cv::cvtColor(*buffer, *tmp, cv::COLOR_BGR2RGB);
            std::swap(buffer, tmp);
        }
        
        return std::make_tuple(index, std::move(buffer));
    }
    catch (const std::exception& e) {
        return tl::unexpected(e.what());
    }
}

std::string VideoSourceVideoSource::toStr() const {
    return "VideoSourceVideoSource<"+Meta::toStr(source)+">";
}
