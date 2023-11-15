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

tl::expected<std::tuple<Frame_t, AbstractBaseVideoSource::gpuMatPtr>, const char*> VideoSourceVideoSource::fetch_next() {
    if (i >= this->source.length()) {
        if(not SETTING(terminate))
            SETTING(terminate) = true;
        return tl::unexpected("EOF");
    }

    try {
        if (not i.valid() or i >= this->source.length()) {
            i = 0_f;
        }

        auto index = i++;
        gpuMatPtr buffer = buffers::get();
        if(not tmp)
            tmp = std::make_unique<useMat>();
        
        if constexpr(are_the_same<useMat, cv::Mat>) {
            source.frame(index, *buffer);
        } else {
            this->source.frame(index, cpuBuffer);
            cpuBuffer.copyTo(*buffer);
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
