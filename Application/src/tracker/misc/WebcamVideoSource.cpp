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

tl::expected<std::tuple<Frame_t, AbstractBaseVideoSource::gpuMatPtr>, const char*> WebcamVideoSource::fetch_next() {
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
        this->source.next(*buffer);
        
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

std::string WebcamVideoSource::toStr() const {
    return "WebcamVideoSource<"+Meta::toStr(source)+">";
}
