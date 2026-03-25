#include "BaslerVideoSource.h"

#include <misc/GlobalSettings.h>

#if WITH_PYLON

BaslerVideoSource::BaslerVideoSource(fg::PylonCamera&& src)
    : AbstractBaseVideoSource({.base = "Basler",
                                .size = src.size(),
                                .framerate = short(std::max(0, READ_SETTING(cam_framerate, int))),
                                .finite = false,
                                .length = Frame_t{}}),
      source(std::move(src))
{
    notify();
}

BaslerVideoSource::~BaslerVideoSource() {
    quit();
}

AbstractBaseVideoSource::VideoFrame_t BaslerVideoSource::fetch_next() {
    try {
        if (not i.valid()) {
            i = 0_f;
        }

        const auto index = i++;
        if(not source.next(frame)) {
            return std::unexpected("Cannot retrieve Basler frame.");
        }

        auto buffer = mat_buffers.get(source_location::current());
        if(not buffer) {
            return std::unexpected("Failed to get Basler frame buffer.");
        }

        const auto size = source.size();
        if(buffer->cols != size.width
           || buffer->rows != size.height
           || buffer->channels() != 3)
        {
            buffer->create(size.height, size.width, CV_8UC3);
        }

        auto mat = frame.get();
        if(mat.empty()) {
            return std::unexpected("Basler frame was empty.");
        }

        if(mat.channels() == 1) {
            cv::cvtColor(mat, *buffer, cv::COLOR_GRAY2RGB);
        } else if(mat.channels() == 3) {
            mat.copyTo(*buffer);
        } else if(mat.channels() == 4) {
            cv::cvtColor(mat, *buffer, cv::COLOR_BGRA2RGB);
        } else {
            return std::unexpected("Unsupported Basler frame channel count.");
        }

        return VideoFrame{
            .index = index,
            .buffer = std::move(buffer)
        };
    } catch(const std::exception& e) {
        return std::unexpected(e.what());
    }
}

uint8_t BaslerVideoSource::channels() const {
    return 3;
}

std::string BaslerVideoSource::toStr() const {
    return "BaslerVideoSource";
}

#endif
