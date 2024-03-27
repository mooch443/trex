#include "PVVideoSource.h"

PVVideoSource::PVVideoSource(pv::File&& source)
    : AbstractBaseVideoSource({ .base = source.filename(),
                                .size = source.size(),
                                .framerate = source.framerate(),
                                .finite = true,
                                .length = source.length()}),
        source(std::move(source))
{
    this->source.header();
}

PVVideoSource::~PVVideoSource() {
    quit();
}

tl::expected<std::tuple<Frame_t, useMatPtr_t>, const char*> PVVideoSource::fetch_next() {
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
            buffer->create(source.size().height, source.size().width, CV_8UC(channels()));
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

std::string PVVideoSource::toStr() const {
    return "VideoSourceVideoSource<"+Meta::toStr(source)+">";
}

uint8_t PVVideoSource::channels() const {
    return source.header().channels;
}

std::set<std::string_view> PVVideoSource::recovered_errors() const {
    return {};
    //return source.recovered_errors();
}
