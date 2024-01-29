#pragma once
#include <misc/AbstractVideoSource.h>
#include <misc/DetectionImageTypes.h>
#include <pv.h>

class PVVideoSource : public AbstractBaseVideoSource {
    pv::File source;
    Image cpuBuffer;
    
public:
    using SourceType = pv::File;
    
public:
    PVVideoSource(pv::File&& source);
    ~PVVideoSource();
    
    tl::expected<std::tuple<Frame_t, useMatPtr_t>, const char*> fetch_next() override;

    uint8_t channels() const override;
    std::string toStr() const override;
    static std::string class_name() { return "PVVideoSource"; }
    
    std::set<std::string_view> recovered_errors() const override;
};
