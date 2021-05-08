#ifndef _VIDEO_SOURCE_H
#define _VIDEO_SOURCE_H

#include "types.h"
#include <video/GenericVideo.h>
#include <file/Path.h>

#define VIDEO_SEQUENCE_INVALID_VALUE (-1)
#define VIDEO_SEQUENCE_UNSPECIFIED_VALUE (-2)

namespace cmn {
    class Video;
    class VideoSource;
}

class cmn::VideoSource : public cmn::GenericVideo {
public:
    class File {
    public:
        enum Type {
            UNKNOWN,
            VIDEO,
            IMAGE
        };
        
        static File *open(size_t index, const std::string& basename, const std::string& ext, bool no_check = false);
        static std::string complete_name(const std::string& basename, const std::string& ext);
        
    private:
        static std::vector<std::pair<std::string, Type>> _extensions;
        
        GETTER(size_t, index)
        GETTER(std::string, filename)
        size_t _length;
        Video *_video;
        Type _type;
        
        GETTER(std::string, format)
        
        std::vector<double> _timestamps;
        cv::Size _size;
        
    private:
        File(size_t index, const std::string& basename, const std::string& extension);
        
    public:
        ~File();
        size_t length() const { return _length; }
        const cv::Size& resolution();
        
        void frame(long_t frameIndex, cv::Mat& output, bool lazy_video = false) const;
        void close() const;
        Type type() const { return _type; }
        bool has_timestamps() const;
        uint64_t timestamp(uint64_t frameIndex) const;
        short framerate();
    };
    
private:
    /**
     * (Video) files
     */
    std::vector<File*> _files_in_seq;
    
    File* _last_file = nullptr;
    cv::Size _size;
    uint64_t _length = 0;
    cv::Mat _average;
    cv::Mat _mask;
    bool _has_timestamps = false;
    short _framerate = -1;
    
public:
    /**
     * Automatically load a range of files with a certain extension called
     * {basename}{seq_start<=number<=seq_end}.{extension}
     */
    VideoSource();
    VideoSource(const std::string& source);
    VideoSource(const std::vector<file::Path>& files);
    void open(const std::string& prefix, const std::string& suffix, const std::string& extension, int seq_start = VIDEO_SEQUENCE_INVALID_VALUE, int seq_end = VIDEO_SEQUENCE_INVALID_VALUE, int padding = 4);
    
    ~VideoSource();
    
    /**
     * ### GENERICVIDEO INTERFACE ###
     **/
#ifdef USE_GPU_MAT
    void frame(uint64_t globalIndex, gpuMat& output) override;
#endif
    void frame(uint64_t globalIndex, cv::Mat& output) override;
    const cv::Size& size() const override { return _size; }
    uint64_t length() const override { return _length; }
    const cv::Mat& average() const override { return _average; }
    cv::Mat& average() { return _average; }
    bool supports_multithreads() const override { return type() == File::Type::IMAGE; }
    
    File::Type type() const { if(_files_in_seq.empty()) return File::Type::UNKNOWN; return _files_in_seq.at(0)->type(); }
    
    virtual bool has_timestamps() const override;
    virtual uint64_t timestamp(uint64_t) const override;
    virtual uint64_t start_timestamp() const override;
    
    virtual short framerate() const override;
    
    virtual bool has_mask() const override { return false; }
    virtual const cv::Mat& mask() const override { return _mask; }
    
    virtual void generate_average(cv::Mat &average, uint64_t frameIndex, std::function<void(float)>&& callback = nullptr) override;
};

#endif
