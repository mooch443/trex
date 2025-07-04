#pragma once
#if WITH_FFMPEG

#include <commons.pc.h>
#include <misc/Image.h>
#include <file/Path.h>
#include <lzo/minilzo.h>
#include "gpuImage.h"
#include <misc/Timer.h>
#include <misc/create_struct.h>
#include <misc/frame_t.h>
#include <video/Video.h>

struct AVCodec;

namespace ffmpeg {

CREATE_STRUCT(Settings,
              (uint32_t, frame_rate));

}

class FFMPEGQueue {
    std::mutex _mutex, _write_mutex;
    std::vector<std::unique_ptr<cmn::Image>> _queue;
    std::condition_variable _condition, _write_condition;
    cmn::Size2 _size;
    cv::Mat _image;
    cmn::file::Path _output_path;
    long pts;
    
    struct Package {
        uchar *memory;
        lzo_uint in_len, out_len;
        cmn::timestamp_t timestamp;
        
        Package() {}
        
        ~Package() {
            free(memory);
        }
        
        void unpack(cmn::Image& image, lzo_uint& new_len) const;
    };
    
    cmn::timestamp_t _last_timestamp;
    GETTER_NCONST(std::atomic_bool, terminate);
    std::vector<uint64_t> timestamps;
    std::vector<long> mp4_indexes;
    std::deque<std::shared_ptr<Package>> packages;
    
    //std::mutex _vacant_mutex;
    //std::deque<cmn::Image*> _vacant_images;
    bool _direct;
    std::thread *write_thread;
    Timer last_printout;
    Timer per_frame;
    
    double average_compressed_size, samples_compressed_size;
    
    bool _finite_source{false};
    cmn::Frame_t _video_length;
    std::function<void(cmn::Image::Ptr&&)> _move_back;
    cmn::ImageMode _mode;
    

    double frame_ms = 1;
    cmn::Frame_t approximate_length; // approximate length in frames
    double approximate_ms = 0;
    uint64_t maximum_memory = 0;  // maximum usage of system memory in bytes
    double maximum_images = 0;

    bool terminated_before = false;
    size_t initial_size = 0;

public:
    FFMPEGQueue(bool direct, const cmn::Size2& size, cmn::ImageMode mode, const cmn::file::Path& output, bool finite_source, cmn::Frame_t video_length, std::function<void(cmn::Image::Ptr&&)> move_back);
    ~FFMPEGQueue();
    
    void loop();
    void loop_once(std::unique_lock<std::mutex>&);
    void write_loop();
    void notify();
    void add(std::unique_ptr<cmn::Image>&& ptr);
    
    //void refill_queue(std::queue<std::unique_ptr<cmn::Image_t>>& queue);
    
private:
    void prints(bool force);
    void process_one_image(cmn::timestamp_t stamp, const std::unique_ptr<cmn::Image>& ptr, bool direct);
    void finalize_one_image(cmn::timestamp_t stamp, const cmn::Image& image);
    void update_cache_strategy(double frame_ms, double compressed_size);
    
    void open_video();
    void close_video();
    void update_statistics(double ms, double image_size);
    static const AVCodec* check_and_select_codec(const cmn::Size2& _codec);
};

#endif
