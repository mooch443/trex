#if WITH_FFMPEG

#include "tomp4.h"
#include <misc/GlobalSettings.h>
#include <misc/Timer.h>

// FFmpeg
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
#include <libavutil/opt.h>
}

#include <cnpy.h>
#include <file/DataFormat.h>
#include <grabber.h>

#if WIN32
#include <windows.h>

unsigned long long getTotalSystemMemory()
{
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return status.ullTotalPhys;
}
#else
#include <unistd.h>

unsigned long long getTotalSystemMemory()
{
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return pages * page_size;
}
#endif

using namespace cmn;

const char *codec_name = "h264_nvenc";
const AVCodec *codec;
AVCodecContext *c= NULL;
int i, ret, x, y;
FILE *f;
AVFrame *frame;
AVPacket *pkt;
AVFrame* input_frame;
SwsContext * ctx;
uint8_t endcode[] = { 0, 0, 1, 0xb7 };

void encode(AVCodecContext *enc_ctx, AVFrame *frame, AVPacket *pkt,
            FILE *outfile)
{
    int ret;
    
    /* send the frame to the encoder */
    //if (frame)
    //    printf("Send frame %3"PRId64"\n", frame->pts);
    
    //if (frame->pts != AV_NOPTS_VALUE)
    //    frame->pts = av_rescale_q(frame->pts, c->time_base, c->time_base);
    //if (pkt->dts != AV_NOPTS_VALUE)
    //    pkt->dts = av_rescale_q(pkt->dts, c->time_base, c->time_base);
    
    ret = avcodec_send_frame(enc_ctx, frame);
    if (ret < 0)
        U_EXCEPTION("Error sending a frame for encoding (%d)", ret);
    
    while (ret >= 0) {
        ret = avcodec_receive_packet(enc_ctx, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            return;
        else if (ret < 0) {
            U_EXCEPTION("Error during encoding (receive) %d", ret);
            exit(1);
        }
        
        //pkt->dts = AV_NOPTS_VALUE;
        
        //printf("Write packet %3"PRId64" (size=%5d)\n", pkt->pts, pkt->size);
        fwrite(pkt->data, 1, pkt->size, outfile);
        av_packet_unref(pkt);
    }
}

FFMPEGQueue::FFMPEGQueue(bool direct, const Size2& size, const file::Path& output) : _size(size), _output_path(output), _last_timestamp(0), _terminate(false), _direct(direct),write_thread(NULL), average_compressed_size(0), samples_compressed_size(0)
{
    if(!direct)
        write_thread = new std::thread([this](){
            this->write_loop();
        });
}

FFMPEGQueue::~FFMPEGQueue() {
    if(write_thread) {
        write_thread->join();
        delete write_thread;
    }
}

#define OUT_LEN(L)     (L + L / 16 + 64 + 3)


/* Work-memory needed for compression. Allocate memory in units
 * of 'lzo_align_t' (instead of 'char') to make sure it is properly aligned.
 */

#define HEAP_ALLOC(var,size) \
lzo_align_t __LZO_MMODEL var [ ((size) + (sizeof(lzo_align_t) - 1)) / sizeof(lzo_align_t) ]
HEAP_ALLOC(wrkmem, LZO1X_1_MEM_COMPRESS);

void FFMPEGQueue::add(std::unique_ptr<Image>&& ptr) {
    /*Image *ptr = new Image(_ptr->rows, _ptr->cols, 1);
    _ptr->get().copyTo(ptr->get());
    ptr->set_index(_ptr->index());
    ptr->set_timestamp(_ptr->timestamp());
    delete _ptr;*/
    /*auto ptr = _ptr->mat();
    _ptr->mat() = nullptr;
    delete _ptr;*/
    
    {
        std::lock_guard<std::mutex> guard(_mutex);
        
        auto stamp = ptr->timestamp();
        _queue.insert(_queue.begin(), std::move(ptr));
        ptr = nullptr;
        
        if(stamp < _last_timestamp)
            Except("Timestamp %lu < last timestamp %lu", stamp, _last_timestamp);
        _last_timestamp = stamp;
    }
    
    _condition.notify_one();
}

void FFMPEGQueue::notify() {
    _condition.notify_all();
}

#define RNDTO2(X) ( ( (X) & 0xFFFFFFFE )
#define RNDTO32(X) ( ( (X) % 32 ) ? ( ( (X) + 32 ) & 0xFFFFFFE0 ) : (X) )

std::mutex fps_mutex;
double frame_average = 0, frame_samples = 0;

void FFMPEGQueue::loop() {
    std::unique_lock<std::mutex> guard(_mutex);
    Timer per_frame;
    
    if(_direct)
        open_video();
    
    Timer last_printout;
    auto prints = [&last_printout, this](bool force){
        if(last_printout.elapsed() >= 30 || force) {
            double samples, ms, compressed_size;
            {
                std::lock_guard<std::mutex> fps_guard(fps_mutex);
                samples = frame_samples;
                ms = frame_average / frame_samples;
                
                if(samples_compressed_size > 0)
                    compressed_size = average_compressed_size / samples_compressed_size;
                else
                    compressed_size = 0;
            }
            
            auto compressed_str = Meta::toStr(FileSize{uint64_t(compressed_size)});
            
            Debug("[FFMPEG] so far we have written %d images to '%S' with %d still in queue (%fms and %S / frame)", pts, &_output_path.str(), _queue.size(), ms * 1000, &compressed_str);
            
            last_printout.reset();
        }
    };
    
    while(true) {
        _condition.wait_for(guard, std::chrono::seconds(1));
        
        while(!_queue.empty()) {
            // per_frame * N > (1/frame_rate)
            
            // if we already terminate, we can process all the frames (let the user wait)
            if(_terminate) {
                static bool terminated_before = false;
                static size_t initial_size = 0;
                if(!terminated_before) {
                    initial_size = _queue.size();
                    terminated_before = true;
                }
                
                if(_queue.size()%size_t(max(1, initial_size * 0.1)) == 0)
                    Debug("Processing remaining queue (%d images)", _queue.size());
                
            } else {
                double samples, ms, compressed_size;
                {
                    std::lock_guard<std::mutex> fps_guard(fps_mutex);
                    samples = frame_samples;
                    ms = frame_average / frame_samples;
                    
                    if(samples_compressed_size > 0)
                        compressed_size = average_compressed_size / samples_compressed_size;
                    else
                        compressed_size = 0;
                }
                
                if(samples > 100)
                    update_cache_strategy(ms * 1000, compressed_size);
            }
            
            if(!_queue.empty()) {
                if(_direct) {
                    per_frame.reset();
                }
                
                auto image = std::move(_queue.back());
                _queue.pop_back();
                
                guard.unlock();
                process_one_image(image->stamp(), image, _direct);
                guard.lock();
                
                if(_direct) {
                    update_statistics(per_frame.elapsed(), _size.width * _size.height);
                }
            }
            
            prints(false);
        }
        
        if(_terminate)
            break;
    }
    
    prints(true);
    
    if(_direct)
        close_video();
    
    Debug("Closed conversion loop.");
}

void FFMPEGQueue::Package::unpack(cmn::Image &image, lzo_uint& new_len) const {
    if(lzo1x_decompress(memory,out_len,image.data(),&new_len,NULL) != LZO_E_OK)
    {
        Except("Uncompression failed!");
        return;
    }
    
    //Debug("Uncompressing %lu bytes to %lu (real:%lu)", package->out_len, package->in_len, new_len);
    
    assert(new_len == image.size());
}

void FFMPEGQueue::process_one_image(uint64_t stamp, const std::unique_ptr<cmn::Image>& image, bool direct) {
    if(direct) {
        finalize_one_image(stamp, *image);
        
        /*if(!_terminate) {
            std::lock_guard<std::mutex> guard(_vacant_mutex);
            _vacant_images.push_back(image);
        } else
            delete image;*/
        return;
    }
    
    assert(image->size() < UINT32_MAX);
    
    auto pack = std::make_shared<Package>();
    
    pack->in_len = (uint32_t)image->size();
    size_t reserved_size = OUT_LEN(pack->in_len);
    
    pack->memory = (uchar*)malloc(reserved_size);
    pack->timestamp = stamp;
    
    //static std::mutex mutex;
    {
        //std::unique_lock<std::mutex> guard(mutex);
        //static Timing timing("compress", 0.01);
        //timing.start_measure();
        // lock for wrkmem
        if(lzo1x_1_compress((uchar*)image->data(), pack->in_len, pack->memory, &pack->out_len, wrkmem) == LZO_E_OK)
        {
            {
                std::lock_guard<std::mutex> write_guard(_write_mutex);
                packages.push_back(pack);
                //Debug("Compressed %lu bytes to %lu", pack->in_len, pack->out_len);
            }
            
            _write_condition.notify_one();
            
        } else {
            Error("Compression of %d bytes failed.", pack->in_len);
        }
    }
    
    /*if(!_terminate) {
        std::lock_guard<std::mutex> guard(_vacant_mutex);
        _vacant_images.push_back(image);
    } else
        delete image;*/
}

/*void FFMPEGQueue::refill_queue(std::queue<std::unique_ptr<cmn::Image>> &queue) {
    std::lock_guard<std::mutex> guard(_vacant_mutex);
    while(!_vacant_images.empty()) {
        queue.push(_vacant_images.front());
        _vacant_images.pop_front();
    }
}*/

void FFMPEGQueue::write_loop() {
    std::unique_lock<std::mutex> guard(_write_mutex);
    open_video();
    
    Timer frame_write_timer;
    
    Image image(_size.height, _size.width);
    lzo_uint new_len;
    
    while(true) {
        _write_condition.wait_for(guard, std::chrono::seconds(1));
        
        while(!packages.empty()) {
            frame_write_timer.reset();
            
            auto pack = packages.front();
            packages.pop_front();
            
            guard.unlock();
            pack->unpack(image, new_len);
            finalize_one_image(pack->timestamp, image);
            guard.lock();
            
            if(_terminate) {
                static bool terminated_before = false;
                static size_t initial_size = 0;
                if(!terminated_before) {
                    initial_size = packages.size();
                    terminated_before = true;
                }
                
                if(packages.size()%size_t(initial_size * 0.1) == 0)
                    Debug("Processing remaining packages (%d packages)", packages.size());
            }
            
            update_statistics(frame_write_timer.elapsed(), pack->out_len + sizeof(Package));
        }
        
        if(_terminate && packages.empty())
            break;
    }
    
    close_video();
    Debug("Quit write_loop");
}

void FFMPEGQueue::update_statistics(double ms, double image_size) {
    std::lock_guard<std::mutex> fps_guard(fps_mutex);
    average_compressed_size += image_size;
    samples_compressed_size += 1;
    
    // prevent datatype overflow
    if(frame_samples < 100000) {
        frame_average += ms;
        ++frame_samples;
    } else {
        frame_average = frame_average / frame_samples;
        frame_samples = 1;
    }
}

void FFMPEGQueue::open_video() {
    pts = 0;
    /* find the mpeg1video encoder */
    codec = avcodec_find_encoder_by_name(codec_name);
    if (!codec) {
        Warning("Cannot record with '%s'. Searching for 'h264'.", codec_name);
        codec = avcodec_find_encoder_by_name("h264_videotoolbox");
    }
    
    if(!codec)
        U_EXCEPTION("Codec '%s' not found, and 'h264_videotoolbox' could not be found either.", codec_name);
    
    c = avcodec_alloc_context3(codec);
    if (!c)
        U_EXCEPTION("Could not allocate video codec context");
    
    pkt = av_packet_alloc();
    if (!pkt)
        U_EXCEPTION("Cannot allocate pkt.");
    
    /* put sample parameters */
    c->bit_rate = 0;//2600 * 1000;
    /* resolution must be a multiple of two */
    c->width = _size.width;
    c->height = _size.height;
    
    auto crf = Meta::toStr(SETTING(ffmpeg_crf).value<uint32_t>());
    av_opt_set(c, "crf", crf.c_str(), AV_OPT_SEARCH_CHILDREN);
    
    if(c->width % 2 || c->height % 2)
        U_EXCEPTION("Dimensions must be a multiple of 2. (%dx%d)", c->width, c->height);
    
    /* frames per second */
    //int frame_rate = SETTING(frame_rate).value<int>();
    c->time_base = AVRational{(int)1, (int)25};
    c->framerate = AVRational{(int)25, (int)1};
    
    /* emit one intra frame every ten frames
     * check frame pict_type before passing frame
     * to encoder, if frame->pict_type is AV_PICTURE_TYPE_I
     * then gop_size is ignored and the output of encoder
     * will always be I frame irrespective to gop_size
     */
    c->gop_size = 10;
    c->max_b_frames = 1;
    c->pix_fmt = AV_PIX_FMT_YUV420P;
    
    if (codec->id == AV_CODEC_ID_H264)
        av_opt_set(c->priv_data, "preset", "fast", 0);
    
    /* open it */
    ret = avcodec_open2(c, codec, NULL);
    if (ret < 0) {
        //auto str = av_err2str(ret);
        U_EXCEPTION("Could not open codec '%s'.", codec_name);
    }
    
    f = _output_path.fopen("wb");
    if (!f)
        U_EXCEPTION("Could not open '%S'.", &_output_path.str());
    
    frame = av_frame_alloc();
    if (!frame)
        U_EXCEPTION("Could not allocate video frame");
    frame->format = c->pix_fmt;
    frame->width  = c->width;
    frame->height = c->height;
    
    ret = av_frame_get_buffer(frame, 0);
    if (ret < 0)
        U_EXCEPTION("Could not allocate the video frame data");
    
    input_frame = av_frame_alloc();
    input_frame->format = AV_PIX_FMT_GRAY8;
    input_frame->width = c->width;
    input_frame->height = c->height;
    
    ret = av_frame_get_buffer(input_frame, 0);
    if (ret < 0)
        U_EXCEPTION("Could not allocate the video frame data");
    
    ctx = sws_getContext(c->width, c->height,
                         AV_PIX_FMT_GRAY8, c->width, c->height,
                         AV_PIX_FMT_YUV420P, 0, 0, 0, 0);
    
    Debug("linesizes: %d, %d, %d, %d", frame->linesize[0], frame->linesize[1], frame->linesize[2], frame->linesize[3]);
    Debug("frame: %dx%d (%dx%d)", c->width, c->height, frame->width, frame->height);
    
    /*for (int i=1; i<3; ++i) {
        if(frame->data[i] == NULL)
            continue;
        
        auto ptr = frame->data[i];
        auto end = ptr + frame->linesize[i] * frame->height;
        
        Debug("Setting 128 from %X to %X (%u, %u)", ptr, end, c->width, frame->linesize[i]);
        
        for (; ptr != end; ptr+=frame->linesize[i]) {
            memset(ptr, 128, frame->linesize[i]);
        }
    }*/
    
    memset(input_frame->data[0], 0, input_frame->linesize[0]);
    sws_scale(ctx, input_frame->data, input_frame->linesize, 0, frame->height, frame->data, frame->linesize);
}

void FFMPEGQueue::close_video() {
    /* flush the encoder */
    encode(c, NULL, pkt, f);
    
    /* add sequence end code to have a real MPEG file */
    if (codec->id == AV_CODEC_ID_MPEG1VIDEO || codec->id == AV_CODEC_ID_MPEG2VIDEO)
        fwrite(endcode, 1, sizeof(endcode), f);
    
    fclose(f);
    
    avcodec_free_context(&c);
    av_frame_free(&frame);
    av_packet_free(&pkt);
    
    Debug("Closed video.");
    
    cnpy::npy_save(_output_path.replace_extension("npy").str(), timestamps);
    cnpy::npy_save(_output_path.remove_extension().str()+"_indexes.npy", mp4_indexes);
    
    file::Path ffmpeg = SETTING(ffmpeg_path);
    if(!ffmpeg.empty()) {
        file::Path save_path = _output_path.replace_extension("mp4");
        std::string cmd = ffmpeg.str()+" -fflags +genpts -i "+_output_path.str()+" -vcodec copy -y "+save_path.str();
        Debug("Remuxing '%S' to '%S'...", &_output_path.str(), &save_path.str());
        system(cmd.c_str());
    } else
        Warning("Cannot do remuxing with empty ffmpeg path.");
}

void FFMPEGQueue::finalize_one_image(uint64_t stamp, const cmn::Image& image) {
    timestamps.push_back(stamp);
    mp4_indexes.push_back(image.index());
    
    /* make sure the frame data is writable */
    ret = av_frame_make_writable(frame);
    if (ret < 0)
        U_EXCEPTION("Cannot make frame writable.");
    
    //Timer timer;
    
    //memcpy(frame->data, image->data(), image->size());
    //assert(image->size() == (size_t)c->height * c->width);
    
    //static cv::Mat mat;
    //cv::cvtColor(image.get(), mat, cv::COLOR_GRAY2BGR);
    //cv::cvtColor(mat, mat, cv::COLOR_BGR2YUV);
    
    //for(int i=0; i<3; ++i) {
        auto ptr = frame->data[0];
        auto end = ptr + frame->linesize[0] * c->height;
        auto srcptr = image.data();
        
        for (; ptr != end; ptr+=frame->linesize[0], srcptr+=image.cols) {
            memcpy(ptr, srcptr, image.cols);
            for(int i=0; i<frame->linesize[0]; ++i)
                *(ptr+i) = saturate(*(ptr+i)+16, 0, 255);
        }
    //}
    
    
    //sws_scale(ctx, input_frame->data, input_frame->linesize, 0, c->height, frame->data, frame->linesize);
    
    //Debug("sws_scale#1: %f", timer.elapsed());
    
    /*timer.reset();
    
    uint8_t * inData[1] = { image.data() };
    int inLinesize[1] = { 1*c->width }; // GRAY stride
    sws_scale(ctx, inData, inLinesize, 0, c->height, frame->data, frame->linesize);
    
    Debug("sws_scale#2 %f", timer.elapsed());*/
    
    frame->pts = pts++;
    //pkt->dts = AV_NOPTS_VALUE;
    //pkt->pts = AV_NOPTS_VALUE;
    
    // have to do muxing https://ffmpeg.org/doxygen/trunk/doc_2examples_2muxing_8c-example.html
    encode(c, frame, pkt, f);
}

void FFMPEGQueue::update_cache_strategy(double needed_ms, double compressed_size) {
    static const double frame_rate = SETTING(frame_rate).value<int>();
    static const double frame_ms = 1000.0 / frame_rate; // ms / frame
    static long_t approximate_length = -1; // approximate length in frames
    static double approximate_ms = 0;
    static uint64_t maximum_memory = 0;  // maximum usage of system memory in bytes
    static double maximum_images = 0;
    
    if(approximate_length == -1 && FrameGrabber::instance->video()) {
           approximate_length = FrameGrabber::instance->video()->length();
    } else if(approximate_length == -1 && GlobalSettings::has("approximate_length_minutes")) {
        approximate_length = SETTING(approximate_length_minutes).value<uint32_t>() * SETTING(frame_rate).value<int>() * 60;
        auto stop_after_minutes = SETTING(stop_after_minutes).value<uint32_t>();
        if(stop_after_minutes > 0) {
            approximate_length = stop_after_minutes * SETTING(frame_rate).value<int>() * 60;
        }
    }
    
    if(approximate_length > 0) {
        maximum_memory = SETTING(system_memory_limit).value<uint64_t>() == 0 ? (uint64_t)(getTotalSystemMemory()*0.9) : SETTING(system_memory_limit).value<uint64_t>();
        approximate_ms = approximate_length / frame_rate;
    }
        
    if(_queue.size() > 0) {
        if(approximate_length > 0) {
            static Timer last_call;
            static long_t skip_step = 0, added_since = 0;
            
            if(last_call.elapsed() > 10 && compressed_size > 0) {
                maximum_images = floor(maximum_memory / compressed_size);
                
                // use approximate_length to determine whether we're going to have a problem
                double current_frame_rate = 1000.0 / double(needed_ms);
                double remaining = approximate_length - approximate_ms * current_frame_rate; // how many frames we will have written to file, how many will be left in memory if we try to write everything
                
                auto compressed_str = Meta::toStr(FileSize{uint64_t(compressed_size)});
                
                if(remaining > maximum_images) {
                    // we need to skip some frames
                    auto str = Meta::toStr(FileSize{maximum_memory});
                    auto needed_str = Meta::toStr(FileSize{uint64_t(remaining * _size.width * _size.height)});
                    skip_step = (remaining-maximum_images) / approximate_ms;
                    
                    Warning("We need to cap memory (%S in remaining images) to %S, that means losing %d images / second (%fms / frame, %S compressed)", &needed_str, &str, skip_step, needed_ms, &compressed_str);
                    
                } else {
                    // we can keep all frames
                    Debug("Cool, we dont need to skip any frames, we can keep it all in memory (%fms / frame, %S compressed).", needed_ms, &compressed_str);
                }
                
                last_call.reset();
            }
            
            ++added_since;
            
            if(skip_step > 0 && added_since >= frame_rate / skip_step) {
                added_since = 0;
                
                auto image = std::move(_queue.back());
                _queue.pop_back();
                
                static Timer last_message_timer;
                if(last_message_timer.elapsed() > 10) {
                    Warning("Skipping frame (queue size = %d)", _queue.size());
                    last_message_timer.reset();
                }
            }
            
        } else if(needed_ms * _queue.size() >= frame_ms * 5) {
            // default to strategy based on needms / frame and queue size skip every 2nd frame or so
            static Timer last_message_timer;
            if(last_message_timer.elapsed() > 10) {
                Warning("Skipping frame (%f >= %f with queue size = %d)", needed_ms * _queue.size(), frame_ms * 5, _queue.size());
                last_message_timer.reset();
            }
            auto image = std::move(_queue.back());
            _queue.pop_back();
        }
    }
}

#endif

