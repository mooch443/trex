#if WITH_FFMPEG

#include "tomp4.h"
#include <misc/GlobalSettings.h>
#include <misc/Timer.h>
#include <misc/frame_t.h>
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
#ifndef __linux__
#include <sys/sysctl.h>
#endif

unsigned long long getTotalSystemMemory()
{
#ifdef __APPLE__
    int mib[2];
    size_t len;
    
    mib[0] = CTL_HW;
    mib[1] = HW_MEMSIZE; /* gives a 64 bit int */
    uint64_t totalphys64;
    len = sizeof(totalphys64);
    sysctl(mib, 2, &totalphys64, &len, NULL, 0);
    return totalphys64;
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return pages * page_size;
#endif
}
#endif

#define FFMPEG_SETTING(NAME) ffmpeg::Settings::get<ffmpeg::Settings:: NAME>()

using namespace cmn;

#if defined(__APPLE__)
const char *codec_name = "h264_videotoolbox";
#else
const char *codec_name = "h264_nvenc";
#endif
const AVCodec *codec;
AVCodecContext *c= NULL;
int i, ret, x, y;
//file::FilePtr f;
AVFrame *frame;
AVPacket *pkt;
AVFrame* input_frame;
SwsContext * ctx;
uint8_t endcode[] = { 0, 0, 1, 0xb7 };

AVFormatContext* outFmtCtx = nullptr;
AVStream* outStream = nullptr;

void encode(AVCodecContext *enc_ctx, AVFrame *frame, AVPacket *pkt)//,
            //FILE *outfile)
{
    int ret;
    
    /* send the frame to the encoder */
    //if (frame)
    //    printf("Send frame %3"PRId64"\n", frame->pts);
    
    //if (frame->pts != AV_NOPTS_VALUE)
    //    frame->pts = av_rescale_q(frame->pts, c->time_base, c->time_base);
    //if (pkt->dts != AV_NOPTS_VALUE)
    //    pkt->dts = av_rescale_q(pkt->dts, c->time_base, c->time_base);

    //frame->pict_type = AV_PICTURE_TYPE_I;

    ret = avcodec_send_frame(enc_ctx, frame);
    if (ret < 0)
        throw U_EXCEPTION("Error sending a frame for encoding (",ret,")");
    
    while (ret >= 0) {
        ret = avcodec_receive_packet(enc_ctx, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            return;
        else if (ret < 0) {
            throw U_EXCEPTION("Error during encoding (receive) ",ret,"");
        }
        
        
        // Rescale timestamps from codec time base to stream time base
        pkt->pts = av_rescale_q(pkt->pts, enc_ctx->time_base, outStream->time_base);
        pkt->dts = av_rescale_q(pkt->dts, enc_ctx->time_base, outStream->time_base);

        //pkt->dts = AV_NOPTS_VALUE;
        // Write the packet
        if (av_interleaved_write_frame(outFmtCtx, pkt) < 0) {
            throw U_EXCEPTION("Error while writing packet to output file");
        }
        //printf("Write packet %3"PRId64" (size=%5d)\n", pkt->pts, pkt->size);
        //fwrite(pkt->data, 1, pkt->size, outfile);
        av_packet_unref(pkt);
    }
}

// Function to flush the encoder's buffers
void flush_encoder(AVCodecContext* enc_ctx) {
    int ret;

    // Sending NULL to the encoder will signal that we're flushing
    ret = avcodec_send_frame(enc_ctx, nullptr);
    if (ret < 0) {
        std::cerr << "Error sending frame to encoder for flushing: " << ret << std::endl;
        return;
    }

    while (ret >= 0) {
        // Create a new packet for the encoded data
        AVPacket pkt;
        av_init_packet(&pkt);
        pkt.data = nullptr;    // packet data will be allocated by the encoder
        pkt.size = 0;

        // Receive the encoded data from the encoder
        ret = avcodec_receive_packet(enc_ctx, &pkt);

        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            // EAGAIN means the encoder needs more input to produce more output
            // EOF means the encoder has flushed all data
            break;
        }
        else if (ret < 0) {
            std::cerr << "Error during encoding while flushing: " << ret << std::endl;
            break;
        }

        // Rescale timestamps from codec time base to stream time base
        pkt.pts = av_rescale_q(pkt.pts, enc_ctx->time_base, outStream->time_base);
        pkt.dts = av_rescale_q(pkt.dts, enc_ctx->time_base, outStream->time_base);

        //pkt->dts = AV_NOPTS_VALUE;
        // Write the packet
        if (av_interleaved_write_frame(outFmtCtx, &pkt) < 0) {
            throw U_EXCEPTION("Error while writing packet to output file");
        }
        //printf("Write packet %3"PRId64" (size=%5d)\n", pkt->pts, pkt->size);
        //fwrite(pkt->data, 1, pkt->size, outfile);
        av_packet_unref(&pkt);
    }
}

// Main function for remuxing
void remux(const std::string &inputFileName, const std::string &outputFileName) {
    AVFormatContext* inputFormatContext = nullptr;
    AVFormatContext* outputFormatContext = nullptr;

    // Initialize FFmpeg
    avformat_network_init();

    // Open input file
    if (avformat_open_input(&inputFormatContext, inputFileName.c_str(), nullptr, nullptr) < 0) {
        throw U_EXCEPTION("Error opening input file");
    }

    // Retrieve stream information
    if (avformat_find_stream_info(inputFormatContext, nullptr) < 0) {
        avformat_close_input(&inputFormatContext);
        throw U_EXCEPTION("Error finding stream information");
    }

    // Allocate output context
    if (avformat_alloc_output_context2(&outputFormatContext, nullptr, nullptr, outputFileName.c_str()) < 0) {
        avformat_close_input(&inputFormatContext);
        throw U_EXCEPTION("Error allocating output context");
    }

    // Copy streams
    for (unsigned int i = 0; i < inputFormatContext->nb_streams; i++) {
        AVStream* inStream = inputFormatContext->streams[i];
        AVStream* outStream = avformat_new_stream(outputFormatContext, nullptr);

        if (!outStream) {
            avformat_close_input(&inputFormatContext);
            avformat_free_context(outputFormatContext);
            throw U_EXCEPTION("Error creating a new stream in the output file");
        }

        if (avcodec_parameters_copy(outStream->codecpar, inStream->codecpar) < 0) {
            avformat_close_input(&inputFormatContext);
            avformat_free_context(outputFormatContext);
            throw U_EXCEPTION("Error copying codec parameters");
        }
        outStream->codecpar->codec_tag = 0;
    }

    // Write the output file header
    if (!(outputFormatContext->oformat->flags & AVFMT_NOFILE)) {
        if (avio_open(&outputFormatContext->pb, outputFileName.c_str(), AVIO_FLAG_WRITE) < 0) {
            avformat_close_input(&inputFormatContext);
            avformat_free_context(outputFormatContext);
            throw U_EXCEPTION("Error opening output file");
        }
    }

    if (avformat_write_header(outputFormatContext, nullptr) < 0) {
        avformat_close_input(&inputFormatContext);
        avformat_free_context(outputFormatContext);
        throw U_EXCEPTION("Error writing header to output file");
    }

    // Remuxing loop
    AVPacket pkt;
    while (av_read_frame(inputFormatContext, &pkt) >= 0) {
        AVStream* inStream = inputFormatContext->streams[pkt.stream_index];
        AVStream* outStream = outputFormatContext->streams[pkt.stream_index];

        pkt.pts = av_rescale_q_rnd(pkt.pts, inStream->time_base, outStream->time_base, (AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
        pkt.dts = av_rescale_q_rnd(pkt.dts, inStream->time_base, outStream->time_base, (AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
        pkt.duration = av_rescale_q(pkt.duration, inStream->time_base, outStream->time_base);
        pkt.pos = -1;

        if (av_interleaved_write_frame(outputFormatContext, &pkt) < 0) {
            av_packet_unref(&pkt);
            avformat_close_input(&inputFormatContext);
            if (outputFormatContext && !(outputFormatContext->oformat->flags & AVFMT_NOFILE))
                avio_closep(&outputFormatContext->pb);
            avformat_free_context(outputFormatContext);
            throw U_EXCEPTION("Error writing frame");
        }
        av_packet_unref(&pkt);
    }

    // Write the trailer to output file
    av_write_trailer(outputFormatContext);

    // Clean up
    avformat_close_input(&inputFormatContext);
    if (outputFormatContext && !(outputFormatContext->oformat->flags & AVFMT_NOFILE))
        avio_closep(&outputFormatContext->pb);
    avformat_free_context(outputFormatContext);
}

FFMPEGQueue::FFMPEGQueue(bool direct, const Size2& size, ImageMode mode, const file::Path& output, bool finite_source, Frame_t video_length, std::function<void(Image::Ptr&&)> move_back)
: _size(size), _output_path(output), _last_timestamp(0), _terminate(false), _direct(direct),write_thread(NULL), average_compressed_size(0), samples_compressed_size(0), _finite_source(finite_source), _video_length(video_length), _move_back(move_back), _mode(mode)
{
    ffmpeg::Settings::init();

    frame_ms = 1000.0 / FFMPEG_SETTING(frame_rate); // ms / frame

    if(!direct)
        write_thread = new std::thread([this](){
            cmn::set_thread_name("FFMPEGQueue::write_loop");
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

void FFMPEGQueue::add(Image::Ptr&& ptr) {
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
            FormatExcept("Timestamp ", stamp," < last timestamp ", _last_timestamp);
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


void FFMPEGQueue::prints(bool force){
    if(last_printout.elapsed() >= 30 || force) {
        double ms, compressed_size;
        {
            std::lock_guard<std::mutex> fps_guard(fps_mutex);
            ms = frame_average / frame_samples;
            
            if(samples_compressed_size > 0)
                compressed_size = average_compressed_size / samples_compressed_size;
            else
                compressed_size = 0;
        }
        
        print("[FFMPEG] so far we have written ",pts," images to ",_output_path," with ", _queue.size(), " still in queue (", ms * 1000, "ms and ", FileSize{uint64_t(compressed_size)}," / frame)");
        last_printout.reset();
    }
}

void FFMPEGQueue::loop() {
    std::unique_lock<std::mutex> guard(_mutex);
    
    if(_direct)
        open_video();
    
    
    while(true) {
        _condition.wait_for(guard, std::chrono::seconds(1));
        loop_once(guard);
        
        if(_terminate && _queue.empty())
            break;
    }
    
    prints(true);
    
    if(_direct)
        close_video();
    
    print("Closed conversion loop.");
}

void FFMPEGQueue::loop_once(std::unique_lock<std::mutex>& guard) {
    while(!_queue.empty()) {
        // per_frame * N > (1/frame_rate)
        
        // if we already terminate, we can process all the frames (let the user wait)
        if(_terminate) {
            if(!terminated_before) {
                initial_size = _queue.size();
                terminated_before = true;
            }
            
            if(_queue.size()%size_t(max(1, initial_size * 0.1)) == 0)
                print("Processing remaining queue (", _queue.size()," images)");
            
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
            
            if(_move_back)
                _move_back(std::move(image));
            
            if(_direct) {
                update_statistics(per_frame.elapsed(), _size.width * _size.height);
            }
        }
        
        prints(false);
    }
}

void FFMPEGQueue::Package::unpack(cmn::Image &image, lzo_uint& new_len) const {
    if(lzo1x_decompress(memory,out_len,image.data(),&new_len,NULL) != LZO_E_OK)
    {
        FormatExcept("Uncompression failed!");
        return;
    }
    
    
    assert(new_len == image.size());
}

void FFMPEGQueue::process_one_image(timestamp_t stamp, const std::unique_ptr<cmn::Image>& image, bool direct) {
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
            }
            
            _write_condition.notify_one();
            
        } else {
            print("Compression of ",pack->in_len," bytes failed.");
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
        if(packages.empty() && not _terminate)
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
                if(!terminated_before) {
                    initial_size = packages.size();
                    terminated_before = true;
                }
                
                if(packages.size()%size_t(initial_size * 0.1) == 0)
                    print("Processing remaining packages (", packages.size()," packages)");
            }
            
            update_statistics(frame_write_timer.elapsed(), pack->out_len + sizeof(Package));
        }
        
        if(_terminate && packages.empty())
            break;
    }
    
    close_video();
    print("Quit write_loop");
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

const AVCodec* FFMPEGQueue::check_and_select_codec(const Size2& _size) {
    struct CodecInfo {
        const char* name;
    };

    static constexpr CodecInfo codecList[] = {
#if defined(__APPLE__)
        {"h264_videotoolbox"},
#else
        {"h264_nvenc"},
#endif
        {"libx264"},
        {"libopenh264"}
    };

    for (auto& codecInfo : codecList) {
        auto _codec = avcodec_find_encoder_by_name(codecInfo.name);
        if (_codec) {
            AVCodecContext* tempContext = avcodec_alloc_context3(_codec);
            if (!tempContext) {
                FormatExcept("Could not allocate temporary video codec context for ", codecInfo.name);
                continue;
            }

            tempContext->bit_rate = 0;
            tempContext->width = _size.width;
            tempContext->height = _size.height;

            auto crf = Meta::toStr(SETTING(ffmpeg_crf).value<uint32_t>());
            av_opt_set(tempContext, "crf", crf.c_str(), AV_OPT_SEARCH_CHILDREN);

            /* frames per second */
            int frame_rate = SETTING(frame_rate).value<uint32_t>();
            if(frame_rate <= 0 || frame_rate > 256)
                frame_rate = 25;
            tempContext->time_base = AVRational{1, frame_rate};
            tempContext->framerate = AVRational{frame_rate, 1}; // For setting frame rate explicitly

            /* emit one intra frame every ten frames
             * check frame pict_type before passing frame
             * to encoder, if frame->pict_type is AV_PICTURE_TYPE_I
             * then gop_size is ignored and the output of encoder
             * will always be I frame irrespective to gop_size
             */
            tempContext->gop_size = 0;
            tempContext->max_b_frames = 1;
            tempContext->pix_fmt = AV_PIX_FMT_YUV420P;
            
            if (_codec->id == AV_CODEC_ID_H264)
                av_opt_set(tempContext->priv_data, "preset", "fast", 0);

            int ret = avcodec_open2(tempContext, _codec, NULL);
            if (ret >= 0) {
                avcodec_close(tempContext);
                avcodec_free_context(&tempContext);
                return _codec;
            }

            char errBuf[AV_ERROR_MAX_STRING_SIZE];
            av_strerror(ret, errBuf, AV_ERROR_MAX_STRING_SIZE);
            //auto str = av_err2str(ret);
            FormatExcept("Could not open codec ",codecInfo.name,": ", errBuf,".");
            avcodec_free_context(&tempContext);
        } else {
            FormatWarning("Could not find encoder for codec ", codecInfo.name);
        }
    }

    throw U_EXCEPTION("No suitable codec found.");
}

void FFMPEGQueue::open_video() {
    pts = 0;

    // try to find a suitable codec
    codec = check_and_select_codec(_size);

    c = avcodec_alloc_context3(codec);
    if (!c)
        throw U_EXCEPTION("Could not allocate video codec context");
    
    pkt = av_packet_alloc();
    if (!pkt)
        throw U_EXCEPTION("Cannot allocate pkt.");
    
    /* put sample parameters */
    c->bit_rate = 0;//2600 * 1000;
    /* resolution must be a multiple of two */
    c->width = _size.width;
    c->height = _size.height;
    
    auto crf = Meta::toStr(SETTING(ffmpeg_crf).value<uint32_t>());
    av_opt_set(c, "crf", crf.c_str(), AV_OPT_SEARCH_CHILDREN);
    
    if(c->width % 2 || c->height % 2)
        throw U_EXCEPTION("Dimensions must be a multiple of 2. (",c->width,"x",c->height,")");
    
    /* frames per second */
    int frame_rate = SETTING(frame_rate).value<uint32_t>();
    if(frame_rate == 0 || frame_rate > 256)
        frame_rate = 25;
    c->time_base = AVRational{1, frame_rate};
    c->framerate = AVRational{frame_rate, 1}; // For setting frame rate explicitly

    /* emit one intra frame every ten frames
     * check frame pict_type before passing frame
     * to encoder, if frame->pict_type is AV_PICTURE_TYPE_I
     * then gop_size is ignored and the output of encoder
     * will always be I frame irrespective to gop_size
     */
    c->gop_size = 0;
    c->max_b_frames = 1;
    c->pix_fmt = AV_PIX_FMT_YUV420P;
    
    if (codec->id == AV_CODEC_ID_H264)
        av_opt_set(c->priv_data, "preset", "fast", 0);
    
    /* open it */
    ret = avcodec_open2(c, codec, NULL);
    if (ret < 0) {
        char errBuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errBuf, AV_ERROR_MAX_STRING_SIZE);
        //auto str = av_err2str(ret);
        throw U_EXCEPTION("Could not open codec ",codec_name,": ", errBuf,".");
    }
    
    //f = _output_path.fopen("wb");
    //if (!f)
    //        throw U_EXCEPTION("Could not open ",_output_path.str(),".");
    if(avformat_alloc_output_context2(&outFmtCtx, nullptr, nullptr, _output_path.c_str()) < 0)
        throw U_EXCEPTION("Could not open ", _output_path," context.");
    
    outStream = avformat_new_stream(outFmtCtx, nullptr);
    avcodec_parameters_from_context(outStream->codecpar, c);
    
    // Ensure the stream time base is the same
    outStream->time_base = c->time_base;
    
    if (!(outFmtCtx->oformat->flags & AVFMT_NOFILE)) {
        if (avio_open(&outFmtCtx->pb, _output_path.c_str(), AVIO_FLAG_WRITE) < 0) {
            throw U_EXCEPTION("Cannot open ", _output_path," output file.");
        }
    }
    
    int error;
    if((error = avformat_write_header(outFmtCtx, nullptr)) < 0) {
        throw U_EXCEPTION("Error creating a header for output: ", error);
    }
    
    frame = av_frame_alloc();
    if (!frame)
        throw U_EXCEPTION("Could not allocate video frame");
    frame->format = c->pix_fmt;
    frame->width  = c->width;
    frame->height = c->height;
    
    ret = av_frame_get_buffer(frame, 0);
    if (ret < 0)
        throw U_EXCEPTION("Could not allocate the video frame data");
    
    input_frame = av_frame_alloc();
    input_frame->format = _mode == ImageMode::GRAY ? AV_PIX_FMT_GRAY8 : AV_PIX_FMT_RGB24;
    input_frame->width = c->width;
    input_frame->height = c->height;
    
    ret = av_frame_get_buffer(input_frame, 0);
    if (ret < 0)
        throw U_EXCEPTION("Could not allocate the video frame data");
    
    ctx = sws_getContext(c->width, c->height,
                         (AVPixelFormat)input_frame->format, c->width, c->height,
                         c->pix_fmt, 0, 0, 0, 0);
    
    print("linesizes: ", frame->linesize[0], " ", frame->linesize[1], " ", frame->linesize[2], " ", frame->linesize[3]);
    print("frame: ", c->width, "x", c->height, " (", frame->width, "x", frame->height, ")");
    
    memset(input_frame->data[0], 0, input_frame->linesize[0]);
    sws_scale(ctx, input_frame->data, input_frame->linesize, 0, frame->height, frame->data, frame->linesize);
}

void FFMPEGQueue::close_video() {
    /* flush the encoder */
    //encode(c, NULL, pkt, f.get());
    /* add sequence end code to have a real MPEG file */
    /*if (codec->id == AV_CODEC_ID_MPEG1VIDEO || codec->id == AV_CODEC_ID_MPEG2VIDEO)
        fwrite(endcode, 1, sizeof(endcode), f.get());
    f.reset();*/

    flush_encoder(c);
    
    av_write_trailer(outFmtCtx);
    if (!(outFmtCtx->oformat->flags & AVFMT_NOFILE)) {
        avio_closep(&outFmtCtx->pb);
    }
    avformat_free_context(outFmtCtx);

    
    avcodec_free_context(&c);
    av_frame_free(&frame);
    av_packet_free(&pkt);
    
    cnpy::npy_save(_output_path.replace_extension("npy").str(), timestamps);
    cnpy::npy_save(_output_path.remove_extension().str()+"_indexes.npy", mp4_indexes);

    print("Closed video.");
    
    //file::Path ffmpeg = SETTING(ffmpeg_path);
    //if(!ffmpeg.empty())
    //{
       // file::Path save_path = _output_path.replace_extension("mp4");
       // print("Remuxing ",_output_path.str()," to ",save_path.str(),"...");
        //remux(_output_path.str(), save_path.str());
        
        /*std::string cmd = ffmpeg.str()+" -fflags +genpts -i "+_output_path.str()+" -vcodec copy -y "+save_path.str();
        
        system(cmd.c_str());*/
    //} //else
        //FormatWarning("Cannot do remuxing with empty ffmpeg path.");
}

void FFMPEGQueue::finalize_one_image(timestamp_t stamp, const cmn::Image& image) {
    timestamps.push_back(stamp.get());
    mp4_indexes.push_back(image.index());

    /* make sure the frame data is writable */
    ret = av_frame_make_writable(frame);
    if (ret < 0)
        throw U_EXCEPTION("Cannot make frame writable.");

    // Prepare a temporary AVFrame for the source RGB data
    /*AVFrame* rgb_frame = av_frame_alloc();
    if (!rgb_frame)
        throw U_EXCEPTION("Could not allocate RGB frame");

    // Assign data and linesize here
    frame->data[0] = image.data();
    rgb_frame->data[0] = image.data(); // Pointer to your RGB data
    rgb_frame->linesize[0] = image.cols * image.dims;
    rgb_frame->width = c->width;
    rgb_frame->height = c->height;
    rgb_frame->format = AV_PIX_FMT_RGB24;*/
    //if(av_frame_make_writable(input_frame) < 0)
    //    throw U_EXCEPTION("Cannot write input frame.");
    
    auto ptr = const_cast<Image*>(&image);
    auto mat = ptr->get();
    cv::putText(mat, Meta::toStr(image.index()), Vec2(100,150), 1, cv::FONT_HERSHEY_PLAIN, gui::Red);
    
    input_frame->data[0] = image.data();

    // Convert the RGB frame to the codec context's pixel format (usually YUV)
    sws_scale(
        ctx,
        (const uint8_t* const*)input_frame->data, input_frame->linesize, 0, c->height,
        frame->data, frame->linesize
    );

    // Free the temporary RGB frame
    //av_frame_free(&rgb_frame);

    // Set the frame's PTS
    // Assuming your desired frame rate is 25 FPS
    //const int framerate = 25;
    //frame->pts = (pts++ * av_q2d(c->time_base)) * framerate;

    //frame->pts = pts * (c->time_base.den / c->time_base.num);
    frame->pts = pts;
    //print("frame ", pts, "->pts = ", frame->pts);
    pts++;

    // Encode the frame
    encode(c, frame, pkt);
}

void FFMPEGQueue::update_cache_strategy(double needed_ms, double compressed_size) {
    
    if(not approximate_length.valid() && _finite_source) {
           approximate_length = _video_length;
    } else if(not approximate_length.valid() && GlobalSettings::has("approximate_length_minutes")) {
        approximate_length = Frame_t(SETTING(approximate_length_minutes).value<uint32_t>() * SETTING(frame_rate).value<uint32_t>() * 60);
        auto stop_after_minutes = SETTING(stop_after_minutes).value<uint32_t>();
        if(stop_after_minutes > 0) {
            approximate_length = Frame_t(stop_after_minutes * SETTING(frame_rate).value<uint32_t>() * 60);
        }
    }
    
    if(approximate_length.valid() && approximate_length > 0_f) {
        maximum_memory = SETTING(system_memory_limit).value<uint64_t>() == 0 ? (uint64_t)(getTotalSystemMemory()*0.9) : SETTING(system_memory_limit).value<uint64_t>();
        approximate_ms = approximate_length.get() / FFMPEG_SETTING(frame_rate);
    }
        
    if(_queue.size() > 0) {
        if(approximate_length.valid() && approximate_length > 0_f) {
            static Timer last_call;
            static long_t skip_step = 0, added_since = 0;
            
            if(last_call.elapsed() > 10 && compressed_size > 0) {
                maximum_images = floor(maximum_memory / compressed_size);
                
                // use approximate_length to determine whether we're going to have a problem
                double current_frame_rate = 1000.0 / double(needed_ms);
                double remaining = approximate_length.get() - approximate_ms * current_frame_rate; // how many frames we will have written to file, how many will be left in memory if we try to write everything
                
                auto compressed = FileSize{uint64_t(compressed_size)};
                
                if(remaining > maximum_images) {
                    // we need to skip some frames
                    auto str = Meta::toStr(FileSize{maximum_memory});
                    auto needed_str = Meta::toStr(FileSize{uint64_t(remaining * _size.width * _size.height)});
                    skip_step = (remaining-maximum_images) / approximate_ms;
                    
                    FormatWarning("We need to cap memory (",needed_str.c_str()," in remaining images) to ",str.c_str(),", that means losing ",skip_step," images / second (",needed_ms,"ms / frame, ",compressed," compressed)");
                    
                } else {
                    // we can keep all frames
                    print("Cool, we dont need to skip any frames, we can keep it all in memory (", needed_ms,"ms / frame, ",compressed," compressed).");
                }
                
                last_call.reset();
            }
            
            ++added_since;
            
            if(skip_step > 0 && added_since >= FFMPEG_SETTING(frame_rate) / skip_step) {
                added_since = 0;
                
                auto image = std::move(_queue.back());
                _queue.pop_back();
                
                static Timer last_message_timer;
                if(last_message_timer.elapsed() > 10) {
                    FormatWarning("Skipping frame (queue size = ", _queue.size(),")");
                    last_message_timer.reset();
                }
            }
            
        } else if(needed_ms * _queue.size() >= frame_ms * 5) {
            // default to strategy based on needms / frame and queue size skip every 2nd frame or so
            static Timer last_message_timer;
            if(last_message_timer.elapsed() > 10) {
                FormatWarning("Skipping frame (",needed_ms * _queue.size()," >= ",frame_ms * 5," with queue size = ",_queue.size(),")");
                last_message_timer.reset();
            }
            auto image = std::move(_queue.back());
            _queue.pop_back();
        }
    }
}

#endif

