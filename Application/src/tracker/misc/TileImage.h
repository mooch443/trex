#pragma once

#include <commons.pc.h>
#include <misc/frame_t.h>
#include <misc/vec2.h>
#include <misc/Image.h>
#include <misc/TaskPipeline.h>

using namespace cmn;

using useMat_t = cv::Mat;

struct GPUMatPtr {
    std::unique_ptr<useMat_t> ptr;
#ifndef NDEBUG
    source_location loc;
#endif
    
    GPUMatPtr(nullptr_t) {}
    GPUMatPtr() { }
    GPUMatPtr(GPUMatPtr&& other) noexcept 
#ifdef NDEBUG
        : ptr(std::move(other.ptr)) 
#endif
    {
#ifndef NDEBUG
        *this = std::move(ptr);
#endif
        assert(ptr);
    }
    GPUMatPtr& operator=(GPUMatPtr&& ptr) noexcept {
#ifndef NDEBUG
        if(this->ptr) {
            print("Destroyed buffer ",loc.file_name(),"::",loc.function_name(),":",loc.line());
        }
        
        //print("Moving buffer ",ptr.loc.file_name(),"::",ptr.loc.function_name(),":",ptr.loc.line());
        this->loc = ptr.loc;
#endif
        this->ptr = std::move(ptr.ptr);
        return *this;
    }
    
    static GPUMatPtr Make(source_location loc) {
        //print("Created buffer ", loc.file_name(),"::",loc.function_name(),":",loc.line());
        GPUMatPtr ptr;
        ptr.ptr = std::make_unique<useMat_t>();
#ifndef NDEBUG
        ptr.loc = loc;
        
        static std::atomic<size_t> counter{0u};
        ++counter;
        if(counter % 100 == 0)
            thread_print("Counted ", counter.load());
#else
        UNUSED(loc);
#endif
        return ptr;
    }
    
    
#ifndef NDEBUG
    ~GPUMatPtr() {
        if(ptr)
            print("Destroyed buffer ",loc.file_name(),"::",loc.function_name(),":",loc.line());
    }
#endif
    
    /*operator useMat_t&() {
        return *ptr;
    }*/
    
    useMat_t* get() const noexcept {
        return ptr.get();
    }
    
    useMat_t& operator*() {
        return *ptr;
    }
    
    useMat_t* operator->() const noexcept {
        return ptr.get();
    }
    
    operator bool() const noexcept {
        return ptr != nullptr;
    }
};

#define MAKE_GPU_MAT GPUMatPtr::Make(cmn::source_location::current())

using useMatPtr_t = GPUMatPtr;

struct TileImage {
    Size2 tile_size;
    SegmentationData data;
    std::vector<Image::Ptr> images;
    inline static useMat_t resized, converted, thresholded;
    inline static cv::Mat download_buffer;
    std::vector<Vec2> _offsets;
    Size2 source_size, original_size;
    std::promise<SegmentationData> promise;
    std::function<void()> callback;

    inline static std::vector<Image::Ptr> buffers;
    static auto& buffer_mutex() {
        static auto m = new LOGGED_MUTEX("TileImage::buffer_mutex");
        return *m;
    }

    static void move_back(Image::Ptr&& ptr) {
        if (auto guard = LOGGED_LOCK( buffer_mutex() );
            ptr)
        {
            buffers.emplace_back(std::move(ptr));
        }
    }
    
    TileImage() = default;
    TileImage(TileImage&&) = default;
    TileImage(const TileImage&) = delete;
    
    TileImage& operator=(TileImage&&) = default;
    TileImage& operator=(const TileImage&) = delete;
    
    TileImage(const useMat_t& source, Image::Ptr&& original, Size2 tile_size, Size2 original_size)
        : tile_size(tile_size),
          source_size(source.cols, source.rows),
          original_size(original_size)
    {
        data.image = std::move(original);
        
        static const auto get_buffer = []() {
            if (auto guard = LOGGED_LOCK( buffer_mutex() );
                not buffers.empty())
            {
                auto buffer = std::move(buffers.back());
                buffers.pop_back();
                return buffer;
            }
            else {
                return Image::Make();
            }
        };
        
        if(tile_size.width == source.cols
           && tile_size.height == source.rows)
        {
            source_size = tile_size;
            auto buffer = get_buffer();
            buffer->create(source);
            images.emplace_back(std::move(buffer));
            //images.emplace_back(Image::Make(source));
            _offsets = {Vec2()};
        }
        else if(tile_size.width > source.cols
             || tile_size.height > source.rows)
        {
            source_size = tile_size;
            cv::resize(source, resized, tile_size);

            auto buffer = get_buffer();
            buffer->create(resized);
            images.emplace_back(std::move(buffer));
            //images.emplace_back(Image::Make(resized));
            _offsets = {Vec2()};
            
        } else {
            useMat_t tile = useMat_t::zeros(tile_size.height, tile_size.width, CV_8UC3);
            for(int y = 0; y < source.rows; y += tile_size.height) {
                for(int x = 0; x < source.cols; x += tile_size.width) {
                    Bounds bds = Bounds(x, y, tile_size.width, tile_size.height);
                    _offsets.push_back(Vec2(x, y));
                    bds.restrict_to(Bounds(0, 0, source.cols, source.rows));
                    
                    source(bds).copyTo(tile(Bounds{bds.size()}));

                    auto buffer = get_buffer();
                    buffer->create(tile);
                    images.emplace_back(std::move(buffer));
                    //images.emplace_back(Image::Make(tile));
                    tile.setTo(0);
                }
            }
        }
        
        //print("Tiling image originally ", this->original->dimensions(), " to ", tile_size, " producing: ", offsets(), " (original_size=", original_size,")");
    }
    
    operator bool() const {
        return not images.empty();
    }
    
    std::vector<Vec2> offsets() const {
        return _offsets;
    }
};
