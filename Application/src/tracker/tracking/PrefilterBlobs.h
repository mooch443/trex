#pragma once

#include <commons.pc.h>
#include <misc/PVBlob.h>
#include <misc/frame_t.h>
#include <misc/BlobSizeRange.h>
#include <processing/CPULabeling.h>

namespace cmn {
class Background;
class GenericThreadPool;
}

namespace track {
struct BlobReceiver;

struct split_expectation {
    size_t number;
    bool allow_less_than;
    std::vector<Vec2> centers;
    
    split_expectation(size_t number = 0, bool allow_less_than = false)
        : number(number), allow_less_than(allow_less_than)
    { }
    
    std::string toStr() const {
        return "{"+std::to_string(number)+","+(allow_less_than ? "true" : "false")+","+Meta::toStr(centers) + "}";
    }
    static std::string class_name() {
        return "split_expectation";
    }
};

struct PrefilterBlobs {
    std::vector<pv::BlobPtr> filtered;
    std::vector<pv::BlobPtr> filtered_out;
    std::vector<pv::BlobPtr> big_blobs;
    
    CPULabeling::ListCache_t cache;
    
    Frame_t frame_index;
    BlobSizeRange fish_size;
    const Background* background;
    int threshold;
    
    size_t overall_pixels = 0;
    size_t samples = 0;
    
    PrefilterBlobs(Frame_t index,
                   int threshold,
                   const BlobSizeRange& fish_size,
                   const Background& background);
    PrefilterBlobs(const PrefilterBlobs&) = delete;
    PrefilterBlobs(PrefilterBlobs&&) noexcept = default;
    PrefilterBlobs& operator=(const PrefilterBlobs&) = delete;
    PrefilterBlobs& operator=(PrefilterBlobs&&) noexcept = default;
    
    void commit(pv::BlobPtr&& b);
    void commit(std::vector<pv::BlobPtr>&& v);
    
    void filter_out(pv::BlobPtr&& b);
    void filter_out(std::vector<pv::BlobPtr>&& v);
    
    void big_blob(pv::BlobPtr&& b);
    
    static void split_big(
        std::vector<pv::BlobPtr> && big_blobs,
        const BlobReceiver& noise,
        const BlobReceiver& regular,
        const robin_hood::unordered_map<pv::bid, split_expectation> &expect,
        bool discard_small = false,
        std::ostream* out = nullptr,
        GenericThreadPool* pool = nullptr);
    
    static bool blob_matches_shapes(const pv::Blob&, const std::vector<std::vector<Vec2>>&);
};

}
