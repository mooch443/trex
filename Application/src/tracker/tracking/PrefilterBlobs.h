#pragma once

#include <commons.pc.h>
#include <processing/PVBlob.h>
#include <misc/frame_t.h>
#include <core/SizeFilters.h>
#include <processing/CPULabeling.h>
#include <tracking/SplitExpectation.h>

namespace cmn {
class Background;
class GenericThreadPool;
}

namespace track {
struct BlobReceiver;
class PPFrame;

using FilterReason = pv::FilterReason;
using namespace cmn;

struct PrefilterBlobs {
private:
    GETTER(std::vector<pv::BlobPtr>, filtered);
    GETTER(std::vector<pv::BlobPtr>, filtered_out);
public:
    std::vector<pv::BlobPtr> big_blobs;
    
private:
    std::vector<FilterReason> filtered_out_reasons;
    
public:
    CPULabeling::ListCache_t cache;
    
    Frame_t frame_index;
    SizeFilters fish_size;
    const Background* background;
    int threshold;
    
    size_t overall_pixels = 0;
    size_t samples = 0;
    
    PrefilterBlobs(Frame_t index,
                   int threshold,
                   const SizeFilters& fish_size,
                   const Background& background);
    PrefilterBlobs(const PrefilterBlobs&) = delete;
    PrefilterBlobs(PrefilterBlobs&&) noexcept = default;
    PrefilterBlobs& operator=(const PrefilterBlobs&) = delete;
    PrefilterBlobs& operator=(PrefilterBlobs&&) noexcept = default;
    
    void commit(pv::BlobPtr&& b);
    void commit(std::vector<pv::BlobPtr>&& v);
    
    void filter_out(pv::BlobPtr&& b, FilterReason reason);
    void filter_out(std::vector<pv::BlobPtr>&& v, FilterReason reason);
    void filter_out(std::vector<pv::BlobPtr>&& v, std::vector<FilterReason>&& reason);
private:
    void filter_out_head(std::vector<pv::BlobPtr>&& v);
    
public:
    void to(PPFrame&) &&;
    void to(PrefilterBlobs&) &&;
    
    void big_blob(pv::BlobPtr&& b);
    void big_blob(std::vector<pv::BlobPtr>&&);
    
    static void split_big(
        Frame_t frame_index,
        std::vector<pv::BlobPtr> && big_blobs,
        const BlobReceiver& noise,
        const BlobReceiver& regular,
        const robin_hood::unordered_map<pv::bid, split_expectation> &expect,
        bool discard_small = false,
        std::ostream* out = nullptr,
        GenericThreadPool* pool = nullptr);
    
    static bool blob_matches_shapes(const pv::Blob&, const std::vector<std::vector<Vec2>>&);
    static bool rect_overlaps_shapes(const Bounds&, const std::vector<std::vector<Vec2>>&);
    static bool is_blob_ignored(Frame_t frame_index, const pv::Blob&, const std::optional<const std::set<pv::bid>*>& track_ignore_bdx_c);
};

}
