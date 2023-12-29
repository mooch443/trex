#include "PrefilterBlobs.h"
#include <processing/Background.h>
#include <misc/ThreadPool.h>
#include <misc/TrackingSettings.h>
#include <tracking/SplitBlob.h>
#include <tracking/Tracker.h>
#include <tracking/PPFrame.h>

//#define TREX_BLOB_DEBUG

namespace track {
using namespace cmn;

PrefilterBlobs::PrefilterBlobs(Frame_t index, int threshold, const BlobSizeRange& fish_size, const Background& background)
: frame_index(index), fish_size(fish_size), background(&background), threshold(threshold)
{
    
}

void PrefilterBlobs::big_blob(pv::BlobPtr&& b) {
#ifdef TREX_BLOB_DEBUG
    print(frame_index, " Big blob ", b);
#endif
    big_blobs.emplace_back(std::move(b));
}

void PrefilterBlobs::commit(pv::BlobPtr&& b) {
#ifdef TREX_BLOB_DEBUG
    print(frame_index, " Commit ", b);
#endif
    overall_pixels += b->num_pixels();
    ++samples;
    _filtered.emplace_back(std::move(b));
}

void PrefilterBlobs::commit(std::vector<pv::BlobPtr>&& v) {
#ifdef TREX_BLOB_DEBUG
    print(frame_index, " Commit ", v);
#endif
    for(const auto &b:v) {
        assert(b != nullptr);
        overall_pixels += b->num_pixels();
    }
    samples += v.size();
    
    _filtered.insert(_filtered.end(),
                    std::make_move_iterator(v.begin()),
                    std::make_move_iterator(v.end()));
}

void PrefilterBlobs::filter_out(pv::BlobPtr&& b, FilterReason reason) {
    overall_pixels += b->num_pixels();
    ++samples;
#ifdef TREX_BLOB_DEBUG
    print(frame_index, " Filter out ", b);
#endif
    _filtered_out.emplace_back(std::move(b));
    filtered_out_reasons.emplace_back(reason);
}

void PrefilterBlobs::filter_out(std::vector<pv::BlobPtr>&& v, std::vector<FilterReason>&& reasons)
{
    filter_out_head(std::move(v));
    
    filtered_out_reasons.insert(filtered_out_reasons.end(),
                                std::make_move_iterator(reasons.begin()),
                                std::make_move_iterator(reasons.end()));
    assert(_filtered_out.size() == filtered_out_reasons.size());
}

void PrefilterBlobs::filter_out(std::vector<pv::BlobPtr>&& v,
                                FilterReason reason)
{
    filter_out_head(std::move(v));
    
    filtered_out_reasons.insert(filtered_out_reasons.end(), v.size(), reason);
    assert(_filtered_out.size() == filtered_out_reasons.size());
}

void PrefilterBlobs::filter_out_head(std::vector<pv::BlobPtr>&& v) {
#ifdef TREX_BLOB_DEBUG
    print(frame_index, " Filter out ", v);
#endif
    for(const auto &b:v) {
        assert(b != nullptr);
        overall_pixels += b->num_pixels();
    }
    samples += v.size();
    _filtered_out.insert(_filtered_out.end(), std::make_move_iterator(v.begin()), std::make_move_iterator(v.end()));
}

void PrefilterBlobs::to(PPFrame &frame) && {
    robin_hood::unordered_flat_set<pv::bid> big_ids;
    for(auto &b : big_blobs)
        big_ids.insert(b->blob_id());
    
    filter_out(std::move(big_blobs), FilterReason::OutsideRange);
    big_blobs.clear();
    
    for(size_t i=0; i<_filtered_out.size(); ++i) {
        if(filtered_out_reasons[i] != FilterReason::Unknown)
            _filtered_out[i]->set_reason(filtered_out_reasons[i]);
    }
    
    frame.add_blobs(std::move(_filtered),
                    std::move(_filtered_out),
                    std::move(big_ids),
                    overall_pixels, samples);
}

void PrefilterBlobs::to(PrefilterBlobs &other) && {
    other.commit(std::move(_filtered));
    other.filter_out(std::move(_filtered_out), std::move(filtered_out_reasons));
    other.big_blobs = std::move(big_blobs);
    other.overall_pixels += overall_pixels;
    other.samples += samples;
}

void PrefilterBlobs::split_big(
    std::vector<pv::BlobPtr> && big_blobs,
    const BlobReceiver& _noise,
    const BlobReceiver& _regular,
    const robin_hood::unordered_map<pv::bid, split_expectation> &expect,
    bool discard_small,
    std::ostream* out,
    GenericThreadPool* pool)
{
    UNUSED(out);
    
    const int threshold = FAST_SETTING(track_threshold);
    const BlobSizeRange fish_size = FAST_SETTING(blob_size_ranges);
    const float cm_sq = SQR(SLOW_SETTING(cm_per_pixel));
    const auto track_ignore = FAST_SETTING(track_ignore);
    const auto track_include = FAST_SETTING(track_include);
    
    std::mutex thread_mutex;
    
    auto check_blob = [&track_ignore, &track_include](const pv::Blob& b) {
        if (!track_ignore.empty()) {
            if (blob_matches_shapes(b, track_ignore))
                return false;
        }

        if (!track_include.empty()) {
            if (!blob_matches_shapes(b, track_include))
                return false;
        }
        
        return true;
    };
    
    auto work = [&](auto, auto start, auto end, auto)
    {
        std::vector<pv::BlobPtr> noise, regular;
        CPULabeling::ListCache_t cache;
        
        for(auto it = start; it != end; ++it) {
            auto &&b = *it;
            if(not b)
                continue;
            
            if(!fish_size.close_to_maximum_of_one(b->pixels()->size() * cm_sq, 1000))
            {
                noise.push_back(std::move(b));
                continue;
            }
            
            auto bdx = b->blob_id();

            split_expectation ex(2, false);
            if(!expect.empty() && expect.count(bdx))
                ex = expect.at(bdx);
            
            auto rec = b->recount(threshold, *Tracker::background());
            if(!fish_size.close_to_maximum_of_one(rec, 10 * ex.number)) {
                noise.push_back(std::move(b));
                continue;
            }
            
            SplitBlob s(&cache, *Tracker::background(), b.get());
            auto ret = s.split(ex.number, ex.centers);
            
            for(auto &ptr : ret) {
                if(b->blob_id() != ptr->blob_id())
                    ptr->set_split(true, b);
            }
            
            if(ex.allow_less_than && ret.empty()) {
                if((!discard_small || fish_size.close_to_minimum_of_one(rec, 0.25))) {
                    regular.push_back(std::move(b));
                } else {
                    noise.push_back(std::move(b));
                }
                
                continue;
            }
            
            std::vector<pv::BlobPtr> for_this_blob;
            std::vector<std::tuple<float, pv::bid, pv::BlobPtr>> found;
            for(auto &ptr : ret) {
                auto recount = ptr->recount(0, *Tracker::background());
                found.push_back({recount, ptr->blob_id(), std::move(ptr)});
            }
            ret.clear();
            
            std::sort(found.begin(), found.end(), std::greater<>{});
            
            size_t counter = 0;
            for(auto && [r, id, ptr] : found) {
                ptr->force_set_recount(threshold, ptr->raw_recount(-1));
                ptr->add_offset(b->bounds().pos());
                ptr->set_split(true, b);

                ptr->calculate_moments();

                if(!check_blob(*ptr)) {
                    noise.emplace_back(std::move(ptr));
                    continue;
                }
                
                if(fish_size.in_range_of_one(r, 0.35, 1)
                   && (!discard_small || counter < ex.number))
                {
                    for_this_blob.emplace_back(std::move(ptr));
                    ++counter;
                } else {
                    noise.emplace_back(std::move(ptr));
                }
            }
            
            
            PPFrame::Log("Spkit blob ", b, " -> regular=", for_this_blob, " noise=", noise);
            
            if(found.empty()) {
                noise.emplace_back(std::move(b));
            } else
                regular.insert(regular.end(),
                               std::make_move_iterator(for_this_blob.begin()),
                               std::make_move_iterator(for_this_blob.end()));
        }
        
        std::unique_lock guard(thread_mutex);
        if(!regular.empty())
            _regular(std::move(regular));
        if(!noise.empty())
            _noise(std::move(noise));
    };
    
    if(big_blobs.size() >= 2 && pool) {
        distribute_indexes(work, *pool, std::make_move_iterator(big_blobs.begin()), std::make_move_iterator(big_blobs.end()));
    } else if(not big_blobs.empty())
        work(0,
             std::make_move_iterator(big_blobs.begin()),
             std::make_move_iterator(big_blobs.end()),
             0);
    
    /*for(auto & b : big_blobs) {
        if(b == nullptr)
            print("Found");
        else
            print(b);
    }*/
    
    big_blobs.erase(std::remove(big_blobs.begin(), big_blobs.end(), nullptr), big_blobs.end());
}

bool PrefilterBlobs::blob_matches_shapes(const pv::Blob & b, const std::vector<std::vector<Vec2> > & shapes) {
    for(auto &rect : shapes) {
        if(rect.size() == 2) {
            // its a boundary
            if(Bounds(rect[0], rect[1] - rect[0]).contains(b.center()))
            {
                return true;
            }
            
        } else if(rect.size() > 2) {
            // its a polygon
            if(pnpoly(rect, b.center())) {
                return true;
            }
        }
#ifndef NDEBUG
        else {
            static bool warned = false;
            if(!warned) {
                print("Array of numbers ",rect," is not a polygon (or rectangle).");
                warned = true;
            }
        }
#endif
    }
    
    return false;
}

bool PrefilterBlobs::rect_overlaps_shapes(const Bounds & b, const std::vector<std::vector<Vec2> > & shapes) {
    for(auto &rect : shapes) {
        if(rect.size() == 2) {
            Bounds bds(rect[0], rect[1] - rect[0]);
            if(bds.overlaps(b))
                return true;
        } else if(rect.size() > 2) {
            Bounds bds(0, 0, FLT_MAX, FLT_MAX);
            for(auto &p : rect) {
                bds.insert_point(p);
            }
            bds.width -= bds.x;
            bds.height -= bds.y;
            if(bds.overlaps(b))
                return true;
        }
#ifndef NDEBUG
        else {
            static bool warned = false;
            if(!warned) {
                print("Array of numbers ",rect," is not a polygon (or rectangle).");
                warned = true;
            }
        }
#endif
    }
    
    return false;
}

}
