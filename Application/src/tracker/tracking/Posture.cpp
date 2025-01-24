#include "Posture.h"
#include <misc/Timer.h>
#include <misc/GlobalSettings.h>
#include <misc/colors.h>

#include <tracking/DebugDrawing.h>
#include <thread>
#include "Tracker.h"
#include <misc/PixelTree.h>
#include <misc/CircularGraph.h>
#include <gui/DrawSFBase.h>

#include <processing/CPULabeling.h>
#include <misc/PVBlob.h>
#include <processing/DLList.h>
#include <misc/ObjectCache.h>
#include <gui/GuiTypes.h>

namespace track {
    static const std::vector<Vec2> neighbors = {
        Vec2(-1,-1),
        Vec2(0,-1),
        Vec2(1, -1),
        Vec2(1, 0),
        Vec2(1, 1),
        Vec2(0, 1),
        Vec2(-1, 1),
        Vec2(-1,0)
    };
    
static ObjectCache<CPULabeling::DLList, 50, std::unique_ptr, ThreadSafePolicy> _dllists;

//#define DEBUG_OUTLINES
using namespace blob;

float distanceBetweenPoints(const Vec2& p1, const Vec2& p2) {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

bool circlesIntersect(const Vec2& center1, float radius1, const Vec2& center2, float radius2) {
    return distanceBetweenPoints(center1, center2) < max(0, (radius1 + radius2) - 2);
}

// Cross product to determine the orientation
float cross(const Vec2& O, const Vec2& A, const Vec2& B) {
    return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
}

void ensureCircleOverlap(std::vector<Vec2>& centers, std::vector<float>& radii) {
#ifdef DEBUG_OUTLINES
    Bounds start(FLT_MAX, FLT_MAX, 0, 0);
    for(auto pt : centers)
        start.combine(Bounds(pt - Vec2(10), Size2(pt) + Size2(10)));
    start = start - Size2(start.pos());
#endif
    
#ifdef DEBUG_OUTLINES
    cv::Mat image = cv::Mat(500, 500, CV_8UC3, cv::Scalar(255, 255, 255));
    for(size_t k = 0; k < centers.size(); ++k) {
        cv::circle(image, centers[k] - start.pos(), radii[k], gui::Blue, 1);
    }
    cv::imshow("Circle Merging", image);
    cv::waitKey(0);
#endif
    
    if(centers.empty())
        return;
    
    //Print("initial = ", centers);
    
    std::optional<size_t> anyMerged;
    do {
        anyMerged = std::nullopt;
        
        for (size_t i = 0; i < centers.size() - 1; ++i) {
            const size_t next = i + 1;
            if (!circlesIntersect(centers[i], radii[i], centers[next], radii[next])) {
                Vec2 direction = centers[next] - centers[i];
                //float distance = distanceBetweenPoints(centers[i], centers[j]);
                Vec2 newPoint = centers[i] + direction * 0.5f;
                float newRadius = (radii[i] + radii[next]) / 2.f + 1.f;
                centers.insert(centers.begin() + next, newPoint);
                radii.insert(radii.begin() + next, newRadius);
                
                //Print("inserting ", newPoint, " at ", next);
                
                anyMerged = next;
                
        #ifdef DEBUG_OUTLINES
                cv::Mat image = cv::Mat(500, 500, CV_8UC3, cv::Scalar(255, 255, 255));
                for(size_t k = 0; k < centers.size(); ++k) {
                    cv::circle(image, centers[k] - start.pos(), radii[k], gui::Blue, 1);
                }
                if(anyMerged != -1)
                    cv::circle(image, centers[anyMerged] - start.pos(), radii[anyMerged], gui::Green, -1);
                cv::imshow("Circle Merging", image);
                cv::waitKey(0);
        #endif
                break;
            }
        }
    } while (anyMerged.has_value());
}

std::vector<Vec2> generateOutline(const Pose& pose, const PoseMidlineIndexes& midline, const std::function<float(float)>& radiusMap) {
    std::vector<Vec2> centers;
    std::vector<float> radii;
    
    /// by default we will just use "all" points if no
    /// indexes are given:
    if(midline.indexes.empty()) {
        for(auto &pt : pose.points) {
            if(pt.valid())
                centers.push_back(pt);
        }
        
    } else {
        /// otherwise, fill with given midline indexes:
        for (uint8_t index : midline.indexes) {
            if (index >= pose.points.size()) {
                FormatWarning("Index ", unsigned(index), " out of range, ignoring it.");
                continue;
            }
            if(pose.points[index].valid()) {
                const Vec2& point = pose.points[index];
                centers.push_back(point);
            }
        }
    }
    
    /// calculate size of each circle statically or based on the index.
    /// this will impact the performance of the algorithm / the number
    /// of points created in the end:
    for(size_t i = 0; i<centers.size(); ++i) {
        float radius = radiusMap && centers.size() > 0
                ? (radiusMap(i / float(centers.size() - 1.f)) + 1.f)
                : 10.0f;
        radii.push_back(radius);
    }

    /// this will add new circles if necessary:
    ensureCircleOverlap(centers, radii);
    
    if(not centers.empty()) {
        /// generate an image that will fit the object.
        /// we should actually make sure that its not too big.
        Bounds bounds(FLT_MAX, FLT_MAX, 0, 0);
        for (size_t i = 0; i < centers.size(); ++i) {
            bounds.combine(Bounds(centers[i] - radii[i], centers[i] + radii[i] * 2));
        }
        
        bounds = bounds - Size2(bounds.pos()) + Size2(2);
        bounds = bounds - Vec2(1);
        
        /// if it is larger than 6000x6000px, we will skip it...
        if(bounds.width * bounds.height > SQR(6000u)) {
            FormatWarning("Object of size ", bounds.size(), " is too large to posture estimate.");
            return {};
        }
        
        cv::Mat merger = cv::Mat::zeros(bounds.height, bounds.width, CV_8UC1);
        for (size_t i = 0; i < centers.size(); ++i) {
            cv::circle(merger, centers.at(i) - bounds.pos(), radii.at(i), cv::Scalar(i / float(centers.size()) * 205.f + 50.f), -1);
        }
        
        /// now detect the merged object(s).
        /// theoretically there should only be one object exactly.
        auto list = _dllists.getObject();
        auto blobs = CPULabeling::run(*list, merger);
        _dllists.returnObject(std::move(list));
        
        if(blobs.empty()) {
            /// this should not happen.
            FormatWarning("This is not a single blob: ", blobs.size(), " pose=", pose, " indexes=", midline, " centers=",centers, " radius=", radii);
            return {};
            
        } else if(blobs.size() != 1) {
#ifndef NDEBUG
            FormatWarning("Not a single blob: ", blobs.size(), " pose=", pose);
#endif
        }
        
        pv::Blob blob{
            std::move(blobs.front().lines),
            std::move(blobs.front().pixels),
            std::move(blobs.front().extra_flags),
            std::move(blobs.front().pred)
        };
        blob.add_offset(bounds.pos());
        
        auto pts = pixel::find_outer_points(pv::BlobWeakPtr{&blob}, 1);
        
#ifdef DEBUG_OUTLINES
        cv::cvtColor(merger, merger, cv::COLOR_GRAY2BGR);
        Vec2 prev = pts.front()->back() - bounds.pos();
        for(auto pt : *pts.front()) {
            pt -= bounds.pos();
            cv::line(merger, prev, pt, gui::Blue, 1);
            prev = pt;
        }
        
        cv::imshow("merger", merger);
        cv::waitKey(0);
#endif
        
        return std::move(*pts.front());
    }
    
    return {};
}

namespace posture {

tl::expected<Result, const char*> calculate_midline(Result&& result) {
    if(result.outline.empty())
        return tl::unexpected("Outline was empty.");
    
    auto r = result.outline.calculate_midline({});
    if(not r) {
        return tl::unexpected(r.error());
    }
    
    result.midline = std::move(r.value());
    assert(result.midline);
    assert(not result.midline->is_normalized());
    
    //result.normalized_midline = result.midline->normalize();
    
    /*if(not result.normalized_midline || result.normalized_midline->size() != FAST_SETTING(midline_resolution)) {
        return tl::unexpected("Unexpected number of points in normalized midline.");
    }*/
    
    return result;
}

tl::expected<Result, const char*> calculate_posture(Frame_t, const BasicStuff& basic, const blob::Pose &pose, const PoseMidlineIndexes &indexes)
{
    Outline::check_constants();
    
    Result result;
    
    {
        auto bds = basic.blob.calculate_bounds();
        
        auto pts = generateOutline(pose, indexes, [m = bds.size().mean() * 0.08_F](float percent) -> float {
            // scale center line by percentage
            return m * (1_F - percent) + 1;
        });
        
        auto ptr = std::make_unique<std::vector<Vec2>>(std::move(pts));
        const auto pos = basic.blob.calculate_bounds().pos();
        for(auto &pt : *ptr)
            pt -= pos;
        
        //result.outline.clear();
        result.outline.replace_points(std::move(ptr));
    }
    
    result.outline.resample(FAST_SETTING(outline_resample));
    result.outline.minimize_memory();
    
    return calculate_midline(std::move(result));
}

tl::expected<Result, const char*> calculate_posture(Frame_t, const BasicStuff &basic, const blob::SegmentedOutlines& outlines) {
    Outline::check_constants();
    
    Result result;
    
    {
        auto ptr = std::make_unique<std::vector<Vec2>>(outlines.original_outline.value());
        const auto pos = basic.blob.calculate_bounds().pos();
        for(auto &pt : *ptr)
            pt -= pos;
        
        result.outline.replace_points(std::move(ptr));
    }
    result.outline.resample(FAST_SETTING(outline_resample));
    result.outline.minimize_memory();
    
    const auto outline_compression = FAST_SETTING(outline_compression);
    if(outline_compression > 0) {
        auto &pts = result.outline.points();
        std::vector<Vec2> reduced;
        reduced.reserve(pts.size());
        gui::reduce_vertex_line(pts, reduced, outline_compression);
        std::swap(reduced, pts);
    }
    
    return calculate_midline(std::move(result));
}

tl::expected<Result, const char*> calculate_posture(Frame_t, pv::BlobWeakPtr blob)
{
    Outline::check_constants();
    
    const int initial_threshold = FAST_SETTING(track_posture_threshold);
    int threshold = initial_threshold;
    
    // order the calculated points to make the outline
    static Timing timing("posture", 100);
    TakeTiming take(timing);
    
    std::optional<std::vector<Vec2>> first_outline;
    
    /// we will store our result here
    Result result;
    
    while(true) {
        // calculate outline points in (almost) random order based on
        // greyscale values, instead of just binary thresholding.
        //auto raw_outline = subpixel_threshold(greyscale, threshold);
        auto thresholded_blob = pixel::threshold_get_biggest_blob(blob, threshold, Tracker::background(), FAST_SETTING(posture_closing_steps), FAST_SETTING(posture_closing_size));
        thresholded_blob->add_offset(-blob->bounds().pos());
        
        periodic::points_t selected = nullptr;
        {
            auto outlines = pixel::find_outer_points(thresholded_blob.get(), threshold);
            
            size_t max_size = 0;
            for(auto &ol : outlines) {
                if(ol->size() > max_size) {
                    max_size = ol->size();
                    selected = std::move(ol);
                }
            }
        }
        
        if(selected != nullptr) {
            result.outline.clear();
            result.outline.replace_points(std::move(selected));
            
            result.outline.resample(FAST_SETTING(outline_resample));
            
            auto r = calculate_midline(std::move(result));
            if(r) {
                /// we found an acceptable configuration with the lowest threshold
                /// possible, so lets just return that:
                return r;
            }
            
            if(not first_outline.has_value()) {
                if(not result.outline.empty())
                    first_outline = result.outline.points();
                else {
                    FormatWarning("Cannot reset since points are empty.");
                    first_outline = std::nullopt;
                }
            }
        }
        
        // increase threshold by 2
        threshold += 2;
        
        if(threshold >= initial_threshold + 50) {
            break;
        }
    }
    
    if(first_outline.has_value()) {
        // move the already resampled points if we have any
        result.outline.replace_points(std::move(first_outline.value()));
        result.midline = nullptr;
        result.normalized_midline = nullptr;
        first_outline = std::nullopt;
        // but midline must have failed, so at least we can return this...
    }
    
    if(not result.outline.empty()) {
        result.outline.minimize_memory();
        return result;
    }
    
    return tl::unexpected("Cannot find valid posture.");
}

/*float Posture::calculate_midline(bool debug) {
    Midline midline;
    _outline.calculate_midline(midline, DebugInfo{frameIndex, fishID, debug});
    _normalized_midline = std::make_unique<Midline>(midline);
    return _outline.confidence();
}*/

}

}
