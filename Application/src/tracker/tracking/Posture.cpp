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
    
    Posture::Posture(Frame_t frameIndex, Idx_t fishID)
        : _outline_points(std::make_shared<std::vector<Vec2>>()), frameIndex(frameIndex), fishID(fishID), _outline(_outline_points, frameIndex), _normalized_midline(nullptr)
    { }

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
    
    int anyMerged;
    do {
        anyMerged = -1;
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
    } while (anyMerged != -1);
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
        float radius = radiusMap
                ? (radiusMap(i / float(centers.size())) + 1.f)
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
        auto blobs = CPULabeling::run(merger);
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

    void Posture::calculate_posture(Frame_t, const BasicStuff& basic, const blob::Pose &pose, const PoseMidlineIndexes &indexes) {
        auto pts = generateOutline(pose, indexes, [](float percent) -> float {
            // scale center line by percentage
            return 40.f * (1.f - percent) + 1.f;
        });
        auto ptr = std::make_shared<std::vector<Vec2>>(pts);
        const auto pos = basic.blob.calculate_bounds().pos();
        for(auto &pt : *ptr)
            pt -= pos;
        
        _outline.clear();
        _outline.replace_points(ptr);
        _outline.minimize_memory();
        _outline.resample(FAST_SETTING(outline_resample));
        
        //std::tuple<pv::bid, Frame_t> gui_show_fish = SETTING(gui_show_fish);
        auto debug = false;//std::get<0>(gui_show_fish) == blob->blob_id() && frame == std::get<1>(gui_show_fish);
        float confidence = calculate_midline(debug);
        bool error = !_normalized_midline || (_normalized_midline->size() != FAST_SETTING(midline_resolution));
        error = !_normalized_midline;
        
        auto norma = _normalized_midline ? _normalized_midline->normalize() : nullptr;
        if(norma && norma->size() != FAST_SETTING(midline_resolution))
            error = true;
        
        //outline_point = ptr;
        
        if(!error && confidence > 0.9f) {
            // found a good configuration! escape.
            return;
        }
    }

void Posture::calculate_posture(Frame_t, const BasicStuff &basic, const blob::SegmentedOutlines& outlines) {
    auto ptr = std::make_shared<std::vector<Vec2>>(outlines.original_outline.value());
    const auto pos = basic.blob.calculate_bounds().pos();
    for(auto &pt : *ptr)
        pt -= pos;
    
    _outline.clear();
    _outline.replace_points(ptr);
    _outline.minimize_memory();
    _outline.resample(FAST_SETTING(outline_resample));
    
    //std::tuple<pv::bid, Frame_t> gui_show_fish = SETTING(gui_show_fish);
    auto debug = false;//std::get<0>(gui_show_fish) == blob->blob_id() && frame == std::get<1>(gui_show_fish);
    float confidence = calculate_midline(debug);
    bool error = !_normalized_midline || (_normalized_midline->size() != FAST_SETTING(midline_resolution));
    error = !_normalized_midline;
    
    auto norma = _normalized_midline ? _normalized_midline->normalize() : nullptr;
    if(norma && norma->size() != FAST_SETTING(midline_resolution))
        error = true;
    
    //outline_point = ptr;
    
    if(!error && confidence > 0.9f) {
        // found a good configuration! escape.
        return;
    }
}

    void Posture::calculate_posture(Frame_t frame, pv::BlobWeakPtr blob)
    {
        const int initial_threshold = FAST_SETTING(track_posture_threshold);
        int threshold = initial_threshold;
        
        // order the calculated points to make the outline
        static Timing timing("posture", 100);
        timing.start_measure();
        
        std::shared_ptr<Outline> first_outline;
        pv::BlobPtr thresholded_blob = nullptr;
        std::vector<std::shared_ptr<std::vector<Vec2>>> _outlines;
        std::vector<Vec2> custom;
        std::shared_ptr<std::vector<Vec2>> outline_point;

        while(true) {
            // calculate outline points in (almost) random order based on
            // greyscale values, instead of just binary thresholding.
            //auto raw_outline = subpixel_threshold(greyscale, threshold);
            thresholded_blob = pixel::threshold_get_biggest_blob(blob, threshold, Tracker::background(), FAST_SETTING(posture_closing_steps), FAST_SETTING(posture_closing_size));
            thresholded_blob->add_offset(-blob->bounds().pos());

            auto outlines = pixel::find_outer_points(thresholded_blob.get(), threshold);
            std::vector<Vec2> interp;
            _outlines = outlines;
            custom = interp;
            
            //if(frame == 177 && blob->blob_id() == 40305448) {
            
            
            decltype(outlines)::value_type selected = nullptr;
            size_t max_size = 0;
            for(auto &ol : outlines) {
                if(ol->size() > max_size) {
                    selected = ol;
                    max_size = ol->size();
                }
            }
            
            if(selected != nullptr) {
                _outline.clear();
                _outline.replace_points(selected);
                _outline.minimize_memory();
                
                
            //}
            
            //if(calculate_outline(raw_outline) > 0.9f) {
                _outline.resample(FAST_SETTING(outline_resample));
                
                std::tuple<pv::bid, Frame_t> gui_show_fish = SETTING(gui_show_fish);
                auto debug = std::get<0>(gui_show_fish) == blob->blob_id() && frame == std::get<1>(gui_show_fish);
                float confidence = calculate_midline(debug);
                bool error = !_normalized_midline || (_normalized_midline->size() != FAST_SETTING(midline_resolution));
                error = !_normalized_midline;
                
                auto norma = _normalized_midline ? _normalized_midline->normalize() : nullptr;
                if(norma && norma->size() != FAST_SETTING(midline_resolution))
                    error = true;
                
                if(first_outline == nullptr) {
                    first_outline = std::make_shared<Outline>(_outline);
                    first_outline->replace_points(std::make_shared<std::vector<Vec2>>(_outline.points()));
                }
                
                outline_point = selected;
                
                if(!error && confidence > 0.9f) {
                    // found a good configuration! escape.
                    break;
                }
            }
            
            // increase threshold by 2
            threshold += 2;
            
            if(threshold >= initial_threshold + 50) {
                break;
            }
        }
        
        if(!_normalized_midline && first_outline)
            _outline = *first_outline;
        
        timing.conclude_measure();
        
        std::tuple<pv::bid, Frame_t> gui_show_fish = SETTING(gui_show_fish);
        if(std::get<0>(gui_show_fish) == blob->blob_id() && frame == std::get<1>(gui_show_fish)
           && outline_point) {
            Print(frame, " ", blob->blob_id(),": threshold ", threshold);
            auto &blob = thresholded_blob;
            auto && [pos, image] = blob->color_image();
            //tf::imshow("image", image->get());
            std::this_thread::sleep_for(std::chrono::seconds(1));
            
            auto curv = periodic::curvature(outline_point, max(1, Outline::get_curvature_range_ratio() * outline_point->size()));
            auto diffs = periodic::differentiate(curv, 2);
            
            auto peak_mode = SETTING(peak_mode).value<default_config::peak_mode_t::Class>() == default_config::peak_mode_t::broad ? periodic::PeakMode::FIND_BROAD : periodic::PeakMode::FIND_POINTY;
            auto && [maxima_ptr, minima_ptr] = periodic::find_peaks(curv, 0, diffs, peak_mode);
            auto str = Meta::toStr(*maxima_ptr);
            Print(frame, ", ", blob->blob_id(),": ", str.c_str());
            Print(*outline_point);
            
            {
                using namespace gui;
                float scale = 20;
                
                cv::Mat colored;
                if(image->channels() == 1)
                    cv::cvtColor(image->get(), colored, cv::COLOR_GRAY2BGR);
                else {
                    assert(image->dims == 3);
                    image->get().copyTo(colored);
                }
                cv::resize(colored, colored, (cv::Size)(Size2(image->cols, image->rows) * scale), 0, 0, cv::INTER_NEAREST);
                
                for(auto &pt : custom) {
                    cv::circle(colored, (Vec2(pt.x, pt.y) - pos + Vec2(0.5)) * scale, 5, DarkCyan, -1);
                }
                
                for (uint i=0; i<image->cols; ++i) {
                    cv::line(colored, Vec2(i, 0) * scale, Vec2(i, image->rows) * scale, Black);
                }
                
                for (uint i=0; i<image->rows; ++i) {
                    cv::line(colored, Vec2(0, i) * scale, Vec2(image->cols, i) * scale, Black);
                }
                
#define OFFSET(X) (((X) - pos) * scale)
                for (auto m : *maxima_ptr) {
                    cv::circle(colored, OFFSET(outline_point->at(m.position.x)), 15, Yellow);
                    auto str = Meta::toStr(m);
                    cv::putText(colored, str, OFFSET(outline_point->at(m.position.x)) + Vec2(-Base::text_dimensions(str).width * 0.5,-10), cv::FONT_HERSHEY_PLAIN, 1.0, Yellow);
                }
                
                if(_normalized_midline) {
                    auto transform = _normalized_midline->transform(default_config::individual_image_normalization_t::none, true);
                    for(auto &seg : _normalized_midline->segments()) {
                        auto trans = transform.transformPoint(seg.pos);
                        cv::circle(colored, OFFSET(trans), 3, Red);
                    }
                    
                    if(_normalized_midline->tail_index() != -1)
                        cv::circle(colored, OFFSET(outline().at(_normalized_midline->tail_index())), 10, Blue, -1);
                    if(_normalized_midline->head_index() != -1)
                        cv::circle(colored, OFFSET(outline().at(_normalized_midline->head_index())), 10, Red, -1);
                    
                    cv::circle(colored, OFFSET(outline().front()), 5, Yellow, -1);
                    
                    Print("tail:", _normalized_midline->tail_index()," head:",_normalized_midline->head_index());
                }
                
                ColorWheel cwheel;
                cwheel.next();
                for (auto &clique: _outlines) {
                    auto color = cwheel.next();
                    
                    auto prev = clique->back();
                    for (auto node : *clique) {
                        cv::circle(colored, OFFSET(node), 5, color);
                        cv::line(colored, OFFSET(prev), OFFSET(node), color);
                        prev = node;
                    }
                }
                
                cv::cvtColor(colored, colored, cv::COLOR_BGR2RGB);
                tf::imshow("image", colored);
                
                Tracker::analysis_state(Tracker::AnalysisState::PAUSED);
            }
        }
    }
    
    float Posture::calculate_midline(bool debug) {
        Midline midline;
        _outline.calculate_midline(midline, DebugInfo{frameIndex, fishID, debug});
        _normalized_midline = std::make_unique<Midline>(midline);
        return _outline.confidence();
    }
}
