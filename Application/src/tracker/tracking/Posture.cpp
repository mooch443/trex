#include "Posture.h"
#include <misc/Timer.h>
#include <misc/GlobalSettings.h>
#include <gui/colors.h>

#include <tracking/DebugDrawing.h>
#include <thread>
#include "Tracker.h"
#include <misc/PixelTree.h>
#include <misc/CircularGraph.h>
#include <gui/DrawSFBase.h>

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
    
    Posture::Posture(long_t frameIndex, uint32_t fishID)
        : _outline_points(std::make_shared<std::vector<Vec2>>()), frameIndex(frameIndex), fishID(fishID), _outline(_outline_points, frameIndex), _normalized_midline(NULL)
    { }

    std::vector<Posture::EntryPoint> Posture::subpixel_threshold(const cv::Mat& greyscale, const int threshold) {
        static Timing timing("subpixel thresholding", 30);
        static std::mutex average_mutex;
        static size_t average_eps_count = 1000;
        static size_t eps_samples = 1;
        
        //auto grid = Tracker::instance()->grid();
        
        timing.start_measure();
        
        std::vector<std::pair<uint32_t, std::vector<Vec2>>> chains; // neighborhood chains
        std::vector<EntryPoint> eps;
        EntryPoint tmp;
        EntryPoint strips[2];
        
        {
            average_mutex.lock();
            eps.reserve(average_eps_count / eps_samples * 1.1f);
            average_mutex.unlock();
        }
        
        const float t = threshold;
        
        for (int y=1; y<greyscale.rows-1; y++) {
            strips[0].clear();
            strips[1].clear();
            
            strips[0].y = y;
            strips[1].y = y;
            
            for (int x=1; x<greyscale.cols-1; x++) {
                const auto& val = greyscale.at<uchar>(y, x);
                //const auto t = threshold * (grid ? grid->relative_threshold(x, y) : 1);
                
                // not fish
                if(val < t) {
                    for(int i=0; i<2; i++) {
                        if (strips[i].x0 != -1) {
                            eps.push_back(strips[i]);
                            strips[i].clear();
                            strips[i].y = y;
                        }
                    }
                }
                // fish
                else if (val && val >= t) {
                    float inside = cmn::abs(float(val) - t);
                    
                    bool in = false;
                    int count = 0;
                    chains.clear();
                    //chains.reserve(4);
                    
                    for (uint32_t n=0; n<neighbors.size(); n++) {
                        const auto &npos = neighbors[n];
                        const Vec2 current(npos.x + x, npos.y + y);
                        
                        auto pixel = greyscale.at<uchar>(round(current.y), round(current.x));
                        float outside = cmn::abs(float(pixel) - t);
                        float middle = outside + inside;
                        
                        if (middle != 0)
                            middle = middle / (cmn::abs(outside) + cmn::abs(inside));
                        
                        if(pixel < t) {
                            const Vec2 value(0.5 * npos * (1 + middle) + 0.5);
                            count++;
                            
                            if(!in) {
                                in = true;
                                chains.push_back({ n, { value } });
                                
                            } else if(chains.back().second.back() != value)
                                chains.back().second.push_back(value);
                            
                        } else {
                            in = false;
                        }
                    }
                    
                    // wrap around
                    if(in) {
                        if (chains.size() == 1 && chains.front().first == 0) {
                            count = 0;
                        } else {
                        
                            if (chains.front().first == 0) {
                                for (auto &e : chains.front().second) {
                                    chains.back().second.push_back(e);
                                }
                                chains.erase(chains.begin());
                            }
                        }
                    }
                    
                    // this pixel is a border pixel
                    if(count > 0) {
                        if (chains.size() == 1 || chains.size() > 2) {
                            // easy, only one average point. attach to closest
                            Vec2 average(0, 0);
                            float samples = 0;
                            
                            for (auto &c : chains) {
                                for (auto &e : c.second) {
                                    average += e;
                                    samples++;
                                }
                            }
                            
                            average /= samples;
                            average += Vec2(x, y);
                            
                            tmp.x0 = x;
                            tmp.x1 = x + 1;
                            tmp.y = y;
                            //tmp.x = x;
                            tmp.interp = {average};
                            
                            eps.push_back(tmp);
                            
                        } else if(chains.size() == 2) {
                            // two sides, axis between them
                            Vec2 points[2];
                            for (uint32_t i=0; i<chains.size(); i++) {
                                Vec2 average(0, 0);
                                for (auto &e : chains[i].second) {
                                    average += e;
                                }
                                average /= float(chains[i].second.size());
                                average += Vec2(x, y);
                                
                                points[i] = average;
                                
                                tmp.x0 = x;
                                tmp.x1 = x + 1;
                                tmp.y = y;
                                //tmp.x = x;
                                tmp.interp = {average};
                                
                                eps.push_back(tmp);
                            }
                            
                        } else if(chains.size()) {
                            // Shouldnt be possible
                            U_EXCEPTION("Did not expect %d chains.", chains.size());
                        }
                        
                    }
                }
                
                // conclude chain
                for (int i=0; i<2; i++) {
                    if (strips[i].x0 != -1 && (strips[i].x1 < x || x == greyscale.cols - 1)) {
                        eps.push_back(strips[i]);
                        
                        strips[i].clear();
                        strips[i].y = y;
                    }
                }
            }
        }
        
        for (uint32_t i=0; i<eps.size(); i++) {
            auto &p = eps[i];
            auto &points = p.interp;
            
            /*for(auto &point : points) {
                point.second += Vec2(1, 1);
            }*/
            
            if (points.size() == 2) {
                if (sqdistance(points.at(0), points.at(1)) >= 1) {
                    tmp.interp = {points[1]};
                    tmp.x0 = p.x0 + 1;//points[1].first;
                    //assert(tmp.x0 == p.x);
                    tmp.x1 = tmp.x0 + 1;
                    tmp.y = p.y;
                    
                    p.x1 = p.x0 + 1;
                    
                    points.erase(points.begin() + 1);
                    eps.insert(eps.begin() + i + 1, tmp);
                }
            }
        }
        
        timing.conclude_measure();
        
        std::lock_guard<std::mutex> guard(average_mutex);
        if(eps_samples < 10000) {
            average_eps_count += eps.size();
            eps_samples++;
        }
        
        return eps;
    }

    void Posture::calculate_posture(long_t frame, pv::BlobPtr blob)
    {
        const int initial_threshold = FAST_SETTINGS(track_posture_threshold);
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
            thresholded_blob = pixel::threshold_get_biggest_blob(blob, threshold, Tracker::instance()->background(), FAST_SETTINGS(posture_closing_steps), FAST_SETTINGS(posture_closing_size));
            thresholded_blob->add_offset(-blob->bounds().pos());

            auto outlines = pixel::find_outer_points(thresholded_blob, threshold);
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
                if(FAST_SETTINGS(outline_resample) != 0) {
                    if(FAST_SETTINGS(outline_resample) >= 1)
                        _outline.resample(FAST_SETTINGS(outline_resample));
                    else
                        _outline.resample(FAST_SETTINGS(outline_resample));
                }
                
                std::pair<pv::bid, long_t> gui_show_fish = SETTING(gui_show_fish);
                auto debug = gui_show_fish.first == blob->blob_id() && frame == gui_show_fish.second;
                float confidence = calculate_midline(debug);
                bool error = !_normalized_midline || (_normalized_midline->size() != FAST_SETTINGS(midline_resolution));
                error = !_normalized_midline;
                
                auto norma = _normalized_midline ? _normalized_midline->normalize() : nullptr;
                if(norma && norma->size() != FAST_SETTINGS(midline_resolution))
                    error = true;
                
                if(first_outline == nullptr) {
                    first_outline = std::make_shared<Outline>(_outline);
                    first_outline->replace_points(std::make_shared<std::vector<Vec2>>(_outline.points()));
                }
                
                outline_point = selected;
                
                if(!error && confidence > 0.9f) {
                    if(/* DISABLES CODE */ (false) && FAST_SETTINGS(debug)) /*&& threshold-initial_threshold > 0*/
                    {
                        printf("raw_outline=np.asarray([");
                        for (auto &a : *selected) {
                            printf("[%f,%f],", a.x, a.y);
                        }
                        printf("])\n");

                        Debug("Frame %d rendered with threshold %d+%d (%d -> %d points).", frameIndex, FAST_SETTINGS(track_posture_threshold), threshold - FAST_SETTINGS(track_posture_threshold), selected->size(), _outline.size());
                    }
                    
                    // found a good configuration! escape.
                    break;
                } else if(FAST_SETTINGS(debug)) {
                    Debug("Error in outline (threshold %d) @%d for %d %d", threshold, frameIndex, fishID, error);
                }
            }
            
            // increase threshold by 2
            threshold += 2;
            
            if(threshold >= initial_threshold + 50) {
                if(FAST_SETTINGS(debug)) {
                    Debug("Outline failed (threshold %d) @%d for %d", threshold, frameIndex, fishID);
                }
                
                break;
            }
        }
        
        if(!_normalized_midline && first_outline)
            _outline = *first_outline;
        
        timing.conclude_measure();
        
        std::pair<pv::bid, long_t> gui_show_fish = SETTING(gui_show_fish);
        if(gui_show_fish.first == blob->blob_id() && frame == gui_show_fish.second) {
        
        //if(frame == 532 && blob->blob_id() == 6357990) {
            Debug("%d %d: threshold %d", frame, blob->blob_id(), threshold);
            auto blob = thresholded_blob;
            auto && [pos, image] = blob->image();
            //tf::imshow("image", image->get());
            std::this_thread::sleep_for(std::chrono::seconds(1));
            
            auto curv = periodic::curvature(outline_point, max(1, Outline::get_curvature_range_ratio() * outline_point->size()));
            auto diffs = periodic::differentiate(curv, 2);
            
            auto peak_mode = SETTING(peak_mode).value<default_config::peak_mode_t::Class>() == default_config::peak_mode_t::broad ? periodic::PeakMode::FIND_BROAD : periodic::PeakMode::FIND_POINTY;
            auto && [maxima_ptr, minima_ptr] = periodic::find_peaks(curv, 0, diffs, peak_mode);
            auto str = Meta::toStr(*maxima_ptr);
            Debug("%d, %d: %S", frame, blob->blob_id(), &str);
            
            str = Meta::toStr(*outline_point);
            Debug("%S", &str);
            
            {
                using namespace gui;
                float scale = 20;
                
                cv::Mat colored;
                cv::cvtColor(image->get(), colored, cv::COLOR_GRAY2BGR);
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
                
                auto midline = _normalized_midline;
                if(midline) {
                    auto transform = midline->transform(default_config::recognition_normalization_t::none, true);
                    for(auto &seg : midline->segments()) {
                        auto trans = transform.transformPoint(seg.pos);
                        cv::circle(colored, OFFSET(trans), 3, Red);
                    }
                    
                    if(midline->tail_index() != -1)
                        cv::circle(colored, OFFSET(outline().at(midline->tail_index())), 10, Blue, -1);
                    if(midline->head_index() != -1)
                        cv::circle(colored, OFFSET(outline().at(midline->head_index())), 10, Red, -1);
                    
                    cv::circle(colored, OFFSET(outline().front()), 5, Yellow, -1);
                    
                    Debug("tail:%d head:%d", midline->tail_index(), midline->head_index());
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
    
    float Posture::calculate_outline(std::vector<Posture::EntryPoint>& entry_points) {
        const size_t N = entry_points.size();
        std::deque<long_t> unassigned;
        for(size_t i=0; i<N; ++i)
            unassigned.push_back((long_t)i);
        
        size_t assigned = 0;
        
#define pdist(A, b) sqdistance(A, entry_points[b].interp.front())
#define rpdist(A, b) sqdistance(A, entry_points[b].interp.back())
        
        float prev_angle;
        Vec2 prev_point(FLT_MAX, FLT_MAX);
        Vec2 direction(FLT_MAX, FLT_MAX);
        //float current_min_rawd = 0;
        
        const float factor = FAST_SETTINGS(outline_resample) ? FAST_SETTINGS(outline_resample) : 1;//0.25;
        
        const auto rdist_points = [&](const Vec2& A, const Vec2& B, float d) -> float {
            if(B == A)
                return 0;

            if(prev_point.x != FLT_MAX && prev_angle != FLT_MAX) {
                assert(!std::isnan(prev_angle));

                float angle = atan2(B.y - A.y, B.x - A.x) - prev_angle;
                if(angle < -M_PI) angle += 2*M_PI;
                if(angle > M_PI) angle -= 2*M_PI;
                assert(cmn::abs(angle) <= M_PI);

                return (d) + cmn::abs(angle) / M_PI * (factor);
            }

            return (d) + (factor);
        };
        
        const auto rdist = [&](const Vec2& A, long_t b) -> float {
            const float d0 = pdist(A, b);
            const float d1 = entry_points[b].interp.size() == 1 ? d0 : rpdist(A, b);
            
            float d;
            if(d0 > d1) {
                d = d1;
                //Debug("Reverse.");
                std::reverse(entry_points[b].interp.begin(), entry_points[b].interp.end());
            } else
                d = d0;
            
            if(d > SQR(factor * 10))
                return FLT_MAX;
            
            const Vec2& B = entry_points[b].interp.front();
            return rdist_points(A, B, d);
        };
        
        auto compare = [](const std::vector<Vec2>& A, const std::vector<Vec2>& B) -> bool {
            return A.size() > B.size();
        };
        
        std::set<std::vector<Vec2>, decltype(compare)> outlines(compare);
        
        /**
         
             GO IN THE OTHER DIRECTION AS WELL, IF ONE OF THE LINES REMAINING
             HAS A VALID CONNECTION TO THE OUTLINE START
         
         */
        // repeat until we found the biggest object
        while (unassigned.size() > N * 0.05) {//unassigned.size() > assigned) {
            _outline.clear();
            assigned = 0;
            //current_min_rawd = FLT_MAX;
            direction = Vec2(FLT_MAX, FLT_MAX);
            prev_angle = FLT_MAX;
            prev_point = Vec2(FLT_MAX, FLT_MAX);
            float back_front = FLT_MAX;

            long_t pt = -1;
            while (!unassigned.empty()) {
                pt = unassigned.front();
                unassigned.pop_front();

                if(!_outline.empty())
                    prev_point = _outline.back();
                _outline.insert(_outline.size(), entry_points[pt].interp.begin(), entry_points[pt].interp.end());
                assigned++;
                
                back_front = _outline.size() > 3 ? rdist_points(_outline.back(), _outline.front(), sqdistance(_outline.back(), _outline.front())) : FLT_MAX;
                
                if(unassigned.size() > 1) {
                    Vec2 A = entry_points[pt].interp.back();
                    
                    if(prev_point.x != FLT_MAX) {
                        Vec2 vec0 = A - prev_point;

                        if(vec0.length() > 0) {
                            if(direction.x == FLT_MAX)
                                direction = vec0.normalize();
                            else
                                direction = direction * 0.6 + vec0.normalize() * 0.4;

                            direction = direction.normalize();
                            prev_angle = atan2(direction.y, direction.x);
                        }
                    }
                    
                    float min_d = FLT_MAX;
                    auto min_idx = unassigned.end();
                    
                    float d = 0;
                    for(auto it = unassigned.begin(); it != unassigned.end(); ++it) {
                        d = rdist(A, *it);
                        if(d < min_d) {
                            min_d = d;
                            min_idx = it;
                        }
                    }
                    
                    if(min_idx != unassigned.end() && min_d < back_front) {
                        if(pt != 0 && min_d > rdist(A, 0))
                            break;
                        std::swap(*unassigned.begin(), *min_idx);
                    }
                    else {
                        // if the end of the line is reached, the possibility exists
                        // that it can be extended from the first point instead of the
                        // last point of the outline.
                        // so if thats the case - reverse the outline and try again
                        // (most of the time this will yield nothing)
                        
                        // MIGHT STILL HAPPEN IF THE FISH IS WEIRD AND THERE ARE
                        // GAPS ON BOTH SIDES
                        min_d = FLT_MAX;
                        min_idx = unassigned.end();
                        
                        d = 0;
                        for(auto it = unassigned.begin(); it != unassigned.end(); ++it) {
                            d = rdist(_outline.points().front(), *it);
                            if(d < min_d) {
                                min_d = d;
                                min_idx = it;
                            }
                        }
                        
                        if(min_idx != unassigned.end() && min_d < back_front) {
                            direction = Vec2(FLT_MAX, FLT_MAX);
                            std::reverse(_outline.points().begin(), _outline.points().end());
                            _outline.minimize_memory();
                            
                            std::swap(*unassigned.begin(), *min_idx);
                        } else
                            break;
                    }
                } else
                    break;
            }
            
            if(_outline.confidence() > 0.9)
                outlines.insert(_outline.points());
            
            /**
             * TEMPORARILY terminating upon biggest-outline-found.
             * need to stitch together outlines potentially.
             */
            if(outlines.begin()->size() > unassigned.size()) {
                //Debug("Stopping %d/%d", outlines.begin()->size(), unassigned.size());
                break;
            }
        }
        
        if(!outlines.empty()) {
            _outline.points() = *outlines.begin();
            _outline.minimize_memory();
            return 1;
        }
        
        //Debug("0 (unassigned: %d, assigned: %d)", unassigned.size(), _outline.size());
        _outline.clear();
        return 0;
    }

    float Posture::calculate_midline(bool debug) {
        Midline midline;
        _outline.calculate_midline(midline, DebugInfo{frameIndex, fishID, debug});
        //_outline.post_process(midline, movement, DebugInfo{frameIndex, fishID, debug});
        //_normalized_midline = midline.normalize();
        _normalized_midline = std::make_shared<Midline>(midline);//.normalize();
        
        /*if(_normalized_midline && _outline.confidence() > 0.9f) {
            auto &segments = _normalized_midline->segments();
            const long_t L = segments.size();
            long_t offset = 1;
            
            for (long_t index=0; index<L; index++) {
                size_t idx01= index < L-offset ? index+offset : index+offset-L;
                size_t idx0 = index >= offset ? index-offset : index-offset+L;
                size_t idx1 = index;
                
                const auto &p3 = segments.at(idx01).pos;
                const auto &p2 = segments.at(idx0).pos;
                const auto &p1 = segments.at(idx1).pos;
                
                float K;
                if(p1 == p2 || p1 == p3 || p2 == p3)
                    K = 0.0;
                else
                    // TODO: this calculation is slow
                    K = 2 * ((p2.x-p1.x)*(p3.y-p2.y) - (p2.y-p1.y)*(p3.x-p2.x))
                    / cmn::sqrt(sqdistance(p1, p2) * sqdistance(p2, p3) * sqdistance(p1, p3));
                
                if((index <= L * 0.25 && abs(K) > 0.17) || (index > L * 0.25 && abs(K) > 0.4f)) {
                    Tracker::increase_midline_errors();
                    //if(FAST_SETTINGS(debug))
                    //    Debug("Midline index %d failed with curvature: %f in frame %d", index, K, frameIndex);
                    //if(index <= L * 0.25)
                    //return 0;
                    //break;
                }
            }
            
            return _outline.confidence();
        }*/
        
        return _outline.confidence();
    }
}
