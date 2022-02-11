#include "SplitBlob.h"
#include <types.h>
#include <processing/CPULabeling.h>
#include <tracking/Tracker.h>
#include <misc/Timer.h>
#include <misc/metastring.h>
#include <misc/PixelTree.h>

using namespace track;

std::atomic_bool registered_callbacks = false;
float blob_split_max_shrink = 0.02f;
float blob_split_global_shrink_limit = 0.01f;
float sqrcm = 1.f;
BlobSizeRange fish_minmax({Rangef(0.001,1000)});
int posture_threshold = 15;

SplitBlob::SplitBlob(const Background& average, pv::BlobPtr blob)
    :   max_objects(0),
        _blob(blob)
{
    // generate greyscale and mask images
    //
    bool result = false;
    if(registered_callbacks.compare_exchange_strong(result, true)) {
        auto callback = "SplitBlob";
        auto fn = [callback](sprite::Map::Signal signal, sprite::Map& map, const std::string& name, const sprite::PropertyType& value)
        {
            if(signal == sprite::Map::Signal::EXIT) {
                map.unregister_callback(callback);
                return;
            }
            
            if(name == "blob_split_max_shrink") {
                blob_split_max_shrink = value.value<float>();
                Debug("blob_split_max_shrink = %f", blob_split_max_shrink);
                
            } else if(name == "blob_split_global_shrink_limit") {
                blob_split_global_shrink_limit = value.value<float>();
                Debug("blob_split_global_shrink_limit = %f", blob_split_global_shrink_limit);
                
            } else if(name == "cm_per_pixel") {
                sqrcm = SQR(value.value<float>());
                Debug("sqrcm = %f", sqrcm);
                
            } else if(name == "track_posture_threshold") {
               posture_threshold = value.value<int>();
               Debug("track_posture_threshold = %d", posture_threshold);
            }
            
            if(name == "blob_size_ranges" || name == "cm_per_pixel") {
                fish_minmax = SETTING(blob_size_ranges).value<BlobSizeRange>();
                //fish_minmax.start /= FAST_SETTINGS(cm_per_pixel);
                //fish_minmax.end /= FAST_SETTINGS(cm_per_pixel);
                auto str = Meta::toStr(fish_minmax);
                Debug("blob_size_ranges = %S", &str);
            }
        };
        GlobalSettings::map().register_callback(callback, fn);
        
        fn(sprite::Map::Signal::NONE, GlobalSettings::map(), "blob_split_max_shrink", SETTING(blob_split_max_shrink).get());
        fn(sprite::Map::Signal::NONE, GlobalSettings::map(), "blob_split_global_shrink_limit", SETTING(blob_split_global_shrink_limit).get());
        fn(sprite::Map::Signal::NONE, GlobalSettings::map(), "cm_per_pixel", SETTING(cm_per_pixel).get());
        fn(sprite::Map::Signal::NONE, GlobalSettings::map(), "track_posture_threshold", SETTING(track_posture_threshold).get());
        fn(sprite::Map::Signal::NONE, GlobalSettings::map(), "blob_size_ranges", SETTING(blob_size_ranges).get());
    }
    
#if DEBUG_ME
    Debug("SplitBlob(%d)", blob->blob_id());
#endif
    
    imageFromLines(blob->hor_lines(), NULL, &_original_grey, &_original, blob->pixels().get(), posture_threshold, &average.image());
    
    blob->set_tried_to_split(true);
}

size_t SplitBlob::apply_threshold(int threshold, std::vector<pv::BlobPtr> &output)
{
    if(_diff_px.empty()) {
        _diff_px.resize(_blob->pixels()->size());
        auto px = _blob->pixels()->data();
        auto out = _diff_px.data();
        auto bg = Tracker::instance()->background();
        //auto grid = Tracker::instance()->grid();
        LuminanceGrid* grid = nullptr;
        
        for (auto &line : _blob->hor_lines()) {
            for (auto x=line.x0; x<=line.x1; ++x, ++px, ++out) {
                *out = (uchar)saturate(float(bg->diff(x, line.y, *px)) / (grid ? float(grid->relative_threshold(x, line.y)) : 1.f));
            }
        }
    }
    
    output = pixel::threshold_blob(_blob, _diff_px, threshold);
    
    for(auto &blob: output) {
        blob->add_offset(-_blob->bounds().pos());
    }
    
    std::sort(output.begin(), output.end(),
              [](const pv::BlobPtr& a, const pv::BlobPtr& b) { return std::make_tuple(a->pixels()->size(), a->blob_id()) > std::make_tuple(b->pixels()->size(), b->blob_id()); });
    
    return output.empty() ? 0 : (*output.begin())->pixels()->size();
    /*auto first_method = timer.elapsed();
    
    static std::mutex local_mutex;
    static double samples, timer1, timer2, ratio;
    std::lock_guard<std::mutex> guard(local_mutex);
    ++samples;
    timer1 += first_method;
    timer2 += second_method;
    ratio += first_method / second_method;
    if(size_t(samples) % 10000 == 0) {
        if(samples >= 100000) {
            timer1 /= samples;
            timer2 /= samples;
            ratio /= samples;
            samples = 1;
        }
        
        Debug("Samples: %lu, timer1: %fms timer2: %fms ratio: %f", size_t(samples), timer1 / samples * 1000, timer2 / samples * 1000, ratio / samples);
    }
    
        return max_size;*/
}

SplitBlob::Action SplitBlob::evaluate_result_single(std::vector<pv::BlobPtr> &result) {
    float size = result.front()->num_pixels() * SQR(FAST_SETTINGS(cm_per_pixel));
    // dont use fish that are too big
    if(size > fish_minmax.max_range().end)
        return REMOVE;
    
#if DEBUG_ME
    // evaluate result
    display_match({0, result});
#endif
    
    // delete unnecessary blobs
    if(result.size() > 1) {
        result.resize(1);
    }
    
    return KEEP;
}

SplitBlob::Action SplitBlob::evaluate_result_multiple(size_t presumed_nr, float first_size, std::vector<pv::BlobPtr>& blobs, ResultProp &r)
{
    // this is way more than we need...
    //assert(presumed_nr <= 2); // only tested with 2 so far
    //display_match({0, result});
    if(blobs.size() > presumed_nr) {
    //    blobs.resize(presumed_nr);
    }
    
    const float min_size_threshold = fish_minmax.max_range().start * blob_split_global_shrink_limit;// * presumed_nr;
    
    size_t offset = 0;
    float ratio = 0;
    while(presumed_nr <= blobs.size() && offset <= blobs.size() - presumed_nr && ratio < 0.3) {
        float fsize = float(blobs.at(0 + offset)->num_pixels()) * sqrcm;
        if(fsize < min_size_threshold || (first_size != 0 && fsize / first_size < blob_split_max_shrink / presumed_nr)) { //)) {
#if DEBUG_ME
            Debug("\tbreaking because fsize %f / %f / %f", fsize, min_size_threshold, first_size);
#endif
            break;
        }
        
        float min_ratio = FLT_MAX;
        for(size_t i=1; i<presumed_nr; ++i) {
            ratio = blobs.size() > 1 ? (fsize / ( float(blobs.at(offset+i)->num_pixels() * sqrcm))) : 1;
            if(ratio > 1)
                ratio = 1/ratio;
            
            if(ratio < min_ratio)
                min_ratio = ratio;
        }
        
        if(min_ratio != FLT_MAX)
            ratio = min_ratio;
        else
            ratio = 0;
        
        ++offset;
    }
    r.ratio = ratio;
    
    if(presumed_nr <= blobs.size() && ratio >= 0.3) {
        return KEEP_ABORT;
    }
    
    return REMOVE;
}

std::vector<pv::BlobPtr> SplitBlob::split(size_t presumed_nr)
{
    std::map<int, ResultProp> best_matches;
    
    size_t calculations = 0;
    Timer timer;
    std::vector<pv::BlobPtr> blobs;
    float first_size = 0;
    size_t more_than_1_times = 0;
    
    const auto fn = [&calculations, &blobs, presumed_nr, &first_size, &more_than_1_times, &best_matches, this](int threshold) {
        calculations++;
        
#if DEBUG_ME
        Debug("T:%d", threshold);
#endif
        float max_size = apply_threshold(threshold, blobs) * sqrcm;
        
        // cant find any blobs anymore...
        if(blobs.empty())
            return ABORT;
        
        // save the maximum number of objects found
        max_objects = max(max_objects, blobs.size());
        
        ResultProp result;
        result.threshold = threshold;
        
        Action action;
        // reduced the blob way too much
        if(first_size != 0 && max_size / first_size <= blob_split_max_shrink / presumed_nr) { //* presumed_nr) {
            action = ABORT;
            
            // differentiate between "search [single/multiple]" fish cases
        } else if(presumed_nr == 1) {
            action = evaluate_result_single(blobs);
        } else {
            action = evaluate_result_multiple(presumed_nr, first_size, blobs, result);
        }
        
        if(first_size == 0)
            first_size = max_size;
        
        if(blobs.size() > 1)
            more_than_1_times++;
        
        // found at least two blobs now
        if(action == KEEP || action == KEEP_ABORT) {
#if DEBUG_ME
            Debug("Found %d blobs at threshold %d (expected %d)", blobs.size(), threshold, presumed_nr);
#endif
            
            result.blobs = blobs;
            best_matches[threshold] = result;
            
        } else {
            blobs.clear();
        }
        
        return action;
    };
    
    auto action = fn(-1);
    
    if(action != KEEP && action != KEEP_ABORT && _blob->pixels()->size() * sqrcm < fish_minmax.max_range().end * 100) {
        static const auto element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(1, 1));
    
        if(presumed_nr > 1) {
            for(int i=posture_threshold+10; i < 255; i+=1) {//max(1, (i-posture_threshold)*0.25)) {
                action = fn(i);
                if(action == ABORT || action == KEEP_ABORT)
                    break;
            }
        }
    }
    
    int best = INT_MIN;
    if(presumed_nr > 1) {
        if(!best_matches.empty())
            best = best_matches.begin()->first;
        
    } else if(presumed_nr == 1 && (max_objects == 1 || more_than_1_times < 2)) {
        // special case, where an object just has part of a wall inside it
        // just take something in the middle.
        
        auto it = best_matches.begin();
        for (size_t i=0; i<=ceil(best_matches.size() * 0.3) && it != best_matches.end(); i++, ++it)
            best = it->first;
        
    } else {
        bool after_two = false;
        size_t max_size = 0;
        float bestp = 0;
        
        for (auto &p : best_matches) {
            float percent = p.second.ratio * p.second.blobs.front()->num_pixels();
            max_size = max(p.second.blobs.front()->num_pixels(), max_size);
            
            if(max_objects >= 2) {
                if(!after_two && p.second.blobs.size() < 2)
                    continue;
                else if(!after_two)
                    after_two = true;
            }
#if DEBUG_ME
            Debug("Match: %d with ratio %f, %d blobs, %f percent", p.first, p.second.ratio, p.second.blobs.size(), percent);
#endif
            if(percent > bestp) {
                best = p.first;
                bestp = percent;
            }
        }
    }
    
#if DEBUG_ME
    Debug("%ld calculations in %.4fs", calculations, timer.elapsed());
#endif
    
    std::vector<pv::BlobPtr> result;
    if(best != INT_MIN) {
        result = { best_matches.at(best).blobs };
        std::vector<uchar> grey;
        
        for (size_t idx = 0; idx < result.size(); idx++) {
            auto& blob = result.at(idx);
            
            if(!blob->pixels()) {
                grey.clear();
                
                for (auto h : blob->hor_lines()) {
                    for (int x=h.x0; x<=h.x1; x++) {
                        assert(h.y < _original_grey.rows && x < _original_grey.cols);
                        grey.push_back(_original_grey.at<uchar>(h.y, x));
                    }
                }
                
                blob->set_pixels(std::make_unique<const std::vector<uchar>>(std::move(grey)));
                //result.pixels.push_back(grey);
            }
        }
        
        
#if DEBUG_ME
        Debug("Best result with threshold %d", best);
        display_match({best, best_matches.at(best).blobs});
#endif
    } else {
#if DEBUG_ME
        auto str = Meta::toStr(best_matches);
        Warning("Not found %d objects. %S", presumed_nr, &str);
        tf::imshow("original", _original);
        
        /*cv::Mat tmp;
        
        for(int i=0; i<100; i++) {
            cv::threshold(_original, tmp, posture_threshold + i*5, 255, cv::THRESH_BINARY);
            //cv::morphologyEx(tmp, tmp, cv::MORPH_OPEN, element);
            //cv::dilate(tmp, tmp, element);
            
            cv::putText(tmp, std::to_string(posture_threshold + i*5), Vec2(10, 10), cv::FONT_HERSHEY_PLAIN, 0.5, cv::Scalar(255, 255, 255));
            
            cv::imshow("erosion", tmp);
            cv::waitKey();
        }*/
#endif
    }
    
    best_matches.clear();
    return result;
}

#if DEBUG_ME
void SplitBlob::display_match(const std::pair<const int, std::vector<pv::BlobPtr>>& blobs)
{
    ColorWheel wheel;
    cv::Mat copy;
    cv::cvtColor(_original, copy, cv::COLOR_GRAY2BGR);
    
    for (auto& blob : blobs.second) {
        auto clr = wheel.next();
        
        cv::Mat fish;
        auto lines = blob->hor_lines();
        lines2mask(lines, fish);
        
        cv::Mat mask;
        fish.copyTo(mask);
        cv::cvtColor(fish, fish, cv::COLOR_GRAY2BGR);
        fish = clr;
        
        int offx, offy;
        offx = (blob->bounds().width - fish.cols)/2;
        offy = (blob->bounds().height - fish.rows)/2;
        
        int x = blob->bounds().x + offx,
        y = blob->bounds().y + offy;
        if(x < 0) x = 0;
        if(y < 0) y = 0;
        
        int w = fish.cols, h = fish.rows;
        w = min(_original.cols-x, w);
        h = min(_original.rows-y, h);
        
        auto subtmp = copy(cv::Rect(x, y, w, h));
        auto submask = mask(cv::Rect(0, 0, w, h));
        
        // make the fish thinner again
        cv::bitwise_and(submask, _original(cv::Rect(x, y, w, h)), submask);
        fish(cv::Rect(0, 0, w, h)).copyTo(subtmp, submask);
    }
    
    tf::imshow("copy", copy);
//cv::waitKey();
}
#endif

