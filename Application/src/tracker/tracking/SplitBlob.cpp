#include "SplitBlob.h"
#include <types.h>
#include <processing/CPULabeling.h>
#include <tracking/Tracker.h>
#include <misc/Timer.h>
#include <misc/metastring.h>
#include <misc/PixelTree.h>
#include <misc/cnpy_wrapper.h>
#include <tracker/misc/default_config.h>

using namespace track;
using namespace default_config;

namespace track::split {

//! Shorthand for defining slow settings cache entries:
#define DEF_SLOW_SETTINGS(X) using X##_t = Settings:: X##_t; inline static X##_t X
#define DEF_SLOW_SETTINGS_T(T, X) using X##_t = T; inline static X##_t X

//! Parameters that are only saved once per frame,
//! but have faster access than the settings cache.
//! Slower update, but faster access.
struct slow {
    DEF_SLOW_SETTINGS_T(float, blob_split_max_shrink);
    DEF_SLOW_SETTINGS_T(float, blob_split_global_shrink_limit);
    DEF_SLOW_SETTINGS(cm_per_pixel);
    DEF_SLOW_SETTINGS(blob_size_ranges);
    DEF_SLOW_SETTINGS(track_threshold);
    DEF_SLOW_SETTINGS(track_posture_threshold);
    DEF_SLOW_SETTINGS_T(blob_split_algorithm_t::Class, blob_split_algorithm);
};

#undef DEF_SLOW_SETTINGS

//! Slow updated, but faster access:
#define SPLIT_SETTING(NAME) (track::split::slow:: NAME)

}

SplitBlob::SplitBlob(CPULabeling::ListCache_t* cache, const Background& average, const pv::BlobPtr& blob)
    :   max_objects(0),
        _blob(blob),
        _cache(cache)
{
    static const volatile auto _ = []{
#define DEF_CALLBACK(X) if( first ) { \
        SPLIT_SETTING( X ) = map[#X].value<track::split::slow:: X##_t >(); \
        print("Setting", #X, "=", SPLIT_SETTING(X), " @ ", (int*)&SPLIT_SETTING(X)); \
    } else if (name == #X) { \
        SPLIT_SETTING( X ) = value.value<track::split::slow:: X##_t >(); \
        print("Setting", #X, "=", SPLIT_SETTING(X), " @ ", (int*)&SPLIT_SETTING(X)); \
    } void()
        
        auto callback = "SplitBlob";
        
        auto fn = [callback](sprite::Map::Signal signal, sprite::Map& map, const std::string& name, const sprite::PropertyType& value)
        {
            static bool first = true;
            
            if(signal == sprite::Map::Signal::EXIT) {
                map.unregister_callback(callback);
                return;
            }
            
            DEF_CALLBACK(blob_split_max_shrink);
            DEF_CALLBACK(blob_split_global_shrink_limit);
            DEF_CALLBACK(cm_per_pixel);
            DEF_CALLBACK(blob_size_ranges);
            DEF_CALLBACK(track_threshold);
            DEF_CALLBACK(track_posture_threshold);
            
            DEF_CALLBACK(blob_split_algorithm);
            
            if(first)
                first = false;
        };
        
        GlobalSettings::map().register_callback(callback, fn);
        fn(sprite::Map::Signal::NONE, GlobalSettings::map(), "", GlobalSettings::map()["cm_per_pixel"].get());
        
        return 0;
    }(); UNUSED(_);
    
    /*static std::once_flag f;
    std::call_once(f, [&]() {

    });*/
    
    // generate greyscale and mask images
    imageFromLines(blob->hor_lines(), NULL, &_original_grey, &_original, blob->pixels().get(), SPLIT_SETTING(track_posture_threshold), &average.image());
    
    blob->set_tried_to_split(true);
}

size_t SplitBlob::apply_threshold(CPULabeling::ListCache_t* cache, int threshold, std::vector<pv::BlobPtr> &output)
{
    if(_diff_px.empty()) {
        _diff_px.resize(_blob->pixels()->size());
        auto px = _blob->pixels()->data();
        auto out = _diff_px.data();
        auto bg = Tracker::instance()->background();
        //auto grid = Tracker::instance()->grid();
        constexpr LuminanceGrid* grid = nullptr;
        min_pixel = 254;
        max_pixel = 0;
        
        for (auto &line : _blob->hor_lines()) {
            for (auto x=line.x0; x<=line.x1; ++x, ++px, ++out) {
                *out = (uchar)saturate(float(bg->diff(x, line.y, *px)) / (grid ? float(grid->relative_threshold(x, line.y)) : 1.f));
                if(*out < min_pixel)
                    min_pixel = *out;
                if(*out > max_pixel)
                    max_pixel = *out;
            }
        }
        
        threshold = max(threshold, (int)min_pixel);
    }
    
    output = pixel::threshold_blob(*cache, _blob, _diff_px, threshold);
    
    for(auto &blob: output)
        blob->add_offset(-_blob->bounds().pos());
    
    std::sort(output.begin(), output.end(),
       [](const pv::BlobPtr& a, const pv::BlobPtr& b) { 
            return std::make_tuple(a->pixels()->size(), a->blob_id()) > std::make_tuple(b->pixels()->size(), b->blob_id()); 
       });
    
    return output.empty() ? 0 : (*output.begin())->pixels()->size();
}

/**
 * This function evaluates the result of `apply_threshold`, looking for acceptable configurations of blobs given the expected number of objects and relative sizes.
 *
 * It expects the blobs to be given in descending order of size.
 *
 * The rules are enforced in this order:
 *  1. given `presumed_number` of expected objects, are enough objects found?
 *  2. the `num_px * sqcmppx` of each object must have a similar value as the preceeding and following one, up to the number of `presumed_number` objects down. this is enforced by finding the minimum ratio between focal and following object sizes - which is a ratio between [0,1] - and comparing to 0.3/0.4 depending on the algorithm. if the min ratio is smaller than e.g. 0.3, then the objects are of vastly different sizes where we would expect similar sized objects.
 *  3. objects have to be larger than a minimum threshold, which is the `blob_size_ranges.max_range[0] * global_shrink_limit`. anything (within objects `0..presumed_nr`) that's smaller than that is not considered a valid object
 *  4. the size of the object that we started out with, before splitting, cannot be shrunk further than `blob_split_max_shrink * first_size`. so each object has to be larger than that divided by the number of expected blobs in order to form the min number of objects required.
 */
split::Action_t SplitBlob::evaluate_result_multiple(size_t presumed_nr, float first_size, std::vector<pv::BlobPtr>& blobs)
{
    const float sqrcm = SQR(SPLIT_SETTING(cm_per_pixel));
    size_t pixels = 0;
    std::optional<size_t> min_size;
    for(size_t i=0; i < blobs.size(); ++i) {
        pixels += blobs.at(i)->num_pixels();
    }
    
    if(pixels * sqrcm < SPLIT_SETTING(blob_split_max_shrink) * first_size) {
        return split::Action::ABORT;
    }
    
    const float min_size_threshold = SPLIT_SETTING(blob_size_ranges).max_range().start * SPLIT_SETTING(blob_split_global_shrink_limit);
    auto it = std::remove_if(blobs.begin(), blobs.end(), [&](const pv::BlobPtr& blob) {
        auto fsize = blob->num_pixels() * sqrcm;
        return fsize < min_size_threshold;
    });
    blobs.erase(it, blobs.end());
    
    for(size_t i=0; i<presumed_nr && i < blobs.size(); ++i) {
        if(!min_size.has_value() || blobs.at(i)->num_pixels() < min_size.value()) {
            min_size = blobs.at(i)->num_pixels();
        }
    }
    
    if(min_size.has_value()
       && min_size.value() * sqrcm > SPLIT_SETTING(blob_size_ranges).max_range().end)
    {
        return split::Action::REMOVE;
    }
    
    if(blobs.size() < presumed_nr) {
        return split::Action::TOO_FEW;
    }
    
    return split::Action::KEEP_ABORT;
}

struct Run {
    int count{0};
    std::atomic<int> best{-1};
    int found_in_step{-1};
    robin_hood::unordered_flat_map<int, split::Action_t> results;
    std::vector<std::tuple<int, split::Action_t>> tried;
    int lowest_non_remove{-1}, highest_remove{-1};
    std::mutex mutex;
    
    bool has_best() const {
        return best != -1;
    }
    
    split::Action_t perform(pv::bid bdx, CPULabeling::ListCache_t* cache, int threshold, int step, auto&& F, auto... args)
    {
        {
            std::unique_lock guard(mutex);
            //if(best != -1 && best < threshold)
            //    return split::Action::ABORT;
            
            if(results.contains(threshold)) {
                auto action = results.at(threshold);
                if(action == split::Action::KEEP_ABORT
                   && (best == -1 || best > threshold))
                {
                    //print("Somehow found this in ", threshold, " in the map.");
                    best = threshold;
                    found_in_step = step;
                }
                
                /*if(action != split::Action::REMOVE
                   && (lowest_non_remove == -1 || threshold < lowest_non_remove)) {
                    lowest_non_remove = threshold;
                } else if(action == split::Action::REMOVE) {
                    if(threshold > highest_remove) {
                        highest_remove = threshold;
                    }
                }*/
                
                return action;
            }
        }
        
        auto action = F(cache, threshold, args...);
        
        
        std::unique_lock guard(mutex);
        if((action == split::Action::KEEP
           || action == split::Action::KEEP_ABORT)
           && (best == -1 || best > threshold))
        {
            assert(best == -1 || best > threshold);
            best = threshold;
            found_in_step = step;
            //print("Found best for ", bdx, " in ", threshold, " step=", step);
        }
        
        results[threshold] = action;
        //tried.push_back({threshold, action});
        /*if(action != split::Action::REMOVE
           && (lowest_non_remove == -1 || threshold < lowest_non_remove)) {
            lowest_non_remove = threshold;
        } else if(action == split::Action::REMOVE) {
            if(threshold > highest_remove) {
                highest_remove = threshold;
            }
        }*/
        
        ++count;
        return action;
    }
};

void commit_run(pv::bid bdx, const Run& naive, const Run& next) {
    static std::atomic<int64_t> thresholds = 0, samples_naive{0};
    static std::atomic<int64_t> samples = 0;
    
    thresholds += next.count;
    ++samples;
    samples_naive += naive.count;
    
    static std::atomic<size_t> matches{0}, mismatches{0};
    static std::atomic<int64_t> offsets{0}, max_offset{0}, not_found{0}, would_find{0}, second_try{0}, preproc{0}, third{0}, count_from_not_found{0};
    static std::mutex m;
    static std::map<int64_t, size_t> often;
    
    if(next.found_in_step == 2)
        ++second_try;
    if(next.found_in_step == 0)
        ++preproc;
    if(next.found_in_step >= 3)
        ++third;
    
    if(naive.best == -1) {
        count_from_not_found += next.count;
    }
    
    if(naive.best != next.best) {
        
        
        if(next.best != -1 && naive.best != -1) {
            if(std::abs(next.best - naive.best) > 2) {
                print(bdx, " Naive count: ", naive.count, " (t=",naive.best.load(),") vs ", next.count, " (t=",next.best.load(),") and map ", next.tried, " vs ", naive.tried);
            }
                auto offset = std::abs(next.best - naive.best);
                offsets += offset;
                max_offset = max(max_offset.load(), offset);
                ++mismatches;
                
                std::unique_lock guard(m);
                ++often[next.best - naive.best];
            //} else {
                // slight deviation is okay
             //   ++matches;
            //}
            
        } else if(next.best == -1) {
            print(bdx, " Not found count: ", naive.count, " (t=",naive.best.load(),") vs ", next.count, " (t=",next.best.load(),") and map ", next.tried, " vs ", naive.tried);
            ++not_found;
            ++mismatches;
        }
        
    } else
        ++matches;
    
    if(samples.load() % 100 == 0) {
        auto m0 = matches.load(), m1 = mismatches.load();
        auto off = offsets.load();
        print("Samples: ", float(thresholds) / float(samples), " (vs. ", float(samples_naive.load()) / float(samples.load()),") ", float(m1) / float(m0 + m1) * 100, "% mismatches (avg. ", float(off) / float(m1), " offset for ", m1," elements) ", float(not_found.load()) / float(m1) * 100, "% not found, ", float(would_find.load()) / float(not_found.load()) * 100, "% could have been found, ", float(second_try.load()) / float(samples.load()) * 100, "% found in second try, ", float(preproc.load()) / float(samples.load()) * 100, "% found in preprocess ", float(third.load()) / float(samples.load()) * 100, "% found in third try, ", float(count_from_not_found.load()) / float(thresholds.load()) * 100, "% from not found objects");
        std::unique_lock guard(m);
        print(often);
    }
}

template<typename F, typename Iterator, typename Pool>
void distribute_indexes(F&& fn, Pool& pool, Iterator start, Iterator end) {
    const auto threads = pool.num_threads();
    int64_t i = 0, N = end - start;
    const int64_t per_thread = max(1, int64_t(N) / int64_t(threads));
#if defined(COMMONS_HAS_LATCH) //&& false
    int64_t enqueued{0};
    
    {
        Iterator nex = start;
        int64_t i = 0;
        
        for(auto it = start; it != end;) {
            auto step = (i + per_thread) < (N - per_thread) ? per_thread : (N - i);
            nex += step;
            if(nex != end) {
                ++enqueued;
            }
            
            it = nex;
            i += step;
        }
    }
    
    std::latch work_done{static_cast<ptrdiff_t>(enqueued)};
    Iterator nex = start;
    std::exception_ptr ex;
    
    size_t j=0;
    for(auto it = start; it != end; ++j) {
        auto step = (i + per_thread) < (N - per_thread) ? per_thread : (N - i);
        nex += step;
        if(nex != end) {
            pool.enqueue([&](auto i, auto it, auto nex, auto step, auto index) {
                try {
                    fn(i, it, nex, step, index);
                } catch(...) {
                    ex = std::current_exception();
                }
                work_done.count_down();
                
            }, i, it, nex, step, j);
            
        } else {
            try {
                // run in local thread
                fn(i, it, nex, step, j);
            } catch(...) {
                ex = std::current_exception();
            }
        }
        
        it = nex;
        i += step;
    }
    
    work_done.wait();
    if(ex)
        std::rethrow_exception(ex);
#else
    std::atomic<int64_t> processed(0);
    int64_t enqueued{0};
    
    {
        Iterator nex = start;
        int64_t i = 0;
        
        for(auto it = start; it != end;) {
            auto step = (i + per_thread) < (N - per_thread) ? per_thread : (N - i);
            assert(step > 0);
            
            nex += step;
            if(nex != end)
                ++enqueued;
            
            it = nex;
            i += step;
        }
    }
    
    Iterator nex = start;
    for(auto it = start; it != end;) {
        auto step = (i + per_thread) < (N - per_thread) ? per_thread : (N - i);
        nex += step;
        
        if(nex == end) {
            fn(i, it, nex, step);
            
        } else {
            pool.enqueue([&](auto i, auto it, auto nex, auto step) {
                fn(i, it, nex, step);
                ++processed;
                
            }, i, it, nex, step);
        }
        
        it = nex;
        i += step;
    }
    
    while(processed < enqueued) { }
#endif
}

std::vector<pv::BlobPtr> SplitBlob::split(size_t presumed_nr, const std::vector<Vec2>& centers)
{
    //std::map<int, ResultProp> best_matches;
    ResultProp best_match;
    
    size_t calculations = 0;
    Timer timer;
    float first_size = 0;
    size_t more_than_1_times = 0;
    
    const float sqrcm = SQR(SPLIT_SETTING(cm_per_pixel));
    
    const auto apply_watershed = [this](const std::vector<Vec2>& centers, std::vector<pv::BlobPtr>& output) {
        static const cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * 1 + 1, 2 * 1 + 1), cv::Point(1, 1));

        auto [offset, img] = _blob->image();
        cv::Mat mask = cv::Mat::zeros(img->rows, img->cols, CV_32SC1);
        for(size_t i = 0; i < img->size(); ++i) {
            if(img->data()[i] == 0)
                mask.ptr<int>()[i] = 1;
        }

        size_t i = 0;
        for(auto c : centers) {
            cv::circle(mask, c, 5, i + 2, cv::FILLED);
            cv::circle(mask, c, 5, 0, 1);
            ++i;
        }

        cv::Mat tmp;
        //mask.convertTo(tmp, CV_8UC1, 255.f / float(centers.size() + 2));

        //tf::imshow("labels1", tmp);
        cv::cvtColor(img->get(), tmp, cv::COLOR_GRAY2BGR);
        cv::watershed(tmp, mask);

        //mask.convertTo(tmp, CV_8UC1, 255.f / float(centers.size() + 1));
        //tf::imshow("labels", tmp);
        //tf::imshow("image", img->get());

        //cv::subtract(mask, 1, mask);
        mask.convertTo(tmp, CV_8UC1);
        cv::threshold(tmp, tmp, 1, 255, cv::THRESH_BINARY);
        //tf::imshow("thresholded", tmp);

        cv::erode(tmp, tmp, element);
        img->get().copyTo(tmp, tmp);

        {
            auto detections = CPULabeling::run(tmp, *_cache);
            //print("Detections: ", detections.size());

            output.clear();
            for(auto&& [lines, pixels, flags] : detections) {
                output.emplace_back(pv::Blob::make(std::move(lines), std::move(pixels), flags));
                //output.back()->add_offset(-_blob->bounds().pos());
            }
        }

        std::sort(output.begin(), output.end(),
            [](const pv::BlobPtr& a, const pv::BlobPtr& b) {
                return std::make_tuple(a->pixels()->size(), a->blob_id()) > std::make_tuple(b->pixels()->size(), b->blob_id());
            });


        //resize_image(tmp, 5, cv::INTER_NEAREST);
        //cv::cvtColor(tmp, tmp, cv::COLOR_GRAY2BGR);

        i = 0;
        ColorWheel wheel;
        for(auto& d : output) {
            auto c = wheel.next();
            for(auto& l : *d->lines())
                cv::line(tmp, (Vec2(l.x0, l.y) + 0.5) * 5, (Vec2(l.x1, l.y) + 0.5) * 5, c, 5);
            //pv::Blob b(std::move(d.lines), std::move(d.pixels));
            //auto [p, img] = b.image();
            //auto [p, img] = d->image();
            //tf::imshow("blob" + Meta::toStr(i), img->get());
            ++i;
        }
        //tf::imshow("blobs", tmp);

        return output.empty() ? 0 : (*output.begin())->pixels()->size();
    };
    
    float max_size;
    const auto min_threshold = (SLOW_SETTING(calculate_posture) ? max(SLOW_SETTING(track_threshold), SPLIT_SETTING(track_posture_threshold)) : SLOW_SETTING(track_threshold)) + 1;

    const auto fn = [&](CPULabeling::ListCache_t* cache, int threshold, bool save)
    {
        std::vector<pv::BlobPtr> blobs;
        calculations++;
        
        bool initial = threshold == -1;
        if(initial)
            threshold = min_threshold;
        
        if(SPLIT_SETTING(blob_split_algorithm) == blob_split_algorithm_t::threshold)
            max_size = apply_threshold(cache, threshold, blobs) * sqrcm;
        else
            max_size = (initial ? apply_threshold(cache, threshold, blobs) : apply_watershed(centers, blobs)) * sqrcm;
        
        // cant find any blobs anymore...
        //if(blobs.empty())
        //    return split::Action::ABORT;
        
        // save the maximum number of objects found
        max_objects = max(max_objects, blobs.size());
        
        ResultProp result;
        result.threshold = threshold;
        
        auto action = evaluate_result_multiple(presumed_nr, first_size, blobs);
        
        if(first_size == 0)
            first_size = max_size;
        
        if(blobs.size() > 1)
            more_than_1_times++;
        
        // we found enough blobs, so we're allowed to keep it
        if(action == split::Action::KEEP
           || action == split::Action::KEEP_ABORT)
        {
            if(save) {
                std::unique_lock guard(mutex);
                if(best_match.threshold == -1
                   || result.threshold < best_match.threshold)
                {
                    result.blobs = std::move(blobs);
                    best_match = std::move(result);
                }
            }
            //print(" inserting ", threshold);
        }

        //blobs.clear();
        return action;
    };
    
    auto action = fn(_cache, -1, true);
    
    if(action != split::Action::KEEP
       && action != split::Action::KEEP_ABORT
        && _blob->pixels()->size() * sqrcm < SPLIT_SETTING(blob_size_ranges).max_range().end * 100)
    {
        
        if(presumed_nr > 1) {
            if(SPLIT_SETTING(blob_split_algorithm) == blob_split_algorithm_t::threshold) {
                
                // start in the center
                // 51, half = 26
                /*
                 t = 26
                 fn(26) = ABORT => half = half * 0.5 + 0.5 = 13; t -= half
                 
                 fn(13) = KEEP_ABORT => 0.5half = 7; t -= half = 6;
                 fn(6) = REMOVE => 0.5half = 4; t += half = 10;
                 fn(10) = KEEP_ABORT => 0.5half = 2; t -= half = 8
                 fn(8) = REMOVE => 0.5half = 1; t += half = 9
                 fn(9) = REMOVE
                 
                 -------------
                 >> use fn(10) since it was the smallest possible threshold
                 
                 RULES:
                    1. IF ABORT => LOWER
                    2. IF REMOVE => INCREASE
                    3. IF KEEP_ABORT => try a lower threshold
                 */
                
                Run run;
                
                //if(false)
                /*{
                    static constexpr std::array<int, 3> offsets {
                        1, 8, 16
                    };
                    
                    const float normalization = 1.f;
                    //const float normalization = 1.f / 64.f * (254.f - (min_threshold + 1));
                    
                    //print("starting at ", posture_threshold + 1, " with offsets ", offsets, " and normalize ", normalization);
                    for (const auto o : offsets) {
                        const auto t = saturate(int(min_threshold + o * std::max(1.f, normalization)), min_threshold, 254);
                        auto action = run.perform(t, 0, fn, true);
                        
                        if(action == split::Action::ABORT
                           || action == split::Action::KEEP_ABORT)
                        {
                            break;
                        }
                    }
                }*/
                
                const int begin_threshold = max(min_threshold, (int)min_pixel);
                
                /*int begin = min(254, begin_threshold);
                int end = run.best == -1 ? max_pixel : (run.best - 1);
                auto half = int((end - begin) * 0.5 + 0.5);
                int t = begin + half;
                int minimal = begin_threshold, maximal = max_pixel;
                int avoided = 0;*/
                
                //print("Solving ", *_blob, " starting at ", t, " begin=",begin," end=",end);
                
                /*while ((run.best == -1 || run.best > min_threshold)
                       && half >= 1)
                {
                    auto action = run.perform(t, 1, fn, true);
                    //print(*_blob, "@t=",t, " half=",half," => ", action, " max_size=",max_size, " (", run.best, " ", run.count,")");
                    
                    if(action == split::Action::ABORT
                       || action == split::Action::KEEP_ABORT
                       || action == split::Action::KEEP
                       || action == split::Action::SKIP)
                    {
                        if(half == 1)
                            half = 0;
                        else
                            half = int(half * 0.5);
                        
                        t -= half;
                        
                    } else if(action == split::Action::REMOVE) {
                        if(half == 1)
                            half = 0;
                        else
                            half = int(half * 0.5);
                        t += half;
                        
                    } else if(action == split::Action::TOO_FEW) {
                        if(half == 1)
                            half = 0;
                        else {
                            half = int(half * 0.5);
                            //q.push({t - half, half});
                        }
                        t += half;
                        
                    } else
                        throw U_EXCEPTION("Invalid action ", action);
                }*/
                
                static constexpr bool accurate = false;
                /*uint8_t max_pixel = 254, min_pixel = min_threshold;
                
                auto [fit, eit] = std::minmax_element(_diff_px.begin(), _diff_px.end());
                if(fit != _blob->pixels()->end()) {
                    max_pixel = max(min_threshold, (int)*eit);
                    min_pixel = max(min_threshold, (int)*fit);
                }*/
                
                if(run.best == -1) {
                    
                    /*auto search_range = [this, &fn, &minimal, &maximal, &avoided](const arange<int> range, Run& run, const int step, const int offset, const int index = 1)
                    {
                        //print("Searching ", range.first, " to ", range.last, " with step ", step, " and offset ", offset, " minimal=", minimal, " and maximal=", maximal);
                        
                        for(int i = range.first + offset; i < range.last; i += step)
                        {
                            auto action = run.perform(_blob->blob_id(), _cache, i, index, fn, true);
                            //if(i < minimal || i > maximal)
                            //    ++avoided;
                            
                            if(action == split::Action::ABORT
                               || action == split::Action::KEEP_ABORT)
                            {
                                return;
                            }
                        }
                    };*/
                    
                    //search_range(arange(min_threshold, maximal), run, 50, 0, 0);
                    
                    const int step = 4;
                    //run.best = -1;
                    int resolution = step;

                    constexpr size_t num_threads = 3;
                    std::latch latch{ptrdiff_t(num_threads)}, end_latch{ptrdiff_t(num_threads)};
                    
                    const float distance = float(max_pixel - begin_threshold);
                    const float _step = distance / float(num_threads);

                    static GenericThreadPool threads(9, "Thresholds", nullptr);
                    static std::queue<CPULabeling::ListCache_t*> caches;
                    static std::mutex clock;

                    struct Guard {
                        CPULabeling::ListCache_t* c{ nullptr };
                        Guard() {
                            std::unique_lock guard(clock);
                            if(caches.empty()) {
                                c = new CPULabeling::ListCache_t;
                            } else {
                                c = caches.front();
                                caches.pop();
                            }
                        }
                        ~Guard() {
                            std::unique_lock guard(clock);
                            caches.push(c);
                        }
                    };
                    
                    if(distance < num_threads || _blob->num_pixels() < 5000) {
                        //print("Distance:", distance, " < ", num_threads);
                        int max_detected = max_pixel;
                        
                        for(auto i = begin_threshold; i < max_detected; i+=3)
                        {
                            auto action = run.perform(_blob->blob_id(), _cache, i, 0, fn, true);
                            
                            if(action == split::Action::ABORT
                               || action == split::Action::KEEP_ABORT)
                            {
                                max_detected = i;
                                break;
                            }
                        }
                        
                        if(!run.has_best()) {
                            for(auto i = begin_threshold + 1; i < run.best.load(); i+=3)
                            {
                                auto action = run.perform(_blob->blob_id(), _cache, i, 2, fn, true);
                                
                                if(action == split::Action::ABORT
                                   || action == split::Action::KEEP_ABORT)
                                {
                                    max_detected = i;
                                    break;
                                }
                            }
                        }
                        
                        if(!run.has_best()) {
                            for(auto i = begin_threshold + 2; i < run.best.load(); i+=3)
                            {
                                auto action = run.perform(_blob->blob_id(), _cache, i, 3, fn, true);
                                
                                if(action == split::Action::ABORT
                                   || action == split::Action::KEEP_ABORT)
                                {
                                    break;
                                }
                            }
                        }
                        
                        if(false && run.has_best()) {
                            auto best = run.best.load();
                            for(auto i = best - 4; i < best; ++i)
                            {
                                if(i <= min_pixel)
                                    continue;
                                
                                auto action = run.perform(_blob->blob_id(), _cache, i, 2, fn, true);
                                
                                if(action == split::Action::ABORT
                                   || action == split::Action::KEEP_ABORT)
                                {
                                    break;
                                }
                            }
                        }
                    }
                    
                    else {
                        distribute_indexes([&](auto, auto start, auto end, auto, auto j)
                            {
                                Guard guard;
                            
                            end = begin_threshold + int(_step * (start + 1) + 0.5);
                            start = begin_threshold + int(_step * start);
                            
                            if(distance == 0 && j == 0) {
                                end++;
                            }
                            
                            int minimal = start;
                            
                            //CPULabeling::ListCache_t cache;
                            
                            const int step = max(1, int(_step * 0.2));
                            //print(_blob->blob_id(), " ",i, ": ", start, "-", end, " step:", _step, " pixels:", Range<int>(min_pixel, max_pixel), " distance:", distance, " _step:", _step, " recounted at:", _blob->last_recount_threshold(), " step:",step);
                            
                            for(auto o=0; o<step; ++o) {
                                //if(run.has_best() && run.best.load() <= start + o)
                                //    return;
                                
                                for(auto i = start + o; i < end; i+=step)
                                {
                                    if(i <  minimal)
                                        continue;
                                    
                                    auto action = run.perform(_blob->blob_id(), guard.c, i, 0, fn, true);
                                    
                                    if(action == split::Action::ABORT
                                       || action == split::Action::KEEP_ABORT)
                                    {
                                        break;
                                    }
                                    
                                    
                                }
                                //print("i:", i, " start:", " end:");
                                
                                if(o == 0) {
                                    //print("Arrived ", j, " ", latch.max());
                                    latch.arrive_and_wait();
                                }
                                
                                if(run.has_best()) {
                                    auto best = run.best.load();
                                     minimal = max((int)begin_threshold, (best - ((best - begin_threshold) % step)) - step);
                                     end = max((int)begin_threshold, best);
                                     //if(minimal < start + o)
                                     //break;
                                    //end = min(run.best.load(), end);
                                    //print(_blob->blob_id(), " found run.best=", run.best.load(), " in ", end, " o:",o, " start:", start);
                                    if(end <= start + o)
                                        break;
                                }
                            }
                            
                            //if(!run.has_best())
                            /*    end_latch.arrive_and_wait();
                            //else
                            //    end_latch.count_down();
                            
                            if(run.has_best()) {
                                float width = step / float(num_threads) + 1;
                                int t = run.best.load() - 1;
                                int start = max((int)min_pixel, t - (j + 1) * width);
                                int end = saturate(t - (j) * width, (int)min_pixel, (int)t);
                                
                                //print(_blob->blob_id(), " third stage from ", start, "-", end, " in ", j, " width=",width, " best=", t+1);
                                
                                for(int i = start; i < end; i+=2) {
                                    if(run.best.load() < i)
                                        break;
                                    
                                    run.perform(_blob->blob_id(), guard.c, i, 3, fn, true);
                                }
                            }*/
                            
                        }, threads, 0, (int)num_threads+1);
                    }
                    
                    /*for(int o = 1; o < step; ++o) {
                        //maximal = run.best == -1 ? 254 : run.best;
                        //if(run.lowest_non_remove != -1)
                        //    minimal = max(min_threshold, run.lowest_non_remove - step);
                        //step = int((maximal - min_threshold) * 0.015) + 1;
                        resolution = step - o;
                        int best = run.best;
                        search_range(arange(minimal, maximal), run, step, o);
                        
                        if(run.best != -1) {
                            if constexpr(accurate)
                                break;
                            else
                                maximal = min(run.best.load(), maximal);
                        }
                        //minimal = max(min_threshold, run.lowest_non_remove + 1 - step * 2);
                        
                        if(!accurate) {
                            if(best == -1 && run.best != -1) {
                                minimal = max(min_threshold, (run.best - ((run.best - min_threshold) % step)) - step);
                                maximal = max(min_threshold, run.best.load());
                            }
                        }
                    }*/
                    
                    /*if(accurate && run.best != -1) {
                        for(int i=minimal; i<run.best; ++i) {
                            auto action = run.perform(_cache, i, 3, fn, true);
                            //if(i < minimal || i > maximal)
                            //    ++avoided;
                            
                            if(action == split::Action::ABORT
                               || action == split::Action::KEEP_ABORT)
                            {
                                break;
                            }
                        }
                    }*/
                    
                    //if(run.best != -1 && resolution > 0)
                    //    search_range(arange(run.best - resolution, run.best), run, 1, 0);
                }
                
                /*if(run.best != -1) {
                    for(int i = max((int)min_pixel, run.best - 3); i < max_pixel && i < run.best; i+=1)
                    {
                        auto action = run.perform(_blob->blob_id(), _cache, i, 3, fn, true);
                        //print("##",*_blob, "@t=",i," => ", action, " max_size=",max_size, " (", run.best, " ", run.count,")");
                        if(action == split::Action::ABORT
                           || action == split::Action::KEEP_ABORT)
                        {
                            break;
                        }
                    }
                }*/
                
                /*Run naive;
                for(int i = begin_threshold; i < 254; ++i)
                {
                    auto action = naive.perform(_blob->blob_id(), _cache, i, 0, fn, false);
                    if(action == split::Action::ABORT
                       || action == split::Action::KEEP_ABORT)
                    {
                        break;
                    }
                }
                
                //naive.best = max(naive.best.load(), (int)min_pixel);
                commit_run(_blob->blob_id(), naive, run);*/
            }
            else
                action = fn(_cache, 0, true);
        }
    }
    
    if(best_match.threshold != -1) {
        std::vector<uchar> grey;
        
        for (auto& blob : best_match.blobs) {
            if(!blob->pixels()) {
                grey.clear();
                int32_t N = 0;
                for (auto& h : blob->hor_lines()) {
                    auto n = int32_t(h.x1) - int32_t(h.x0) + 1;
                    auto ptr = _original_grey.ptr(h.y, h.x0);
                    grey.resize(N + n);
                    std::copy(ptr, ptr + n, grey.data() + N);
                    N += n;
                }
                
                blob->set_pixels(std::make_unique<std::vector<uchar>>(std::move(grey)));
                //result.pixels.push_back(grey);
            }
        }
        
        return best_match.blobs;
    }
    
    //! could not find anything
    return {};
}
