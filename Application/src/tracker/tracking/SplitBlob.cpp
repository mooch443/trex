#include "SplitBlob.h"
#include <types.h>
#include <processing/CPULabeling.h>
#include <tracking/Tracker.h>
#include <misc/Timer.h>
#include <misc/metastring.h>
#include <misc/PixelTree.h>
#include <misc/cnpy_wrapper.h>
#include <tracker/misc/default_config.h>

//#define TREX_SPLIT_DEBUG
//#define TREX_SPLIT_DEBUG_TIMINGS

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
    //! Settings initializer:
    static const auto _ = []{
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
 * This function evaluates the result of `apply_threshold`, looking for acceptable configurations of blobs given the expected number of objects and their sizes.
 *
 * It expects the blobs to be given in descending order of size.
 *
 * The rules are enforced in this order:
 *  1. given `presumed_number` of expected objects, are enough objects found?
 *  2. the overall number of pixels should not shrink further than `blob_split_max_shrink * first_size`, i.e. a certain percentage of the unthresholded image
 *  3. all objects smaller than `blob_size_ranges.max.start * blob_split_global_shrink_limit` are removed
 *  4. if the smallest found object is bigger than `blob_size_ranges.max.end`, ignore the results
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

template<bool thread_safe>
struct Run {
    int count{0};
    std::atomic<int> best{-1};
    int found_in_step{-1};
    robin_hood::unordered_flat_map<int, split::Action_t> results;
    std::vector<std::tuple<int, split::Action_t>> tried;
    std::mutex mutex;
    
    bool has_best() const {
        return best != -1;
    }
    
    template<bool TF = thread_safe>
        requires TF
    split::Action_t from_cache(int threshold) {
        std::unique_lock guard(mutex);
        return _unsafe_from_cache(threshold);
    }
    
    template<bool TF = thread_safe>
        requires (!TF)
    split::Action_t from_cache(int threshold) {
        return _unsafe_from_cache(threshold);
    }
    
    split::Action_t _unsafe_from_cache(int threshold) {
        auto it = results.find(threshold);
        if(it != results.end()) {
            return it->second;
        }
        return split::Action::NO_CHANCE;
    }
    
    template<bool TF=thread_safe>
        requires TF
    void add_result(split::Action_t action, int threshold) {
        std::unique_lock guard(mutex);
        _unsafe_add_result(action, threshold);
    }
    
    template<bool TF=thread_safe>
        requires (!TF)
    void add_result(split::Action_t action, int threshold) {
        _unsafe_add_result(action, threshold);
    }
    
    void _unsafe_add_result(split::Action_t action, int threshold) {
        results[threshold] = action;
#ifdef TREX_SPLIT_DEBUG
        tried.push_back({threshold, action});
#endif
    }
    
    bool check_viable_option(split::Action_t action, int threshold, int step) {
        if(action == split::Action::KEEP
           || action == split::Action::KEEP_ABORT)
        {
            if(best == -1 || threshold < best) {
                best = threshold;
                found_in_step = step;
                
                add_result(action, threshold);
            }
            
            return true;
        }
        
        return false;
    }
    
    split::Action_t perform(CPULabeling::ListCache_t* cache, int threshold, int step, auto&& F, auto... args)
    {
        auto action = from_cache(threshold);
        if(action == split::Action::NO_CHANCE) {
            action = F(cache, threshold, args...);
            ++count;
        }
        
        check_viable_option(action, threshold, step);
        return action;
    }
};

#ifdef TREX_SPLIT_DEBUG
template<bool thread_safe>
void commit_run(pv::bid bdx, const Run<false>& naive, const Run<thread_safe>& next)
{
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
            //if(std::abs(next.best - naive.best) > 2)
            {
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
#endif

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
                
                const int begin_threshold = max(min_threshold, (int)min_pixel);
                static constexpr bool accurate = false;
                
                constexpr size_t num_threads = 3;
                
                const float distance = float(max_pixel - begin_threshold);
                const float _step = distance / float(num_threads);
                
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
                
                const int segments = 3;
                
                auto work = [&](auto cache, auto& run, auto j)
                {
                    const int step = segments * 2;
                    int start = begin_threshold;
                    int end = max_pixel;
                    
                    Range<int> first_stage(start,
                                           start + int((end - start) * 0.3) + 1);
                    //print("Thread ", j, "/", num_threads, " => ", step, " ", Range<int>(start, end), " first:", first_stage);
                    
                    for(auto i = first_stage.start + j * 2; i < first_stage.end; i += step)
                    {
                        auto action = run.perform(cache, i, 0, fn, true);
                        
                        if(action == split::Action::ABORT
                           || action == split::Action::KEEP_ABORT)
                        {
                            if(action == split::Action::KEEP_ABORT)
                                return;
                            break;
                        }
                        
                        if(run.has_best() && i >= run.best) {
                            break;
                        }
                    }
                    
                    if(!run.has_best()) {
                        for(auto i = first_stage.start + j * 2 + 1; i < first_stage.end; i+=step)
                        {
                            if(run.has_best() && i >= run.best) {
                                break;
                            }
                            
                            auto action = run.perform(cache, i, 1, fn, true);
                            
                            if(action == split::Action::ABORT
                               || action == split::Action::KEEP_ABORT)
                            {
                                if(action == split::Action::KEEP_ABORT)
                                    return;
                                break;
                            }
                        }
                    }
                    
                    if(!run.has_best()) {
                        for(auto i = first_stage.end + j; i < end; i+=step / 2)
                        {
                            if(run.has_best() && i >= run.best) {
                                break;
                            }
                            
                            auto action = run.perform(cache, i, 3, fn, true);
                            
                            if(action == split::Action::ABORT
                               || action == split::Action::KEEP_ABORT)
                            {
                                if(action == split::Action::KEEP_ABORT)
                                    return;
                                break;
                            }
                        }
                    }
                };
                
                
#ifdef TREX_SPLIT_DEBUG
                Run<false> naive;
                for(int i = begin_threshold; i < 254; ++i)
                {
                    auto action = naive.perform(_cache, i, 0, fn, false);
                    if(action == split::Action::ABORT
                       || action == split::Action::KEEP_ABORT)
                    {
                        break;
                    }
                }
#endif
                
                static std::map<int, std::tuple<uint64_t, double>> tens_times, tens_times_thread;
                static std::mutex mutex;
                
                if(distance < num_threads || _blob->num_pixels() < 16000)
                {
                    Timer timer;
                    Run<false> run;
                    
                    work(_cache, run, 0);
                    if(!run.has_best())
                        work(_cache, run, 1);
                    if(!run.has_best())
                        work(_cache, run, 2);
                    
#ifdef TREX_SPLIT_DEBUG
                    commit_run(_blob->blob_id(), naive, run);
#endif
                } else {
                    Timer timer;
                    static GenericThreadPool threads(9, "Thresholds", nullptr);
                    Run<true> run;
                    
                    //! Counts down for all threads, to synchronize
                    //! the first stage of the algorithm.
                    //! This reduces the number of misses.
                    //std::latch latch{ptrdiff_t(num_threads)};
                    distribute_indexes([&](auto, auto, auto, auto j) {
                        //! protects the usage of CPULabeling caches
                        //! via RAII
                        const Guard guard{};
                        work(guard.c, run, j);
                        
                    }, threads, 0, (int)segments + 1);
                    
#ifdef TREX_SPLIT_DEBUG
                    commit_run(_blob->blob_id(), naive, run);
#endif
                }
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
