#include "SplitBlob.h"
#include <processing/CPULabeling.h>
#include <tracking/Tracker.h>
#include <misc/Timer.h>

#include <misc/PixelTree.h>
#include <misc/cnpy_wrapper.h>
#include <tracker/misc/default_config.h>

//#define TREX_SPLIT_DEBUG
//#define TREX_SPLIT_DEBUG_TIMINGS

using namespace track;
using namespace default_config;

namespace track::split {

//! secures the CPULabeling caches while in use
//! + queue & dequeue and reusing
struct Guard {
    static auto& caches() {
        static std::queue<CPULabeling::ListCache_t*> c;
        return c;
    }
    static auto& clock() {
        static auto m = LOGGED_MUTEX("track::split::clock");
        return m;
    }
    
    CPULabeling::ListCache_t* c{ nullptr };
    Guard() {
        auto guard = LOGGED_LOCK(clock());
        if(caches().empty()) {
            c = new CPULabeling::ListCache_t;
        } else {
            c = caches().front();
            caches().pop();
        }
    }
    ~Guard() {
        auto guard = LOGGED_LOCK(clock());
        caches().push(c);
    }
};


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
    DEF_SLOW_SETTINGS(track_size_filter);
    DEF_SLOW_SETTINGS(track_threshold);
    DEF_SLOW_SETTINGS(track_posture_threshold);
    DEF_SLOW_SETTINGS_T(blob_split_algorithm_t::Class, blob_split_algorithm);
};

#undef DEF_SLOW_SETTINGS

//! Slow updated, but faster access:
#define SPLIT_SETTING(NAME) (track::split::slow:: NAME)

}

SplitBlob::SplitBlob(CPULabeling::ListCache_t* cache, const Background& average, pv::BlobWeakPtr blob)
    :   max_objects(0),
        _blob(blob),
        _cache(cache)
{
    //! Settings initializer:
    static const auto _ = []{
#define DEF_CALLBACK(X) if( first ) { \
        SPLIT_SETTING( X ) = map[#X].value<track::split::slow:: X##_t >(); \
    } else if (name == #X) { \
        SPLIT_SETTING( X ) = value.value<track::split::slow:: X##_t >(); \
    } void()
        
        auto fn = [](std::string_view name) {
            static bool first = true;
            auto &map = GlobalSettings::map();
            auto &value = map.at(name).get();
            
            DEF_CALLBACK(blob_split_max_shrink);
            DEF_CALLBACK(blob_split_global_shrink_limit);
            DEF_CALLBACK(cm_per_pixel);
            DEF_CALLBACK(track_size_filter);
            DEF_CALLBACK(track_threshold);
            DEF_CALLBACK(track_posture_threshold);
            
            DEF_CALLBACK(blob_split_algorithm);
            
            if(first)
                first = false;
        };
        
        GlobalSettings::map().register_callbacks({
            "blob_split_max_shrink",
            "blob_split_global_shrink_limit",
            "cm_per_pixel",
            "track_size_filter",
            "track_threshold",
            "track_posture_threshold",
            "blob_split_algorithm"
            
        }, fn);
        
#undef DEF_CALLBACK
        
        return 0;
    }(); UNUSED(_);
    
    // generate greyscale and mask images
    imageFromLines(blob->input_info(), blob->hor_lines(), NULL, &_original_grey, &_original, blob->pixels().get(), SPLIT_SETTING(track_posture_threshold), &average);
    
    blob->set_tried_to_split(true);
}

size_t SplitBlob::apply_threshold(CPULabeling::ListCache_t* cache, int threshold, std::vector<pv::BlobPtr> &output, const Background& background)
{
    if(_diff_px.empty()) {
        _diff_px.resize(_blob->pixels()->size());
        auto px = _blob->pixels()->data();
        auto out = _diff_px.data();
        //auto grid = Tracker::instance()->grid();
        //constexpr LuminanceGrid* grid = nullptr;
        min_pixel = 254;
        max_pixel = 0;
        
        auto work = [&]<InputInfo input, OutputInfo output, DifferenceMethod method>() {
            static_assert(output.channels == 1, "Only support single value.");
            
            for (auto &line : _blob->hor_lines()) {
                for (auto x=line.x0; x<=line.x1; ++x, px += input.channels, out += output.channels) {
                    auto value = diffable_pixel_value<input, output>(px);
                    *out = (uchar)saturate(float(background.diff<output, method>(x, line.y, value)) / 1.f);//(grid ? float(grid->relative_threshold(x, line.y)) : 1.f));
                    if(*out < min_pixel)
                        min_pixel = *out;
                    if(*out > max_pixel)
                        max_pixel = *out;
                }
            }
        };
        
        call_image_mode_function<OutputInfo{
            .channels = 1u,
            .encoding = meta_encoding_t::gray
        }>(_blob->input_info(), KnownOutputType{}, work);
        
        threshold = max(threshold, (int)min_pixel);
    }
    
    output = pixel::threshold_blob(*cache, _blob, _diff_px, threshold, background);
    
    for(auto &blob: output)
        blob->add_offset(-_blob->bounds().pos());
    
    std::sort(output.begin(), output.end(),
       [](const pv::BlobPtr& a, const pv::BlobPtr& b) { 
            return std::make_tuple(a->pixels()->size(), a->blob_id()) > std::make_tuple(b->pixels()->size(), b->blob_id()); 
       });
    
    return output.empty() ? 0 : (*output.begin())->pixels()->size() / (*output.begin())->channels();
}

/**
 * This function evaluates the result of `apply_threshold`, looking for acceptable configurations of blobs given the expected number of objects and their sizes.
 *
 * It expects the blobs to be given in descending order of size.
 *
 * The rules are enforced in this order:
 *  1. given `presumed_number` of expected objects, are enough objects found?
 *  2. the overall number of pixels should not shrink further than `blob_split_max_shrink * first_size`, i.e. a certain percentage of the unthresholded image
 *  3. all objects smaller than `track_size_filter.max.start * blob_split_global_shrink_limit` are removed
 *  4. if the smallest found object is bigger than `track_size_filter.max.end`, ignore the results
 */
split::Action_t SplitBlob::evaluate_result_multiple(size_t presumed_nr, float first_size, std::vector<pv::BlobPtr>& blobs)
{
    const Float2_t sqrcm = SQR(SPLIT_SETTING(cm_per_pixel));
    size_t pixels = 0;
    std::optional<size_t> min_size;
    for(size_t i=0; i < blobs.size(); ++i) {
        pixels += blobs.at(i)->num_pixels();
    }
    
    if(pixels * sqrcm < SPLIT_SETTING(blob_split_max_shrink) * first_size) {
        return split::Action::ABORT;
    }
    
    if(not SPLIT_SETTING(track_size_filter).empty()) {
        const auto min_size_threshold = SPLIT_SETTING(track_size_filter).max_range().start * SPLIT_SETTING(blob_split_global_shrink_limit);
        auto it = std::remove_if(blobs.begin(), blobs.end(), [&](const pv::BlobPtr& blob) {
            auto fsize = blob->num_pixels() * sqrcm;
            return fsize < min_size_threshold;
        });
        blobs.erase(it, blobs.end());
    }
    
    for(size_t i=0; i<presumed_nr && i < blobs.size(); ++i) {
        if(!min_size.has_value() || blobs.at(i)->num_pixels() < min_size.value()) {
            min_size = blobs.at(i)->num_pixels();
        }
    }
    
    if(SPLIT_SETTING(track_size_filter)
       && min_size.has_value()
       && min_size.value() * sqrcm > SPLIT_SETTING(track_size_filter).max_range().end)
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
#ifdef TREX_SPLIT_DEBUG
    int count{0};
    std::vector<std::tuple<int, split::Action_t>> tried;
    int found_in_step{-1};
#endif
    
    robin_hood::unordered_flat_map<int, split::Action_t> results;
    using best_t = std::conditional_t<thread_safe, std::atomic<int>, int>;
    best_t best{-1};
    
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
        if(is_in(action, split::Action::KEEP, split::Action::KEEP_ABORT))
        {
            if(best == -1 || threshold < best) {
                best = threshold;
#ifdef TREX_SPLIT_DEBUG
                found_in_step = step;
#else
                UNUSED(step);
#endif
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
#ifdef TREX_SPLIT_DEBUG
            ++count;
#endif
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
    
    static std::atomic<size_t> matches{0}, mismatches{0};
    static std::atomic<int64_t> offsets{0}, max_offset{0}, not_found{0}, would_find{0}, second_try{0}, preproc{0}, third{0}, count_from_not_found{0};
    static std::mutex m;
    static std::map<int64_t, size_t> often;
    
    thresholds += next.count;
    ++samples;
    samples_naive += naive.count;
    
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
                Print(thread_safe ? "(ts)" : "(single)", " ", bdx, " Naive count: ", naive.count, " (t=",(int)naive.best,") vs ", next.count, " (t=",(int)next.best,") and map ", next.tried, " vs ", naive.tried);
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
            Print(thread_safe ? "(ts)" : "(single)", " ", bdx, " Not found count: ", naive.count, " (t=",(int)naive.best,") vs ", next.count, " (t=",(int)next.best,") and map ", naive.tried, " vs ", next.tried);
            ++not_found;
            ++mismatches;
        }
        
    } else
        ++matches;
    
    if(samples.load() % 100 == 0) {
        auto m0 = matches.load(), m1 = mismatches.load();
        auto off = offsets.load();
        Print(thread_safe ? "(ts)" : "(single)", " ", bdx, " Samples: ", float(thresholds) / float(samples), " (vs. ", float(samples_naive.load()) / float(samples.load()),") ", float(m1) / float(m0 + m1) * 100, "% mismatches (avg. ", float(off) / float(m1), " offset for ", m1," elements) ", float(not_found.load()) / float(m1) * 100, "% not found, ", float(would_find.load()) / float(not_found.load()) * 100, "% could have been found, ", float(second_try.load()) / float(samples.load()) * 100, "% found in second try, ", float(preproc.load()) / float(samples.load()) * 100, "% found in preprocess ", float(third.load()) / float(samples.load()) * 100, "% found in third try, ", float(count_from_not_found.load()) / float(thresholds.load()) * 100, "% from not found objects");
        std::unique_lock guard(m);
        Print(often);
    }
}
#endif

std::vector<pv::BlobPtr> SplitBlob::split(size_t presumed_nr, const std::vector<std::vector<Vec2>>& centers, const Background& background)
{
    if(SPLIT_SETTING(blob_split_algorithm) == blob_split_algorithm_t::none)
        return {};
    
    ResultProp best_match;
    float first_size = 0;
    
    const Float2_t sqrcm = SQR(SPLIT_SETTING(cm_per_pixel));
    
    const auto apply_watershed = [this](const std::vector<std::vector<Vec2>>& centers, std::vector<pv::BlobPtr>& output) {
        static const cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * 1 + 1, 2 * 1 + 1), cv::Point(1, 1));

        //auto [offset, img] = _blob->binary_image();
        auto [offset, img] = _blob->color_image();
        cv::Mat mask = cv::Mat::zeros(img->rows, img->cols, CV_32SC1);
        for(size_t i = 0; i < img->size() / img->channels(); ++i) {
            bool px_set = false;
            for(uint p = 0; p < img->channels(); ++p) {
                if(img->data()[i * img->channels() + p] != 0) {
                    px_set = true;
                    break;
                }
            }
            if(not px_set)
                mask.ptr<int>()[i] = 1;
        }

        size_t i = 0;
        for(auto& c : centers) {
            for(auto &pt : c) {
                cv::circle(mask, pt, 5, i + 2, cv::FILLED);
                cv::circle(mask, pt, 5, 0, 1);
            }
            ++i;
        }

        cv::Mat tmp;
        mask.convertTo(tmp, CV_8UC1, 255.f / float(centers.size() + 2));
        //tf::imshow("labels1", tmp);
        
        if(img->channels() == 1)
            cv::cvtColor(img->get(), tmp, cv::COLOR_GRAY2BGR);
        else {
            assert(img->channels() == 3);
            img->get().copyTo(tmp);
        }
        cv::watershed(tmp, mask);

        //cv::Mat ltmp;
        //mask.convertTo(ltmp, CV_8UC1, 255.f / float(centers.size() + 1));
        //tf::imshow("labels", ltmp);
        //tf::imshow("image", img->get());

        //cv::subtract(mask, 1, mask);
        mask.convertTo(mask, CV_8UC1);
        cv::threshold(mask, mask, 1, 255, cv::THRESH_BINARY);
        //tf::imshow("thresholded", tmp);

        cv::Mat mask2;
        
        //tf::imshow("mask before", mask);
        cv::erode(mask, mask2, element);
        //mask2.convertTo(ltmp, CV_8UC1, 255.f / float(centers.size() + 1));
        //tf::imshow("mask eroded", ltmp);
        //if(img->channels() == 1) {
        //assert(img->channels() == tmp.channels());
        cv::Mat tmp2;
        if(img->channels() == 3) {
            tmp.copyTo(tmp2, mask2);
            
        } else if(img->channels() == 1) {
            std::vector<cv::Mat> separate_channels;
            cv::split(tmp, separate_channels);
            separate_channels.front().copyTo(tmp2, mask2);
        }
        
        //tf::imshow("mask copy", tmp2);

        {
            auto detections = CPULabeling::run(tmp2, *_cache);
            //Print("Detections: ", detections.size());

            output.clear();
            for(auto&& [lines, pixels, flags, pred] : detections) {
                flags |= pv::Blob::copy_flags(*_blob);
                output.emplace_back(pv::Blob::Make(std::move(lines), std::move(pixels), flags, std::move(pred)));
                //output.back()->add_offset(-_blob->bounds().pos());
            }
        }

        std::sort(output.begin(), output.end(),
            [](const pv::BlobPtr& a, const pv::BlobPtr& b) {
                return std::make_tuple(a->pixels()->size(), a->blob_id()) > std::make_tuple(b->pixels()->size(), b->blob_id());
            });


        /*resize_image(tmp, 5, cv::INTER_NEAREST);
        if(tmp.channels() == 1)
            cv::cvtColor(tmp, tmp, cv::COLOR_GRAY2BGR);

        i = 0;
        cmn::gui::ColorWheel wheel;
        for(auto& d : output) {
            auto c = wheel.next();
            for(auto& l : *d->lines())
                cv::line(tmp, (Vec2(l.x0, l.y) + 0.5) * 5, (Vec2(l.x1, l.y) + 0.5) * 5, c, 5);
            pv::Blob b{
                *d->lines(),
                *d->pixels(),
                pv::Blob::copy_flags(*_blob),
                {}
            };
            auto [p, img] = b.color_image();
            //auto [p, img] = d->image();
            tf::imshow("blob" + Meta::toStr(i), img->get());
            ++i;
        }
        tf::imshow("blobs", tmp);*/

        return output.empty() ? 0 : (*output.begin())->pixels()->size() / (*output.begin())->channels();
    };
    
    std::atomic<float> max_size;
    const auto initial_threshold = (SLOW_SETTING(calculate_posture) ? max(SLOW_SETTING(track_threshold), SPLIT_SETTING(track_posture_threshold)) : SLOW_SETTING(track_threshold)) + 1;

    const auto try_threshold = [&](CPULabeling::ListCache_t* cache, int threshold, bool save)
    {
        std::vector<pv::BlobPtr> blobs;
        
        bool initial = threshold == -1;
        if(initial)
            threshold = initial_threshold;
        
        if(is_in(SPLIT_SETTING(blob_split_algorithm), blob_split_algorithm_t::threshold, blob_split_algorithm_t::threshold_approximate))
            max_size = apply_threshold(cache, threshold, blobs, background) * sqrcm;
        else
            max_size = (initial ? apply_threshold(cache, threshold, blobs, background) : apply_watershed(centers, blobs)) * sqrcm;
        
        // save the maximum number of objects found
        max_objects = max(max_objects.load(), blobs.size());
        
        ResultProp result;
        result.threshold = threshold;
        
        auto action = evaluate_result_multiple(presumed_nr, first_size, blobs);
        
        if(first_size == 0)
            first_size = max_size;
        
        //PPFrame::Log("Resolved action ", action, " for presumed=",presumed_nr, " of ", *_blob, " to ", blobs);
        
        // we found enough blobs, so we're allowed to keep it
        if(is_in(action, split::Action::KEEP, split::Action::KEEP_ABORT))
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
            //Print(" inserting ", threshold);
        }

        //blobs.clear();
        return action;
    };
    
    auto action = try_threshold(_cache, -1, true);
    
    if(action != split::Action::KEEP
       && action != split::Action::KEEP_ABORT
       && (not SPLIT_SETTING(track_size_filter)
            || _blob->num_pixels() * sqrcm < SPLIT_SETTING(track_size_filter).max_range().end * 100))
    {
        
        if(presumed_nr > 1) {
            if(is_in(SPLIT_SETTING(blob_split_algorithm), blob_split_algorithm_t::threshold, blob_split_algorithm_t::threshold_approximate))
            {
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
                
                const int begin_threshold = max(initial_threshold, (int)min_pixel);
                const float distance = float(max_pixel - begin_threshold);
                constexpr int segments = 3;
                
                auto work = [&]<bool accurate>(auto cache, auto& run, int64_t thread_index)
                {
                    //! we run two samplings with a step of 2 each
                    //! for this thread
                    constexpr int sampling_runs = 2;
                    //! across threads sample with `segments` step,
                    //! so here we do `global_step * local_step`
                    constexpr int step = segments * sampling_runs;
                    
                    //! we start at `max(threshold, min_pixel)`
                    //! and end at maximum pixel value
                    int start = begin_threshold;
                    int end = max_pixel;
                    
                    //! traverse the lower 30% of thresholds first
                    //! this is the most likely place that we will
                    //! find a useful threshold.
                    Range<int> first_stage{
                        start,
                        start + int((end - start) * 0.3)
                    };
                    
                    //! iterate through the local sampling runs,
                    //! with offsets 0 to 2
                    for(int offset = 0; offset < sampling_runs; ++offset) {
                        // stopping after a solution has been found
                        // makes the method inaccurate, since we then
                        // do not traverse all remaining offsets at all:
                        if(!accurate && run.has_best())
                            break;
                        
                        // go from start, but offset by thread index
                        // as well as sampling offset, go towards the end
                        // skipping big steps (segments * sampling_runs)
                        for(int threshold = first_stage.start + narrow_cast<int32_t>(thread_index) * sampling_runs + offset;
                            threshold < first_stage.end;
                            threshold += step)
                        {
                            // we really can abort, if the found best
                            // is larger than our current threshold
                            if(run.has_best() && threshold >= run.best)
                                break;
                            
                            auto action = run.perform(cache, threshold, offset, try_threshold, true);
                            
                            if(is_in(action, split::Action::ABORT, split::Action::KEEP_ABORT))
                            {
                                if constexpr (!accurate) {
                                    if(action == split::Action::KEEP_ABORT)
                                        return;
                                }
                                break;
                            }
                        }
                    }
                    
                    //! potentially traverse the upper part of possible
                    //! thresholds. this is unlikely to yield a good result,
                    //! but if we are forced to be deterministic, do it.
                    //! although we dont need to if the found best is below
                    //! the high thresholds we havent searched yet:
                    if(run.has_best() && (!accurate || run.best < first_stage.end))
                        return;
                    
                    //! could use step (without /2) here for the upper 70%
                    //! if we want to accept solutions not being found
                    //! in like 1% of cases.
                    constexpr int use_step = step / 2;//accurate ? step / 2 : step;
                    for(int threshold = first_stage.end + narrow_cast<int32_t>(thread_index);
                        threshold < end;
                        threshold += use_step)
                    {
                        // although if the current best is already
                        // lower, stop (since we go in single steps
                        // when accurate):
                        if(run.has_best() && threshold >= run.best)
                            break;
                        
                        auto action = run.perform(cache, threshold, 3, try_threshold, true);
                        
                        if(is_in(action, split::Action::ABORT, split::Action::KEEP_ABORT))
                        {
                            if constexpr (!accurate) {
                                if(action == split::Action::KEEP_ABORT)
                                    return;
                            }
                            break;
                        }
                    }
                };
                
                
#ifdef TREX_SPLIT_DEBUG
                Run<false> naive;
                for(int i = begin_threshold; i < 254; ++i)
                {
                    auto action = naive.perform(_cache, i, 0, try_threshold, false);
                    if(is_in(action, split::Action::ABORT, split::Action::KEEP_ABORT))
                    {
                        break;
                    }
                }
#endif
                const bool complete_search = SPLIT_SETTING(blob_split_algorithm) == blob_split_algorithm_t::threshold;
                
                if(distance < segments || _blob->num_pixels() < 16000)
                {
                    Timer timer;
                    Run<false> run;
                    
                    if(complete_search) {
                        for(int i = begin_threshold; i < max_pixel; ++i)
                        {
                            auto action = run.perform(_cache, i, 0, try_threshold, true);
                            if(is_in(action, split::Action::ABORT, split::Action::KEEP_ABORT))
                            {
                                break;
                            }
                        }
                        
                    } else {
                        for(int i = 0; i < segments; ++i) {
                            if(run.has_best())
                                break;
                            work.operator()<false>(_cache, run, i);
                        }
                    }
                    
                    
#ifdef TREX_SPLIT_DEBUG
                    commit_run(_blob->blob_id(), naive, run);
#endif
                } else {
                    static GenericThreadPool threads(9, "Thresholds", nullptr);
                    Timer timer;
                    Run<true> run;
                    
                    //! Counts down for all threads, to synchronize
                    //! the first stage of the algorithm.
                    //! This reduces the number of misses.
                    //std::latch latch{ptrdiff_t(num_threads)};
                    distribute_indexes([&](auto, auto, auto, int64_t j) {
                        //! protects the usage of CPULabeling caches
                        //! via RAII
                        const split::Guard guard{};
                        if(complete_search)
                            work.operator()<true>(guard.c, run, j);
                        else
                            work.operator()<false>(guard.c, run, j);
                        
                    }, threads, 0, (int)segments + 1);
                    
#ifdef TREX_SPLIT_DEBUG
                    commit_run(_blob->blob_id(), naive, run);
#endif
                }
            }
            else
                action = try_threshold(_cache, 0, true);
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
        
        return std::move(best_match.blobs);
    }
    
    //! could not find anything
    return {};
}
