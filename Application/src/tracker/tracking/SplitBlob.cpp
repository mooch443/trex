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

std::atomic_bool registered_callbacks = false;
float blob_split_max_shrink = 0.02f;
float blob_split_global_shrink_limit = 0.01f;
float sqrcm = 1.f;
BlobSizeRange fish_minmax({Rangef(0.001,1000)});
int posture_threshold = 15;
auto blob_split_algorithm = blob_split_algorithm_t::threshold;

SplitBlob::SplitBlob(CPULabeling::ListCache_t* cache, const Background& average, pv::BlobPtr blob)
    :   max_objects(0),
        _blob(blob),
        _cache(cache)
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
                
            } else if(name == "blob_split_global_shrink_limit") {
                blob_split_global_shrink_limit = value.value<float>();
                
            } else if(name == "cm_per_pixel") {
                sqrcm = SQR(value.value<float>());
                
            } else if(name == "track_posture_threshold") {
               posture_threshold = value.value<int>();
            } else if(name == "blob_split_algorithm")
                blob_split_algorithm = value.value<blob_split_algorithm_t::Class>();
            
            if(name == "blob_size_ranges" || name == "cm_per_pixel") {
                fish_minmax = SETTING(blob_size_ranges).value<BlobSizeRange>();
            }
        };
        GlobalSettings::map().register_callback(callback, fn);
        
        fn(sprite::Map::Signal::NONE, GlobalSettings::map(), "blob_split_max_shrink", SETTING(blob_split_max_shrink).get());
        fn(sprite::Map::Signal::NONE, GlobalSettings::map(), "blob_split_global_shrink_limit", SETTING(blob_split_global_shrink_limit).get());
        fn(sprite::Map::Signal::NONE, GlobalSettings::map(), "cm_per_pixel", SETTING(cm_per_pixel).get());
        fn(sprite::Map::Signal::NONE, GlobalSettings::map(), "track_posture_threshold", SETTING(track_posture_threshold).get());
        fn(sprite::Map::Signal::NONE, GlobalSettings::map(), "blob_size_ranges", SETTING(blob_size_ranges).get());
        fn(sprite::Map::Signal::NONE, GlobalSettings::map(), "blob_split_algorithm", SETTING(blob_split_algorithm).get());
    }
    
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
        constexpr LuminanceGrid* grid = nullptr;
        
        for (auto &line : _blob->hor_lines()) {
            for (auto x=line.x0; x<=line.x1; ++x, ++px, ++out) {
                *out = (uchar)saturate(float(bg->diff(x, line.y, *px)) / (grid ? float(grid->relative_threshold(x, line.y)) : 1.f));
            }
        }
    }
    
    output = pixel::threshold_blob(*_cache, _blob, _diff_px, threshold);
    
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
    size_t pixels = 0;
    std::optional<size_t> min_size;
    for(size_t i=0; i < blobs.size(); ++i) {
        pixels += blobs.at(i)->num_pixels();
    }
    
    if(pixels * sqrcm < blob_split_max_shrink * first_size) {
        return split::Action::ABORT;
    }
    
    const float min_size_threshold = fish_minmax.max_range().start * blob_split_global_shrink_limit;
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
       && min_size.value() * sqrcm > fish_minmax.max_range().end)
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
    int best{-1};
    int found_in_step{-1};
    robin_hood::unordered_flat_map<int, split::Action_t> results;
    std::vector<std::tuple<int, split::Action_t>> tried;
    int lowest_non_remove{-1}, highest_remove{-1};
    
    split::Action_t perform(int threshold, int step, auto&& F, auto... args) {
        if(best != -1 && best < threshold)
            return split::Action::ABORT;
        if(results.contains(threshold)) {
            auto action = results.at(threshold);
            if(action == split::Action::KEEP_ABORT
               && (best == -1 || best > threshold))
            {
                print("Somehow found this in ", threshold, " in the map.");
                best = threshold;
                found_in_step = step;
            }
            
            //if(action != split::Action::REMOVE
            //   && (lowest_non_remove == -1 || threshold < lowest_non_remove)) {
            //    lowest_non_remove = threshold;
            //} else if(action == split::Action::REMOVE) {
            //    if(threshold > highest_remove) {
            //        highest_remove = threshold;
            //    }
            //}
            
            return action;
        }
        
        auto action = F(threshold, args...);
        if(action == split::Action::KEEP
           || action == split::Action::KEEP_ABORT)
        {
            assert(best == -1 || best > threshold);
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
        
        //tried.push_back({threshold, action});
        results[threshold] = action;
        ++count;
        return action;
    }
};

void commit_run(pv::bid bdx, Run naive, Run next) {
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
                print(bdx, " Naive count: ", naive.count, " (t=",naive.best,") vs ", next.count, " (t=",next.best,") and map ", next.tried, " vs ", naive.tried);
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
            ++not_found;
            ++mismatches;
        }
        
    } else
        ++matches;
    
    if(samples.load() % 2000 == 0) {
        auto m0 = matches.load(), m1 = mismatches.load();
        auto off = offsets.load();
        print("Samples: ", float(thresholds) / float(samples), " (vs. ", float(samples_naive.load()) / float(samples.load()),") ", float(m1) / float(m0 + m1) * 100, "% mismatches (avg. ", float(off) / float(m1), " offset for ", m1," elements) ", float(not_found.load()) / float(m1) * 100, "% not found, ", float(would_find.load()) / float(not_found.load()) * 100, "% could have been found, ", float(second_try.load()) / float(samples.load()) * 100, "% found in second try, ", float(preproc.load()) / float(samples.load()) * 100, "% found in preprocess ", float(third.load()) / float(samples.load()) * 100, "% found in third try, ", float(count_from_not_found.load()) / float(thresholds.load()) * 100, "% from not found objects");
        std::unique_lock guard(m);
        print(often);
    }
}

std::vector<pv::BlobPtr> SplitBlob::split(size_t presumed_nr, const std::vector<Vec2>& centers)
{
    //std::map<int, ResultProp> best_matches;
    ResultProp best_match;
    
    size_t calculations = 0;
    Timer timer;
    std::vector<pv::BlobPtr> blobs;
    float first_size = 0;
    size_t more_than_1_times = 0;
    
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
    const auto fn = [&](int threshold, bool save)
    {
        calculations++;
        
        if(blob_split_algorithm == blob_split_algorithm_t::threshold)
            max_size = apply_threshold(threshold, blobs) * sqrcm;
        else
            max_size = (threshold == -1 ? apply_threshold(threshold, blobs) : apply_watershed(centers, blobs)) * sqrcm;
        
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
                if(best_match.threshold == -1
                   || result.threshold < best_match.threshold)
                {
                    result.blobs = std::move(blobs);
                    best_match = std::move(result);
                }
            }
            //print(" inserting ", threshold);
        }

        blobs.clear();
        return action;
    };
    
    auto action = fn(-1, true);
    
    if(action != split::Action::KEEP
       && action != split::Action::KEEP_ABORT
        && _blob->pixels()->size() * sqrcm < fish_minmax.max_range().end * 100) 
    {
        
        if(presumed_nr > 1) {
            if(blob_split_algorithm == blob_split_algorithm_t::threshold) {
                
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
                const auto min_threshold = (SLOW_SETTING(calculate_posture) ? max(SLOW_SETTING(track_threshold), posture_threshold) : SLOW_SETTING(track_threshold)) + 1;
                
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
                
                int begin = min(254, min_threshold);
                int end = run.best == -1 ? 254 : (run.best - 1);
                auto half = int((end - begin) * 0.5 + 0.5);
                int t = begin + half;
                int minimal = min_threshold, maximal = 254;
                int avoided = 0;
                
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
                
                if(run.best == -1) {
                    
                    auto search_range = [&fn, &minimal, &maximal, &avoided](const arange<int> range, Run& run, const int step, const int offset, const int index = 1)
                    {
                        //print("Searching ", range.first, " to ", range.last, " with step ", step, " and offset ", offset, " minimal=", minimal, " and maximal=", maximal);
                        
                        for(int i = range.first + offset; i < range.last; i += step)
                        {
                            auto action = run.perform(i, index, fn, true);
                            //if(i < minimal || i > maximal)
                            //    ++avoided;
                            
                            if(action == split::Action::ABORT
                               || action == split::Action::KEEP_ABORT)
                            {
                                return;
                            }
                        }
                    };
                    
                    //search_range(arange(min_threshold, maximal), run, 50, 0, 0);
                    
                    const int step = 4;
                    //run.best = -1;
                    int resolution = step;
                    
                    for(int o = 0; o < step; ++o) {
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
                                maximal = min(run.best, maximal);
                        }
                        //minimal = max(min_threshold, run.lowest_non_remove + 1 - step * 2);
                        
                        if(!accurate) {
                            if(best == -1 && run.best != -1) {
                                minimal = max(min_threshold, (run.best - ((run.best - min_threshold) % step)) - step);
                                maximal = max(min_threshold, run.best);
                            } /*else if(run.lowest_non_remove != -1 && run.best == -1) {
                               minimal = max(min_threshold, (run.lowest_non_remove - ((run.lowest_non_remove - min_threshold) % step)) - step);
                               }*/
                        }
                    }
                    
                    if(accurate && run.best != -1) {
                        for(int i=minimal; i<run.best; ++i) {
                            auto action = run.perform(i, 3, fn, true);
                            //if(i < minimal || i > maximal)
                            //    ++avoided;
                            
                            if(action == split::Action::ABORT
                               || action == split::Action::KEEP_ABORT)
                            {
                                break;
                            }
                        }
                    }
                    
                    //if(run.best != -1 && resolution > 0)
                    //    search_range(arange(run.best - resolution, run.best), run, 1, 0);
                }
                
                /*if(run.best != -1) {
                    for(int i = max(min_threshold, run.best - 9); i < 255 && i < run.best; i+=2)
                    {
                        auto action = run.perform(i, 3, fn, true);
                        //print("##",*_blob, "@t=",i," => ", action, " max_size=",max_size, " (", run.best, " ", run.count,")");
                        if(action == split::Action::ABORT
                           || action == split::Action::KEEP_ABORT)
                        {
                            break;
                        }
                    }
                }*/
                
                /*Run naive;
                for(int i = min_threshold; i < 254; ++i)
                {
                    auto action = naive.perform(i, 0, fn, false);
                    if(action == split::Action::ABORT
                       || action == split::Action::KEEP_ABORT)
                    {
                        break;
                    }
                }
                
                commit_run(_blob->blob_id(), naive, run);*/
            }
            else
                action = fn(0, true);
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
