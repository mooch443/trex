#include "SplitBlob.h"
#include <types.h>
#include <processing/CPULabeling.h>
#include <tracking/Tracker.h>
#include <misc/Timer.h>
#include <misc/metastring.h>
#include <misc/PixelTree.h>
#include <misc/cnpy_wrapper.h>

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
                
            } else if(name == "blob_split_global_shrink_limit") {
                blob_split_global_shrink_limit = value.value<float>();
                
            } else if(name == "cm_per_pixel") {
                sqrcm = SQR(value.value<float>());
                
            } else if(name == "track_posture_threshold") {
               posture_threshold = value.value<int>();
            }
            
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
    }
    
#if DEBUG_ME
    print("SplitBlob(", blob->blob_id(),")");
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
        constexpr LuminanceGrid* grid = nullptr;
        
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
       [](const pv::BlobPtr& a, const pv::BlobPtr& b) { 
            return std::make_tuple(a->pixels()->size(), a->blob_id()) > std::make_tuple(b->pixels()->size(), b->blob_id()); 
       });
    
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
        
        print("Samples: ",size_t(samples),"u, timer1: ",timer1 / samples * 1000,"ms timer2: ",timer2 / samples * 1000,"ms ratio: ",ratio / samples);
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
    display_match({0, result}, {});
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
            print("\tbreaking because fsize ", fsize," / ", min_size_threshold," / ",first_size);
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
    
    if(presumed_nr <= blobs.size() && ratio >= 0.4) {
        return KEEP_ABORT;
    }
    
    return REMOVE;
}

std::vector<pv::BlobPtr> SplitBlob::split(size_t presumed_nr, const std::vector<Vec2>& centers)
{
    std::map<int, ResultProp> best_matches;

    /* {
        auto [p, img] = _blob->image(Tracker::instance()->background());
        auto [p0, img0] = _blob->binary_image();

        file::Path path = "C:/Users/tristan/Videos/locusts/last_blob.npz";
        npz_save(path.str(), "image", img0->data(), std::vector<size_t>{img0->rows, img0->cols}, "w");
        std::vector<float> coords;
        for(auto c : centers)
            coords.insert(coords.end(),  { c.x, c.y });
        npz_save(path.str(), "coords", coords.data(), std::vector<size_t>{centers.size(), 2}, "a");
        tf::imshow("analyse blob", img0->get());
    }*/
    
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

        auto detections = CPULabeling::run(tmp);
        //print("Detections: ", detections.size());

        output.clear();
        for(auto&& [lines, pixels] : detections) {
            output.emplace_back(pv::Blob::make(std::move(lines), std::move(pixels)));
            //output.back()->add_offset(-_blob->bounds().pos());
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
    
    const auto fn = [&apply_watershed, &calculations, &blobs, &centers, presumed_nr, &first_size, &more_than_1_times, &best_matches, this](int threshold) {
        calculations++;
        
#if DEBUG_ME
        print("T:", threshold);
#endif
        //float max_size = apply_threshold(threshold, blobs) * sqrcm;
        float max_size = (threshold == -1 ? apply_threshold(threshold, blobs) : apply_watershed(centers, blobs)) * sqrcm;
        
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
        
        // we found enough blobs, so we're allowed to keep it
        if(action == KEEP || action == KEEP_ABORT) {
#if DEBUG_ME
            print("Found ", blobs.size()," blobs at threshold ", threshold," (expected ",presumed_nr,") with centers: ", centers);
#endif
            
            result.blobs = blobs;
            best_matches[threshold] = result;
        }

        blobs.clear();
        return action;
    };
    
    auto action = fn(-1);
    
    if(action != KEEP 
        && action != KEEP_ABORT 
        && _blob->pixels()->size() * sqrcm < fish_minmax.max_range().end * 100) 
    {
        
        if(presumed_nr > 1) {
            action = fn(0);

            /*for(int i = posture_threshold + 10; i < 255; i += 1) {//max(1, (i-posture_threshold)*0.25)) {
                action = fn(i);
                if(action == ABORT || action == KEEP_ABORT)
                    break;
            }*/
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
            print("Match: ",p.first," with ratio ",p.second.ratio,", ",p.second.blobs.size()," blobs, ",percent," percent");
#endif
            if(percent > bestp) {
                best = p.first;
                bestp = percent;
            }
        }
    }
    
#if DEBUG_ME
    print(calculations," calculations in ",dec<4>(timer.elapsed()),"s");
#endif
    
    std::vector<pv::BlobPtr> result;
    if(best != INT_MIN) {
        result = { best_matches.at(best).blobs };
        std::vector<uchar> grey;
        
        for (size_t idx = 0; idx < result.size(); idx++) {
            auto& blob = result.at(idx);
            
            if(!blob->pixels()) {
                grey.clear();
                int32_t N = 0;
                for (auto& h : blob->hor_lines()) {
                    auto n = int32_t(h.x1) - int32_t(h.x0) + 1;
                    auto ptr = _original_grey.ptr(h.y, h.x0);
                    grey.resize(N + n);
                    std::copy(ptr, ptr + n, grey.data() + N);
                    N += n;
                    /*for(int x = h.x0; x <= h.x1; x++) {
                        assert(h.y < _original_grey.rows && x < _original_grey.cols);
                        grey.push_back(_original_grey.at<uchar>(h.y, x));
                    }*/
                }
                
                blob->set_pixels(std::make_unique<std::vector<uchar>>(std::move(grey)));
                //result.pixels.push_back(grey);
            }
        }
        
        
#if DEBUG_ME
        print("Best result with threshold ", best);
        display_match({best, best_matches.at(best).blobs}, centers);
#endif
    } else {
#if DEBUG_ME
        FormatWarning("Not found ", presumed_nr," objects. ", best_matches);
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
void SplitBlob::display_match(const std::pair<const int, std::vector<pv::BlobPtr>>& blobs, const std::vector<Vec2>& centers)
{
    ColorWheel wheel;
    cv::Mat copy;
    cv::cvtColor(_original, copy, cv::COLOR_GRAY2BGR);
    
    //for(size_t i=0; i<blobs.size(); ++i) {
    //    auto& blob = blobs.second[i];
        
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

    for(auto& p : centers) {
        cv::circle(copy, p, 2, gui::Red, 1);
        cv::circle(copy, p, 2, gui::White, cv::FILLED);
    }

    tf::imshow("copy", copy);
//cv::waitKey();
}
#endif

