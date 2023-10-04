#include "UnitTest.h"

#include <misc/CommandLine.h>
#include <processing/CPULabeling.h>
#include <misc/PVBlob.h>
#include <tracking/Tracker.h>
#include <misc/default_config.h>
#include <misc/Timer.h>
#include <tracking/SplitBlob.h>
#include <gui/IMGUIBase.h>
#include <gui/SFLoop.h>
#include "Data.h"
#include <gui/Graph.h>
#include "fish_frames.h"
#include "termite_frames.h"
#include "stickleback_frames.h"
#include "stickleback_tags_frames.h"
#include "placo_frames.h"
#include <tracking/Posture.h>
#include <misc/cnpy_wrapper.h>
#include <misc/PixelTree.h>
#include <processing/RawProcessing.h>
#include "noisy_frame.h"

std::vector<Vec2> extract_outline(pv::BlobPtr blob, track::StaticBackground*, int threshold);

class TestLabeling : public UnitTest {
public:
    TestLabeling() : UnitTest("CPULabeling") {}
    
    static int _gaussCircleProblem(int radius) {
        int allPoints=0; //holds the sum of points
        double y=0; //will hold the precise y coordinate of a point on the circle edge for a given x coordinate.
        long inscribedSquare=(long) std::sqrt(radius*radius/2); //the length of the side of an inscribed square in the upper right quarter of the circle
        int x=(int)inscribedSquare; //will hold x coordinate - starts on the edge of the inscribed square
        while(x<=radius){
            allPoints+=(long) y; //returns floor of y, which is initially 0
            x++; //because we need to start behind the inscribed square and move outwards from there
            y=std::sqrt(radius*radius-x*x); // Pythagorean equation - returns how many points there are vertically between the X axis and the edge of the circle for given x
        }
        allPoints*=8; //because we were counting points in the right half of the upper right corner of that circle, so we had just one-eightth
        allPoints+=(4*inscribedSquare*inscribedSquare); //how many points there are in the inscribed square
        allPoints+=(4*radius+1); //the loop and the inscribed square calculations did not touch the points on the axis and in the center
        return allPoints;
    }
    
    static double gaussCircleProblem(float radius, int line) {
        cv::Mat circle = cv::Mat::zeros(radius*2 + 2, radius*2 + 2, CV_8UC1);
        cv::circle(circle, Vec2(circle.cols * 0.5).map(ceilf), radius, 1, line);
        return cv::sum(circle)[0];
    }
    
    static std::tuple<size_t, size_t> generate_circles(cv::Mat& image, Vec2 offset = Vec2(), float scale = 1) {
        assert(!image.empty());
        
        constexpr size_t number = 5;
        const size_t circle_width = image.cols / number;
        const size_t radius = circle_width * 0.5;
        size_t number_objects = 0;
        size_t width = image.cols, height = image.rows;
        double num_pixels = 0;
        
        for (size_t i=0; i<number; ++i) {
            for (size_t j=0; j<number; ++j) {
                Vec2 pos = Vec2(i * circle_width + radius, j * circle_width + radius);
                if(pos.x <= width - radius
                   && pos.y <= height - radius
                   && pos.x >= radius
                   && pos.y >= radius)
                {
                    pos += offset;
                    
                    float w = roundf(radius * 0.9 * scale);
                    if(i%2 == 0)
                    {
                        cv::circle(image, pos, w, White, j %2 == 0 ? cv::FILLED : 1);
                        
                        /*if(j%2 == 0) {
                            if(_gaussCircleProblem(w) != gaussCircleProblem(w, cv::FILLED)) {
                                U_EXCEPTION("No");
                            }
                        }*/
                        
                        num_pixels += gaussCircleProblem(w, j %2 == 0 ? cv::FILLED : 1);
                    }
                    else {
                        cv::rectangle(image, pos - Vec2(w), pos + Vec2(w), White, j %2 == 0 ? cv::FILLED : 1);
                        num_pixels += j % 2 ? (SQR((pos.x + w - (pos.x - w)) + 1)) : ((pos.x + w - (pos.x - w)) * 4);
                    }
                    ++number_objects;
                }
            }
        }
        
        //auto dim = Size2(image) - 1;
        //cv::rectangle(image, Vec2(), dim, White, 1);
        //num_pixels += (dim.width * 4);
        
        return {number_objects, num_pixels};
    }
    
    static size_t generate_ellipses(cv::Mat& image, Vec2 offset = Vec2(), float scale = 1) {
        assert(!image.empty());
        
        constexpr size_t number = 10;
        const size_t circle_width = image.cols / number;
        const size_t radius = circle_width * 0.5;
        size_t number_objects = 0;
        size_t width = image.cols, height = image.rows;
        
        for (size_t i=0; i<number; ++i) {
            for (size_t j=0; j<number; ++j) {
                Vec2 pos = Vec2(i * circle_width + radius, j * circle_width + radius) + offset;
                if(pos.x <= width
                   && pos.y <= height
                   && pos.x >= radius
                   && pos.y >= radius)
                {
                    pos += offset;
                    cv::ellipse(image, pos + radius, (cv::Size)(Size2(radius, radius * 0.5) * 0.9 * scale), 45, 0, 360, White, cv::FILLED);
                    
                    ++number_objects;
                }
            }
        }
        
        return number_objects;
    }
    
    virtual void evaluate() override {
        cv::Mat image = cv::Mat::zeros(200, 200, CV_8UC1);
        auto expected = generate_circles(image);
        cv::imshow("image", image);
        cv::waitKey(1);
        
        auto blobs = CPULabeling::run_fast(image);
        ASSERT(blobs.size() == std::get<0>(expected));
        
        cv::Mat colored;
        cv::cvtColor(image, colored, cv::COLOR_GRAY2BGR);
        
        ColorWheel wheel;
        size_t npixels = 0;
        for(auto && [lines, pixels] : blobs) {
            npixels += pixels->size();
            
            auto color = wheel.next();
            for(auto &l : *lines) {
                for (auto x=l.x0; x<=l.x1; ++x) {
                    colored.at<cv::Vec3b>(l.y, x) = cv::Vec3b(color.r, color.g, color.b);
                }
            }
        }
        
        cv::imshow("blobs", colored);
        cv::waitKey(0);
        
        size_t pixel_count = 0;
        for(int y=0; y<image.rows; ++y) {
            for (int x=0; x<image.cols; ++x) {
                if(image.at<uchar>(y, x) > 0)
                    ++pixel_count;
            }
        }
        ASSERT(npixels == pixel_count);
        
        cv::Mat compare = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
        
        for(auto && [lines, pixels] : blobs) {
            for(auto &line : *lines) {
                for(auto x = line.x0; x <= line.x1; ++x) {
                    compare.at<uchar>(line.y, x) = 255;
                }
            }
        }
        
        // --- check whether the pixels in the blobs are
        //     actually all the pixels in the image:
        image.convertTo(image, CV_32FC1);
        compare.convertTo(compare, CV_32FC1);
        
        auto sum = cv::sum(compare - image);
        ASSERT(sum[0] == 0);
    }
};

using namespace track;

class Split {
public:
    static std::vector<pv::BlobPtr> split(const cv::Mat& greyscale, std::shared_ptr<Blob> original_blob, size_t expected = 2) {
        cv::Mat tmp;
        std::vector<pv::BlobPtr> result;
        
        Timer timer;
        // Perform the distance transform algorithm
        cv::distanceTransform(greyscale, tmp, cv::DIST_L2, 3);
        
        // Normalize the distance image for range = {0.0, 1.0}
        // so we can visualize and threshold it
        cv::normalize(tmp, tmp, 0, 1.0, cv::NORM_MINMAX);
        tmp.mul(tmp);
        
        Debug("Took %fms", timer.elapsed() * 1000);
        
        cv::imshow("dt", tmp);
        
        tmp.convertTo(tmp, CV_8UC1, 255.0);
        
        cv::Mat markersbg = cv::Mat::ones(tmp.rows, tmp.cols, CV_32SC1);
        for(auto &line : original_blob->hor_lines()) {
            for (auto x=line.x0; x<=line.x1; ++x)
                markersbg.at<int>(line.y - original_blob->bounds().y, x - original_blob->bounds().x) = 0;
        }
        
        
        for(int i=100; i<255; i+=5) {
            cv::Mat binary;
            cv::threshold(tmp, binary, i, 255, cv::THRESH_BINARY);
            
            auto blobs = CPULabeling::run_fast(binary);
            if(blobs.size() >= 2) {
                size_t overall_pixels = 0;
                std::set<std::tuple<size_t, blobs_t::value_type>, std::greater<>> sorted;
                for(auto && [lines, pixels] : blobs) {
                    overall_pixels += pixels->size();
                    sorted.insert({pixels->size(), {lines, pixels}});
                }
                
                //Debug("All pixels: %d", overall_pixels);
                //auto str = Meta::toStr(sorted);
                //Debug("%S", &str);
                
                size_t index = 0;
                size_t collected = 0;
                for(auto && [c, b] : sorted) {
                    collected += c;
                    
                    if(overall_pixels - collected < overall_pixels * (1.f / expected) * 0.5) {
                        break;
                    }
                    ++index;
                }
                
                //Debug("Index: %d", index);
                if(index < expected - 1)
                    continue;
                
                //cv::cvtColor(binary, binary, cv::COLOR_GRAY2BGR);
                
                cv::Mat markers;
                markersbg.copyTo(markers);
                
                int idx = 1;
                
                for (auto && [c, blob]: sorted) {
                    ++idx;
                    
                    for (auto &line : *std::get<0>(blob)) {
                        int* start = (int*)markers.data + line.y * markers.cols;
                        std::fill(start + line.x0, start + line.x1, idx);
                    }
                    
                    //if(idx >= expected+1)
                    //    break;
                }
                
                cv::Mat colored;
                cv::cvtColor(greyscale, colored, cv::COLOR_GRAY2BGR);
                cv::watershed(colored, markers);
                
                cv::Mat temp;
                cv::normalize(markers, temp, 0, 255, cv::NORM_MINMAX);
                temp.convertTo(temp, CV_8UC1);
                cv::imshow("temp", temp);
                cv::waitKey();
                
                cv::Mat masked = cv::Mat::zeros(markers.rows, markers.cols, CV_8UC1);
                std::set<int> value_set;
                
                std::vector<std::shared_ptr<std::vector<HorizontalLine>>> line_arrays;
                std::vector<std::shared_ptr<std::vector<uchar>>> pixel_arrays;
                
                for(int i=0; i<idx-1; ++i) {
                    line_arrays.push_back(std::make_shared<std::vector<HorizontalLine>>());
                    pixel_arrays.push_back(std::make_shared<std::vector<uchar>>());
                }
                
                std::vector<HorizontalLine> unsorted;
                size_t count_sorted = 0, count_unsorted = 0;
                
                for(auto &line : original_blob->hor_lines()) {
                    int y = line.y - original_blob->bounds().y;
                    for(auto x = line.x0 - original_blob->bounds().x; x<= line.x1 - original_blob->bounds().x; ++x) {
                        int val = markers.at<int>(y, x);
                        if(val < 2)
                            val = 0;
                        else --val;
                        
                        if(!val) {
                            if(unsorted.empty() || unsorted.back().y != y || x > unsorted.back().x1 + 1) {
                                unsorted.push_back(HorizontalLine(y, x, x));
                            } else
                                unsorted.back().x1 = x;
                            ++count_unsorted;
                        }
                        else {
                            assert(val >= 1 && val < idx);
                            if(!line_arrays[val - 1]->empty() && line_arrays[val - 1]->back().y == y && (int)line_arrays[val - 1]->back().x1 == x-1)
                                line_arrays[val - 1]->back().x1 = x; // continue line
                            else
                                line_arrays[val - 1]->push_back(HorizontalLine(y, x, x)); // create new line
                            pixel_arrays[val - 1]->push_back(greyscale.at<uchar>(y, x));
                            ++count_sorted;
                        }
                    }
                }
                
                Debug("Unsorted pixels %d / %d, lines: %d", count_unsorted, count_unsorted + count_sorted, unsorted.size());
                std::map<pv::BlobPtr, int> blob2index;
                for(size_t i=0; i<line_arrays.size(); ++i) {
                    if(pixel_arrays.at(i)->empty())
                        continue;
                    
                    Debug("Lines %d pixels %d", line_arrays.at(i)->size(), pixel_arrays.at(i)->size());
                    result.push_back(std::make_shared<pv::Blob>(line_arrays.at(i), pixel_arrays.at(i)));
                    result.back()->add_offset(original_blob->bounds().pos());
                    
                    blob2index[result.back()] = i + 2;
                    
                    auto && [pos, img] = result.back()->image();
                    cv::imshow(Meta::toStr(result.back()->blob_id()), img->get());
                }
                
                
                //cv::Mat temp;
                
                
                for(auto &line : unsorted) {
                    auto start = (int*)markers.data + markers.cols * line.y + line.x0;
                    auto end = start + line.x1 - line.x0 + 1;
                    const auto s = start;
                    
                    for(; start != end; ++start) {
                        pv::BlobPtr found = nullptr;
                        float min_d = infinity<float>();
                        for (auto blob : result) {
                            auto d = sqdistance(blob->bounds().pos() + blob->bounds().size() * 0.5, Vec2(line.x0 + start - s, line.y));
                            if(d < min_d) {
                                found = blob;
                                min_d = d;
                            }
                        }
                        
                        if(found) {
                            *start = blob2index.at(found);
                        }
                    }
                    
                    //std::fill(start, end, 0);
                }
                
                cv::normalize(markers, temp, 0, 255, cv::NORM_MINMAX);
                temp.convertTo(temp, CV_8UC1);
                cv::imshow("markers", temp);
                
                Debug("Blobs: %d (%d) -> %d", blobs.size(), i, result.size());
                break;
            }
        }
        
        cv::imshow("thresh", tmp);
        return result;
    }
};

class TestTracking : public UnitTest {
public:
    TestTracking() : UnitTest("Tracking"), current_frame(0) {
        bg = cv::Mat::zeros(1000, 1000, CV_8UC1);
    }
    
    size_t current_frame;
    cv::Mat bg;
    
    std::tuple<cv::Mat, blobs_t> generate_frame(Tracker& tracker, const Vec2& offset) {
        PPFrame frame;
        frame.frame().set_index(current_frame++);
        
        cv::Mat image;
        
        bg.copyTo(image);
        TestLabeling::generate_ellipses(image, offset, 0.5);
        //TestLabeling::generate_circles(image, offset, 0.5);
        cv::imshow("circles", image);
        
        auto blobs = CPULabeling::run_fast(image);
        for(auto &&[lines, pixels] : blobs)
            frame.frame().add_object(*lines, image);
        double time = 0.05 * frame.index();
        frame.frame().set_timestamp(1000 * 1000 * time);
        
        if(frame.index() > 0) {
            PPFrame tmp;
            tracker.preprocess_frame(frame, tracker.active_individuals(frame.index()-1), NULL, nullptr);
            //auto automatic_assignments = Tracker::blob_automatically_assigned(frame.index());
            
            for(auto && [id, fish]: tracker.individuals()) {
                auto cache = fish->cache_for_frame(tmp.index(), tmp.time);
                //std::map<long, Individual::Probability> probs;
                Individual::Probability max_p;
                max_p.p = 0;
                long blob_id = -1;
                for(auto blob : tmp.blobs) {
                    auto p = fish->probability(cache, frame.index(), blob);
                    Debug("%f %f", p.p_time, p.p_pos);
                    if(p.p > max_p.p) {
                        max_p = p;
                        blob_id = blob->blob_id();
                    }
                }
                
                Debug("Individual %d -> %d: %f", id, blob_id, max_p.p);
            }
        }
        
        tracker.add(frame);
        
        Tracker::LockGuard guard("testtracking");
        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
        
        for(auto && [id, fish] : tracker.individuals()) {
            Vec2 prev(infinity<Float2_t>());
            // visual studio 2017
            const auto range = arange<long>(fish->start_frame(), fish->end_frame());
            for(auto idx : range) {
                if(fish->has(idx)) {
                    if(!isinf(prev.x)) {
                        cv::line(image, prev, fish->centroid(idx)->pos(Units::PX_AND_SECONDS), fish->identity().color());
                    }
                    prev = fish->centroid(idx)->pos(Units::PX_AND_SECONDS);
                }
            }
            
            if(fish->has(frame.index())) {
                cv::circle(image, fish->centroid(frame.index())->pos(Units::PX_AND_SECONDS), 3, fish->identity().color(), cv::FILLED);
                cv::putText(image, fish->identity().raw_name(), fish->centroid(frame.index())->pos(Units::PX_AND_SECONDS), cv::FONT_HERSHEY_PLAIN, 0.5, White);
            }
            auto outline = fish->outline(frame.index());
            
            if(outline) {
                Vec2 pos = fish->blob(frame.index())->bounds().pos();
                auto points = outline->uncompress();
                for(size_t i = 0; i< points.size(); ++i) {
                    cv::line(image, pos + points.at(i % points.size()), pos + points.at((i+1) % points.size()), Red);
                }
            }
        }
        
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        
        return {image, blobs};
    }
    
    virtual void evaluate() override {
        
        
        DebugHeader("LOADING DEFAULT SETTINGS");
        default_config::get(GlobalSettings::map(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
        default_config::get(GlobalSettings::set_defaults(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
        
        if(SETTING(meta_real_width).value<float>() == 0)
            SETTING(meta_real_width) = float(30.0);
        
        // setting cm_per_pixel after average has been generated (and offsets have been set)
        if(!GlobalSettings::map().has("cm_per_pixel") || SETTING(cm_per_pixel).value<float>() == 0)
            SETTING(cm_per_pixel) = SETTING(meta_real_width).value<float>() / float(bg.cols);
        
        SETTING(track_max_speed) = float(5);
        SETTING(blob_size_range) = Rangef(0.0001, 10000);
        SETTING(track_max_reassign_time) = float(0.1);
        SETTING(frame_rate) = int(30);
        SETTING(matching_probability_threshold) = 0.5f;
        //SETTING(history_matching_log) = file::Path("history.html");
        //SETTING(debug) = true;
        
        auto tmp_points = std::make_shared<std::vector<Vec2>>();
        Outline out(tmp_points);
        
        using namespace track;
        Tracker tracker;
        
        SETTING(video_size) = Size2(bg.cols, bg.rows);
        SETTING(video_mask) = false;
        SETTING(video_length) = size_t(3);
        SETTING(track_max_individuals) = idx_t(100);
        SETTING(track_posture_threshold) = int(30);
        SETTING(outline_resample) = 1.f;
        SETTING(midline_resolution) = int(20);
        SETTING(posture_head_percentage) = float(0.2);
        
        if(SETTING(manual_identities).value<std::vector<idx_t>>().empty() && SETTING(track_max_individuals).value<idx_t>() != 0)
        {
            std::set<idx_t> vector;
            for(idx_t i=0; i<SETTING(track_max_individuals).value<idx_t>(); ++i) {
                vector.insert(i);
            }
            SETTING(manual_identities) = vector;
        }
        
        tracker.set_average(bg);
        
        
#if __APPLE__
        std::string root = "../../../";
#else
        std::string root = "../../";
#endif
        
        std::vector<cv::Mat> vec;
        {
            cv::Mat image = cv::imread(root+"penbackground.png");
            if(image.empty())
                U_EXCEPTION("Cannot load background.png");
            cv::split(image, vec);
        }
        
        Image bgimage(vec[0]);
        LuminanceGrid luminance(vec[0]);
        StaticBackground blob_background(bgimage, &luminance);
        
        std::vector<std::string> images = {
            "dot.png",
            "blob40305448.png",
            "image_fish_overlap_2.png",
            "image_fish_overlap.png",
            "piece.png",
            "termite_overlap_full.png",
            "termites.png",
            "termites_three.png",
            
            "image_fish.png"
        };
        
        for(auto name : images) {
            pv::Frame frame;
            
            auto path = root + name;
            cv::Mat image = cv::imread(path);
            if(image.empty())
                U_EXCEPTION("Cannot load image %S", &path);
            std::vector<cv::Mat> vec;
            cv::split(image, vec);
            image = vec[0];
            
            if(name == "blob40305448.png") {
                /*auto pixels = std::make_shared<std::vector<uchar>>(std::vector<uchar>{24,26,31,32,21,24,28,30,31,31,21,24,28,30,26,17,17,17,17,18,18,18,18,37,26,21,18,17,17,17,17,18,18,18,18,27,24,21,18,14,14,17,17,17,17,18,18,18,18,37,17,11,14,14,14,14,17,17,17,17,18,18,18,18,37,26,20,16,9,14,14,17,18,14,13,12,12,14,14,14,14,35,18,16,14,9,21,20,18,17,12,12,11,11,14,14,14,14,26,15,16,10,5,2,20,14,11,11,11,11,14,14,14,14,11,1,13,7,4,4,21,14,11,11,11,11,14,14,14,14,19,5,11,2,3,4,20,8,5,5,5,5,11,11,11,28,18,6,12,2,0,0,26,12,5,5,5,5,5,11,11,19,29,11,1,8,7,2,2,9,5,1,0,5,5,5,5,11,11,28,28,8,0,2,1,1,2,0,0,0,0,5,5,5,5,11,19,17,20,19,12,10,8,11,14,4,4,4,4,1,5,11,14,19,24,22,22,20,11,10,8,6,9,2,2,2,2,0,4,9,13,23,27,30,19,11,8,4,4,7,7,7,7,7,5,8,14,18,28,34,27,23,15,10,7,3,2,5,7,7,7,7,5,8,14,18,23,13,8,7,7,7,7,13,9,11,19,22,23,25,15,12,9,8,9,9,11,15,18,25,23,21,19,13,10,12,11,11,18,24,26,20,18,21,16,10,16,20,19,25,25,30,24,23,17,12,18,28,26,23,20,19,19,25,24,25,23,19,13,17,21,25,23,20,17,18,19,14,19,29,23,24,14,13,12,14,24,29,26,20,20,14,22,21,24,24,20,16,17,12,16,16,16,17,16,24,18,7,12,13,19,19,22,28,23,15,9,5,10,10,19,19,29,16,15,14,8,7,8,13,17,32,18,11,14,8,7,7,8,9,16,9,11,6,6,6,6,7,14,23,25,14,6,8,4,8,8,9,9,13,23,29,15,7,9,6,9,10,11,11,15,25,28,22,19,14,10,13,13,11,11,16,28,29,28,17,14,15,15,15,16,18,20,16,18,17,18,20,24,31,28,30});
                auto lines = std::make_shared<std::vector<HorizontalLine>>(std::vector<HorizontalLine>{HorizontalLine(4,28,31),HorizontalLine(5,28,31),HorizontalLine(6,26,31),HorizontalLine(7,23,32),HorizontalLine(8,21,31),HorizontalLine(9,18,32),HorizontalLine(10,18,32),HorizontalLine(11,16,32),HorizontalLine(12,16,31),HorizontalLine(13,14,19),HorizontalLine(13,22,31),HorizontalLine(14,14,19),HorizontalLine(14,22,31),HorizontalLine(15,14,19),HorizontalLine(15,22,31),HorizontalLine(16,14,19),HorizontalLine(16,21,30),HorizontalLine(17,13,30),HorizontalLine(18,13,29),HorizontalLine(19,12,29),HorizontalLine(20,12,29),HorizontalLine(21,13,29),HorizontalLine(22,12,27),HorizontalLine(23,13,25),HorizontalLine(24,13,23),HorizontalLine(25,13,23),HorizontalLine(26,13,22),HorizontalLine(27,13,20),HorizontalLine(28,15,19),HorizontalLine(29,12,19),HorizontalLine(30,11,18),HorizontalLine(31,9,16),HorizontalLine(32,8,14),HorizontalLine(33,7,11),HorizontalLine(34,7,12),HorizontalLine(35,6,13),HorizontalLine(36,4,12),HorizontalLine(37,4,11),HorizontalLine(37,13,13),HorizontalLine(38,4,11),HorizontalLine(39,4,14),HorizontalLine(40,4,14),HorizontalLine(41,4,14),HorizontalLine(42,4,13),HorizontalLine(43,4,12),HorizontalLine(44,6,12),HorizontalLine(45,6,8)});*/
                /*auto pixels = std::make_shared<std::vector<uchar>>(std::vector<uchar>{ 133,136,133,130,129,135,131,131,135,135,135,126,121,123,117,105,112,110,117,110,111,104,95,89,102,131,114,111,100,89,83,79,78,79,78,83,93,94,89,76,65,48,58,90,128,135,132,130,122,112,97,87,85,79,66,61,58,58,58,58,64,73,73,71,66,60,53,58,74,107,133,130,125,123,122,118,115,104,96,81,79,67,67,52,47,41,43,44,44,50,59,58,54,57,60,67,73,90,109,125,135,130,126,124,122,119,111,108,103,94,87,88,66,68,52,53,51,47,43,43,43,43,48,58,53,48,52,59,72,83,103,115,118,130,128,125,123,118,124,117,122,119,118,115,114,112,111,108,105,96,91,85,79,64,61,58,57,45,45,45,45,45,46,50,52,53,53,54,57,60,71,100,121,125,133,132,128,124,122,119,116,115,110,116,109,114,111,110,107,108,112,114,110,107,100,95,85,74,67,65,62,58,53,52,51,52,55,57,60,64,69,67,61,58,44,45,75,109,131,136,135,130,126,125,121,126,119,124,122,121,117,117,117,117,114,112,115,109,102,94,87,85,82,78,76,75,74,75,79,80,83,87,89,87,81,78,66,62,85,114,135,132,132,132,132,129,126,131,125,124,124,123,123,119,116,114,114,112,112,112,114,117,117,116,116,114,112,107,107,117,130,133,131,129,132,133,132,131
                });
                auto lines = std::make_shared<std::vector<HorizontalLine>>(std::vector<HorizontalLine>{ HorizontalLine(0,36,36),HorizontalLine(1,27,27),HorizontalLine(1,29,29),HorizontalLine(1,31,31),HorizontalLine(1,36,37),HorizontalLine(1,39,41),HorizontalLine(2,26,41),HorizontalLine(3,24,43),HorizontalLine(4,19,44),HorizontalLine(5,16,44),HorizontalLine(6,12,44),HorizontalLine(7,4,44),HorizontalLine(8,0,44),HorizontalLine(9,3,43),HorizontalLine(10,10,10),HorizontalLine(10,14,43),HorizontalLine(11,26,26),HorizontalLine(11,28,28),HorizontalLine(11,30,30),HorizontalLine(11,33,33),HorizontalLine(11,35,35),HorizontalLine(11,42,43)
                });*/
                
                /*auto pixels = std::make_shared<std::vector<uchar>>(std::vector<uchar>{59,57,55,53,41,48,58,61,59,64,64,58,57,61,61,61,64,65,55,51,54,54,54,54,53,52,51,51,50,50,50,50,51,51,52,53,60,54,53,54,51,50,48,47,46,47,46,46,46,46,46,47,47,47,51,60,62,62,54,55,55,53,51,48,46,45,45,45,45,44,44,45,47,47,48,51,54,59,64,54,41,43,53,61,61,60,57,52,43,41,44,44,43,44,44,48,50,53,57,60,65,50,38,43,55,65,65,53,44,41,46,46,47,50,54,53,59,62,66,64,61,65});
                auto lines = std::make_shared<std::vector<HorizontalLine>>(std::vector<HorizontalLine>{HorizontalLine(0,0,2),HorizontalLine(1,0,4),HorizontalLine(1,9,18),HorizontalLine(2,1,19),HorizontalLine(3,2,21),HorizontalLine(4,1,22),HorizontalLine(5,0,21),HorizontalLine(6,0,4),HorizontalLine(6,8,18),HorizontalLine(7,1,2),HorizontalLine(7,10,12)});*/
                
                /*
                 HorizontalLine(7,11,16),HorizontalLine(8,10,21),HorizontalLine(8,32,34),HorizontalLine(9,8,35),HorizontalLine(10,7,38),HorizontalLine(10,40,41),HorizontalLine(11,6,45),HorizontalLine(12,5,48),HorizontalLine(12,50,50),HorizontalLine(13,4,52),HorizontalLine(14,4,55),HorizontalLine(15,4,57),HorizontalLine(16,4,60),HorizontalLine(17,4,63),HorizontalLine(18,4,66),HorizontalLine(18,87,87),HorizontalLine(19,5,71),HorizontalLine(19,86,86),HorizontalLine(20,5,75),HorizontalLine(20,85,85),HorizontalLine(20,88,89),HorizontalLine(20,94,97),HorizontalLine(21,6,89),HorizontalLine(21,93,98),HorizontalLine(22,7,99),HorizontalLine(23,8,100),HorizontalLine(24,9,101),HorizontalLine(25,11,100),HorizontalLine(26,14,99),HorizontalLine(27,19,98),HorizontalLine(28,20,96),HorizontalLine(29,15,95),HorizontalLine(30,18,94),HorizontalLine(31,21,93),HorizontalLine(32,25,92),HorizontalLine(33,27,91),HorizontalLine(34,28,34),HorizontalLine(34,36,86),HorizontalLine(35,31,31),HorizontalLine(35,41,77),HorizontalLine(35,79,81),HorizontalLine(36,67,67)
                 
                 */
                
                auto lines = std::make_shared<std::vector<HorizontalLine>>(std::vector<HorizontalLine>{HorizontalLine(7,11,16),HorizontalLine(8,10,21),HorizontalLine(8,32,34),HorizontalLine(9,8,35),HorizontalLine(10,7,38),HorizontalLine(10,40,41),HorizontalLine(11,6,45),HorizontalLine(12,5,48),HorizontalLine(12,50,50),HorizontalLine(13,4,52),HorizontalLine(14,4,55),HorizontalLine(15,4,57),HorizontalLine(16,4,60),HorizontalLine(17,4,63),HorizontalLine(18,4,66),HorizontalLine(18,87,87),HorizontalLine(19,5,71),HorizontalLine(19,86,86),HorizontalLine(20,5,75),HorizontalLine(20,85,85),HorizontalLine(20,88,89),HorizontalLine(20,94,97),HorizontalLine(21,6,89),HorizontalLine(21,93,98),HorizontalLine(22,7,99),HorizontalLine(23,8,100),HorizontalLine(24,9,101),HorizontalLine(25,11,100),HorizontalLine(26,14,99),HorizontalLine(27,19,98),HorizontalLine(28,20,96),HorizontalLine(29,15,95),HorizontalLine(30,18,94),HorizontalLine(31,21,93),HorizontalLine(32,25,92),HorizontalLine(33,27,91),HorizontalLine(33,83,89),HorizontalLine(34,28,34),HorizontalLine(34,36,86),HorizontalLine(34,60,80),HorizontalLine(34,84,86),HorizontalLine(35,31,31),HorizontalLine(35,41,77),HorizontalLine(35,79,81),HorizontalLine(35,50,52),HorizontalLine(35,53,55),HorizontalLine(35,56,58),HorizontalLine(35,61,68),HorizontalLine(35,69,76),HorizontalLine(35,77,77),HorizontalLine(35,79,79),HorizontalLine(36,67,67)});
                auto pixels = std::make_shared<std::vector<uchar>>(std::vector<uchar>{187,181,175,174,177,181,179,170,155,143,133,133,146,156,166,175,179,181,182,178,178,183,173,155,132,113,101,93,91,99,110,122,140,155,158,161,164,168,174,178,179,178,178,175,171,167,167,170,177,177,167,156,134,103,81,69,65,65,68,76,89,107,124,133,137,139,144,148,153,157,156,156,155,151,147,147,155,164,167,170,173,177,179,177,163,153,144,126,96,72,60,54,53,55,61,73,89,102,112,115,116,121,124,128,131,132,131,131,127,125,127,132,137,139,144,145,150,154,153,159,164,167,172,178,164,153,146,140,130,106,78,62,53,50,51,58,68,78,87,93,97,99,104,107,108,110,110,107,107,106,103,104,103,105,108,110,115,115,120,121,129,134,136,145,152,162,166,177,178,168,158,146,139,138,131,116,95,74,61,56,55,59,64,69,74,80,83,87,91,91,92,90,86,87,87,87,83,81,80,81,79,82,83,88,90,92,98,102,109,115,120,129,140,151,151,173,173,180,167,154,144,139,141,138,125,109,92,76,66,62,63,64,65,69,73,74,78,79,79,79,76,70,69,70,70,64,63,64,62,60,66,65,66,67,71,72,79,86,89,97,106,116,121,134,138,147,161,165,175,175,164,155,145,144,145,141,130,119,104,86,76,68,64,61,61,64,65,66,70,73,71,68,66,62,59,58,57,53,52,51,50,50,52,53,57,55,58,63,63,69,75,80,86,93,106,106,115,121,129,142,153,159,170,175,163,151,144,145,144,142,133,120,106,89,78,69,63,60,59,61,60,60,64,67,65,62,59,56,55,53,51,49,47,45,44,46,46,46,51,49,51,54,55,62,63,70,77,80,85,89,95,103,110,118,129,138,146,154,165,176,174,160,146,140,141,140,135,125,112,102,87,77,69,62,60,59,58,57,58,60,61,61,59,56,53,53,49,48,46,45,44,42,41,43,43,46,46,48,50,52,57,59,63,64,68,73,76,82,89,95,102,109,114,124,132,142,151,160,170,177,180,164,147,135,133,134,124,112,101,93,82,73,68,63,60,60,58,56,56,56,57,57,57,54,52,51,50,48,46,44,45,42,42,42,44,44,45,46,48,50,52,52,60,59,64,68,70,74,77,83,88,95,102,107,115,124,133,137,144,153,161,166,173,177,173,154,141,130,123,110,95,81,72,69,66,65,63,61,61,59,58,58,57,57,55,56,55,52,51,50,48,46,45,46,44,43,44,44,44,45,44,50,47,50,51,52,57,60,62,64,68,73,76,79,83,90,96,103,112,120,123,127,132,136,143,149,156,163,168,173,180,177,183,167,151,130,109,89,74,63,57,56,59,62,63,61,62,60,61,61,62,59,57,57,55,51,50,51,47,45,44,42,43,43,44,46,43,44,45,46,50,50,50,53,56,56,56,56,63,66,69,74,77,84,87,93,101,108,112,118,121,123,129,132,136,142,146,152,158,161,167,171,174,176,171,171,160,152,145,136,182,162,139,105,80,65,56,51,52,55,62,64,65,66,66,67,69,67,64,60,59,54,52,49,49,46,46,44,42,41,42,44,45,45,43,45,48,49,49,50,49,52,55,55,56,59,61,65,70,72,77,80,87,93,99,104,110,114,119,122,124,126,127,128,132,135,138,143,148,153,160,161,162,165,167,166,168,166,167,168,169,167,163,164,156,147,141,133,126,116,176,155,117,87,72,61,56,54,56,63,69,74,74,73,75,74,71,68,66,60,55,53,51,49,49,46,46,46,43,44,45,44,45,44,47,48,48,50,50,49,52,54,55,56,58,59,62,68,70,73,76,82,87,94,99,105,110,113,117,119,121,121,119,121,125,125,127,130,132,136,137,139,141,144,144,146,151,148,146,145,145,147,142,142,142,138,133,128,126,118,111,106,103,176,148,121,102,85,74,67,65,71,77,84,85,83,82,79,78,73,71,65,61,55,55,52,48,48,48,45,44,45,45,47,47,46,48,47,50,50,51,52,52,55,55,56,57,58,61,65,68,72,75,78,81,89,95,100,103,107,111,114,115,117,118,117,117,117,121,122,121,122,124,126,127,126,129,131,127,127,126,129,127,125,126,124,123,120,116,114,110,108,103,99,94,95,173,161,143,125,115,104,94,96,97,101,99,96,93,90,90,83,78,73,68,63,58,57,53,50,49,48,47,47,47,45,48,49,51,52,54,52,53,55,55,57,57,58,57,59,62,65,68,70,74,77,79,84,90,95,98,103,105,107,109,111,113,113,112,111,114,115,115,117,118,121,122,121,122,117,117,119,118,118,117,116,117,111,115,111,106,103,104,99,97,96,92,90,90,174,162,159,153,140,134,130,129,119,113,110,106,103,98,92,87,80,75,68,66,60,58,55,55,50,51,53,49,49,50,51,54,54,55,57,58,60,60,61,62,63,63,65,69,70,71,74,78,80,84,87,91,96,99,102,104,105,105,105,107,107,107,109,112,113,114,117,117,118,120,116,116,118,115,116,113,112,105,106,108,105,104,101,101,100,94,96,94,92,91,177,172,169,166,159,151,142,137,131,126,120,114,105,97,89,83,79,74,69,66,65,59,61,56,56,57,55,55,56,56,59,61,58,62,63,64,65,67,68,70,71,74,75,77,81,82,86,90,92,95,99,101,103,104,103,105,108,107,104,106,109,111,113,112,113,112,111,110,112,116,109,107,106,104,103,103,100,100,100,99,101,98,93,94,91,89,178,167,165,160,155,149,141,132,120,109,101,95,91,86,87,81,74,72,69,64,63,62,63,60,63,63,63,66,68,67,69,68,70,73,75,74,78,79,82,86,87,90,95,97,98,100,103,105,106,106,107,107,109,107,104,105,107,109,108,107,108,108,108,106,106,107,105,100,102,102,101,103,102,101,98,97,96,92,91,88,178,175,175,175,169,161,158,148,132,124,115,110,113,112,106,98,91,86,81,75,74,75,73,72,73,72,74,75,75,76,76,77,78,79,81,84,86,87,90,92,94,98,100,100,103,105,106,107,107,108,107,107,107,106,108,108,111,109,108,108,106,107,108,107,107,106,108,106,108,103,106,106,101,98,95,94,90,173,174,176,175,174,174,172,170,167,167,165,163,157,152,141,134,121,120,127,132,131,126,116,103,96,93,88,89,88,86,83,85,87,87,86,87,86,87,89,90,90,93,95,96,97,98,99,101,102,103,104,103,106,105,105,108,108,104,105,106,107,109,108,107,106,110,109,103,108,106,106,106,104,106,104,108,105,101,98,95,91,155,155,154,155,153,146,150,151,148,147,143,139,137,130,124,127,134,135,133,133,128,124,115,109,107,104,102,101,101,101,100,98,98,98,100,101,102,101,102,103,102,104,104,103,104,105,104,104,104,105,105,105,105,105,104,104,104,104,106,108,109,104,107,105,107,106,104,101,101,104,108,106,108,104,100,97,92,133,129,128,129,129,127,123,122,122,119,119,119,121,119,126,125,122,119,116,114,111,112,110,107,106,107,107,108,106,105,105,105,106,107,106,107,107,107,107,106,106,107,108,107,107,109,109,110,109,109,107,106,109,108,107,108,107,107,105,109,106,103,104,106,103,101,105,108,103,104,101,99,96,113,110,114,112,109,110,108,107,110,115,118,116,115,110,110,110,105,103,103,103,102,104,105,105,105,103,103,103,104,104,103,105,105,105,106,105,105,107,107,107,107,108,106,108,108,106,104,105,106,107,104,107,104,104,104,103,102,102,100,101,98,99,101,98,99,99,97,98,102,103,101,102,101,102,102,106,112,115,114,110,107,101,101,100,99,98,99,100,100,100,100,100,99,100,101,100,101,101,100,100,102,102,102,102,101,100,101,102,101,101,102,100,100,99,100,101,99,102,102,100,102,97,99,96,96,96,99,95,97,95,96,94,94,96,96,99,95,97,95,96,100,97,98,97,97,99,100,109,112,111,106,102,101,100,99,98,98,97,96,96,97,97,96,97,97,97,99,99,98,98,98,98,97,96,96,98,99,101,99,98,98,98,98,95,95,96,97,94,96,96,96,95,93,94,93,93,96,91,98,97,96,96,98,99,101,99,98,98,98,98,95,95,96,97,94,96,96,96,95,93,96,91,97,100,98,98,96,98,97,94,95,96,95,94,95,96,95,96,98,97,96,97,97,95,95,96,96,95,97,95,95,94,96,94,93,95,95,95,94,95,94,92,93,95,94,95,96,95,96,98,97,96,95,95,96,96,95,97,95,95,94,96,94,93,95,95,95,94,95,94,92});
                
                HorizontalLine::repair_lines_array(*lines, *pixels);
                auto pxstr = Meta::toStr(*pixels);
                auto hlstr = Meta::toStr(*lines);
                
                printf("%s\n%s\n\n", pxstr.c_str(), hlstr.c_str());
                
                auto ptr = std::make_shared<pv::Blob>(lines, pixels);
                auto && [pos, image] = ptr->binary_image();
                cv::imwrite("/Users/tristan/blobby.png", image->get());
                cv::imshow("image", image->get());
                cv::waitKey(1);
                extract_outline(ptr, NULL, 0);
            } else {
                
                cv::Mat tmp;
                cv::threshold(image, tmp, 0, 255, cv::THRESH_BINARY);
                cv::imshow("thresholded", tmp);
                cv::waitKey(1);
                Debug("%S", &name);
                auto blobs = CPULabeling::run_fast(tmp);
                for(auto &&[lines, pixels] : blobs) {
                    auto pxstr = Meta::toStr(*pixels);
                    auto hlstr = Meta::toStr(*lines);
                    
                    printf("%s\n%s\n\n", pxstr.c_str(), hlstr.c_str());
                    frame.add_object(*lines, image);
                }
                double time = 0.05 * frame.index();
                frame.set_timestamp(1000 * 1000 * time);
                
                //assert(blobs.size() == 1);
                if(name == "blob40305448.png") {
                    //std::make_shared<pv::Blob>();
                    
                    extract_outline(std::make_shared<pv::Blob>(frame.mask().front(), frame.pixels().front()), NULL, 0);
                } else
                    extract_outline(std::make_shared<pv::Blob>(frame.mask().front(), frame.pixels().front()), NULL, 0);
            }
            
            /*Split split;
            blobs.front()->calculate_properties();
            split.split(image, blobs.front());*/
            
            cv::imshow("image", image);
            cv::waitKey();
        }
        
        Vec2 pos(200, 200);
        Vec2 head_pos(-50), v(-1);
        float angle = 0;
        bool circular = false;
        
        cv::Mat image = cv::Mat::zeros(500, 500, CV_8UC1), image_head = cv::Mat::zeros(500, 500, CV_8UC1);
        cv::Mat tmp;
        
        cv::imshow("image", image);
        cv::waitKey(0);
        
        int key = -1;
        while(key != 27) {
            static Timing timing("loop");
            TakeTiming take(timing);
            
            image.setTo(0);
            image_head.setTo(0);
            
            Vec2 p = pos + head_pos;
            
            if(circular) {
                v = Vec2(cos(angle), -sin(angle)) * 50 - head_pos;
                angle += M_PI * 0.05;
                
                if(angle >= M_PI * 2) {
                    circular = false;
                    angle = 0;
                    v = Vec2(-1);
                    head_pos = Vec2();
                }
                
            } else {
                if(head_pos.y < -50 && v.y < 0 && v.y == v.x) {
                    v = Vec2(1);
                } else if(head_pos.y > 50 && !circular && v.y == v.x) {
                    circular = true;
                }
            }
            
            head_pos += v;
            
            for (int i=0; i<255; ++i) {
                auto radius = (255 - i) / 255.f * 100;
                cv::ellipse(image, pos, (cv::Size)(Size2(radius, radius * 0.5) * 0.9), 45, 0, 360, cv::Scalar(i), cv::FILLED);
                cv::circle(image_head, p, (255 - i) / 255.f * 50, cv::Scalar(i), -1);
            }
            
            for(int y=0; y<image.rows; ++y) {
                for (int x=0; x<image.cols; ++x) {
                    image.at<uchar>(y, x) = min(255, float(image.at<uchar>(y, x)) + float(image_head.at<uchar>(y, x)));
                }
            }
            
            auto blobs = CPULabeling::run_fast(image);
            if(blobs.empty()) {
                Warning("No blobs?");
            } else {
                PPFrame frame;
                for(auto && [lines, pixels] : blobs)
                    frame.frame().add_object(*lines, image);
                double time = 0.05 * frame.index();
                frame.frame().set_timestamp(1000 * 1000 * time);
            
                PPFrame pp;
                tracker.preprocess_frame(frame, std::set<Individual*>(), NULL, nullptr);
                
                if(pp.blobs.empty()) {
                    cv::cvtColor(image, tmp, cv::COLOR_GRAY2BGR);
                    ColorWheel wheel;
                    for(auto && [lines, pixels] : blobs) {
                        auto color = wheel.next();
                        for(auto &line : *lines) {
                            for (auto x=line.x0; x<=line.x1; ++x) {
                                tmp.at<cv::Vec3b>(line.y, x) = Color(tmp.at<cv::Vec3b>(line.y, x)) + color;
                            }
                        }
                    }
                    
                    cv::imshow("image", tmp);
                    key = cv::waitKey(1);
                    Except("Blobs empty in PPFrame!");
                    continue;
                }
            
                auto blob = pp.blobs.front();
                auto && [pos, image] = blob->image();
                
                Posture posture(0, 0);
                //posture.calculate_posture(0, image->get(), Vec2());
                cv::cvtColor(image->get(), tmp, cv::COLOR_GRAY2BGR);
                
                if(posture.normalized_midline()) {
                    auto outline = posture.outline();
                    for(auto& pt : outline.points()) {
                        cv::circle(tmp, pt, 1, Red, -1);
                    }
                    
                    auto midline = posture.normalized_midline();
                    float head_offset = midline->len() * FAST_SETTINGS(posture_head_percentage);
                    float L = infinity<float>();
                    Vec2 prev;
                    Vec2 head_point(infinity<Float2_t>());
                    
                    for(auto &seg : midline->segments()) {
                        if(!isinf(L)) {
                            L += (prev - seg.pos).length();
                        } else
                            L = 0;
                        
                        if(L >= head_offset) {
                            float percent = head_offset == 0 ? 1 : L / head_offset;
                            //assert(percent <= 1);
                            
                            head_point = percent * seg.pos + (1 - percent) * prev;
                            break;
                        }
                        
                        prev = seg.pos;
                    }
                    
                    
                    ASSERT(!isinf(head_point.x));
                    
                    //size_t head_index = roundf(midline->segments().size() * FAST_SETTINGS(posture_head_percentage));
                    //Debug("%d", head_index);
                    auto pt = head_point;//midline->segments().at(head_index).pos;
                    
                    float angle = midline->angle() + M_PI;
                    float x = (pt.x * cmn::cos(angle) - pt.y * cmn::sin(angle));
                    float y = (pt.x * cmn::sin(angle) + pt.y * cmn::cos(angle));
                    
                    pt = Vec2(x, y);
                    pt += /*blob.bounds().pos() +*/ midline->offset();
                    
                    cv::circle(tmp, pt, 10, Red);
                }
                
                /*SplitBlob split(*Tracker::instance()->background(), blob);
                auto blobs = split.split(2);
                if(blobs.empty())
                    Debug("Cannot split.");*/
                {
                    Split split;
                    auto && [pos, image] = blob->image();//difference_image(*Tracker::instance()->background(), 0);
                    
                    if(image) {
                    cv::Mat mat;
                    image->get().copyTo(mat);
                    
                    cv::imshow("image", mat);
                    cv::waitKey();
                    
                    split.split(mat, blob);
                    } else
                        U_EXCEPTION("No pointer");
                }
                
                /*ColorWheel wheel;
                for(auto b : blobs) {
                    auto color = wheel.next();
                    for(auto &line : b->hor_lines()) {
                        for (auto x=line.x0; x<=line.x1; ++x) {
                            tmp.at<cv::Vec3b>(line.y, x) = color;
                        }
                    }
                }
                
                cv::cvtColor(tmp, tmp, cv::COLOR_BGR2RGB);*/
                
                cv::imshow("image", tmp);
            }
            
            key = cv::waitKey(10);
        }
        
        return;
        
        
        PPFrame frame;
        frame.frame().set_index(current_frame++);
        frame.set_index(frame.frame().index());
        tracker.add(frame);
        
        {
            Tracker::LockGuard guard("tests");
            
            ASSERT(tracker.properties(0) != NULL);
            //ASSERT(tracker.individuals().empty());
            ASSERT(tracker.active_individuals(0).empty());
        }
        
        long start_frame = current_frame;
        
        {
            auto [image, blobs] = generate_frame(tracker, Vec2(0));
            
            //cv::imshow("circles", image);
            //cv::waitKey(1);
            
        }
        
        for(int i=1; i<bg.cols; i+=1){
            auto [image, blobs] = generate_frame(tracker, Vec2(i));
            
            cv::imshow("circles", image);
            cv::waitKey(0);
            //ASSERT(tracker.individuals().size() == blobs.size());
        }
        
        Tracker::LockGuard guard("");
        for(auto && [id, fish] : tracker.individuals()) {
            ASSERT(fish->has(start_frame));
            ASSERT(!fish->has(current_frame-1));
        }
    }
};

class TestMisc : public UnitTest {
public:
    TestMisc() : UnitTest("Misc") {
        
    }
    
    virtual void evaluate() override {
        std::string pair = "\"key\":\"\"";
        Debug("Parsing '%S':", &pair);
        auto parts = parse_array_parts(pair, ':');
        for(auto &str : parts) {
            Debug("\t'%S'", &str);
        }
        
        sprite::Map map;
        map["key"] = std::string("bla");
        pair = "{\"key\":\"\"}";
        sprite::parse_values(map, pair);
    }
};

#include <misc/CircularGraph.h>
using namespace periodic;

class TestCircularGraphs : public UnitTest {
public:
    TestCircularGraphs() : UnitTest("CircularGraphs") {
        
    }
    
    virtual void evaluate() override {
        /**
         * Test EFT:
         */
        auto ranges = ExampleData::ranges();
        auto test_shape = std::make_shared<points_t::element_type>(ranges.points);
        auto diff = periodic::differentiate(test_shape)[0];
        auto && [cache_xy_sum, phi, cumsum, dt] = EFT::dt(diff);
        auto T = cumsum->back();
        for(auto &ptr : *phi) {
            ptr /= T;
        }
        
        auto coeffs = eft(test_shape, ranges.order);
        
        ASSERT(ranges.compare(diff, {}, dt, phi, cache_xy_sum, coeffs));
        
        auto fish = ExampleData::fish();
        
        coeffs = eft(std::make_shared<points_t::element_type>(fish.points), 10);
        auto points = ieft(coeffs, 10, 100);
        ASSERT(points.size() == 10);
        ASSERT(points.front()->size() == 100);
        
        ASSERT(fish.compare(nullptr, points, nullptr, nullptr, nullptr, coeffs));
        
        /**
         * Test basic curve functions.
         */
        auto rectangle = std::make_shared<points_t::element_type>();
        *rectangle = {
            Vec2(0, 0),
            Vec2(1, 0),
            Vec2(1, 1),
            Vec2(0, 1)
        };
        
        // out[n] = a[n+1] - a[n]
        std::vector<point_t> first_derive {
            Vec2(1, 0),
            Vec2(0, 1),
            Vec2(-1, 0),
            Vec2(0, -1)
        };
        
        {
            auto && [sum, diff] = periodic::differentiate_and_test_clockwise(rectangle);
            ASSERT(sum == 2);
        }
        
        Curve curve;
        curve.set_points(rectangle);
        
        ASSERT(curve.is_clockwise());
        ASSERT(curve.derivatives().size() == 1);
        ASSERT(*curve.derivatives().at(0) == first_derive);
        
        std::reverse(rectangle->begin(), rectangle->end());
        
        {
            auto && [sum, diff] = periodic::differentiate_and_test_clockwise(rectangle);
            ASSERT(sum == -2);
        }
        
        curve.set_points(rectangle);
        
        ASSERT(*curve.points() == *rectangle);
        
        ASSERT(!curve.is_clockwise());
        curve.make_clockwise();
        
        ASSERT(curve.is_clockwise());
        ASSERT(curve.derivatives().size() == 1);
        ASSERT(*curve.derivatives().at(0) == first_derive);
        
        /**
         TEST find_peaks
         */
        
        // x = np.linspace(0, 6 * np.pi, 1000)
        // c = np.sin(x) + 0.6 * np.sin(2.6 * x)
        
        {
            auto range = arange<scalar_t>(0, 6 * M_PI, 6 * M_PI / 999.f);
            auto x = std::vector<scalar_t>(range.begin(), range.end());
            std::vector<scalar_t> c;
            for (auto x : range)
                c.push_back(sin(x) + 0.6 * sin(2.6 * x));
            
            std::vector<double> expected_x {
                0.0, 0.018868424345884642, 0.037736848691769284, 0.056605273037653926, 0.07547369738353857, 0.09434212172942322, 0.11321054607530785, 0.1320789704211925, 0.15094739476707714, 0.16981581911296179, 0.18868424345884643, 0.20755266780473106, 0.2264210921506157, 0.24528951649650035, 0.264157940842385, 0.2830263651882696, 0.3018947895341543, 0.3207632138800389, 0.33963163822592357, 0.3585000625718082, 0.37736848691769287, 0.39623691126357746, 0.4151053356094621, 0.43397375995534676, 0.4528421843012314, 0.47171060864711606, 0.4905790329930007, 0.5094474573388853, 0.52831588168477, 0.5471843060306546, 0.5660527303765392, 0.5849211547224239, 0.6037895790683085, 0.6226580034141932, 0.6415264277600778, 0.6603948521059625, 0.6792632764518471, 0.6981317007977318, 0.7170001251436164, 0.7358685494895011, 0.7547369738353857, 0.7736053981812703, 0.7924738225271549, 0.8113422468730396, 0.8302106712189242, 0.8490790955648089, 0.8679475199106935, 0.8868159442565782, 0.9056843686024628, 0.9245527929483475, 0.9434212172942321, 0.9622896416401168, 0.9811580659860014, 1.000026490331886, 1.0188949146777706, 1.0377633390236554, 1.05663176336954, 1.0755001877154247, 1.0943686120613092, 1.113237036407194, 1.1321054607530785, 1.1509738850989633, 1.1698423094448478, 1.1887107337907326, 1.207579158136617, 1.2264475824825016, 1.2453160068283864, 1.264184431174271, 1.2830528555201557, 1.3019212798660402, 1.320789704211925, 1.3396581285578095, 1.3585265529036943, 1.3773949772495788, 1.3962634015954636, 1.4151318259413481, 1.4340002502872329, 1.4528686746331174, 1.4717370989790022, 1.4906055233248867, 1.5094739476707715, 1.528342372016656, 1.5472107963625406, 1.5660792207084253, 1.5849476450543099, 1.6038160694001946, 1.6226844937460791, 1.641552918091964, 1.6604213424378484, 1.6792897667837332, 1.6981581911296177, 1.7170266154755025, 1.735895039821387, 1.7547634641672718, 1.7736318885131563, 1.792500312859041, 1.8113687372049256, 1.8302371615508102, 1.849105585896695, 1.8679740102425795, 1.8868424345884642, 1.9057108589343488, 1.9245792832802335, 1.943447707626118, 1.9623161319720028, 1.9811845563178874, 2.000052980663772, 2.0189214050096567, 2.037789829355541, 2.056658253701426, 2.0755266780473107, 2.0943951023931953, 2.11326352673908, 2.1321319510849643, 2.1510003754308493, 2.169868799776734, 2.1887372241226184, 2.207605648468503, 2.226474072814388, 2.2453424971602725, 2.264210921506157, 2.2830793458520415, 2.3019477701979265, 2.320816194543811, 2.3396846188896956, 2.35855304323558, 2.377421467581465, 2.3962898919273496, 2.415158316273234, 2.4340267406191187, 2.4528951649650033, 2.4717635893108882, 2.490632013656773, 2.5095004380026573, 2.528368862348542, 2.547237286694427, 2.5661057110403114, 2.584974135386196, 2.6038425597320805, 2.6227109840779654, 2.64157940842385, 2.6604478327697345, 2.679316257115619, 2.698184681461504, 2.7170531058073886, 2.735921530153273, 2.7547899544991576, 2.773658378845042, 2.792526803190927, 2.8113952275368117, 2.8302636518826962, 2.849132076228581, 2.8680005005744658, 2.8868689249203503, 2.905737349266235, 2.9246057736121194, 2.9434741979580044, 2.962342622303889, 2.9812110466497734, 3.000079470995658, 3.018947895341543, 3.0378163196874275, 3.056684744033312, 3.0755531683791966, 3.094421592725081, 3.113290017070966, 3.1321584414168506, 3.151026865762735, 3.1698952901086197, 3.1887637144545047, 3.207632138800389, 3.2265005631462738, 3.2453689874921583, 3.2642374118380433, 3.283105836183928, 3.3019742605298124, 3.320842684875697, 3.3397111092215814, 3.3585795335674664, 3.377447957913351, 3.3963163822592355, 3.41518480660512, 3.434053230951005, 3.4529216552968895, 3.471790079642774, 3.4906585039886586, 3.5095269283345436, 3.528395352680428, 3.5472637770263127, 3.566132201372197, 3.585000625718082, 3.6038690500639667, 3.6227374744098513, 3.641605898755736, 3.6604743231016204, 3.6793427474475053, 3.69821117179339, 3.7170795961392744, 3.735948020485159, 3.754816444831044, 3.7736848691769285, 3.792553293522813, 3.8114217178686975, 3.8302901422145825, 3.849158566560467, 3.8680269909063516, 3.886895415252236, 3.905763839598121, 3.9246322639440057, 3.94350068828989, 3.9623691126357747, 3.9812375369816593, 4.000105961327544, 4.018974385673428, 4.037842810019313, 4.056711234365198, 4.075579658711082, 4.094448083056967, 4.113316507402852, 4.1321849317487365, 4.1510533560946214, 4.1699217804405055, 4.1887902047863905, 4.2076586291322755, 4.22652705347816, 4.245395477824045, 4.264263902169929, 4.283132326515814, 4.302000750861699, 4.320869175207583, 4.339737599553468, 4.358606023899353, 4.377474448245237, 4.396342872591122, 4.415211296937006, 4.434079721282891, 4.452948145628776, 4.47181656997466, 4.490684994320545, 4.50955341866643, 4.528421843012314, 4.547290267358199, 4.566158691704083, 4.585027116049968, 4.603895540395853, 4.622763964741737, 4.641632389087622, 4.660500813433506, 4.679369237779391, 4.698237662125276, 4.71710608647116, 4.735974510817045, 4.75484293516293, 4.773711359508814, 4.792579783854699, 4.811448208200583, 4.830316632546468, 4.849185056892353, 4.8680534812382374, 4.886921905584122, 4.9057903299300065, 4.9246587542758915, 4.9435271786217765, 4.962395602967661, 4.981264027313546, 5.0001324516594305, 5.019000876005315, 5.0378693003512, 5.056737724697084, 5.075606149042969, 5.094474573388854, 5.113342997734738, 5.132211422080623, 5.151079846426507, 5.169948270772392, 5.188816695118277, 5.207685119464161, 5.226553543810046, 5.245421968155931, 5.264290392501815, 5.2831588168477, 5.302027241193584, 5.320895665539469, 5.339764089885354, 5.358632514231238, 5.377500938577123, 5.396369362923008, 5.415237787268892, 5.434106211614777, 5.452974635960661, 5.471843060306546, 5.490711484652431, 5.509579908998315, 5.5284483333442, 5.547316757690084, 5.566185182035969, 5.585053606381854, 5.603922030727738, 5.622790455073623, 5.641658879419508, 5.6605273037653925, 5.6793957281112775, 5.698264152457162, 5.7171325768030465, 5.7360010011489315, 5.754869425494816, 5.773737849840701, 5.792606274186585, 5.81147469853247, 5.830343122878355, 5.849211547224239, 5.868079971570124, 5.886948395916009, 5.905816820261893, 5.924685244607778, 5.943553668953662, 5.962422093299547, 5.981290517645432, 6.000158941991316, 6.019027366337201, 6.037895790683086, 6.05676421502897, 6.075632639374855, 6.094501063720739, 6.113369488066624, 6.132237912412509, 6.151106336758393, 6.169974761104278, 6.188843185450162, 6.207711609796047, 6.226580034141932, 6.245448458487816, 6.264316882833701, 6.283185307179586, 6.30205373152547, 6.320922155871355, 6.339790580217239, 6.358659004563124, 6.377527428909009, 6.3963958532548935, 6.415264277600778, 6.4341327019466625, 6.4530011262925475, 6.4718695506384325, 6.490737974984317, 6.509606399330202, 6.528474823676087, 6.547343248021971, 6.566211672367856, 6.58508009671374, 6.603948521059625, 6.62281694540551, 6.641685369751394, 6.660553794097279, 6.679422218443163, 6.698290642789048, 6.717159067134933, 6.736027491480817, 6.754895915826702, 6.773764340172587, 6.792632764518471, 6.811501188864356, 6.83036961321024, 6.849238037556125, 6.86810646190201, 6.886974886247894, 6.905843310593779, 6.924711734939664, 6.943580159285548, 6.962448583631433, 6.981317007977317, 7.000185432323202, 7.019053856669087, 7.037922281014971, 7.056790705360856, 7.07565912970674, 7.094527554052625, 7.11339597839851, 7.132264402744394, 7.151132827090279, 7.170001251436164, 7.1888696757820485, 7.2077381001279335, 7.226606524473818, 7.2454749488197026, 7.2643433731655875, 7.283211797511472, 7.302080221857357, 7.320948646203241, 7.339817070549126, 7.358685494895011, 7.377553919240895, 7.39642234358678, 7.415290767932665, 7.434159192278549, 7.453027616624434, 7.471896040970318, 7.490764465316203, 7.509632889662088, 7.528501314007972, 7.547369738353857, 7.566238162699741, 7.585106587045626, 7.603975011391511, 7.622843435737395, 7.64171186008328, 7.660580284429165, 7.679448708775049, 7.698317133120934, 7.717185557466818, 7.736053981812703, 7.754922406158588, 7.773790830504472, 7.792659254850357, 7.811527679196242, 7.830396103542126, 7.849264527888011, 7.868132952233895, 7.88700137657978, 7.905869800925665, 7.9247382252715495, 7.9436066496174345, 7.9624750739633186, 7.9813434983092035, 8.000211922655089, 8.019080347000973, 8.037948771346857, 8.056817195692743, 8.075685620038627, 8.09455404438451, 8.113422468730397, 8.13229089307628, 8.151159317422165, 8.17002774176805, 8.188896166113935, 8.207764590459819, 8.226633014805705, 8.245501439151589, 8.264369863497473, 8.283238287843357, 8.302106712189243, 8.320975136535127, 8.339843560881011, 8.358711985226897, 8.377580409572781, 8.396448833918665, 8.415317258264551, 8.434185682610435, 8.45305410695632, 8.471922531302205, 8.49079095564809, 8.509659379993973, 8.528527804339857, 8.547396228685743, 8.566264653031627, 8.585133077377511, 8.604001501723397, 8.622869926069281, 8.641738350415165, 8.660606774761051, 8.679475199106935, 8.69834362345282, 8.717212047798705, 8.73608047214459, 8.754948896490474, 8.773817320836358, 8.792685745182244, 8.811554169528128, 8.830422593874012, 8.849291018219898, 8.868159442565782, 8.887027866911666, 8.905896291257552, 8.924764715603436, 8.94363313994932, 8.962501564295206, 8.98136998864109, 9.000238412986974, 9.01910683733286, 9.037975261678744, 9.056843686024628, 9.075712110370512, 9.094580534716398, 9.113448959062282, 9.132317383408166, 9.151185807754052, 9.170054232099936, 9.18892265644582, 9.207791080791706, 9.22665950513759, 9.245527929483474, 9.26439635382936, 9.283264778175244, 9.302133202521128, 9.321001626867012, 9.339870051212898, 9.358738475558782, 9.377606899904666, 9.396475324250552, 9.415343748596436, 9.43421217294232, 9.453080597288206, 9.47194902163409, 9.490817445979975, 9.50968587032586, 9.528554294671745, 9.547422719017629, 9.566291143363513, 9.585159567709399, 9.604027992055283, 9.622896416401167, 9.641764840747053, 9.660633265092937, 9.67950168943882, 9.698370113784707, 9.71723853813059, 9.736106962476475, 9.75497538682236, 9.773843811168245, 9.792712235514129, 9.811580659860013, 9.830449084205899, 9.849317508551783, 9.868185932897667, 9.887054357243553, 9.905922781589437, 9.924791205935321, 9.943659630281207, 9.962528054627091, 9.981396478972975, 10.000264903318861, 10.019133327664745, 10.03800175201063, 10.056870176356513, 10.0757386007024, 10.094607025048283, 10.113475449394167, 10.132343873740053, 10.151212298085937, 10.170080722431821, 10.188949146777707, 10.207817571123591, 10.226685995469476, 10.245554419815361, 10.264422844161246, 10.28329126850713, 10.302159692853014, 10.3210281171989, 10.339896541544784, 10.358764965890668, 10.377633390236554, 10.396501814582438, 10.415370238928322, 10.434238663274208, 10.453107087620092, 10.471975511965976, 10.490843936311862, 10.509712360657746, 10.52858078500363, 10.547449209349516, 10.5663176336954, 10.585186058041284, 10.604054482387168, 10.622922906733054, 10.641791331078938, 10.660659755424822, 10.679528179770708, 10.698396604116592, 10.717265028462476, 10.736133452808362, 10.755001877154246, 10.77387030150013, 10.792738725846016, 10.8116071501919, 10.830475574537784, 10.849343998883668, 10.868212423229554, 10.887080847575438, 10.905949271921322, 10.924817696267208, 10.943686120613092, 10.962554544958977, 10.981422969304862, 11.000291393650746, 11.01915981799663, 11.038028242342516, 11.0568966666884, 11.075765091034285, 11.094633515380169, 11.113501939726055, 11.132370364071939, 11.151238788417823, 11.170107212763709, 11.188975637109593, 11.207844061455477, 11.226712485801363, 11.245580910147247, 11.264449334493131, 11.283317758839017, 11.3021861831849, 11.321054607530785, 11.339923031876669, 11.358791456222555, 11.377659880568439, 11.396528304914323, 11.415396729260209, 11.434265153606093, 11.453133577951977, 11.472002002297863, 11.490870426643747, 11.509738850989631, 11.528607275335517, 11.547475699681401, 11.566344124027285, 11.58521254837317, 11.604080972719055, 11.62294939706494, 11.641817821410823, 11.66068624575671, 11.679554670102593, 11.698423094448477, 11.717291518794363, 11.736159943140247, 11.755028367486132, 11.773896791832017, 11.792765216177902, 11.811633640523786, 11.83050206486967, 11.849370489215556, 11.86823891356144, 11.887107337907324, 11.90597576225321, 11.924844186599094, 11.943712610944978, 11.962581035290864, 11.981449459636748, 12.000317883982632, 12.019186308328518, 12.038054732674402, 12.056923157020286, 12.075791581366172, 12.094660005712056, 12.11352843005794, 12.132396854403824, 12.15126527874971, 12.170133703095594, 12.189002127441478, 12.207870551787364, 12.226738976133248, 12.245607400479132, 12.264475824825018, 12.283344249170902, 12.302212673516786, 12.321081097862672, 12.339949522208556, 12.35881794655444, 12.377686370900324, 12.39655479524621, 12.415423219592094, 12.434291643937978, 12.453160068283864, 12.472028492629748, 12.490896916975633, 12.509765341321518, 12.528633765667402, 12.547502190013287, 12.566370614359172, 12.585239038705057, 12.60410746305094, 12.622975887396825, 12.64184431174271, 12.660712736088595, 12.679581160434479, 12.698449584780365, 12.717318009126249, 12.736186433472133, 12.755054857818019, 12.773923282163903, 12.792791706509787, 12.811660130855673, 12.830528555201557, 12.849396979547441, 12.868265403893325, 12.887133828239211, 12.906002252585095, 12.92487067693098, 12.943739101276865, 12.962607525622749, 12.981475949968633, 13.000344374314519, 13.019212798660403, 13.038081223006287, 13.056949647352173, 13.075818071698057, 13.094686496043941, 13.113554920389825, 13.132423344735711, 13.151291769081595, 13.17016019342748, 13.189028617773365, 13.20789704211925, 13.226765466465134, 13.24563389081102, 13.264502315156903, 13.283370739502788, 13.302239163848673, 13.321107588194558, 13.339976012540442, 13.358844436886326, 13.377712861232212, 13.396581285578096, 13.41544970992398, 13.434318134269866, 13.45318655861575, 13.472054982961634, 13.49092340730752, 13.509791831653404, 13.528660255999288, 13.547528680345174, 13.566397104691058, 13.585265529036942, 13.604133953382826, 13.623002377728712, 13.641870802074596, 13.66073922642048, 13.679607650766366, 13.69847607511225, 13.717344499458134, 13.73621292380402, 13.755081348149904, 13.773949772495788, 13.792818196841674, 13.811686621187558, 13.830555045533442, 13.849423469879328, 13.868291894225212, 13.887160318571096, 13.90602874291698, 13.924897167262866, 13.94376559160875, 13.962634015954634, 13.98150244030052, 14.000370864646404, 14.019239288992289, 14.038107713338174, 14.056976137684059, 14.075844562029943, 14.094712986375828, 14.113581410721713, 14.132449835067597, 14.15131825941348, 14.170186683759367, 14.18905510810525, 14.207923532451135, 14.22679195679702, 14.245660381142905, 14.264528805488789, 14.283397229834675, 14.302265654180559, 14.321134078526443, 14.340002502872329, 14.358870927218213, 14.377739351564097, 14.396607775909981, 14.415476200255867, 14.434344624601751, 14.453213048947635, 14.472081473293521, 14.490949897639405, 14.50981832198529, 14.528686746331175, 14.54755517067706, 14.566423595022943, 14.58529201936883, 14.604160443714713, 14.623028868060597, 14.641897292406481, 14.660765716752367, 14.679634141098251, 14.698502565444135, 14.717370989790021, 14.736239414135905, 14.75510783848179, 14.773976262827675, 14.79284468717356, 14.811713111519444, 14.83058153586533, 14.849449960211214, 14.868318384557098, 14.887186808902982, 14.906055233248868, 14.924923657594752, 14.943792081940636, 14.962660506286522, 14.981528930632406, 15.00039735497829, 15.019265779324176, 15.03813420367006, 15.057002628015944, 15.07587105236183, 15.094739476707714, 15.113607901053598, 15.132476325399482, 15.151344749745368, 15.170213174091252, 15.189081598437136, 15.207950022783022, 15.226818447128906, 15.24568687147479, 15.264555295820676, 15.28342372016656, 15.302292144512444, 15.32116056885833, 15.340028993204214, 15.358897417550098, 15.377765841895984, 15.396634266241868, 15.415502690587752, 15.434371114933636, 15.453239539279522, 15.472107963625406, 15.49097638797129, 15.509844812317176, 15.52871323666306, 15.547581661008945, 15.56645008535483, 15.585318509700715, 15.604186934046599, 15.623055358392484, 15.641923782738369, 15.660792207084253, 15.679660631430137, 15.698529055776023, 15.717397480121907, 15.73626590446779, 15.755134328813677, 15.77400275315956, 15.792871177505445, 15.81173960185133, 15.830608026197215, 15.849476450543099, 15.868344874888985, 15.887213299234869, 15.906081723580753, 15.924950147926637, 15.943818572272523, 15.962686996618407, 15.981555420964291, 16.000423845310177, 16.01929226965606, 16.038160694001945, 16.05702911834783, 16.075897542693713, 16.0947659670396, 16.113634391385485, 16.132502815731367, 16.151371240077253, 16.17023966442314, 16.18910808876902, 16.207976513114907, 16.226844937460793, 16.245713361806676, 16.26458178615256, 16.283450210498447, 16.30231863484433, 16.321187059190216, 16.3400554835361, 16.358923907881984, 16.37779233222787, 16.396660756573755, 16.415529180919638, 16.434397605265524, 16.45326602961141, 16.472134453957292, 16.491002878303178, 16.509871302649064, 16.528739726994946, 16.54760815134083, 16.566476575686714, 16.5853450000326, 16.604213424378486, 16.623081848724368, 16.641950273070254, 16.66081869741614, 16.679687121762022, 16.698555546107908, 16.717423970453794, 16.736292394799676, 16.755160819145562, 16.774029243491448, 16.79289766783733, 16.811766092183216, 16.830634516529102, 16.849502940874984, 16.86837136522087, 16.887239789566756, 16.90610821391264, 16.924976638258524, 16.94384506260441, 16.962713486950292, 16.98158191129618, 17.000450335642064, 17.019318759987947, 17.038187184333832, 17.057055608679715, 17.0759240330256, 17.094792457371486, 17.11366088171737, 17.132529306063255, 17.15139773040914, 17.170266154755023, 17.18913457910091, 17.208003003446795, 17.226871427792677, 17.245739852138563, 17.26460827648445, 17.28347670083033, 17.302345125176217, 17.321213549522103, 17.340081973867985, 17.35895039821387, 17.377818822559757, 17.39668724690564, 17.415555671251525, 17.43442409559741, 17.453292519943293, 17.47216094428918, 17.491029368635065, 17.509897792980947, 17.528766217326833, 17.547634641672715, 17.5665030660186, 17.585371490364487, 17.60423991471037, 17.623108339056255, 17.64197676340214, 17.660845187748023, 17.67971361209391, 17.698582036439795, 17.717450460785678, 17.736318885131563, 17.75518730947745, 17.77405573382333, 17.792924158169217, 17.811792582515103, 17.830661006860986, 17.84952943120687, 17.868397855552757, 17.88726627989864, 17.906134704244526, 17.92500312859041, 17.943871552936294, 17.96273997728218, 17.981608401628066, 18.000476825973948, 18.019345250319834, 18.03821367466572, 18.057082099011602, 18.075950523357488, 18.09481894770337, 18.113687372049256, 18.132555796395142, 18.151424220741024, 18.17029264508691, 18.189161069432796, 18.208029493778678, 18.226897918124564, 18.24576634247045, 18.264634766816332, 18.283503191162218, 18.302371615508104, 18.321240039853986, 18.340108464199872, 18.358976888545758, 18.37784531289164, 18.396713737237526, 18.415582161583412, 18.434450585929294, 18.45331901027518, 18.472187434621066, 18.49105585896695, 18.509924283312834, 18.52879270765872, 18.547661132004603, 18.56652955635049, 18.58539798069637, 18.604266405042257, 18.623134829388142, 18.642003253734025, 18.66087167807991, 18.679740102425797, 18.69860852677168, 18.717476951117565, 18.73634537546345, 18.755213799809333, 18.77408222415522, 18.792950648501105, 18.811819072846987, 18.830687497192873, 18.84955592153876
            };
            std::vector<double> expected_y {
                0.0, 0.04829024152765479, 0.096502968834755, 0.14456084044420436, 0.1923868599550714, 0.23990454755477855, 0.2870381103029792, 0.3337126107822793, 0.37985413371488086, 0.4253899501491125, 0.4702486788256478, 0.5143604443399862, 0.5576570317254672, 0.6000720370896848, 0.6415410139466479, 0.6820016148973717, 0.7213937283227547, 0.7596596097645781, 0.7967440076832166, 0.8325942832941607, 0.867160524199666, 0.9003956515467467, 0.9322555204582794, 0.9626990135001325, 0.9916881269639646, 1.0191880497625794, 1.0451672347524714, 1.0695974623163726, 1.0924538960571957, 1.113715130473711, 1.133363230507542, 1.1513837628705808, 1.1677658190816496, 1.1825020301611457, 1.195588572952423, 1.2070251680587578, 1.2168150694048807, 1.2249650454521372, 1.2314853521163849, 1.2363896974576316, 1.2396951982301658, 1.241422328401449, 1.2415948597673165, 1.2402397948099741, 1.237387291963892, 1.2330705834728888, 1.2273258860394767, 1.2201923044847964, 1.2117117286542314, 1.201928723819972, 1.1908904148473745, 1.1786463644068954, 1.1652484455276388, 1.1507507088020872, 1.1352092445643942, 1.1186820403766267, 1.1012288341685579, 1.0829109633869987, 1.0637912105201761, 1.0439336453712995, 1.02340346446322, 1.0022668279628835, 0.980590694520184, 0.958442654420747, 0.9358907614561555, 0.9130033639181201, 0.8898489351251327, 0.866495903891172, 0.8430124853460804, 0.8194665125163121, 0.7959252690728166, 0.7724553236499402, 0.7491223661353509, 0.7259910463261672, 0.7031248153406735, 0.6805857701683096, 0.6584345017329492, 0.6367299468359675, 0.6155292443361471, 0.5948875959132032, 0.5748581317505743, 0.5554917814612108, 0.5368371505673686, 0.5189404028319762, 0.5018451487249609, 0.48559234029306175, 0.4702201726861751, 0.45576399257714095, 0.4422562136952296, 0.4297262396763498, 0.4182003944153265, 0.40770186008743425, 0.3982506229878373, 0.38986342731867774, 0.38255373703434425, 0.376331705835965, 0.3712041553864739, 0.36717456179773, 0.36424305042116445, 0.36240639895337523, 0.3616580488479759, 0.36198812500494393, 0.363383463688695, 0.3658276486062315, 0.3693010550569833, 0.37378090204645553, 0.379241312236541, 0.38565337958642554, 0.3929852445194049, 0.40120217643275763, 0.41026666335005196, 0.420138508498002, 0.4307749335732397, 0.4421306884481953, 0.45415816704969947, 0.4668075291289871, 0.48002682762753773, 0.4937621413296261, 0.5079577124796765, 0.5225560890304797, 0.5374982711771196, 0.5527238618210707, 0.568171220599386, 0.5837776211052619, 0.5994794109185042, 0.6152121740575907, 0.6309108954591327, 0.6465101270855866, 0.6619441552580944, 0.6771471688082978, 0.6920534276409595, 0.7065974312981559, 0.720714087115749, 0.7343388775637706, 0.747408026364238, 0.7598586629828298, 0.7716289850946867, 0.7826584186294354, 0.7928877750062941, 0.8022594051768427, 0.8107173501006519, 0.8182074872875185, 0.8246776730494606, 0.8300778801159061, 0.8343603302766202, 0.8374796217288443, 0.8393928508178095, 0.8400597278732485, 0.8394426868586939, 0.8375069885652022, 0.8342208170966412, 0.8295553694097872, 0.8234849376891584, 0.8159869843537242, 0.8070422095103229, 0.7966346106867803, 0.7847515346962639, 0.7713837215033208, 0.7565253399812719, 0.7401740154701254, 0.7223308490638858, 0.7030004285760149, 0.6821908311518121, 0.6599136175165725, 0.6361838178684859, 0.611019909445344, 0.5844437858141471, 0.5564807179526016, 0.5271593072112684, 0.4965114302646306, 0.4645721761786464, 0.43137977574129827, 0.39697552322130303, 0.3614036907383357, 0.3247114354459295, 0.28694869974550224, 0.24816810476674261, 0.20842483736580864, 0.16777653090838787, 0.12628314011965458, 0.08400681029743329, 0.04101174119848028, -0.002635954079381775, -0.046868394884481135, -0.09161608312179839, -0.136808057330266, -0.18237205036081205, -0.2282346502803849, -0.27432146411943725, -0.3205572840734244, -0.3668662557629827, -0.4131720481524283, -0.45939802472222774, -0.5054674154879858, -0.5513034894564696, -0.5968297271080657, -0.6419699924949619, -0.6866487045452534, -0.730791007164986, -0.7743229377330514, -0.8171715935876196, -0.8592652961076228, -0.9005337519984852, -0.9409082113980185, -0.9803216224259631, -1.0187087818091882, -1.0560064812239103, -1.0921536490065709, -1.127091486896084, -1.1607636014820577, -1.1931161300462678, -1.2240978604980957, -1.2536603451187864, -1.2817580078442037, -1.308348244831258, -1.3333915180692515, -1.356851441814071, -1.378694861640352, -1.3988919259244108, -1.4174161495889055, -1.4342444699587302, -1.449357294596516, -1.4627385410054021, -1.4743756681061573, -1.484259699415507, -1.4923852378723774, -1.498750472278784, -1.503357175342218, -1.5062106933264632, -1.5073199273379365, -1.5066973062946438, -1.504358751644803, -1.500323633921937, -1.4946147212428011, -1.487258119873817, -1.478283207010664, -1.4677225559343587, -1.455611853725375, -1.4419898117352004, -1.4268980690320578, -1.4103810890543196, -1.392486049721427, -1.3732627272677496, -1.3527633740798504, -1.3310425908319543, -1.308157193228052, -1.2841660736719382, -1.2591300581986298, -1.2331117590118839, -1.2061754229830377, -1.1783867764760187, -1.1498128668720924, -1.1205219011758227, -1.0905830820905753, -1.0600664419579457, -1.029042674960543, -0.9975829679915793, -0.9657588305979415, -0.9336419244054407, -0.901303892436169, -0.8688161887280589, -0.8362499086658405, -0.8036756204309193, -0.771163197974783, -0.7387816559168716, -0.7065989867631216, -0.6746820008356539, -0.6430961692975481, -0.6119054706490559, -0.5811722410631612, -0.5509570289191245, -0.5213184538823741, -0.49231307086816234, -0.4639952392145172, -0.4364169973774114, -0.4096279434477128, -0.3836751217753581, -0.3586029159714459, -0.3344529485434925, -0.3112639874030593, -0.2890718594683327, -0.2679093715671216, -0.24780623882802355, -0.2287890207294948, -0.21088106495798697, -0.19410245920748348, -0.17846999103358474, -0.16399711585580568, -0.15069393318210378, -0.13856717110978534, -0.12762017913693502, -0.11785292929848168, -0.10926202562087539, -0.10184072186929427, -0.09557894754126794, -0.09046334204069162, -0.08647729694646733, -0.08360100627046008, -0.08181152458017527, -0.08108283284257278, -0.08138591182680788, -0.08268882288542068, -0.08495679591571725, -0.08815232428571312, -0.09223526649223862, -0.09716295430252508, -0.10289030711494479, -0.10936995225958407, -0.11655235094494909, -0.12438592954352035, -0.1328172158959287, -0.14179098030145348, -0.1512503808511954, -0.1611371127497867, -0.17139156126189742, -0.18195295791098168, -0.1927595395499151, -0.20374870991615646, -0.2148572032781244, -0.22602124977437296, -0.23717674204304962, -0.24825940273602168, -0.259204952509866, -0.2699492780847474, -0.2804285999620659, -0.29057963939248876, -0.30033978418783036, -0.30964725297296963, -0.3184412574777368, -0.3266621624734166, -0.33425164296417237, -0.34115283825025206, -0.3473105024874351, -0.35267115137548377, -0.35718320461781666, -0.360797123804651, -0.3634655453829874, -0.3651434083885453, -0.3657880766274332, -0.3653594550086295, -0.36382009974250296, -0.3611353221352806, -0.3572732857248902, -0.3522050965195302, -0.34590488611703174, -0.3383498875001272, -0.32952050332045935, -0.3194003665021941, -0.30797639301463525, -0.2952388266820977, -0.2811812759184618, -0.2658007422933069, -0.2490976408561914, -0.23107581216550233, -0.21174252598830745, -0.19110847665767866, -0.16918777009410785, -0.14599790251768202, -0.12155973089774624, -0.09589743520669342, -0.06903847256427253, -0.04101352337835151, -0.0118564296074255, 0.018395874710937732, 0.04970344050275566, 0.08202338809026005, 0.11530999881548487, 0.1495148136384568, 0.18458673847680868, 0.22047215603738152, 0.25711504387461326, 0.29445709839564754, 0.33243786451750595, 0.37099487066823933, 0.41006376881086803, 0.44957847915688115, 0.4894713392246937, 0.5296732568879219, 0.5701138670486889, 0.6107216915624224, 0.6514243020325858, 0.6921484850870157, 0.7328204097412896, 0.7733657964495945, 0.8137100874393804, 0.8537786179229225, 0.8934967877767479, 0.9327902332786693, 0.9715849984919519, 1.0098077058869313, 1.047385725792127, 1.0842473442696563, 1.120321929013476, 1.1555400928736028, 1.1898338546151648, 1.2231367965276567, 1.2553842185072859, 1.2865132882437043, 1.3164631871516956, 1.3451752516985451, 1.3725931097888495, 1.3986628118802236, 1.4233329565161261, 1.446554809975166, 1.4682824197505169, 1.488472721587672, 1.5070856398242691, 1.5240841807917074, 1.5394345190549146, 1.5531060762836928, 1.565071592566825, 1.5753071899980884, 1.5837924283819418, 1.5905103529254343, 1.5954475338021292, 1.5985940974932764, 1.5999437498311684, 1.599493790689489, 1.5972451202854643, 1.593202237078727, 1.5873732272719223, 1.5797697459382039, 1.570406989820825, 1.559303661869972, 1.5464819276017794, 1.5319673633840485, 1.515788896772532, 1.4979787390406718, 1.4785723100643895, 1.4576081557417915, 1.435127858145592, 1.4111759386233833, 1.3857997540777836, 1.3590493866748476, 1.3309775272448099, 1.3016393526542964, 1.2710923974436565, 1.2393964200366634, 1.2066132638428526, 1.1728067135850004, 1.1380423471956096, 1.1023873836368443, 1.0659105270080809, 1.028681807314127, 0.9907724182750959, 0.9522545525658778, 0.9132012348794094, 0.8736861532129762, 0.8337834887809861, 0.7935677449609533, 0.7531135756816358, 0.7124956136634611, 0.6717882989218179, 0.6310657079429149, 0.5904013839403637, 0.549868168597799, 0.5095380356993949, 0.469481927045384, 0.42976959104411977, 0.3904694243658432, 0.35164831703572, 0.3133715013354435, 0.27570240487351433, 0.23870250817414146, 0.20243120712376117, 0.1669456806025436, 0.13230076361555326, 0.09854882622507966, 0.06573965857151864, 0.033920362255612124, 0.003135248339364849, -0.026574257792967737, -0.0551697044889507, -0.0826156945893754, -0.1088799587068956, -0.13393342084974874, -0.15775025623696753, -0.1803079411703099, -0.20158729484723348, -0.2215725130187317, -0.24025119341551243, -0.2576143528857871, -0.27365643620798435, -0.2883753165617241, -0.301772287660532, -0.313852047569819, -0.32462267425373154, -0.3340955929143981, -0.342285535206855, -0.34921049043252195, -0.3548916488334365, -0.3593533371284505, -0.36262294645130955, -0.36473085286882434, -0.3657103306752194, -0.3655974586761246, -0.3644310196925805, -0.36225239353173383, -0.3591054436866503, -0.3550363980427355, -0.3500937238827222, -0.3443279974958663, -0.33779176871002775, -0.33053942067746656, -0.32262702525666354, -0.31411219434303655, -0.3050539275111741, -0.2955124563400702, -0.2855490858008525, -0.27522603309348853, -0.2646062643251503, -0.2537533294280461, -0.24273119571880575, -0.23160408050467582, -0.22043628314414354, -0.2092920169708486, -0.1982352414899477, -0.18732949525547185, -0.17663772983546883, -0.166222145269151, -0.15614402741662092, -0.14646358759714986, -0.13723980490649534, -0.12853027159718655, -0.12039104189838112, -0.11287648464353556, -0.10603914006493359, -0.09992958110409222, -0.09459627957608896, -0.09008547751419743, -0.08644106400866303, -0.0837044578402264, -0.08191449619495944, -0.08110732973234525, -0.08131632426317192, -0.08257196927785504, -0.08490179354927219, -0.08833028801711929, -0.09287883614320169, -0.0985656519090704, -0.10540572560893435, -0.1134107775719867, -0.1225892199291313, -0.13294612651967108, -0.14448321101387995, -0.15719881330754915, -0.17108789422460724, -0.18614203854387879, -0.20234946634592865, -0.21969505265587508, -0.2381603553379863, -0.2577236511779668, -0.278359980069092, -0.30004119719871225, -0.322736033112373, -0.3464101615137736, -0.3710262746400288, -0.39654416603346665, -0.4229208205132955, -0.4501105111330387, -0.47806490289276443, -0.5067331629588547, -0.5360620771282071, -0.5659961722587604, -0.5964778443737193, -0.6274474921331956, -0.6588436553538664, -0.690603158245138, -0.722661257018814, -0.7549517915186343, -0.787407340506305, -0.8199593802318241, -0.8525384459077812, -0.8850742957003345, -0.9174960768434312, -0.9497324934775188, -0.9817119758099191, -1.0133628501906249, -1.0446135096950895, -1.0753925848041903, -1.1056291137712408, -1.135252712266667, -1.1641937418924628, -1.1923834771612865, -1.2197542705386424, -1.2462397151510771, -1.2717748047689628, -1.2962960906788343, -1.319741835067639, -1.3420521605495719, -1.363169195475377, -1.3830372146740317, -1.4016027752876068, -1.4188148473718556, -1.434624938947578, -1.4489872152010135, -1.4618586115455934, -1.4731989402719914, -1.4829709905287796, -1.4911406213919376, -1.4976768477980378, -1.5025519191329653, -1.505741390285649, -1.5072241849943013, -1.5069826513311457, -1.5050026091904123, -1.5012733896635602, -1.495787866205106, -1.488542477512115, -1.4795372420602604, -1.468775764259349, -1.456265232211294, -1.4420164070736503, -1.426043604051911, -1.4083646650638815, -1.389000923139346, -1.3679771586381055, -1.345321547389011, -1.3210656008720576, -1.295244098584608, -1.2678950127515594, -1.2390594255576552, -1.208781439098029, -1.1771080782604817, -1.1440891867700445, -1.1097773166426368, -1.0742276113104763, -1.0374976826970799, -0.9996474825341533, -0.9607391682264645, -0.9208369635838276, -0.8800070147516834, -0.8383172416832035, -0.7958371855064619, -0.7526378521501719, -0.7087915526003246, -0.6643717401681123, -0.6194528451567463, -0.5741101073209296, -0.5284194065179337, -0.4824570919536978, -0.43629981043056587, -0.3900243340056283, -0.34370738747015606, -0.2974254760607976, -0.25125471381278686, -0.20527065296367947, -0.15954811481369854, -0.11416102244516468, -0.06918223569890536, -0.024683388800245898, 0.019265268979366257, 0.06259502924638355, 0.10523887662421835, 0.147131636746017, 0.18821011974981708, 0.22841325893541384, 0.2676822442542187, 0.30596065031557895, 0.3431945586064724, 0.3793326736351792, 0.4143264327244108, 0.4481301091945537, 0.4807009086935705, 0.511999058446651, 0.5419878892158215, 0.5706339097770915, 0.5979068737408846, 0.6237798385598796, 0.6482292165871416, 0.6712348180666211, 0.6927798859574584, 0.7128511225132219, 0.7314387075569863, 0.7485363084131765, 0.7641410814771736, 0.7782536654237371, 0.7908781660754621, 0.8020221329725199, 0.8116965277048775, 0.8199156840879995, 0.8266972602826557, 0.8320621829788015, 0.8360345837825767, 0.8386417279642316, 0.8399139357430863, 0.8398844963036229, 0.8385895747542222, 0.8360681122570534, 0.8323617195739661, 0.8275145642890924, 0.82157325198404, 0.8145867016560302, 0.806606015683196, 0.7976843446543267, 0.7878767473926251, 0.7772400465146088, 0.7658326798759754, 0.7537145482660377, 0.7409468597214673, 0.7275919708379796, 0.7137132254658869, 0.6993747911816657, 0.6846414939330139, 0.6695786512590658, 0.6542519044910322, 0.6387270503407311, 0.6230698722860082, 0.6073459721624024, 0.5916206023699038, 0.5759584991020704, 0.5604237170022837, 0.5450794656483763, 0.5299879482624357, 0.5152102030371635, 0.5008059474637707, 0.4868334260391533, 0.4733492617217936, 0.46040831149690387, 0.44806352640117086, 0.4363658163467856, 0.4253639200727885, 0.41510428053935283, 0.40563092606740286, 0.3969853575121185, 0.3892064417442779, 0.38233031169808385, 0.3763902732282933, 0.3714167190029555, 0.36743704964108703, 0.3644756022870841, 0.36255358679568084, 0.3616890296828835, 0.36189672597953426, 0.3631881991050415, 0.3655716688594508, 0.3690520276123892, 0.3736308247476222, 0.37930625940198837, 0.3860731815174475, 0.39392310120488216, 0.40284420639819407, 0.41282138875719576, 0.42383627775789434, 0.4358672828889114, 0.4488896438532439, 0.4628754886551846, 0.4777938994331664, 0.49361098588056174, 0.5102899660781189, 0.5277912545437936, 0.5460725572882679, 0.5650889736475073, 0.5847931046473045, 0.6051351676389454, 0.6260631169299398, 0.6475227701193147, 0.669457939832985, 0.6918105705418284, 0.7145208801326535, 0.7375275058907278, 0.7607676545418139, 0.7841772559919367, 0.807691120393887, 0.8312430981615986, 0.8547662425461409, 0.8781929743810647, 0.9014552485992557, 0.9244847221193139, 0.9472129226959769, 0.9695714183267474, 0.9914919868054399, 1.0129067850129183, 1.0337485175358667, 1.0539506042059166, 1.073447346154077, 1.0921740899786982, 1.1100673896299198, 1.12706516561872, 1.1431068611652522, 1.158133594908206, 1.1720883098053676, 1.1849159178644817, 1.196563440353493, 1.2069801431501064, 1.2161176669021563, 1.223930151682737, 1.230374355837189, 1.2354097687329357, 1.2389987171377954, 1.2411064649676145, 1.2417013061599858, 1.2407546504472502, 1.2382411018190442, 1.2341385294821428, 1.2284281311433378, 1.22109448845953, 1.2121256145179633, 1.2015129932286557, 1.1892516105305013, 1.1753399773321243, 1.159780144128419, 1.142577707253682, 1.1237418067523002, 1.1032851158680943, 1.081223822173495, 1.0575776003798547, 1.0323695768901122, 1.0056262861748602, 0.9773776190725265, 0.9476567631337363, 0.916500135149009, 0.8839473060177668, 0.8500409181349828, 0.8148265954897058, 0.7783528466873662, 0.740670961124573, 0.7018348985616301, 0.6619011723539124, 0.6209287266183912, 0.5789788076261884, 0.5361148297258747, 0.49240223611544626, 0.4479083547932521, 0.40270225002957927, 0.356854569711643, 0.3104373889243901, 0.26352405013867075, 0.21618900038649597, 0.16850762581029383, 0.12055608397938442, 0.07241113437230426, 0.024149967427957005, -0.02414996742794783, -0.07241113437229332, -0.12055608397937953, -0.16850762581028475, -0.21618900038648695, -0.263524050138666, -0.3104373889243853, -0.3568545697116383, -0.4027022500295746, -0.44790835479324737, -0.49240223611544165, -0.5361148297258649, -0.5789788076261839, -0.6209287266183866, -0.6619011723539032, -0.7018348985616227, -0.740670961124567, -0.778352846687362, -0.8148265954897018, -0.8500409181349748, -0.8839473060177631, -0.9165001351490054, -0.9476567631337292, -0.9773776190725213, -1.0056262861748566, -1.032369576890106, -1.0575776003798516, -1.081223822173492, -1.103285115868089, -1.1237418067522973, -1.1425777072536791, -1.159780144128415, -1.175339977332122, -1.1892516105305, -1.2015129932286528, -1.2121256145179613, -1.2210944884595287, -1.2284281311433358, -1.2341385294821423, -1.2382411018190438, -1.2407546504472493, -1.2417013061599846, -1.2411064649676145, -1.2389987171377954, -1.2354097687329348, -1.2303743558371896, -1.2239301516827363, -1.2161176669021556, -1.206980143150108, -1.1965634403534926, -1.1849159178644815, -1.17208830980537, -1.158133594908206, -1.143106861165256, -1.1270651656187232, -1.11006738962992, -1.0921740899786987, -1.0734473461540806, -1.0539506042059212, -1.033748517535864, -1.0129067850129225, -0.9914919868054406, -0.9695714183267489, -0.9472129226959813, -0.9244847221193148, -0.9014552485992572, -0.8781929743810692, -0.8547662425461419, -0.8312430981616001, -0.8076911203938917, -0.7841772559919378, -0.7607676545418186, -0.737527505890729, -0.7145208801326584, -0.6918105705418293, -0.6694579398329861, -0.6475227701193158, -0.6260631169299442, -0.6051351676389465, -0.5847931046473056, -0.5650889736475113, -0.5460725572882718, -0.5277912545437948, -0.5102899660781226, -0.49361098588056274, -0.4777938994331675, -0.4628754886551876, -0.4488896438532448, -0.4358672828889122, -0.42383627775789556, -0.41282138875719665, -0.4028442063981936, -0.39392310120488416, -0.3860731815174484, -0.3793062594019885, -0.37363082474762355, -0.36905202761238987, -0.3655716688594516, -0.3631881991050422, -0.3618967259795348, -0.36168902968288374, -0.36255358679568017, -0.36447560228708453, -0.3674370496410866, -0.3714167190029558, -0.37639027322829366, -0.38233031169808285, -0.3892064417442781, -0.3969853575121189, -0.40563092606740137, -0.415104280539353, -0.4253639200727858, -0.4363658163467836, -0.44806352640117114, -0.4604083114969064, -0.47334926172179126, -0.48683342603914986, -0.5008059474637734, -0.515210203037161, -0.5299879482624357, -0.5450794656483753, -0.5604237170022811, -0.5759584991020706, -0.5916206023699028, -0.6073459721623997, -0.6230698722860084, -0.6387270503407284, -0.6542519044910324, -0.6695786512590661, -0.6846414939330114, -0.6993747911816661, -0.7137132254658832, -0.7275919708379815, -0.7409468597214679, -0.7537145482660382, -0.7658326798759737, -0.7772400465146095, -0.787876747392626, -0.7976843446543255, -0.8066060156831933, -0.8145867016560298, -0.8215732519840394, -0.8275145642890938, -0.8323617195739658, -0.8360681122570534, -0.8385895747542237, -0.8398844963036227, -0.8399139357430896, -0.8386417279642333, -0.836034583782579, -0.8320621829788033, -0.8266972602826577, -0.8199156840880019, -0.8116965277048799, -0.8020221329725211, -0.7908781660754653, -0.7782536654237397, -0.7641410814771764, -0.7485363084131809, -0.731438707556989, -0.7128511225132249, -0.6927798859574635, -0.6712348180666243, -0.6482292165871449, -0.6237798385598857, -0.597906873740888, -0.5706339097770933
            };
            ASSERT(ExampleData::float_equals(expected_x, x, 1e-5));
            ASSERT(ExampleData::float_equals(expected_y, c, 1e-5));
            
            peaks_t maxima, minima;
            for (size_t i = 0; i<1000; ++i) {
                auto&& [ma, mi] = find_peaks(std::make_shared<decltype(c)>(c));
                maxima = ma, minima = mi;
            }
            Debug("%d maxima", maxima->size());
            
            Size2 size(640, 300);
            Vec2 scale = size.div(Vec2(c.size(), M_PI * 2));
            cv::Mat mat = cv::Mat::zeros(size.height, size.width, CV_8UC3);
            for (size_t i=0; i<c.size(); ++i) {
                cv::circle(mat, Vec2(i, -c.at(i) + M_PI).mul(scale), 2, Red);
            }
            
            ColorWheel wheel;
            for(auto &peak : *maxima) {
                auto pos = peak.position.mul(Vec2(1, -1)) + Vec2(0, M_PI);
                auto left = Vec2(peak.range.start, peak.position.y).mul(Vec2(1, -1)) + Vec2(0, M_PI);
                auto right = Vec2(peak.range.end, peak.position.y).mul(Vec2(1, -1)) + Vec2(0, M_PI);
                
                auto color = wheel.next();
                
                cv::circle(mat, pos.mul(scale), 4, color);
                cv::line(mat, left.mul(scale), right.mul(scale), color);
                
                if(peak.range.start < 0) {
                    auto left = Vec2(c.size() + peak.range.start, peak.position.y).mul(Vec2(1, -1)) + Vec2(0, M_PI);
                    auto right = Vec2(c.size(), peak.position.y).mul(Vec2(1, -1)) + Vec2(0, M_PI);
                    cv::line(mat, left.mul(scale), right.mul(scale), color);
                    cv::circle(mat, left.mul(scale), 3, color);
                } else
                    cv::circle(mat, left.mul(scale), 3, color);
                
                if(peak.range.end >= c.size()) {
                    auto left = Vec2(0, peak.position.y).mul(Vec2(1, -1)) + Vec2(0, M_PI);
                    auto right = Vec2(peak.range.start - c.size(), peak.position.y).mul(Vec2(1, -1)) + Vec2(0, M_PI);
                    cv::line(mat, left.mul(scale), right.mul(scale), color);
                    cv::circle(mat, right.mul(scale), 3, color);
                } else
                    cv::circle(mat, right.mul(scale), 3, color);
                
                Debug("Maximum @%f has height %f and width %f (%f-%f)", peak.position.x, peak.position.y, peak.width, peak.range.start, peak.range.end);
            }
            
            cv::imshow("image", mat);
            cv::waitKey(1);
        }
        
        if(true) {
            /**
             DISPLAY A TERMITE
             */
            
            struct FrameData {
                std::vector<std::vector<point_t>> points;
                PeakMode mode;
            };
            
            auto noisy_frame = ExampleData::noisy_frame();
            auto fish_frames = ExampleData::fish_frames();
            auto termite_frames = ExampleData::termite_frames();
            auto stickleback_frames = ExampleData::stickleback_frames();
            auto stickleback_tags_frames = ExampleData::stickleback_tags_frames();
            auto placo_frames = ExampleData::placo_frames();
            
            size_t example_data_index = 0;
            std::vector<FrameData> example_frames {
                FrameData{noisy_frame, PeakMode::FIND_POINTY},
                FrameData{stickleback_tags_frames, PeakMode::FIND_POINTY},
                FrameData{fish_frames, PeakMode::FIND_POINTY},
                FrameData{termite_frames, PeakMode::FIND_BROAD},
                FrameData{stickleback_frames, PeakMode::FIND_POINTY},
                FrameData{placo_frames, PeakMode::FIND_POINTY}
            };
            
            scalars_t curv, curv_smooth;
            
            auto expected_points = ExampleData::get_termite();
            auto ptr = std::make_shared<decltype(expected_points)>(expected_points);
            coeffs = eft(ptr, 50);
            
            Debug("Running a few iterations of curvature for %d points...", ptr->size());
            for (size_t i=0; i<10000; ++i)
                curv = curvature(ptr, 10);
            
            auto expect = ExampleData::get_termite_curvature();
            ASSERT(ExampleData::float_equals(*curv, expect, 1e-5));
            
            curv = curvature(ptr, max(1, ptr->size() * 0.03));
            auto diff = differentiate(curv)[0];
            auto [maxima, minima] = ExampleData::find_peaks(curv);
            
            DrawStructure base(1024, 768);
            
            int order = 3;
            const int maxpoints = 100;
            float scale = base.width() / 800 * 2;
            Timer timer;
            Vec2 center;
            
            std::vector<Vertex> original;
            auto minima_ptr = minima, maxima_ptr = maxima;
            float highlighted_x = infinity<float>();
            
            const Size2 dims(base.width(), base.height() * 0.25);
            static Graph graph(Bounds(Vec2(0, base.height()-dims.height), dims), "curvature", Rangef(0, 0), Rangef(-1,-1));
            
            size_t current_index = 0;
            
            auto update_points = [&](size_t index){
                if(example_data_index == 4)
                    scale = base.width() / 800 * 0.8;
                else
                    scale = base.width() / 800 * 4;
                
                auto idx = example_data_index;
                if(idx >= example_frames.size())
                    idx = 0;
                if(index >= example_frames.at(idx).points.size())
                    index = 0;
                
                ptr = std::make_shared<points_t::element_type>(example_frames.at(idx).points.at(index));
                
                Curve curve;
                curve.set_points(ptr);
                curve.make_clockwise();
                ptr = curve.points();
                
                //coeffs = eft(ptr, order);
                curv = curvature(ptr, max(1, ptr->size() * 0.03));
                auto diff = differentiate(curv)[0];
                auto [maxima, minima] = find_peaks(curv, 0, {}, example_frames[idx].mode);
                
                center = Vec2(0, 0);
                for(auto &pt : *ptr) {
                    center += pt;
                }
                center /= ptr->size();
                
                // precalculate original outline + curvature
                Color positive_clr(0, 255, 125);
                Color negative_clr(Red);
                
                float n_mi = FLT_MAX, n_mx = 0;
                float p_mi = FLT_MAX, p_mx = 0;
                
                for (auto c: *curv) {
                    auto cabs = cmn::abs(c);
                    
                    if(c < 0) {
                        n_mi = min(n_mi, cabs);
                        n_mx = max(n_mx, cabs);
                    } else {
                        p_mi = min(p_mi, cabs);
                        p_mx = max(p_mx, cabs);
                    }
                }
                
                original.clear();
                for (size_t i=0; i<=curv->size(); ++i) {
                    auto &pt = i == curv->size() ? ptr->front() : ptr->at(i);
                    auto c = i == curv->size() ? curv->front() : curv->at(i);
                    
                    float percent = (c + n_mx) / (n_mx+p_mx);
                    auto clr = negative_clr * (1.f - percent) + positive_clr * percent;
                    //clr = clr.alpha(0.8 * 255);
                    
                    original.push_back(Vertex((pt - center) * scale + Size2(base.width(), base.height()) * 0.5, clr));
                }
                
                minima_ptr = minima;
                maxima_ptr = maxima;
                highlighted_x = infinity<float>();
                
                graph.clear();
                graph.set_ranges(Rangef(0, original.size()), Rangef(-n_mx, p_mx));
            };
            
            current_index = 18;
            update_points(current_index);
            Circle butt(Vec2(), 10, Cyan, Cyan);
            
            IMGUIBase window("Test", base, [](){
                return true;
            }, [&](auto& e){
                auto idx = example_data_index;
                if(idx >= example_frames.size())
                    idx = 0;
                
                if(e.type == EventType::KEY && e.key.pressed) {
                    if(e.key.code == Keyboard::Escape) {
                        SETTING(terminate) = true;
                    } else if(e.key.code == Keyboard::Right) {
                        ++current_index;
                        if(current_index >= example_frames[idx].points.size())
                            current_index = 0;
                        update_points(current_index);
                    } else if(e.key.code == Keyboard::Left) {
                        if(current_index == 0)
                            current_index = example_frames[idx].points.size() - 1;
                        else
                            --current_index;
                        update_points(current_index);
                    } else if(e.key.code == Keyboard::Down) {
                        ++example_data_index;
                        if(example_data_index >= example_frames.size())
                            example_data_index = 0;
                        
                        if(current_index >= example_frames[idx].points.size())
                            current_index = 0;
                        update_points(current_index);
                    } else if(e.key.code == Keyboard::RBracket) {
                        ++order;
                        if(order >= maxpoints)
                            order = 1;
                        update_points(current_index);
                    } else if(e.key.code == Keyboard::Slash) {
                        --order;
                        if(order < 1)
                            order = maxpoints-1;
                        update_points(current_index);
                    }
                }
                
            });
            
            SFLoop loop(base, &window, [&](auto&){
                auto idx = example_data_index;
                if(idx >= example_frames.size())
                    idx = 0;
                
                base.text(Str("order:"+Meta::toStr(order)), Loc(100), TextClr(Red), Font(scale / 2 * 0.8));
                base.text(Str("frame:"+Meta::toStr(current_index)+"/"+Meta::toStr(example_frames.at(idx).points.size())), Loc(100, 150), TextClr(Red), Font(scale / 2 * 0.8));
                
                {
                    static Timing timing("ieft");
                    TakeTiming take(timing);
                    coeffs = eft(ptr, order);
                    points = ieft(coeffs, coeffs->size(), ptr->size(), Vec2(), false);
                    curv_smooth = curvature(points.back(), max(1, points.back()->size() * 0.03));
                }
                
                const auto zero = Size2(base.width(), base.height()) * 0.5;
                
                std::vector<Vertex> vertices;
                for (auto&pt : *points.front()) {
                    vertices.push_back(Vertex(pt * scale + zero, Red.alpha(150)));
                }
                
                base.line(vertices, 5);
                base.line(original, 5);
                
                for(auto &pt : *minima_ptr) {
                    auto pos = (ptr->at(pt.position.x) - center) * scale + zero;
                    if(pt.position.x != highlighted_x)
                        base.circle(pos, 5, Red);
                    else
                        base.circle(pos, 10, White, Red.alpha(200));
                }
                
                for(auto &pt : *maxima_ptr) {
                    auto pos = (ptr->at(pt.position.x) - center) * scale + zero;
                    if(pt.position.x != highlighted_x)
                        base.circle(pos, 5, Green);
                    else
                        base.circle(pos, 10, White, Green.alpha(200));
                }
                
                if(graph.functions().empty()) {
                    graph.add_function(Graph::Function("curvature", Graph::DISCRETE, [&curv](float x) -> float {
                        while (x < 0)
                            x += curv->size();
                        while (x >= curv->size())
                            x -= curv->size();
                        
                        return curv->at(x);
                    }));
                    
                    graph.add_function(Graph::Function("curvature'", Graph::DISCRETE, [&diff](float x) -> float {
                        while (x < 0)
                            x += diff->size();
                        while (x >= diff->size())
                            x -= diff->size();
                        
                        return diff->at(x);
                    }));
                    
                    graph.add_function(Graph::Function("smooth_curvature", Graph::DISCRETE, [&curv_smooth](float x) -> float {
                        while (x < 0)
                            x += curv_smooth->size();
                        while (x >= curv_smooth->size())
                            x -= curv_smooth->size();
                        
                        return curv_smooth->at(x);
                    }));
                    
                    std::vector<Vec2> mini;
                    for (auto &peak : *minima_ptr) {
                        mini.push_back(peak.position);
                    }
                    graph.add_points("minima", mini, [&](auto&, auto x) {
                        highlighted_x = x;
                        //base.set_dirty(&window);
                    });
                    
                    std::vector<Vec2> maxi;
                    float max_h = -1, max_int = -1;
                    scalar_t max_int_index = 0;
                    std::set<Vec2> max_int_pts;
                    for (auto &peak : *maxima_ptr) {
                        auto h = cmn::abs(peak.range.end - peak.range.start);
                        max_h = max(max_h, h);
                        if(peak.integral > max_int) {
                            max_int_index = peak.position.x;
                            max_int = peak.integral;
                            max_int_pts = peak.points;
                        }
                    }
                    
                    auto is_in_periodic_range = [&ptr](const range_t& range, scalar_t x) -> std::tuple<bool, scalar_t> {
                        if(range.start < 0) {
                            if(x - ptr->size() >= range.start)
                                return {true, x - ptr->size()};
                            return {x <= range.end, x};
                        } else if(range.end >= ptr->size()) {
                            if(x + ptr->size() <= range.end) {
                                return {true, x + ptr->size()};
                            }
                            return {x >= range.start, x};
                        }
                        return {range.contains(x), x};
                    };
                    
                    std::vector<Peak> high_peaks;
                    
                    for (auto &peak : *maxima_ptr) {
                        maxi.push_back(peak.position);
                        
                        auto h = cmn::abs(peak.range.end - peak.range.start) / max_h;
                        auto percent = peak.integral / max_int;
                        auto y = h / M_PI;
                        y = percent / M_PI;
                        h = percent;
                        
                        graph.add_line(Vec2(peak.position.x, 0), Vec2(peak.position.x, y), Blue.alpha(percent * 200 + 55));
                        
                        if(abs(peak.integral - max_int) <= 1e-5) {
                            //peak.position.x == max_int_index) {
                            high_peaks.push_back(peak);
                        }
                        //graph.add_line(Vec2(peak.position.x - peak.integral / max_int * 50 * scale, y), Vec2(peak.position.x + peak.integral / max_int * 50 * scale, y), Blue);
                        if(abs(percent - 1) < 1e-5)
                            graph.add_points("MAX", {Vec2(peak.position.x,  y)});
                        //graph.add_line(Vec2(peak.range.start, peak.position.y), Vec2(peak.range.end, peak.position.y), Blue);
                    }
                    graph.add_points("maxima", maxi, [&](auto&, auto x) {
                        highlighted_x = x;
                        Debug("%f", x);
                        //base.set_dirty(&window);
                    });
                    
                    scalar_t idx = 0;
                    std::set<point_t> pts;
                    if(example_frames[idx].mode == PeakMode::FIND_POINTY) {
                        idx = max_int_index;
                        pts.insert(max_int_pts.begin(), max_int_pts.end());
                    } else {
                        range_t merged;
                        scalar_t max_y = 0;
                        
                        for(size_t i=0; i<high_peaks.size(); ++i) {
                            if(!i)
                                merged = high_peaks.at(i).range;
                            else
                                merged = range_t(min(high_peaks.at(i).range.start, merged.start), max(high_peaks.at(i).range.end, merged.end));
                            if(high_peaks.at(i).max_y > max_y)
                                max_y = high_peaks.at(i).max_y;
                            
                            pts.insert(high_peaks.at(i).points.begin(), high_peaks.at(i).points.end());
                        }
                        
                        scalar_t start = merged.end, end = merged.start;
                        for(auto & pt : pts) {
                            auto && [in_range, cx] = is_in_periodic_range(merged, pt.x);
                            if(pt.y >= max_y * 0.9 && in_range) {
                                if(start > cx)
                                    start = cx;
                                if(end < cx)
                                    end = cx;
                            }
                        }
                        
                        //Debug("peak has range %f-%f (%f) - %f-%f", merged.start, merged.end, merged.length(), start, end);
                        idx = round(start + (end - start)*0.5);
                        if(idx < 0)
                            idx += ptr->size();
                        if(idx >= ptr->size())
                            idx -= ptr->size();
                    }
                    
                    auto pos = (ptr->at(idx) - center) * scale + zero;
                    butt.set_pos(pos);
                    graph.add_points("butt_points", std::vector<Vec2>(pts.begin(), pts.end()), Cyan);
                }
                
                base.wrap_object(graph);
                base.wrap_object(butt);
                
                /*if(timer.elapsed() * 1000 >= 500) {
                    ++order;
                    if(order >= maxpoints)
                        order = 1;
                    timer.reset();
                }*/
                
                std::this_thread::sleep_for(std::chrono::milliseconds(25));
                
            });
        }
    }
};

std::vector<std::vector<Vec2>> calculate_outlines(const cv::Mat& original, float scale, Vec2 pos, std::vector<Vec2>& entry_points) {
    //static Timing timing("calculate_outlines", 0.1);
    //TakeTiming take(timing);
    
    const size_t N = entry_points.size();
    const arange<long> range(0, N-1);
    std::deque<long> unassigned(range.begin(), range.end());
    cv::Mat colored;
    
    size_t assigned = 0;
    
#define pdist(A, b) sqdistance(A, entry_points[b])
    
    float prev_angle;
    Vec2 prev_point(FLT_MAX, FLT_MAX);
    Vec2 direction(FLT_MAX, FLT_MAX);
    //float current_min_rawd = 0;
    
    //const float factor = 0.01;//FAST_SETTINGS(outline_resample) ? FAST_SETTINGS(outline_resample) : 1;//0.25;
    ////
    const auto rdist_points = [&](const Vec2& A, const Vec2& B, float d) -> float {
        if(B == A)
            return FLT_MAX;
        return d;
        
        float component = 1;
        if(prev_point.x != FLT_MAX && prev_angle != FLT_MAX) {
            auto v = (B - A).normalize();
            //auto v1 = (A - B).normalize();
            auto dir = direction.normalize();
            component = sqdistance(dir, v);//min(sqdistance(dir, v), sqdistance(dir, v1));
        }
        
        return (d * 0.9 + component * 0.1);
        
        /*if(prev_point.x != FLT_MAX && prev_angle != FLT_MAX) {
            assert(!std::isnan(prev_angle));
            
            float angle = atan2(B.y - A.y, B.x - A.x) - prev_angle;
            if(angle < -M_PI) angle += 2*M_PI;
            if(angle > M_PI) angle -= 2*M_PI;
            assert(abs(angle) <= M_PI);
            
            return (d) + abs(angle) / M_PI * (factor) * 0.1;
        }
        
        return (d) + (factor);*/
    };
    
    const auto rdist = [&](const Vec2& A, long b) -> float {
        const float d0 = pdist(A, b);
        
        //if(d0 > SQR(factor * 10))
        //    return FLT_MAX;
        
        const Vec2& B = entry_points[b];
        return rdist_points(A, B, d0);
    };
    
    auto compare = [](const std::vector<Vec2>& A, const std::vector<Vec2>& B) -> bool {
        return A.size() > B.size();
    };
    
    std::multiset<std::vector<Vec2>, decltype(compare)> outlines(compare);
    std::vector<Vec2> _outline;
    _outline.reserve(unassigned.size());
    
    //cv::imshow("colored", original);
    //cv::waitKey();
    /*auto draw = [&]() {
        Debug("%d finished outlines, %d current len", outlines.size(), _outline.size());
        
        original.copyTo(colored);
        for(auto idx : unassigned) {
            auto &pt = entry_points[idx];
            cv::circle(colored, (Vec2(pt.x, pt.y) - pos) * scale, 1, Yellow, -1);
        }
        
        Vec2 prev = _outline.empty() ? Vec2(-1) : _outline.back();
        for(auto &pt : _outline) {
            cv::line(colored, (prev - pos) * scale, (pt - pos) * scale, Cyan, 2);
            prev = pt;
            //cv::circle(colored, (Vec2(pt.x, pt.y) - pos) * scale, 3, Cyan);
        }
        
        if(!unassigned.empty()) {
            auto current = entry_points[unassigned.front()];
            cv::circle(colored, (Vec2(current.x, current.y) - pos) * scale, 5, Cyan);
            
            if(direction.x != FLT_MAX) {
                Vec2 p0 = current;
                Vec2 p1 = current + direction * 2;
                
                cv::line(colored, (p0 - pos) * scale, (p1 - pos) * scale, Cyan);
            }
        }
            
        ColorWheel wheel;
        for(auto &o : outlines) {
            auto color = wheel.next();
            Vec2 prev = o.back();
            for(auto &pt : o) {
                cv::line(colored, (prev - pos) * scale, (pt - pos) * scale, color, 2);
                //cv::circle(colored, (pt - pos) * scale, 6, color, pt == o.front() ? -1 : 1);
                prev = pt;
            }
        }
        
        cv::cvtColor(colored, colored, cv::COLOR_BGR2RGB);
        cv::imshow("colored", colored);
        cv::waitKey(1);
    };*/
    
    /**
     
     GO IN THE OTHER DIRECTION AS WELL, IF ONE OF THE LINES REMAINING
     HAS A VALID CONNECTION TO THE OUTLINE START
     
     */
    // repeat until we found the biggest object
    while (unassigned.size() > N * 0.05) {//unassigned.size() > assigned) {
        _outline.clear();
        //assigned = 0;
        //current_min_rawd = FLT_MAX;
        direction = Vec2(FLT_MAX, FLT_MAX);
        prev_angle = FLT_MAX;
        prev_point = Vec2(FLT_MAX, FLT_MAX);
        float back_front = FLT_MAX;
        
        long pt = -1;
        while (!unassigned.empty()) {
            //draw();
            
            pt = unassigned.front();
            unassigned.pop_front();
            
            if(!_outline.empty())
                prev_point = _outline.back();
            _outline.push_back(entry_points[pt]);
            assigned++;
            
            back_front = _outline.size() > 3 ? rdist_points(_outline.back(), _outline.front(), sqdistance(_outline.back(), _outline.front())) : FLT_MAX;
            
            if(unassigned.size() >= 1) {
                Vec2 A = entry_points[pt];
                
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
                    auto front_d = rdist(A, 0);
                    if(pt != 0 && _outline.size() > 3 && min_d > front_d)
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
                        d = rdist(_outline.front(), *it);
                        if(d < min_d) {
                            min_d = d;
                            min_idx = it;
                        }
                    }
                    
                    if(min_idx != unassigned.end() && min_d < back_front) {
                        direction = Vec2(FLT_MAX, FLT_MAX);
                        std::reverse(_outline.begin(), _outline.end());
                        
                        std::swap(*unassigned.begin(), *min_idx);
                    } else
                        break;
                }
            } else
                break;
        }
        
        //if(_outline.confidence() > 0.9)
            outlines.insert(_outline);
        
        /**
         * TEMPORARILY terminating upon biggest-outline-found.
         * need to stitch together outlines potentially.
         */
        //if(outlines.begin()->size() > unassigned.size()) {
            //Debug("Stopping %d/%d", outlines.begin()->size(), unassigned.size());
        //    break;
        //}
    }
    
    
    //draw();
    //Debug("0 (unassigned: %d, assigned: %d)", unassigned.size(), assigned);
    //_outline.clear();
    return std::vector<std::vector<Vec2>>(outlines.begin(), outlines.end());
}

std::vector<Vec2> extract_outline(pv::BlobPtr blob, StaticBackground* bg, int threshold)
{
    //threshold = 25;
    
    {
        auto lines = std::make_shared<std::vector<HorizontalLine>>();
        *lines = {
            HorizontalLine(0, 0, 100),
            HorizontalLine(1, 0, 50),
            HorizontalLine(2, 0, 25)
        };
        
        auto pixels = std::make_shared<std::vector<uchar>>();
        for(auto &line : *lines) {
            for (auto x=line.x0; x<=line.x1; ++x) {
                pixels->push_back(255);
            }
        }
        
        lines->push_back(HorizontalLine(3, 5, 10));
        pixels->insert(pixels->end(), 6, 150);
        
        lines->push_back(HorizontalLine(4, 0, 25));
        pixels->insert(pixels->end(), 26, 255);
        
        auto test_blob = std::make_shared<pv::Blob>(lines, pixels);
        auto && [pos, image] = test_blob->image();
        cv::imshow("test_blob", image->get());
        
        {
            auto thresholded = pixel::threshold_get_biggest_blob(test_blob, 200, NULL);
            auto && [pos, image] = thresholded->image();
            cv::imshow("thresholded", image->get());
        }
        cv::waitKey(1);
        
        
    }
    
    Timer generate_image_timer;
    auto && [pos, image] = blob->image();
    
    auto original_pos = blob->bounds().pos();
    if(threshold > 0)
        blob = pixel::threshold_get_biggest_blob(blob, threshold, bg);
    auto thresholded_pos = blob->bounds().pos();
    //auto generate_image_time = generate_image_timer.elapsed();
    double findcont_seconds = 0, cont_samples = 0;
    std::vector<cv::Mat> cv_contours;
    cv::Mat cv_hierarchy;
    
    for (int i=0; i<500; ++i) {
        Timer timer;
        auto && [pos, image] = blob->image();
        
        std::vector<cv::Mat> contours;
        cv::Mat hierarchy;
        cv::findContours(image->get(), contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
        
        findcont_seconds += timer.elapsed();
        ++cont_samples;
        
        cv_contours = contours;
        cv_hierarchy = hierarchy;
    }
    
    Debug("cv::findContours: %fms (%d contours)", (findcont_seconds) / cont_samples * 1000, cv_contours.size());
    
    std::vector<Posture::EntryPoint> points;
    Timer timer;
    double seconds = 0, samples = 0;
    std::vector<Vec2> outline;
    size_t point_nr = 0;
    
    for(int i=0; i<500; ++i) {
        timer.reset();
        points = Posture::subpixel_threshold(image->get(), threshold);
        seconds += timer.elapsed();
        
        point_nr = 0;
        for(auto &ep : points) {
            point_nr += ep.interp.size();
        }
        ++samples;
    }
    
    seconds = seconds / samples * 1000;
    auto subpixel_time = seconds;
    
    cv::Mat colored;
    cv::Mat white;
    cv::cvtColor(image->get(), colored, cv::COLOR_GRAY2BGR);
    white = cv::Mat(image->rows, image->cols, CV_8UC4);
    white = cv::Scalar(0,0,0,0);
    
    
    for(auto &line : blob->hor_lines()) {
        for (auto x=line.x0; x<=line.x1; ++x) {
            colored.at<cv::Vec3b>(line.y - pos.y, x - pos.x) = //Color(colored.at<cv::Vec3b>(line.y - pos.y, x - pos.x)) +
            Viridis::value(colored.at<cv::Vec3b>(line.y - pos.y, x - pos.x)[0] / 255.f);
            white.at<cv::Vec4b>(line.y - pos.y, x - pos.x) = //Color(colored.at<cv::Vec3b>(line.y - pos.y, x - pos.x)) +
            Viridis::value(colored.at<cv::Vec3b>(line.y - pos.y, x - pos.x)[0] / 255.f);
            //colored.at<cv::Vec3b>(line.y - pos.y, x - pos.x) = saturate((int)colored.at<cv::Vec3b>(line.y - pos.y, x - pos.x)[0] + 100);
            //colored.at<cv::Vec3b>(line.y, x)[0] = saturate((int)colored.at<cv::Vec3b>(line.y, x)[0] + 100);
        }
    }
    
    white.copyTo(colored);
    cv::imshow("image", colored);
    //cv::waitKey();
    
    //assert(colored.cols == blob->bounds().width + 2);
    std::vector<Vec2> custom, interp, grad;
    std::vector<std::shared_ptr<std::vector<Vec2>>> outlines;
    //pixel::Tree tree;
    samples = seconds = 0;
    
    for(int i=0; i<1; ++i) {
        timer.reset();
        outlines = pixel::find_outer_points(blob, threshold);
        //interp = in;
        //grad = gr;
        //tree = tr33;
        seconds += timer.elapsed();
        ++samples;
    }
    
    float scale = 27;
    cv::Mat original;
    cv::resize(colored, original, (cv::Size)(Size2(image->cols, image->rows) * scale), 0, 0, cv::INTER_NEAREST);
    original.copyTo(colored);
    cv::copyMakeBorder(colored, colored, 0, 1, 0, 1, cv::BorderTypes::BORDER_CONSTANT);
    
    for (uint i=0; i<image->cols; ++i) {
        cv::line(colored, Vec2(i, 0) * scale, Vec2(i, image->rows) * scale, Black);
    }
    
    for (uint i=0; i<image->rows; ++i) {
        cv::line(colored, Vec2(0, i) * scale, Vec2(image->cols, i) * scale, Black);
    }
    
    for(auto &pt : points) {
        cv::line(colored, Vec2(pt.x0, pt.y)*scale, Vec2(pt.x1, pt.y)*scale, Red);
        for(auto &p : pt.interp)
            cv::circle(colored, Vec2(p.x, p.y) * scale, 2, Red, -1);
    }
    
    std::vector<std::vector<Vec2>> ol;
    {
        std::vector<Posture::EntryPoint> entry_points;
        Posture::EntryPoint ep;
        std::set<std::shared_ptr<std::vector<Vec2>>, std::function<bool(std::shared_ptr<std::vector<Vec2>>, std::shared_ptr<std::vector<Vec2>>)>> ols([](std::shared_ptr<std::vector<Vec2>> A, std::shared_ptr<std::vector<Vec2>> B) -> bool {
            return A->size() > B->size();
        });
        for (auto ol : outlines) {
            ols.insert(ol);
        }
        
        if(!ols.empty()) {
            auto interp = *ols.begin();
            for(auto &pt : *interp) {
                ep.y = pt.y;
                ep.x0 = ep.x1 = pt.x;
                ep.interp = {pt};
                entry_points.push_back(ep);
            }
        }
        
        double seconds = 0, samples = 0;
        /*for(size_t i=0; i<100; ++i) {
            Timer timer;
            ol = calculate_outlines(original, scale, pos, interp);
            outline = ol.front();
            seconds += timer.elapsed();
            samples++;
        }
        
        Debug("calculate_outlines\t%fms", seconds / samples * 1000);*/
        
        seconds = samples = 0;
        for(size_t i=0; i<50; ++i) {
            Timer timer;
            Posture posture(0, 0);
            posture.calculate_outline(entry_points);
            outline = posture.outline().points();
            seconds += timer.elapsed();
            samples++;
        }
        
        subpixel_time += seconds / samples * 1000;
        Debug("Posture::calculate_outlines\t%fms overall: %fms", seconds / samples * 1000, subpixel_time);
    }
    
    seconds = seconds / samples * 1000;
    
    Debug("subpixel_threshold: %fms,  %d (%d) outline points vs. %d", subpixel_time, outline.size(), point_nr, ol.empty() ? 0 : ol.front().size());
    Debug("find_outer: %fms (%d outlines)", seconds, outlines.size());
    
    Posture posture(0,0);
    posture.calculate_posture(0, blob);
    auto midline = posture.normalized_midline();
    
    cv::Mat mat = image->get();
    /*for(auto &pt : custom) {
        int x0 = pt.x - pos.x;
        int y0 = pt.y - pos.y;
        
        auto old = pt;
        cv::circle(colored, (Vec2(pt.x, pt.y) - pos + 0.5) * scale, 2, Blue, -1);
        
        float center = threshold / 255.f;
        center = mat.at<uchar>(y0, x0) / 255.f;
        Vec2 vec(0, 0);
        float weights = 1;
        for (int x=-2; x<=2; ++x) {
            for (int y=-2; y<=2; ++y) {
                if(x == y && x == 0)
                    continue;
                
                if (x0 + x >= 0 && y0 + y >= 0) {
                    float w = (float)mat.at<uchar>(y0 + y, x0 + x) / 255.f;
                    vec -= Vec2(x < 0 ? -1 : 1, y < 0 ? -1 : 1) * (w - center);
                    weights += 1;
                    //neighbors[x+1 + (y + 1) * 3] = ;
                }
            }
        }
        
        //if(weights)
            vec /= 2;
        
        auto str = Meta::toStr(vec);
        Debug("%S", &str);
        
        pt += vec;
    }*/
    
    auto draw = [&, pos = pos, image = image]() {
        //original.copyTo(colored);
        for(auto &pt : custom) {
            cv::circle(colored, (Vec2(pt.x, pt.y) - pos + Vec2(0.5)) * scale, 5, DarkCyan, -1);
        }
        
        /*if(!grad.empty()) {
            for(size_t i=0; i<interp.size(); ++i) {
                auto &pt = interp.at(i);
                auto &g = grad.at(i);
                //cv::line(colored, (Vec2(pt.x, pt.y) - pos) * scale, (Vec2(pt.x, pt.y) + g - pos) * scale, DarkCyan);
            }
        }
        
        for(auto &pt : interp) {
            //cv::circle(colored, (Vec2(pt.x, pt.y) - pos) * scale, 3, Cyan);
        }*/
        
        //Debug("CV:");
        for(auto &cont : cv_contours) {
            //print_mat("cont", cont);
            for(int i=0; i<cont.rows; ++i) {
                auto v = cont.at<cv::Vec2i>(i, 0);
                Vec2 pt(v[0], v[1]);
                cv::circle(colored, (pt + (thresholded_pos - original_pos) + Vec2(0.5)) * scale, 3, Yellow, -1);
            }
            //Debug("\t%dx%d", cont.cols, cont.rows);
        }
        
        //float seconds = 0;
        //ColorWheel wheel;
        //for(auto &o : ol) {
            /*float order = o.size() > 8 ? min((float)o.size(), o.size() * 0.05) : o.size(); //o.size() > 8 ? max(4, float(o.size()) * 0.05) : o.size();
            if(order < 8)
                order = 8;
            
            Vec2 mean(0, 0);
            float samples = 0;
            
            for(auto &pt : o) {
                mean += pt;
            }
            mean /= float(o.size());
            
            auto points = std::make_shared<std::vector<Vec2>>(o);
            Timer timer;
            coeff_t ptr = periodic::eft(points, order);
            seconds = timer.elapsed();
            auto pts = periodic::ieft(ptr, ptr->size(), max(10, o.size() * 0.25)).back();
            
            
            Debug("Time for EFT / iEFT of size %d: %fms", o.size(), seconds* 1000);
            
            o = *pts;*/
            
            /*auto color = wheel.next();
            Vec2 prev = o.back();
            for(auto &pt : o) {
                //cv::line(colored, (prev - pos) * scale, (pt - pos) * scale, color, 2);
                //cv::circle(colored, (pt - pos) * scale, 6, color, pt == o.front() ? -1 : 1);
                prev = pt;
            }*/
        //}
        
#define OFFSET(X) (((X) - pos) * scale)
        
        /*for(auto node : tree.nodes()) {
            cv::circle(colored, (node->position - pos) * scale, 5, White, -1);
            for(auto d : node->border) {
                auto offset = pixel::vectors[(size_t)d] * 0.5;
                cv::circle(colored, (node->position + offset - pos) * scale, 3, White, -1);
            }
        }*/
        
        if(midline) {
            auto transform = midline->transform(default_config::recognition_normalization_t::posture, true);
            for(auto &seg : midline->segments()) {
                auto trans = transform.transformPoint(seg.pos);
                cv::circle(colored, OFFSET(trans), 3, Black);
            }
            
            if(posture.normalized_midline()) {
                auto midline = posture.normalized_midline();
                //if(midline->tail_index() != -1)
                //    cv::circle(colored, OFFSET(posture.outline().at(midline->tail_index())), 10, Blue, -1);
                //if(midline->head_index() != -1)
                //    cv::circle(colored, OFFSET(posture.outline().at(midline->head_index())), 10, Red, -1);
                
                Debug("tail:%d head:%d", midline->tail_index(), midline->head_index());
            }
        }
        
        std::stringstream ss;
        ss << "<svg viewBox='0 0 "+Meta::toStr(colored.cols)+" "+Meta::toStr(colored.rows)+"' xmlns='http://www.w3.org/2000/svg'>\n";
        
        auto color2svg = [](Color c) {
            return "rgba(" + Meta::toStr(c.r ) + "," + Meta::toStr(c.g) + "," + Meta::toStr(c.b) + "," + Meta::toStr(c.a ) + ")";
        };
        
        for(uint x=0; x<image->cols; ++x) {
            for (uint y=0; y<image->rows; ++y) {
                //auto clr = Viridis::value(float(image->get().at<uchar>(y, x)) / 255.f);
                auto v = 255 - image->get().at<uchar>(y, x);
                auto clr = Color(v);
                if(v == 255)
                    clr = Transparent;
                else
                    clr = Color(128,200,234,255);
                ss << "\t<rect x='"+Meta::toStr(x*scale)+"' y='"+Meta::toStr(y*scale)+"' width='"+Meta::toStr(scale)+"' height='"+Meta::toStr(scale)+"' style='stroke:"+color2svg(Black)+";fill:"+color2svg(clr)+"' />\n";
            }
        }
        
        {
            std::set<std::tuple<size_t, size_t>, std::greater<>> size_indexes;
            size_t i=0;
            for(auto && cont : cv_contours)
                size_indexes.insert({cont.rows, i++});
            
            ColorWheel wheel;
            std::map<size_t, Color> colors;
            for(auto && [n, i] : size_indexes) {
                colors[i] = wheel.next().brighten(0.9);
            }
            
            float c = 0;
            for(auto &cont : cv_contours) {
                auto color = colors.at((size_t)c);
                //print_mat("cont", cont);
                for(int i=0; i<cont.rows; ++i) {
                    auto v = cont.at<cv::Vec2i>(i, 0);
                    Vec2 pt(v[0], v[1]);
                    pt = (pt + (thresholded_pos - original_pos) + Vec2(0.5) + Vec2(0.125 + 0.5 * (c / float(cv_contours.size()) - 0.5), 0)) * scale;
                    //cv::circle(colored, (pt + (thresholded_pos - original_pos) + Vec2(0.5)) * scale, 3, Yellow, -1);
                    ss << "\t<ellipse cx='"+Meta::toStr(pt.x)+"' cy='"+Meta::toStr(pt.y)+"' rx='2' ry='2' stroke-width='2' style='stroke:transparent;fill:"+color2svg(color)+"' />\n";
                }
                //Debug("\t%dx%d", cont.cols, cont.rows);
                ++c;
            }
        }
        
        {
            std::set<std::tuple<size_t, size_t>, std::greater<>> size_indexes;
            size_t i=0;
            for(auto && cont : outlines)
                size_indexes.insert({cont->size(), i++});
            
            ColorWheel wheel;
            std::map<size_t, Color> colors;
            for(auto && [n, i] : size_indexes) {
                colors[i] = wheel.next().brighten(0.9);
            }
            
            i = 0;
            for (auto &clique: outlines) {
                auto color = colors.at(i++);
                auto prev = clique->back();
                for (auto node : *clique) {
                    cv::circle(colored, OFFSET(node), 5, color);
                    ss << "\t<ellipse cx='"+Meta::toStr(OFFSET(node).x)+"' cy='"+Meta::toStr(OFFSET(node).y)+"' rx='5' ry='5' style='stroke:"+color2svg(color)+";fill:transparent' stroke-width='2' />\n";
                    cv::line(colored, OFFSET(prev), OFFSET(node), color);
                    ss << "\t<line x1='"<<OFFSET(prev).x<<"' y1='"<<OFFSET(prev).y<<"' x2='"<<OFFSET(node).x<<"' y2='"<<OFFSET(node).y<<"' stroke='" << color2svg(color) << "' stroke-width='2' />\n";
                    prev = node;
                }
            }
        }
        
        ss << "</svg>\n";
        
        auto str = ss.str();
        printf("SVG:\n%s\n", str.c_str());
        
        cv::cvtColor(colored, colored, cv::COLOR_BGR2RGB);
        cv::imshow("test", colored);
        cv::waitKey(1);
    };
    
    draw();
    cv::imwrite("image.png", colored);
    //cnpy::npy_save("/Users/tristan/Desktop/outer_points.npy", (Float2_t*)custom.data(), {custom.size(), 2});
    
    /*for(size_t i=0; i<custom.size(); ++i) {
        
        draw();
    }*/
    
    return {};
}

int main(int argc, char**argv) {
    print(info, "%s", TT_RESOURCE_FOLDER);
    CommandLine cmd(argc, argv, true);
    cmd.cd_home();
#if __APPLE__
    std::string _wd = "../Resources/";
    if (!chdir(_wd.c_str()))
        Debug("Changed directory to '%S'.", &_wd);
    else
        Error("Cannot change directory to '%S'.", &_wd);
#endif
    
    bool tests_failed = false;
    
    
    
    if(!run<TestLabeling>())
        tests_failed = true;
    
    if(!run<TestMisc>())
        tests_failed = true;
    
    if(!run<TestTracking>())
        tests_failed = true;
    
    if(!run<TestCircularGraphs>())
        tests_failed = true;
    
    std::vector<scalar_t> curvature{0.114597,0.035283,0.031355,0.009789,-0.013756,0.036372,0.043067,0.040647,0.045483,0.007744,-0.013387,0.009376,0.048119,0.053089,0.013066,-0.025695,-0.052966,-0.021509,0.028664,0.023191,-0.001357,-0.035390,-0.032412,0.031978,0.070197,0.073208,0.034292,-0.044179,-0.068274,-0.053141,-0.031008,-0.006992,-0.004797,-0.006707,-0.006637,-0.003873,0.016444,0.047246,0.088715,0.138593,0.119754,0.037764,-0.026926,-0.056396,-0.021126,0.038295,0.061995,0.040882,-0.021667,-0.055249,-0.060522,-0.052327,-0.018167,-0.005216,0.006166,0.058633,0.117017,0.152949,0.178732,0.170193,0.156252,0.204226,0.221592,0.218812,0.209488,0.146080,0.121031,0.151100,0.168466,0.189766,0.208820,0.215967,0.223249,0.235570,0.245512,0.212956,0.141721,0.046050,-0.038359,-0.064789,-0.062828,-0.059291,-0.062028,-0.052696,-0.021870,0.038858,0.123939,0.122466,0.040126,-0.027858,-0.058433,-0.019874,0.038295,0.058327,0.030576,-0.012729,-0.002373,0.012289,0.018298,0.030610,0.001999,-0.012954,0.026416,0.052862,0.027893,0.005623,-0.040877,-0.100829,-0.071703,-0.010402,0.017944,0.045753,0.026186,-0.042103,-0.063299,-0.057186,-0.028406,0.025317,0.055210,0.049689,-0.019993,-0.102458,-0.092822,-0.045728,0.017070,0.044967,-0.032907,-0.061595,-0.014689,0.017184,0.012265,-0.029139,-0.072752,-0.039677,0.401047,0.548849,0.593275,0.346801,0.515890,0.250496};
    std::vector<scalar_t> curvature_okay{0.210828,0.109928,0.069095,0.049157,0.039071,0.036336,0.032832,0.028997,0.025637,0.021105,0.017453,0.014792,0.012941,0.010330,0.007766,0.006374,0.004838,0.004932,0.005420,0.004044,0.003079,0.001932,0.000941,0.001339,0.001098,0.000212,-0.001853,-0.004575,-0.006378,-0.006399,-0.003522,0.000095,0.003711,0.006909,0.009408,0.013345,0.018679,0.025170,0.030518,0.033586,0.032773,0.028083,0.023010,0.017841,0.014200,0.011457,0.008778,0.006226,0.003878,0.004173,0.006312,0.012012,0.021878,0.034126,0.049888,0.068711,0.089749,0.112482,0.137254,0.162647,0.186367,0.207292,0.221305,0.227198,0.227564,0.224243,0.222316,0.223927,0.226885,0.229325,0.228357,0.221856,0.208318,0.188116,0.163235,0.134308,0.105918,0.080252,0.057959,0.040990,0.026881,0.016249,0.008830,0.004784,0.004496,0.005394,0.008657,0.010820,0.011466,0.013497,0.014691,0.016693,0.018391,0.019104,0.018246,0.015735,0.014531,0.011730,0.009594,0.008938,0.006908,0.005798,0.004649,0.002280,-0.001473,-0.005295,-0.009299,-0.013024,-0.013371,-0.012172,-0.010921,-0.010011,-0.010958,-0.013280,-0.014349,-0.013510,-0.012358,-0.010664,-0.010124,-0.012052,-0.015424,-0.019437,-0.022095,-0.022897,-0.021227,-0.017757,-0.013182,-0.006400,0.003187,0.015046,0.030386,0.058555,0.119880,0.286510,0.733997,1.151744,1.414560,1.261370,0.952151,0.474025};
    
    {
        auto curv = std::make_shared<std::vector<scalar_t>>(curvature);
        
    }
    
    {
        auto curv = std::make_shared<std::vector<scalar_t>>(curvature);
        auto diffs = differentiate(curv, 2);
        auto && [maxima_ptr, minima_ptr] = find_peaks(curv, 0, diffs, PeakMode::FIND_BROAD);
        auto str = Meta::toStr(*maxima_ptr);
        Debug("%S", &str);
    }
    
    {
        auto curv = std::make_shared<std::vector<scalar_t>>(curvature_okay);
        auto diffs = differentiate(curv, 2);
        auto && [maxima_ptr, minima_ptr] = find_peaks(curv, 0, diffs, PeakMode::FIND_BROAD);
        auto str = Meta::toStr(*maxima_ptr);
        Debug("%S", &str);
    }
    
    return tests_failed;
}
