#include "DetectTag.h"
#include <misc/GlobalSettings.h>
#include <types.h>
#include <tracking/Tracker.h>
#include <misc/Timer.h>
#include <processing/PadImage.h>

namespace track {
    namespace tags {
        std::vector<result_t> prettify_blobs(const std::vector<blob_pixel>& fish, const std::vector<blob_pixel>& noise, const Image& average)
        {
            std::vector<result_t> result;
            std::vector<result_t> noise_images;
            
            //cv::Mat correct;
            //Tracker::instance()->grid()->relative_brightness().copyTo(correct);
            
            for(auto blob : noise) {
                cv::Mat greyscale, bin;
                imageFromLines(blob->hor_lines(), &bin, &greyscale, NULL, blob->pixels().get(), 0, &average);
                
                noise_images.push_back({ blob, Image::Make(greyscale), Image::Make(bin) });
            }
            
            //! for all fish, try to correct their images by adding smaller noise images
            //  to the overall image
            for(auto blob : fish) {
                cv::Mat mgrey, mmask;
                imageFromLines(blob->hor_lines(), &mmask, &mgrey, NULL, blob->pixels().get(), 0, &average);
                
                for(auto && [nb, grey, mask] : noise_images) {
                    if(blob->bounds().contains(nb->bounds())) {
                        Bounds bounds(nb->bounds().pos() - blob->bounds().pos(), grey->bounds().size());
                        bounds.restrict_to(Bounds(mgrey));
                        
                        if(bounds.width > 0 && bounds.height > 0) {
                            auto m = mask->get();
                            grey->get()(Bounds(bounds.size())).copyTo(mgrey(bounds), m(Bounds(bounds.size())));
                            m(Bounds(bounds.size())).copyTo(mmask(bounds), m(Bounds(bounds.size())));
                        }
                    }
                }
                
                cv::Mat tmp2;
                cv::Rect outrect = Bounds(blob->bounds().pos(), Size2(mgrey));
                average.get()(outrect).copyTo(tmp2);
                mgrey.copyTo(tmp2, mmask);
                
                //tmp2.convertTo(tmp2, CV_32FC1, 1.f/255.f);
                //cv::divide(tmp2, correct(outrect), tmp2);
                //tmp2 = tmp2.mul(0.5 + tmp2);
                //tmp2.convertTo(tmp2, CV_8UC1, 255);
                
                result.push_back({blob, Image::Make(tmp2), Image::Make(mmask) });
            }
            
            return result;
        }
        
        Tag is_good_image(const result_t& result, const Image& average) {
            using namespace gui;
            static Timing timing("is_good_image");
            TakeTiming take(timing);
            
            Bounds bounds(average.bounds());
            
            auto && [blob, grey, mask] = result;
            
            cv::Mat tmp3;
            cv::threshold(grey->get(), tmp3, 150, 255, cv::THRESH_BINARY);
            grey->get().copyTo(tmp3, 255 - tmp3);
            
            cv::equalizeHist(tmp3, tmp3);
            
            std::vector<cv::Mat> output;
            cv::Mat inverted;
            
            //static int morph_size = 3;
            //static const cv::Mat element = cv::getStructuringElement( cv::MORPH_ELLIPSE, cv::Size( 2*morph_size + 1, 2*morph_size+1 ), cv::Point( morph_size, morph_size ) );
            //cv::morphologyEx(tmp3, tmp3, cv::MORPH_CLOSE,element );
            
            cv::Canny(tmp3, inverted, 250, 255);
            
            //static const cv::Mat element1 = cv::getStructuringElement( cv::MORPH_ELLIPSE, cv::Size( 2*1 + 1, 2*1+1 ), cv::Point( 1, 1 ) );
            //cv::morphologyEx(inverted, inverted, cv::MORPH_CLOSE,element1 );
            
            cv::findContours(inverted, output, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            cv::cvtColor(inverted, inverted, cv::COLOR_GRAY2BGRA);
            
            if(!output.empty()) {
                for(auto &o : output) {
                    cv::Mat shape;
                    cv::approxPolyDP(o, shape, 0.1 * cv::arcLength(o, true), true);
                    
                    if(shape.rows == 4) {
                        Vec2 previous = shape.at<cv::Point>(shape.rows - 1, 0);
                        int correct = 0;
                        Bounds bounding(FLT_MAX, FLT_MAX, 0, 0);
                        
                        for (int i=0; i<shape.rows; i++) {
                            Vec2 next = shape.at<cv::Point>(i < shape.rows-1 ? i+1 : 0);
                            Vec2 current = shape.at<cv::Point>(i, 0);
                            Vec2 v0(current - previous);
                            Vec2 v1(next - current);
                            
                            float angle = angle_between_vectors(v0, v1);
                            
                            if(DEGREE(angle) >= 75 && DEGREE(angle) <= 105 && v0.length() > 5 && v1.length() > 5) {
                                correct++;
                            }
                            
                            bounding.combine(Bounds(current, Size2(1)));
                            previous = current;
                        }
                        
                        if(correct > 5 || correct < 3)
                            continue;
                        
                        float area = bounding.width * bounding.height / float(grey->cols * grey->rows);
                        if(area > 0.4)
                            continue; // rectangles that are almost the size of the image are too big for a tag inside a blob
                        
                        cv::Mat tmp, hist;
                        grey->get()(bounding).copyTo(tmp);
                        
                        cv::Mat laplace, mean, stdv;
                        cv::Laplacian(tmp, laplace, CV_32F);
                        cv::meanStdDev(laplace, mean, stdv);
                        
                        /*if(stdv.at<double>(0, 0) <= 20) {
                            //print_mat("reject", stdv);
                            //tf::imshow("reject", tmp);
                            continue;
                        }*/
                        
                        const float range[] = {0, 255};
                        const float* ranges[] = {range};
                        const int channels[] = {0};
                        const int histSize[] = {4};
                        const cv::Mat images[] = {tmp};
                        
                        cv::calcHist(images, 1, channels, cv::Mat(), hist, 1, histSize, ranges);
                        
                        float sum = cv::sum(hist)(0);
                        hist /= sum;
                        
                        //if(blob->blob_id() == 6)
                        //    print_mat("hist", hist);
                        
                        //if((hist.at<float>(1,0) > 0.21 || hist.at<float>(2,0) > 0.02) || hist.at<float>(2,0) > hist.at<float>(1,0) + 0.01) {
                            
                        if(hist.at<float>(0, 0) >= 0.99) {
                            //print_mat("hist", hist);
                            //tf::imshow("rejected", tmp);
                            return {0.f, std::numeric_limits<decltype(Tag::blob_id)>::max(), nullptr};
                        }
                        
                        static constexpr Size2 normal_dimensions(128, 128);
                        static constexpr Vec2 offset(128, 128);
                        bounding.pos() -= offset;
                        bounding.size() += offset * 2;
                        
                        if(bounding.width > normal_dimensions.width) {
                            auto o = bounding.width - normal_dimensions.width;
                            bounding.x += o * 0.5;
                            bounding.width -= o;
                        }
                        if(bounding.height > normal_dimensions.height) {
                            auto o = bounding.height - normal_dimensions.height;
                            bounding.y += o * 0.5;
                            bounding.height -= o;
                        }
                        
                        bounding.restrict_to(grey->bounds());
                        
                        cv::Mat padded;
                        grey->get()(bounding).copyTo(tmp);
                        
                        assert(tmp.cols <= normal_dimensions.width && tmp.rows <= normal_dimensions.height);
                        pad_image(tmp, padded, normal_dimensions);
                        
                        //tf::imshow("crop", padded);
                        
                        //tf::imshow("tmp3", inverted);
                        
                        float var = stdv.at<double>(0, 0);
                        return {var, blob->blob_id(), Image::Make(padded)};
                    }
                }
            }
            
            //tf::imshow("no rectangle", inverted);
            return {0.f, std::numeric_limits<decltype(Tag::blob_id)>::max(), nullptr};
        }
    }
}
