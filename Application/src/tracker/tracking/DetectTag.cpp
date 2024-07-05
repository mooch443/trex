#include "DetectTag.h"
#include <misc/GlobalSettings.h>
#include <tracking/Tracker.h>
#include <misc/Timer.h>
#include <processing/PadImage.h>
#include <opencv2/features2d.hpp>
#include <tracking/PPFrame.h>

namespace track {
    namespace tags {
        std::vector<result_t> prettify_blobs(const std::vector<blob_pixel>& fish, const std::vector<blob_pixel>& noise, const std::vector<blob_pixel>& original, const Image& average)
        {
            std::vector<result_t> result;
            std::vector<result_t> noise_images;
            
            //cv::Mat correct;
            //Tracker::instance()->grid()->relative_brightness().copyTo(correct);
            
            for(auto &blob : noise) {
                cv::Mat greyscale, bin;
                imageFromLines(blob->input_info(), blob->hor_lines(), &bin, &greyscale, NULL, blob->pixels().get(), 0);//, &average);
                
                noise_images.push_back({ blob->blob_id(), Image::Make(greyscale), Image::Make(bin) });
            }
            
            //! for all fish, try to correct their images by adding smaller noise images
            //  to the overall image
            for(auto &blob : fish) {
                cv::Mat mgrey, mmask;
                Vec2 offset(0);

                pv::Blob* parent = nullptr;
                if (blob->parent_id().valid()) {
                    for (auto& p : original) {
                        if (p->blob_id() == blob->parent_id()) {
                            parent = p.get();
                            offset = parent->bounds().pos() - blob->bounds().pos();
                            break;
                        }
                    }
                }

                if (parent)
                    imageFromLines(blob->input_info(), parent->hor_lines(), &mmask, &mgrey, NULL, parent->pixels().get(), 0);//, &average);
                else
                    imageFromLines(blob->input_info(), blob->hor_lines(), &mmask, &mgrey, NULL, blob->pixels().get(), 0);//, &average);

                Size2 normal_dimensions = SETTING(tags_image_size).value<Size2>();
                cv::Mat padded;
                Vec2 crop_offset(0);
                if (normal_dimensions.width > mgrey.cols || normal_dimensions.height > mgrey.rows) {
                    auto size = Size2(max(normal_dimensions.width, mgrey.cols), max(normal_dimensions.height, mgrey.rows));
                    crop_offset += size - Size2(mgrey);
                    pad_image(mgrey, padded, size);
                    cv::Mat t;
                    pad_image(mmask, t, size);
                    t.copyTo(mmask);
                }
                else
                    padded = mgrey;

                cv::Mat local;
                resize_image(mgrey, local, 5);
                tf::imshow("mgrey", local);

                resize_image(padded, local, 5);
                tf::imshow("padded", local);
                mgrey = padded;
                
                cv::equalizeHist(mgrey, mgrey);

                cv::Mat tmp2;
                cv::Rect outrect = Bounds((parent ? parent->bounds().pos() : blob->bounds().pos()) - crop_offset, Size2(mgrey));
                average.get()(outrect).copyTo(tmp2);
                mgrey.copyTo(tmp2, mmask);
                
                result.emplace_back(result_t{
                    blob->blob_id(),
                    Image::Make(tmp2),
                    Image::Make(mmask)
                });
            }
            
            return result;
        }
        
        Tag is_good_image(const result_t& result) {
            using namespace gui;
            static Timing timing("is_good_image");
            TakeTiming take(timing);
            
            auto && [bdx, grey, mask] = result;
            
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
            
            for(auto &o : output) {
                cv::Mat shape;
                cv::approxPolyDP(o, shape, 0.1 * cv::arcLength(o, true), true);
                
                if(shape.rows != 4)
                    continue;
                
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
                
                cv::Mat qr = cv::Mat::zeros(tmp3.rows, tmp3.cols, CV_8UC3);
                tmp3.copyTo(qr);
                cv::cvtColor(qr, qr, cv::COLOR_GRAY2BGR);
                for (int i=0; i<shape.rows; i++) {
                    Vec2 next = shape.at<cv::Point>(i < shape.rows-1 ? i+1 : 0);
                    Vec2 current = shape.at<cv::Point>(i, 0);
                    cv::line(qr, current, next, Red);
                }
                
                cv::cvtColor(qr, qr, cv::COLOR_BGR2RGB);
                cv::resize(qr, qr, (cv::Size)Size2(1024, float(qr.rows) / float(qr.cols) * 1024));
                tf::imshow("qr", qr);
                
                cv::Mat tmp, hist;
                const Size2 normal_dimensions = SETTING(tags_image_size).value<Size2>();
                const Vec2 offset(normal_dimensions);

                if (bounding.width > normal_dimensions.width) {
                    auto o = bounding.width - normal_dimensions.width;
                    bounding.x += o * 0.5;
                    bounding.width -= o;
                }
                if (bounding.height > normal_dimensions.height) {
                    auto o = bounding.height - normal_dimensions.height;
                    bounding.y += o * 0.5;
                    bounding.height -= o;
                }
                if (bounding.width < normal_dimensions.width) {
                    auto o = normal_dimensions.width - bounding.width;
                    if (bounding.x >= o / 2) {
                        bounding.x -= floor(o / 2.0);
                        bounding.width += o;
                        print("bounding: ", bounding.x,",",bounding.width);
                    }
                }
                if (bounding.height < normal_dimensions.height) {
                    auto o = normal_dimensions.height - bounding.height;
                    if (bounding.y >= o / 2) {
                        bounding.y -= floor(o / 2.0);
                        bounding.height += o;
                        print("bounding: ", bounding.y,",",bounding.height);
                    }
                }

                bounding.restrict_to(Bounds(grey->get()));

                grey->get()(bounding).copyTo(tmp);

                auto ret = Image::Make(tmp);

                cv::Mat laplace, mean, stdv;
                cv::Laplacian(tmp, laplace, CV_32F);
                cv::meanStdDev(laplace, mean, stdv);
                
                const float range[] = {0, 255};
                const float* ranges[] = {range};
                const int channels[] = {0};
                const int histSize[] = {4};
                const cv::Mat images[] = {tmp};
                
                cv::calcHist(images, 1, channels, cv::Mat(), hist, 1, histSize, ranges);
                
                float sum = cv::sum(hist)(0);
                hist /= sum;
                
                if(hist.at<float>(0, 0) >= 0.99) {
                    break;
                }
                
                bounding.restrict_to(grey->bounds());
                
                cv::Mat padded;
                grey->get()(bounding).copyTo(tmp);
                
                assert(tmp.cols <= normal_dimensions.width && tmp.rows <= normal_dimensions.height);

                float var = stdv.at<double>(0, 0);
                return {var, bdx, Image::Make(tmp) };
            }
            
            return {0.f, decltype(Tag::blob_id){}, nullptr};
        }
    }
}
