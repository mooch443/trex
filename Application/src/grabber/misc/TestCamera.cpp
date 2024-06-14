#include "TestCamera.h"
#include <misc/GlobalSettings.h>
#include <misc/SpriteMap.h>

namespace fg {
    using namespace cmn;

    TestCamera::TestCamera(cv::Size size, size_t element_size) : _size(size) {
        _image = cv::Mat::zeros(_size.height, _size.width, CV_8UC1);
        
        if(GlobalSettings::map().has("test_image")) {
            std::string test_image = SETTING(test_image);
            if (test_image == "checkerboard") {
                static cv::Mat checkerboard;
                checkerboard = cv::Mat::ones(_image.rows, _image.cols, CV_8UC1) * 255;
                const int size = sign_cast<int>(element_size), padding = 1;
                for (int j=0; j<_image.rows; j++) {
                    if(j%(size+padding) >= size)
                        continue;
                    
                    for (int i=0; i<_image.cols-size; i+=size+padding) {
                        auto ptr = checkerboard.ptr(j, i);
                        memset(ptr, 0, size);
                    }
                }
                
                _image = checkerboard;
                
            } else if(test_image == "fullscreen") {
                _image = 255;
                
            } else if(test_image == "vertical") {
                static cv::Mat checkerboard;
                checkerboard = cv::Mat::ones(_image.rows, _image.cols, CV_8UC1) * 255;
                const int padding = 2;
                for (int j=0; j<_image.rows; j++) {
                    for (int i=0; i<_image.cols; i+=padding) {
                        checkerboard.at<uchar>(j, i) = 0;
                    }
                }
                
                _image = checkerboard;
            
            } else if(test_image == "diagonal") {
                static cv::Mat checkerboard;
                checkerboard = cv::Mat::ones(_image.rows, _image.cols, CV_8UC1) * 255;
                const int size = sign_cast<int>(element_size), padding = 1;
                int running = 0;
                for (int j=0; j<_image.rows; j++) {
                    if(j%(size+padding) >= size)
                        continue;
                    
                    for (int i=0; i<_image.cols-size; i+=size+padding+running*0.1) {
                        auto ptr = checkerboard.ptr(j, i);
                        memset(ptr, 0, size);
                    }
                    
                    running++;
                }
                
                _image = checkerboard;
                
            } else if(test_image == "all") {
                cv::Mat image = cv::Mat::ones(size.height, size.width, CV_8UC1) * 255;
                // element_size determines size of playing field
                const size_t number_of_fields = max(size.width, size.height) / (element_size*0.5) + 1;
                print("Size: ", size,", Nr: ", number_of_fields,", width: ", element_size);
                
                for(size_t i=0; i<number_of_fields; i++) {
                    for(size_t j=0; j<number_of_fields; j++) {
                        const size_t padding = 1;
                        const size_t layers = 6;
                        float layer_size = (element_size) / (layers) - padding*3;
                        
                        std::vector<Vec2> points = {
                            Vec2(padding, padding),
                            Vec2(layer_size+padding*2, padding),
                            Vec2(layer_size*2+padding*3, padding),
                            Vec2(layer_size*2+padding*3, layer_size+padding*2),
                            Vec2(layer_size*2+padding*3, layer_size*2+padding*3),
                            Vec2(layer_size+padding*2, layer_size+padding*2),
                            Vec2(padding, layer_size+padding*2),
                            Vec2(padding, layer_size*2+padding*3),
                        };
                        
                        Vec2 local(i * (element_size*0.5), j * (element_size*0.5));
                        cv::rectangle(image,
                                      local+points[0],
                                      local+points[0]+Vec2(layer_size,layer_size),
                                      cv::Scalar(0), -1);
                        
                        // draw circle
                        cv::circle(image,
                                   local + points[1] + Vec2(layer_size*0.5,layer_size*0.5),
                                   layer_size*0.4,
                                   cv::Scalar(0),
                                   -1);
                        
                        for(int k=0; k<5; k++) {
                            cv::line(image,
                                     local + points[2] + Vec2(k*layer_size/4+1, layer_size-1),
                                     local + points[2] + Vec2(k*layer_size/4+1, 1), cv::Scalar(0));
                        }
                        
                        for(int k=0; k<5; k++) {
                            auto angle = RADIANS(-90);
                            Vec2 pt0(-0.5, -0.5);
                            Vec2 pt1(-0.5, 0.5);
                            
                            pt0.x = cos(angle) * pt0.x - sin(angle) * pt0.y;
                            pt0.y = sin(angle) * pt0.x + cos(angle) * pt0.y;
                            
                            pt1.x = cos(angle) * pt1.x - sin(angle) * pt1.y;
                            pt1.y = sin(angle) * pt1.x + cos(angle) * pt1.y;
                            
                            pt0 += Vec2(0.5, 0.5);
                            pt1 += Vec2(0.5, 0.5);
                            
                            cv::line(image,
                                     local + points[3] + pt0 * (layer_size/k-2) + Vec2(0, 1),
                                     local + points[3] + pt1 * (layer_size/k-2) + Vec2(0, 1), cv::Scalar(0));
                        }
                        
                        float w = max(1, layer_size/10.f*0.8), p = max(1,layer_size/10.f*0.1);
                        for(int k=0; k<=10; k+=2) {
                            for(int h=0; h<=10; h+=2) {
                                auto s = Vec2(k*(w+p),h*(w+p));
                                if(s.x > layer_size || s.y > layer_size)
                                    continue;
                                    
                                cv::rectangle(image,
                                              local + points[4] + s,
                                              local + points[4] + s + Vec2(w,w),
                                              cv::Scalar(0), -1);
                            }
                        }
                        
                        cv::circle(image,
                                   local + points[5] + Vec2(layer_size*0.5,layer_size*0.5),
                                   layer_size*0.4,
                                   cv::Scalar(0), -1);
                        cv::circle(image,
                                   local + points[5] + Vec2(layer_size*0.5,layer_size*0.5),
                                   layer_size*0.2,
                                   cv::Scalar(255), -1);
                        
                        {
                            cv::RotatedRect rRect((cv::Point2f)(local + points[6]) + cv::Point2f(layer_size*0.5, layer_size*0.5), cv::Point2f(layer_size*0.9, layer_size*0.15), 45);
                            cv::Point2f vertices2f[4];
                            cv::Point vertices[4];
                            rRect.points(vertices2f);
                            for(int i = 0; i < 4; ++i){
                                vertices[i] = vertices2f[i];
                            }
                            cv::fillConvexPoly(image,
                                               vertices,
                                               4,
                                               cv::Scalar(0));
                        }
                        
                        {
                            cv::RotatedRect rRect((cv::Point2f)(local + points[6] + Vec2(layer_size*0.5, layer_size*0.5)), cv::Point2f(layer_size*0.9, layer_size*0.15), -45);
                            cv::Point2f vertices2f[4];
                            cv::Point vertices[4];
                            rRect.points(vertices2f);
                            for(int i = 0; i < 4; ++i){
                                vertices[i] = vertices2f[i];
                            }
                            cv::fillConvexPoly(image,
                                               vertices,
                                               4,
                                               cv::Scalar(0));
                        }
                        
                        cv::circle(image,
                                   local + points[7] + Vec2(layer_size*0.5,layer_size*0.5),
                                   layer_size*0.4,
                                   cv::Scalar(0), -1);
                        cv::circle(image,
                                   local + points[7] + Vec2(layer_size*0.3-1,layer_size*0.5),
                                   layer_size*0.3,
                                   cv::Scalar(255), -1);
                    }
                }
                
                //cv::imwrite("all.png", image);
                _image = image;
            }
            
            //tf::imshow("test image", _image);
        }
    }
    
    bool TestCamera::next(cmn::Image &image) {
        image.create(_image, image.index());
        return true;
    }
}
