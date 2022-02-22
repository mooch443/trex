#include "DebugDrawing.h"
#include <misc/GlobalSettings.h>

#include <gui/Graph.h>
#include <gui/DrawCVBase.h>
#include <tracking/Tracker.h>
#include <misc/curve_discussion.h>

using namespace track;
using namespace gui;

void DebugDrawing::paint_raster() {
    if (raster_image.empty()) {
        raster_image = cv::Mat::zeros(height * scale, width * scale, CV_8UC3);
        
        for (int y=0; y<height; y++) {
            cv::line(raster_image, cv::Point(0, y*scale), cv::Point(raster_image.cols, y * scale), cv::Scalar(100, 100, 100));
        }
        
        for (int x=0; x<width; x++) {
            cv::line(raster_image, cv::Point(x*scale, 0), cv::Point(x*scale, raster_image.rows), cv::Scalar(100, 100, 100));
        }
    }
}

void DebugDrawing::reset_image() {
    if (image.empty()) {
        image = cv::Mat::zeros(height * scale, width * scale, CV_8UC3);
    } else {
        image = cv::Scalar(0);
    }
    
    paint_raster();
    image += raster_image;
}

void DebugDrawing::paint(const Outline &outline, bool erase) {
    //Tracker::LockGuard guard(*Tracker::instance());
    
    print("First: ", outline.points().front().x,",",outline.points().front().y);
    
    if(erase)
        reset_image();
    
    auto prev = outline.back();
    cv::Scalar clr(255, 225, 0);
    
    float max_slope = 0.0;
    int idx = -1;
    assert(outline.size() < INT_MAX);
    std::vector<float> curvature;
    curvature.resize(outline.size());
    
    auto curvature_range = Outline::calculate_curvature_range(outline.size());
    for (size_t i=0; i<outline.size(); i++) {
        float slope = Outline::calculate_curvature(curvature_range, outline.points(), i);
        curvature[i] = slope;
        
        if (cmn::abs(slope) > max_slope) {
            max_slope = cmn::abs(slope);
            idx = int(i);
        }
    }
    
    {
        float max_curvature;
        std::vector<float> corrected;
        outline.smooth_array(curvature, corrected, &max_curvature);
        /*std::vector<float> curvature;
        
        curvature.resize(corrected.size());
        for(size_t i=0; i<corrected.size(); i++) {
            curvature[i] = Outline::calculate_curvature(outline.points(), i, 2) * 0.5 + corrected[i] * 0.5;
        }*/
        
        std::vector<float> io;
        io.resize(corrected.size());
        
        cv::Mat output(1, (int)corrected.size(), CV_32FC1);
        cv::dft(cv::Mat(1, (int)corrected.size(), CV_32FC1, corrected.data()), output);
        
        const size_t start = min(corrected.size()-1, max(5u, size_t(corrected.size()*0.03)));
        auto ptr = output.ptr<float>(0, 0);
        std::fill(ptr + start, ptr + corrected.size(), 0.f);
        
        cv::Mat tmp(1, (int)io.size(), CV_32FC1, io.data());
        cv::dft(output, tmp, cv::DFT_INVERSE + cv::DFT_SCALE);
        
        if(!corrected.empty()) {
            float max_curvature = 0;
            
            auto points = outline.points();
            
            /*long L = corrected.size();
             printf("\nCurvature: ");
            for (long i=0; i<L; i++) {
                printf("%f, ", corrected[i]);
                max_curvature = max(abs(corrected[i]), max_curvature);
            }
            printf("\n");*/
            
            print("Maxmimum curvature: ", max_curvature);
            
            auto derivative = curves::derive(corrected);
            auto derivative2 = curves::derive(io);
            auto area = curves::area_under_maxima(io);
            
            struct Area {
                float idx;
                float area;
            };
            std::vector<Area> areas;
            float max_area = 0;
            for(auto &a : area)
                max_area = max(max_area, cmn::abs(io[a.first]));
            for(auto &a : area)
                //if(a.second / max_area > 0.25)
                areas.push_back(Area{a.first, curves::interpolate(corrected, a.first)}); //corrected2[a.first]});
            
            std::sort(areas.begin(), areas.end(), [](Area a, Area b) -> bool {
                return (a.area) > (b.area);
            });
            
            auto e = curves::find_extreme_points(io, derivative2);
            auto e1 = curves::find_extreme_points(corrected, derivative);
            
            auto &minima = e.minima;
            auto &maxima = e.maxima;
            
            for(auto &a : areas) {
                print("Area[", a.idx,"]: ",a.area);
            }
            
            if(minima.empty())
                print("minima empty.");
            
            Graph graph(Bounds(Size2(800, 600)), "outline", Rangef(0, outline.size()+5), Rangef(-max_slope*1.5, max_slope*1.5));
            graph.set_zero(0);
            
            /*graph.add_function(Graph::Function("derivative", Graph::Type::DISCRETE, [&](float x) {
                while(x < 0)
                    x += corrected.size();
                while(x >= corrected.size())
                    x -= corrected.size();
                
                return derivative.at(x) * 10;
            }));*/
            
            std::vector<Vec2> converted_points;
            for(auto m : e1.minima)
                converted_points.push_back(Vec2(m, corrected[m]));
            graph.add_points("minima2", converted_points);
            converted_points.clear();
            for(auto m : e1.maxima)
                converted_points.push_back(Vec2(m, corrected[m]));
            graph.add_points("maxima2", converted_points);
            converted_points.clear();
            for(auto m : minima)
                converted_points.push_back(Vec2(m, io[m]));
            graph.add_points("minima", converted_points);
            converted_points.clear();
            for(auto m : maxima)
                converted_points.push_back(Vec2(m, io[m]));
            graph.add_points("maxima", converted_points);
            /*graph.add_function(Graph::Function("curvature", Graph::Type::DISCRETE, [&outline, &max_slope, &corrected, &io](float x) -> float
            {
                while(x < 0)
                    x += outline.size();
                while(x >= outline.size())
                    x -= outline.size();
                
                return o[x];//corrected[x];//outline.curvature(x);
            }));*/
            graph.add_function(Graph::Function("curvature", Graph::Type::DISCRETE, [&outline, &corrected](float x) -> float
           {
               while(x < 0)
                   x += outline.size();
               while(x >= outline.size())
                   x -= outline.size();
               
               return corrected[narrow_cast<uint>(x)];
           }));
            graph.add_function(Graph::Function("idft", Graph::Type::DISCRETE, [&outline, &io](float x) -> float
           {
               while(x < 0)
                   x += outline.size();
               while(x >= outline.size())
                   x -= outline.size();
               
               return io[narrow_cast<uint>(x)];//- (corrected2[x] - corrected[x]);
           }));
            graph.add_function(Graph::Function("area", Graph::Type::POINTS, [&](float x) -> float
           {
               while(x < 0)
                   x += corrected.size();
               while(x >= corrected.size())
                   x -= corrected.size();
               
               if(area.find(x) != area.end())
                   return area[narrow_cast<uint>(x)] / 50;
               return gui::Graph::invalid();
               //return output_area[x] / 100;
           }));
            
            //graph.save_file("dicks.csv");
            
            cv::Mat window = cv::Mat::zeros(graph.size().height, graph.size().width, CV_8UC4);
            gui::CVBase base(window);
            gui::DrawStructure s(window.cols, window.rows);
            s.wrap_object(graph);
            
            base.paint(s);
            base.display();
            
        } else {
            FormatWarning("Smoothed curvature is empty.");
        }
    }
    
    Vec2 zero(width * scale * 0.5, height * scale * 0.5);
    cv::Mat graph = cv::Mat::zeros(400, (int)outline.size()*5, CV_8UC3);
    prev = outline.back();
    /*if(!inv_mat.empty()) {
        prev = (inv_mat * prev);
    }*/
    prev += _center;
    prev *= scale;
    
    for (size_t i=0; i<outline.size(); i++) {
        float percent = cmn::abs(curvature[i]) / max_slope;
        
        clr = cv::Scalar(cv::Scalar(255, 125, 0) * (1.0 - percent) + cv::Scalar(0, 0, 255) * percent);
        
        auto pt = outline.at(i);
        /*pt += _center;
        pt *= scale;*/
        //pt += _pos;
        pt = pt + Vec2(image.cols, image.rows) * 0.5 / scale;
        
        pt *= scale;
        //pt += zero;
        
        if (i == 0)
            cv::circle(image, pt, 4, cv::Scalar(255, 255, 0));
        else if(i == outline.size()-1)
            cv::circle(image, pt, 4, cv::Scalar(255, 0, 255));
        else
            cv::circle(image, pt, 2, clr);
        
        cv::line(image, prev, pt, clr);
        prev = pt;
    }
    
    //tf::imshow("graph_posture", graph);
    //cv::imshow(window_name, image);
    tf::imshow(window_name, image);
}

void DebugDrawing::paint(const cv::Mat& greyscale, const std::vector<track::Posture::EntryPoint>& pts) {
    reset_image();
    
    const int threshold = SETTING(track_posture_threshold);
    
    auto g = greyscale;
    cv::Mat resized;
    resize_image(g, resized, scale);
    cv::cvtColor(resized, resized, cv::COLOR_GRAY2BGR);
    
    for (int j=0; j<g.rows; j++) {
        for (int i=0; i<g.cols; i++) {
            Vec2 pt(i, j);
            pt += Vec2(0.0, 0.5);
            
            auto val = g.at<uchar>(j, i);
            if (val >= threshold) {
                //cv::circle(resized, pt * scale, 1, cv::Scalar(100, 250, 0));
                std::stringstream ss;
                ss << int(int(val) - threshold);
                
                cv::putText(resized, ss.str(), pt * scale, cv::FONT_HERSHEY_PLAIN, 0.5, cv::Scalar(100, 250, 0));
            }
        }
    }
    
    for (auto &e : pts) {
        for (uint32_t i=0; i<e.interp.size(); i++) {
            auto pt = e.interp.at(i);
            pt = (pt + Vec2(0.5, 0.5)) * scale;
            if (i) {
                cv::line(resized, (e.interp.at(i-1) + Vec2(0.5, 0.5)) * scale, pt, cv::Scalar(255, 0, 0, 255));
            }
            
            cv::circle(resized, pt, 5, cv::Scalar(255, 0, 255, 255));
        }
    }
    
    image += resized;
    
    tf::imshow(window_name, image);
}

void DebugDrawing::paint(const track::Posture &posture, const cv::Mat& greyscale) {
    reset_image();
    
    const int threshold = SETTING(track_posture_threshold);
    
    auto g = greyscale;
    cv::Mat resized;
    resize_image(g, resized, scale);
    cv::cvtColor(resized, resized, cv::COLOR_GRAY2BGR);
    
    for (int j=0; j<g.rows; j++) {
        for (int i=0; i<g.cols; i++) {
            Vec2 pt(i, j);
            pt += Vec2(0.0, 0.5);
            
            auto val = g.at<uchar>(j, i);
            if (val >= threshold) {
                //cv::circle(resized, pt * scale, 1, cv::Scalar(100, 250, 0));
                std::stringstream ss;
                ss << int(val - threshold);
                
                //cv::putText(resized, ss.str(), pt * scale, CV_FONT_HERSHEY_PLAIN, 0.5, cv::Scalar(100, 250, 0));
            }
        }
    }
    
    image += resized;
    
    std::string label;
    if(!posture.outline_empty()) {
        paint(posture.outline(), false);
        label = std::to_string(posture.outline().size());
    }
    
    auto nmidline = posture.normalized_midline();
    if(nmidline) {
        paint(nmidline.get());
    }
    
    tf::imshow(window_name, image, label);
}

void DebugDrawing::paint(const Midline *midline) {
    print("Midline curvature:");
    auto &segments = midline->segments();
    long L = segments.size();
    long offset = 1;
    
    for (long index=0; index<L; index++) {
        size_t idx01= index < L-offset ? index+offset : index+offset-L;
        size_t idx0 = index >= offset  ? index-offset : index-offset+L;
        size_t idx1 = index;
        
        const auto &p3 = segments.at(idx01).pos;
        const auto &p2 = segments.at(idx0).pos;
        const auto &p1 = segments.at(idx1).pos;
        
        float K;
        if(p1 == p2 || p1 == p3 || p2 == p3)
            K = 0.0;
        else
            // TODO: this calculation is slow
            K = 2 * ((p2.x-p1.x)*(p3.y-p2.y) - (p2.y-p1.y)*(p3.x-p2.x))
            / cmn::sqrt(sqdistance(p1, p2) * sqdistance(p2, p3) * sqdistance(p1, p3));
        
        printf("%ld: %f, ", index, K);
    }
    printf("\n");
    
    for (uint32_t i=0; i<midline->size(); i++) {
        auto pt = midline->segments().at(i).pos;
        //auto pt = segment.pos;
        //float angle = midline->angle() + M_PI;
        //float x = (pt.x * cmn::cos(angle) - pt.y * cmn::sin(angle));
        //float y = (pt.x * cmn::sin(angle) + pt.y * cmn::cos(angle));
        
        //Vec2 pt = segment.pos;
        //auto pt = segment.pos;
        
        //pt = (inv_mat * pt);
        
        float angle = midline->angle() + M_PI;
        float x = (pt.x * cmn::cos(angle) - pt.y * cmn::sin(angle));
        float y = (pt.x * cmn::sin(angle) + pt.y * cmn::cos(angle));
        
        pt = Vec2(x, y) + midline->offset();
        pt = pt + Vec2(image.cols, image.rows) * 0.5 / scale;
        pt *= scale;
        
        //pt = do_rotate(pt, angle);
        //pt = scale * (Vec2(x, y) + midline->offset()) + _center;
        //pt +=  _pos /*+ offset*/ + midline->offset();
        //pt *= scale;
        
        /* auto pt_r = (outline._midline_pairs.at(i).first);
         auto pt_l = (outline._midline_pairs.at(i).second);
         
         if(!inv_mat.empty()) {
         pt_l = (inv_mat * pt_l);
         pt_r = inv_mat * pt_r;
         pt = inv_mat *pt;
         }
         pt_l = pt_l * scale + zero;
         pt_r = pt_r * scale + zero;*/
//pt = pt * scale + zero;
        
        cv::circle(image, pt, 5, cv::Scalar(255, 255, 255));
        //cv::line(image, pt_r, pt_l, cv::Scalar(200, 200, 200));
    }
}
