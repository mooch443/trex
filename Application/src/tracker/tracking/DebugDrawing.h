#ifndef _DEBUG_DRAWING_H
#define _DEBUG_DRAWING_H

#include <tracking/Posture.h>

namespace track { class DebugDrawing; }

class track::DebugDrawing {
    cv::Mat image;
    cv::Mat raster_image;
    const std::string window_name;
    const float scale;
    const int width, height;
    cv::Mat inv_mat;
    Vec2 _pos, _center;
    
public:
    DebugDrawing(const Vec2& pos, const Vec2& center, const std::string& name, float scale, int width, int height, const cv::Mat &rot_mat = cv::Mat()) : window_name(name), scale(scale), width(width), height(height), _pos(pos), _center(center)
    {
        if(!rot_mat.empty()) {
            rot_mat.copyTo(inv_mat);
            cv::invertAffineTransform(rot_mat, inv_mat);
        }
    }
    void paint(const Outline& outline, bool erase = true);
    void paint(const Midline* midline);
    //void paint(const track::Posture& posture, const cv::Mat& greyscale);
    //void paint(const cv::Mat& greyscale, const std::vector<track::Posture::EntryPoint>& pts);
    void paint_raster();
    
private:
    void reset_image();
};

#endif
