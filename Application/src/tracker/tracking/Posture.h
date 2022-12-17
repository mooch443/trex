#ifndef _POSTURE_H
#define _POSTURE_H

#include <types.h>

#include "Outline.h"
#include <misc/bid.h>

//#define POSTURE_DEBUG
namespace track {
    class Posture {
    private:
    public:
        struct EntryPoint {
            int y;
            int x0, x1;
            //int x;
            std::vector<Vec2> interp;
            
            EntryPoint() : y(-1),x0(-1),x1(-1) { }
            void clear() {
                y = -1; x0 = -1; x1 = -1;
                interp.clear();
            }
        };
        
    private:
        friend class DebugDrawing;
        
        std::shared_ptr<std::vector<Vec2>> _outline_points;
        
        Frame_t frameIndex;
        uint32_t fishID;
        GETTER(Outline, outline)
        GETTER_PTR(Midline::Ptr, normalized_midline)
        
    public:
        Posture(Frame_t frameIndex, uint32_t fishID);
        ~Posture() {
        }
        
        void calculate_posture(Frame_t frameIndex, pv::BlobWeakPtr blob);//const cv::Mat& greyscale, Vec2 previous_direction);
        
        bool outline_empty() const { return _outline.empty(); }
        static std::vector<EntryPoint> subpixel_threshold(const cv::Mat& greyscale, int threshold) 
#ifndef WIN32
            __attribute__((deprecated("use new method please")))
#endif
            ;
        
        float calculate_outline(std::vector<EntryPoint>&) 
#ifndef WIN32
            __attribute__((deprecated("use new method please")))
#endif
            ;
        float calculate_outline(const std::vector<Vec2>&);
    private:
        float calculate_midline(bool debug);
        
        
    };
}

#endif
