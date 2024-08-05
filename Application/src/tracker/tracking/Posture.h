#ifndef _POSTURE_H
#define _POSTURE_H

#include <commons.pc.h>

#include "Outline.h"
#include <misc/bid.h>
#include <misc/idx_t.h>

//#define POSTURE_DEBUG
namespace cmn::blob {
struct Pose;
}

namespace track {
struct PoseMidlineIndexes;
struct BasicStuff;

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
        Idx_t fishID;
        GETTER(Outline, outline);
        Midline::Ptr _normalized_midline;
        
    public:
        Posture(Frame_t frameIndex = {}, Idx_t fishID = {});
        ~Posture() {
        }
        
        Midline::Ptr&& steal_normalized_midline() && { return std::move(_normalized_midline); }
        const Midline::Ptr& normalized_midline() const { return _normalized_midline; }
        
        void calculate_posture(Frame_t, pv::BlobWeakPtr);
        void calculate_posture(Frame_t, const BasicStuff&, const blob::Pose &, const PoseMidlineIndexes &);
        void calculate_posture(Frame_t, const BasicStuff &, const blob::SegmentedOutlines&);
        
        bool outline_empty() const { return _outline.empty(); }
    private:
        float calculate_midline(bool debug);
        
        
    };


std::vector<Vec2> generateOutline(const blob::Pose& pose,
                                  const PoseMidlineIndexes& midline,
                                  const std::function<float(float)>& radius = nullptr);

}

#endif
