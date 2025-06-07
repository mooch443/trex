#pragma once

#include <commons.pc.h>
#include <gui/DrawStructure.h>
#include <tracking/Individual.h>

namespace cmn::gui {
    class DrawStructure;
}

namespace track {    
    class VisualField {
    public:
        using Scalar64 = double;
        using Vec64 = Vector2D<Scalar64, true>;
        
    public:
        static constexpr uint8_t layers = 2;
        static constexpr uint16_t field_resolution = 512;
        static constexpr Scalar64 symmetric_fov = RADIANS(130);
        static constexpr int32_t custom_id = 1234;
        static constexpr Scalar64 invalid_value = FLT_MAX;
        //static constexpr double eye_separation = RADIANS(60);
        
        struct eye {
        public:
            gui::Color clr;
            Scalar64 angle;
            Vec64 pos;
            Vec64 rpos;
            
            std::array<uchar, field_resolution * layers> _fov;
            std::array<Scalar64, field_resolution * layers> _depth;
            std::array<Vec2, field_resolution * layers> _visible_points;
            std::array<long_t, field_resolution * layers> _visible_ids;
            std::array<Scalar64, field_resolution * layers> _visible_head_distance;
            
            eye() {
                std::fill(_fov.begin(), _fov.end(), 0u);
                std::fill(_depth.begin(), _depth.end(), invalid_value);
                std::fill(_visible_points.begin(), _visible_points.end(), Vec2(0,0));
                std::fill(_visible_ids.begin(), _visible_ids.end(), -1);
                std::fill(_visible_head_distance.begin(), _visible_head_distance.end(), -1.f);
            }
        };
        
    protected:
        const Scalar64 max_d;
        std::array<eye, 2> _eyes;
        GETTER(Vec64, fish_pos);
        GETTER(Scalar64, fish_angle);
        
        GETTER(Idx_t, fish_id);
        GETTER(Frame_t, frame);
        
    public:
        VisualField(Idx_t fish_id, Frame_t frame,const BasicStuff& basic, const PostureStuff* posture, bool blocking);
        
        const decltype(_eyes)& eyes() const { return _eyes; }
        void calculate(const BasicStuff& basic, const PostureStuff* posture, bool blocking = true);
        //void show(gui::DrawStructure &graph);
        //static void show_ts(gui::DrawStructure &graph, Frame_t frameNr, Individual* selected);
        void plot_projected_line(eye& e, std::tuple<Scalar64, Scalar64>& tuple, Scalar64 d, const Vec64& point, Idx_t id, Scalar64 hd);
        
        static void remove_frames_after(Frame_t);
        
        static std::tuple<std::array<eye, 2>, Vec64> generate_eyes(Frame_t frame, Idx_t fdx, const BasicStuff& basic, const std::vector<Vec2>& outline, const Midline::Ptr& midline, Scalar64 angle);
        
        static std::vector<Vec64> tesselate_outline(const std::vector<Vec2>& outline, Scalar64 max_distance = Scalar64(5));
    };
}

