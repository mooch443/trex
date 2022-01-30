#pragma once

#include <misc/defines.h>
#include <gui/DrawStructure.h>
#include <tracking/Individual.h>

namespace gui {
    class DrawStructure;
}

namespace track {    
    class VisualField {
    public:
        static constexpr uint8_t layers = 2;
        static constexpr uint16_t field_resolution = 512;
        static constexpr double symmetric_fov = RADIANS(130);
        static constexpr int32_t custom_id = 1234;
        //static constexpr double eye_separation = RADIANS(60);
        
        struct eye {
        public:
            gui::Color clr;
            float angle;
            Vec2 pos;
            Vec2 rpos;
            
            std::array<uchar, field_resolution * layers> _fov;
            std::array<float, field_resolution * layers> _depth;
            std::array<Vec2, field_resolution * layers> _visible_points;
            std::array<long_t, field_resolution * layers> _visible_ids;
            std::array<float, field_resolution * layers> _visible_head_distance;
            
            eye() {
                std::fill(_fov.begin(), _fov.end(), 0);
                std::fill(_depth.begin(), _depth.end(), FLT_MAX);
                std::fill(_visible_points.begin(), _visible_points.end(), Vec2(0,0));
                std::fill(_visible_ids.begin(), _visible_ids.end(), -1);
                std::fill(_visible_head_distance.begin(), _visible_head_distance.end(), -1);
            }
        };
        
    protected:
        const float max_d;
        std::array<eye, 2> _eyes;
        GETTER(Vec2, fish_pos)
        GETTER(double, fish_angle)
        
        GETTER(Idx_t, fish_id)
        GETTER(Frame_t, frame)
        
    public:
        VisualField(Idx_t fish_id, Frame_t frame,const std::shared_ptr<Individual::BasicStuff>& basic, const std::shared_ptr<Individual::PostureStuff>& posture, bool blocking);
        
        const decltype(_eyes)& eyes() const { return _eyes; }
        void calculate(const std::shared_ptr<Individual::BasicStuff>& basic, const std::shared_ptr<Individual::PostureStuff>& posture, bool blocking = true);
        void show(gui::DrawStructure &graph);
        static void show_ts(gui::DrawStructure &graph, Frame_t frameNr, Individual* selected);
        void plot_projected_line(eye& e, std::tuple<float, float>& tuple, double d, const Vec2& point, Idx_t id, float hd);
        
        static std::tuple<std::array<eye, 2>, Vec2> generate_eyes(const Individual* fish, const std::shared_ptr<Individual::BasicStuff>& basic, const std::vector<Vec2>& outline, const Midline::Ptr& midline, float angle);
    };
}

