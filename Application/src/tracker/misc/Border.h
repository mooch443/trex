#pragma once

#include <commons.pc.h>
#include <misc/ranges.h>
#include <misc/Image.h>

namespace pv {
class File;
}

namespace track {
    ENUM_CLASS(recognition_border_t, none, heatmap, outline, shapes, grid, circle)
    ENUM_CLASS_HAS_DOCS(recognition_border_t)

    //! Contains cached properties of the border of a given setup
    //  (this might be a grid, a single rectangular box, or a circle)
    class Border {
    public:
        using Type = recognition_border_t::data::values;
        
    protected:
        GETTER(Type, type);
        GETTER(float, max_distance);
        GETTER(float, min_distance);
        std::vector<Rangef> x_range;
        std::vector<Rangef> y_range;
        GETTER(std::vector<std::vector<Vec2>>, vertices);
        bool poly_set;
        std::mutex mutex;
        GETTER(Image::Ptr, mask);
        std::vector<bool> x_valid, y_valid;
        std::map<std::tuple<uint16_t, uint16_t>, uint32_t> grid_cells;
        float _recognition_border_size_rescale;
        GETTER(std::vector<std::shared_ptr<std::vector<cmn::Vec2>>>, polygons);
        
    public:
        Border();
        void update(pv::File&);
        float distance(const Vec2& pt) const;
        bool in_recognition_bounds(const Vec2& pt) const;
        void clear();
        
    private:
        void update_outline(pv::File& video);
        void update_heatmap(pv::File& video);
        void update_polygons();
    };
}
