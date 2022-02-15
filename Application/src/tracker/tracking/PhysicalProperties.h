#ifndef PHYSICAL_PROPERTIES_H
#define PHYSICAL_PROPERTIES_H

#include <misc/defines.h>
#include <misc/Blob.h>
#include <misc/GlobalSettings.h>
#include <misc/frame_t.h>

#define SMOOTH_RECURSIVELY false

namespace track {
class Individual;

class PairDistance {
    GETTER_SETTER_PTR(const Individual*, fish0)
    GETTER_SETTER_PTR(const Individual*, fish1)
    GETTER_SETTER(float, d)
        
public:
    PairDistance(const Individual* fish0, const Individual* fish1, float d)
        : _fish0(fish0), _fish1(fish1), _d(d)
    {}
        
    bool operator==(const PairDistance& other) const {
        return ((other.fish0() == _fish0 || other.fish1() == _fish0)
            && other.fish1() != other.fish0()
            && (other.fish1() == _fish1 || other.fish0() == _fish1));
    }
    bool operator<(const PairDistance& other) const {
        return _d < other.d();
    }
    bool operator>(const PairDistance& other) const {
        return _d > other.d();
    }
    bool operator<=(const PairDistance& other) const {
        return _d <= other.d();
    }
    bool operator>=(const PairDistance& other) const {
        return _d >= other.d();
    }
};

struct FrameProperties {
    double time;
    uint64_t org_timestamp;
    Frame_t frame;
    long_t active_individuals;
        
    FrameProperties(Frame_t frame, double t, uint64_t ot)
        : time(t), org_timestamp(ot), frame(frame), active_individuals(-1)
    {}
        
    FrameProperties()
        : time(-1), org_timestamp(0), frame(-1), active_individuals(-1)
    {}
        
    bool operator<(Frame_t frame) const {
        return this->frame < frame;
    }
};

struct CacheHints {
    Frame_t current;
    std::vector<const FrameProperties *> _last_second;
        
    CacheHints(size_t size = 0);
    void push(Frame_t index, const FrameProperties* ptr);
    //void push_front(Frame_t index, const FrameProperties* ptr);
    void clear(size_t size = 0);
    size_t size() const;
    bool full() const;
    void remove_after(Frame_t);
    const FrameProperties* properties(Frame_t) const;
};

enum class Units {
    //PX_AND_FRAMES,
    PX_AND_SECONDS,
    CM_AND_SECONDS,
        
    DEFAULT = PX_AND_SECONDS
};
    
enum class PropertyType { POSITION, ANGLE };
    
/**
    * This class describes the physical properties of a moving entity
    * for a specific point in time.
    * TODO: Make linked list where it knows its predecessor so that
    * more complicated properties can be calculated and saved
    */
class PhysicalProperties {
public:
    static constexpr const size_t max_derivatives = 3;

protected:
    friend class DataFormat;
    GETTER(double, time)

    std::array<Vec2, PhysicalProperties::max_derivatives> _pos;
    std::array<float, PhysicalProperties::max_derivatives> _angle;
        
public:
    PhysicalProperties() = default;
    PhysicalProperties(const PhysicalProperties* previous, Frame_t frame, double time, const Vec2& pos, float angle, const CacheHints* hints = nullptr);
        
    static size_t saved_midlines();
        
    template<Units to> float speed(bool smooth) const { return v<to>(smooth).length(); }
    template<Units to> float speed() const { return v<to>().length(); }
        
    template<Units to> float acceleration(bool smooth) const { return length(a<to>(smooth)); }
    template<Units to> float acceleration() const { return length(a<to>()); }
        
    float angle(bool smooth) const { return value<float>(0, smooth); };
    float angle() const { return get<float>(0); };
        
    template<Units to> float angular_velocity(bool smooth) const { return value<to, float>(1, smooth); };
    template<Units to> float angular_velocity() const { return value<to, float>(1); };
        
    template<Units to> float angular_acceleration(bool smooth) const { return value<to, float>(2, smooth); };
    template<Units to> float angular_acceleration() const { return value<to, float>(2); };
        
    template<Units to> Vec2 pos(bool smooth) const { return value<to, Vec2>(0, smooth); }
    template<Units to> Vec2 pos() const { return value<to, Vec2>(0); }
        
    template<Units to> Vec2 v(bool smooth) const { return value<to, Vec2>(1, smooth); }
    template<Units to> Vec2 v() const { return value<to, Vec2>(1); }
        
    template<Units to> Vec2 a(bool smooth) const { return value<to, Vec2>(2, smooth); }
    template<Units to> Vec2 a() const { return value<to, Vec2>(2); }
        
    void flip(const PhysicalProperties* previous, const CacheHints* hints);
    //static Frame_t smooth_window();
    static float cm_per_pixel();
        
private:
    //void update_derivatives();

    template<typename T>
    const T& get(size_t derivative = 0) const {
        if constexpr (std::is_same_v<T, Vec2>)
            return _pos[derivative];
        else if constexpr (std::is_same_v<T, float>)
            return _angle[derivative];
        else
            static_assert(std::same_as<T, float>);
    }

    template<typename T>
    constexpr T& get(size_t derivative = 0) {
        if constexpr (std::same_as<T, Vec2>)
            return _pos[derivative];
        else
            return _angle[derivative];
    }

    template<typename T>
    T value(size_t derivative, bool smooth) const
    {
        if (smooth)
            return value<Units::DEFAULT, Units::DEFAULT, true, T>(derivative);
        else
            return value<Units::DEFAULT, Units::DEFAULT, false, T>(derivative);
    }

    template<Units from, Units to, typename T>
    T value(size_t derivative, bool smooth) const
    {
        if (smooth)
            return value<from, to, true, T>(derivative);
        else
            return value<from, to, false, T>(derivative);
    }

    template<Units to, typename T>
    T value(size_t derivative, bool smooth) const
    {
        return value<Units::DEFAULT, to, T>(derivative, smooth);
    }

    template<Units from, Units to, bool smooth, typename T>
    T value(size_t derivative = 0) const
    {
        //if constexpr(smooth)
        //    return convert<from, to>(smooth_value(derivative));
        //else
        return convert<from, to>(get<T>(derivative));
    }

    template<Units from, Units to, typename T>
    T value(size_t derivative = 0) const
    {
        return convert<from, to, false>(get<T>(derivative));
    }

    template<Units to, typename T>
    T value(size_t derivative = 0) const
    {
        return value<Units::DEFAULT, to, false, T>(derivative);
    }

    template<Units from, typename T>
    void value(const PhysicalProperties* previous, const T& val, size_t derivative = 0, const CacheHints* hints = nullptr)
    {
        // save
        set<T>(derivative, convert<from, Units::DEFAULT>(val));

        // calculate the next higher derivative
        for (size_t i = derivative + 1; i < PhysicalProperties::max_derivatives; i++) {
            calculate_derivative<T>(previous, i, hints);
        }
    }

    template<track::Units from, track::Units to, typename T>
    static T convert(const T& val)
    {
        if constexpr (from == Units::PX_AND_SECONDS) {
            if constexpr (to == Units::CM_AND_SECONDS) {
                return val * PhysicalProperties::cm_per_pixel();
            }

        }
        else {
            if constexpr (to == Units::PX_AND_SECONDS) {
                return val / PhysicalProperties::cm_per_pixel();
            }
        }

        return val;
    }

    size_t memory_size() const {
        return (sizeof(Vec2) + sizeof(float)) * max_derivatives;
    }

private:
    template<typename T>
    void calculate_derivative(const PhysicalProperties* prev, size_t index, const CacheHints* hints);

public:
    //T smooth_value(size_t derivative) const {
    //    return update_smooth(derivative);
    //}

protected:
    friend class PropertyBase;
    friend class PhysicalProperties;

    template<typename T>
    void set(size_t derivative, const T& value) {
        assert(derivative < max_derivatives);
        assert(!cmn::isnan(value));
        get<T>(derivative) = value;
    }
   // T update_smooth(size_t derivative) const;

    //static T _update_smooth(const PhysicalProperties::Property<T>* prop, size_t derivative);
};

template void PhysicalProperties::calculate_derivative<Vec2>(const PhysicalProperties* prev, size_t index, const CacheHints* hints);
template void PhysicalProperties::calculate_derivative<float>(const PhysicalProperties* prev, size_t index, const CacheHints* hints);
//template<> Vec2 PhysicalProperties::Property<Vec2>::update_smooth(size_t derivative) const;
//template<> float PhysicalProperties::Property<float>::update_smooth(size_t derivative) const;
}

#endif
