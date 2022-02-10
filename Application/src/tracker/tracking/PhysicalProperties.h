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
        
        //! Basic properties
        template<typename T>
        class Property {
            PhysicalProperties* _mother;
            
        public:
            Property() = default;
            constexpr Property(PhysicalProperties* mother) : _mother(mother) {}
            
            constexpr const T& value(size_t derivative = 0) const {
                return _values[derivative];
            }
            
            T value(size_t derivative, bool smooth) const
            {
                if(smooth)
                    return value<Units::DEFAULT, Units::DEFAULT, true>(derivative);
                else
                    return value<Units::DEFAULT, Units::DEFAULT, false>(derivative);
            }
            
            template<Units from, Units to>
            T value(size_t derivative, bool smooth) const
            {
                if(smooth)
                    return value<from, to, true>(derivative);
                else
                    return value<from, to, false>(derivative);
            }
            
            template<Units to>
            T value(size_t derivative, bool smooth) const
            {
                return value<Units::DEFAULT, to>(derivative, smooth);
            }
            
            template<Units from, Units to, bool smooth>
            T value(size_t derivative = 0) const
            {
                //if constexpr(smooth)
                //    return convert<from, to>(smooth_value(derivative));
                //else
                    return convert<from, to>(value(derivative));
            }
            
            template<Units from, Units to>
            T value(size_t derivative = 0) const
            {
                return convert<from, to, false>(value(derivative));
            }
            
            template<Units to>
            T value(size_t derivative = 0) const
            {
                return value<Units::DEFAULT, to, false>(derivative);
            }
            
            template<Units from>
            void value(const PhysicalProperties* previous, const T& val, size_t derivative = 0, const CacheHints* hints = nullptr)
            {
                // save
                set_value(derivative, convert<from, Units::DEFAULT>(val));
                
                // calculate the next higher derivative
                for(size_t i=derivative+1; i<PhysicalProperties::max_derivatives; i++) {
                    calculate_derivative(previous, i, hints);
                }
            }
            
            template<track::Units from, track::Units to>
            static T convert(const T &val)
            {
                if constexpr(from == Units::PX_AND_SECONDS) {
                    if constexpr(to == Units::CM_AND_SECONDS) {
                        return val * PhysicalProperties::cm_per_pixel();
                    }
                    
                } else {
                    if constexpr(to == Units::PX_AND_SECONDS) {
                        return val / PhysicalProperties::cm_per_pixel();
                    }
                }
                
                return val;
            }
            
            size_t memory_size() const {
                return sizeof(Property<T>);
            }
            
        private:
            void calculate_derivative(const PhysicalProperties* prev, size_t index, const CacheHints* hints);
        private:
            std::array<T, PhysicalProperties::max_derivatives> _values;
            
        public:
            T smooth_value(size_t derivative) const {
                return update_smooth(derivative);
            }
            
        protected:
            friend class PropertyBase;
            friend class PhysicalProperties;
            
            void set_value(size_t derivative, const T& value);
            T update_smooth(size_t derivative) const;
            
            static T _update_smooth(const PhysicalProperties::Property<T> *prop, size_t derivative);
        };
        
    private:
        friend class DataFormat;
        
        GETTER_PTR(Individual*, fish)
        GETTER(Frame_t, frame)
        //GETTER(double, time)
        
        // contains either float or Point2f pointers
        //std::array<PropertyBase*, 2> _derivatives;
        Property<Vec2> _pos;
        Property<float> _angle;
        //std::map<Type, PropertyBase*> _derivatives;
        
    public:
        PhysicalProperties() = default;
        PhysicalProperties(const PhysicalProperties* previous, Individual* fish, Frame_t frame, const Vec2& pos, float angle, const CacheHints* hints = nullptr);
        
        //const decltype(_derivatives)& derivatives() const { return _derivatives; }
        static size_t saved_midlines();
        
        //! Gets a properties value with given derivative depth,
        //  type is either float or Vec2
        /*inline const PropertyBase& get(const Type& name) const {
            return *_derivatives[(size_t)name];
        }
        inline PropertyBase& get(const Type& name) {
            return *_derivatives[(size_t)name];
        }*/
        
        //template<> inline Property<Vec2>& get() { return _pos; }
        //template<> inline Property<float>& get() { return _angle; }
        
        //void set_next(PhysicalProperties* ptr);
        
        template<Units to> float speed(bool smooth) const { return v<to>(smooth).length(); }
        template<Units to> float speed() const { return v<to>().length(); }
        
        template<Units to> float acceleration(bool smooth) const { return length(a<to>(smooth)); }
        template<Units to> float acceleration() const { return length(a<to>()); }
        
        float angle(bool smooth) const { return _angle.value(0, smooth); };
        float angle() const { return _angle.value(0); };
        
        template<Units to> float angular_velocity(bool smooth) const { return _angle.value<to>(1, smooth); };
        template<Units to> float angular_velocity() const { return _angle.value<to>(1); };
        
        template<Units to> float angular_acceleration(bool smooth) const { return _angle.value<to>(2, smooth); };
        template<Units to> float angular_acceleration() const { return _angle.value<to>(2); };
        
        template<Units to> Vec2 pos(bool smooth) const { return _pos.value<to>(0, smooth); }
        template<Units to> Vec2 pos() const { return _pos.value<to>(0); }
        
        template<Units to> Vec2 v(bool smooth) const { return _pos.value<to>(1, smooth); }
        template<Units to> Vec2 v() const { return _pos.value<to>(1); }
        
        template<Units to> Vec2 a(bool smooth) const { return _pos.value<to>(2, smooth); }
        template<Units to> Vec2 a() const { return _pos.value<to>(2); }
        
        void flip();
        size_t memory_size() const;
        static Frame_t smooth_window();
        static float cm_per_pixel();
        
    private:
        void update_derivatives();
    };
    


template<> Vec2 PhysicalProperties::Property<Vec2>::update_smooth(size_t derivative) const;
template<> float PhysicalProperties::Property<float>::update_smooth(size_t derivative) const;
}

#endif
