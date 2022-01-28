#pragma once

#include <types.h>
#include <gui/types/Basic.h>
#include <gui/colors.h>
#include <misc/ranges.h>

namespace track {
    //! Frames of interest
    class FOI {
    public:
        struct Properties {
            long_t id;
            gui::Color color;
        };
        
        struct fdx_t {
            uint32_t id;
            explicit fdx_t(uint32_t i) : id(i) {}
            bool operator <(const fdx_t& other) const {
                return id < other.id;
            }
            bool operator !=(const fdx_t& other) const {
                return other.id != id;
            }
            bool operator ==(const fdx_t& other) const {
                return other.id == id;
            }
            std::string toStr() const;
            static std::string class_name() {
                return "fdx_t";
            }
        };
        
        struct bdx_t {
            uint32_t id;
            explicit bdx_t(uint32_t i) : id(i) {}
            bool operator <(bdx_t other) const {
                return id < other.id;
            }
            std::string toStr() const;
            static std::string class_name() {
                return "bdx_t";
            }
            bool operator==(const bdx_t& other) const {
                return id == other.id;
            }
        };
        
        typedef std::map<long_t, std::set<FOI, std::less<FOI>>> foi_type;
        typedef std::chrono::high_resolution_clock change_clock_t;
        //typedef std::chrono::duration<double, std::ratio<1> > second_;
        typedef std::chrono::time_point<change_clock_t> time_point_t;
        
    protected:
        static std::map<std::string, Properties> _string_to_id;
        static std::map<long_t, std::string> _id_to_string;
        static std::set<long_t> _ids;
        static foi_type _frames_of_interest;
        static std::recursive_mutex _mutex;
        static ColorWheel _wheel;
        
        const Properties* _props;
        
    protected:
        GETTER(Rangel, frames)
        GETTER(std::set<fdx_t>, fdx)
        GETTER(std::set<bdx_t>, bdx)
        GETTER(std::string, description)
        
    public:
        FOI() : _props(NULL), _frames(-1, -1) {}
        bool valid() const { return _frames.start != _frames.end || _frames.start != -1; }

        FOI(const Rangel& frames, std::set<fdx_t> fdx, const std::string& reason, const std::string& _description = "");
        FOI(const Rangel& frames, std::set<bdx_t> bdx, const std::string& reason, const std::string& _description = "");
        
        FOI(long_t frame, std::set<fdx_t> fdx, const std::string& reason, const std::string& _description = "");
        FOI(long_t frame, std::set<bdx_t> bdx, const std::string& reason, const std::string& _description = "");
        
        FOI(long_t frame, const std::string& reason, const std::string& _description = "");
        FOI(const Rangel& frames, const std::string& reason, const std::string& _description = "");
        
    public:
        bool operator<(const FOI& other) const;
        bool operator==(const FOI& other) const;
        
        static const gui::Color& color(const std::string& name);
        long_t id() const;
        
        static const std::string& name(long_t id);
        static long_t to_id(const std::string& name);
        static void add(FOI&&);
        static foi_type::mapped_type foi(long_t id);
        static std::set<long_t> ids();
        static void remove_frames(long_t frameIndex);
        static void remove_frames(long_t frameIndex, long_t id);
        static void clear();
        static uint64_t last_change();
        static foi_type all_fois();
        std::string toStr() const;
        static std::string class_name() { return "FOI"; }

    private:
        static void changed();
    };
}
