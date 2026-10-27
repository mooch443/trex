#pragma once

#include <commons.pc.h>
#include <misc/colors.h>
#include <misc/ranges.h>
#include <core/idx_t.h>
#include <misc/frame_t.h>

namespace track {
    using namespace cmn;

    //! Frames of interest
    class FOI {
    public:
        using ID_t = long_t;

        static constexpr std::string_view none = "none";
        static constexpr std::string_view correcting = "correcting";
        static constexpr std::string_view lost_1_fish = "lost >=1 fish";
        static constexpr std::string_view lost_2_fish = "lost >=2 fish";
        static constexpr std::string_view huge_time_jump = "huge time jump";
        static constexpr std::string_view manual_match = "manual match";
        static constexpr std::string_view long_tracklet = "long tracklet";
        static constexpr std::string_view weird_distance = "weird distance";
        static constexpr std::string_view weird_distance_mm = "weird distance + mm";
        static constexpr std::string_view weird_angle = "weird angle";
        static constexpr std::string_view tracklet_end = "tracklet end";
        static constexpr std::string_view global_tracklet = "global tracklet";
        static constexpr std::string_view failed_matches = "failed matches";
        static constexpr std::string_view apprx_matching = "apprx matching";
        static constexpr std::string_view split_up = "split_up";
        static constexpr std::string_view annotated = "annotated";

        struct Info {
            std::string_view name;
            std::string_view symbol;
            std::string_view tooltip;

            glz::json_t to_json() const {
                glz::json_t json;
                if(not glz::read_json(json, toStr()))
                    throw InvalidArgumentException("Failed to serialize Info to JSON.");
                return json;
            }
            static Info fromStr(StringLike auto&& str) {
                Info info;
                if(not glz::read_json(info, str))
                    throw InvalidArgumentException("Failed to parse Info from ", str);
                return info;
            }
            std::string toStr() const {
                return glz::write_json(*this).value();
            }
        };

        static consteval Info info(std::string_view key) {
            if(key == correcting)
                return {correcting, "⚠", "Frames to review after a time jump, a manual match, a lost individual, or unusually fast movement."};
            if(key == lost_1_fish)
                return {lost_1_fish, "⮟", "At least one individual tracked in the previous frame is missing in this frame."};
            if(key == lost_2_fish)
                return {lost_2_fish, "⮟⮟", "The number of tracked individuals dropped by at least two since the previous frame."};
            if(key == huge_time_jump)
                return {huge_time_jump, "⏱", "A large time gap occurred between consecutive frame timestamps."};
            if(key == manual_match)
                return {manual_match, "☞", "An individual was assigned to a detection using a manual match."};
            if(key == long_tracklet)
                return {long_tracklet, "⛓", "Last frame before losing an individual whose tracklet reached the minimum output length."};
            if(key == weird_distance)
                return {weird_distance, "↯", "An individual's speed reached the threshold for unusually fast movement."};
            if(key == weird_distance_mm)
                return {weird_distance_mm, "↯☞", "Unusually fast movement and a manual match occurred in the same frame."};
            if(key == weird_angle)
                return {weird_angle, "⮥", "An individual's orientation changed sharply between consecutive frames."};
            if(key == tracklet_end)
                return {tracklet_end, "⏹", "Last frame of an individual tracklet before it ends or a new tracklet begins."};
            if(key == global_tracklet)
                return {global_tracklet, "◎", "A continuous interval in which all expected individuals were tracked."};
            if(key == failed_matches)
                return {failed_matches, "✖", "Manual matches could not be applied directly and required a recovery attempt."};
            if(key == apprx_matching)
                return {apprx_matching, "⏳", "An interval marked for approximate matching after optimal matching failed."};
            if(key == split_up)
                return {split_up, "✂", "Start of a tracklet section created from changes in visual identity predictions."};
            if(key == annotated)
                return {annotated, "✎", "A frame with manual detection annotations or frame tags."};
            return {key, {}, {}};
        }
        
        struct Properties {
            ID_t id{0};
            gui::Color color;
            bool is_global{false};
        };
        
        struct fdx_t {
            Idx_t id;
            explicit fdx_t(Idx_t i) : id(i) {}
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
            static consteval std::string_view class_name() {
                return "fdx_t";
            }
            [[nodiscard]] operator Idx_t() const { return Idx_t{id}; }
        };
        
        struct bdx_t {
            uint32_t id;
            explicit bdx_t(uint32_t i) : id(i) {}
            bool operator <(bdx_t other) const {
                return id < other.id;
            }
            std::string toStr() const;
            static consteval std::string_view class_name() {
                return "bdx_t";
            }
            bool operator==(const bdx_t& other) const {
                return id == other.id;
            }
        };
        
        using foi_type = std::map<ID_t, std::set<FOI, std::less<FOI>>>;
        typedef std::chrono::high_resolution_clock change_clock_t;
        //typedef std::chrono::duration<double, std::ratio<1> > second_;
        typedef std::chrono::time_point<change_clock_t> time_point_t;
        
    protected:
        static std::unordered_map<std::string, Properties, MultiStringHash, MultiStringEqual> _string_to_id;
        static std::map<ID_t, std::string> _id_to_string;
        static std::set<ID_t> _ids;
        static foi_type _frames_of_interest;
        static std::recursive_mutex _mutex;
        static cmn::gui::ColorWheel _wheel;
        
        const Properties* _props;
        
    protected:
        GETTER(Range<Frame_t>, frames);
        GETTER(std::set<fdx_t>, fdx);
        GETTER(std::set<bdx_t>, bdx);
        GETTER(std::string, description);
        
    public:
        FOI() : _props(NULL), _frames({}, {}) {}
        bool valid() const { return _frames.start != _frames.end || _frames.start.valid(); }

        FOI(const Range<Frame_t>& frames, std::set<fdx_t> fdx, std::string_view reason, const std::string& _description = "");
        FOI(const Range<Frame_t>& frames, std::set<bdx_t> bdx, std::string_view reason, const std::string& _description = "");
        
        FOI(Frame_t frame, std::set<fdx_t> fdx, std::string_view reason, const std::string& _description = "");
        FOI(Frame_t frame, std::set<bdx_t> bdx, std::string_view reason, const std::string& _description = "");
        
        FOI(Frame_t frame, std::string_view reason, bool is_global = false, const std::string& _description = "");
        FOI(const Range<Frame_t>& frames, std::string_view reason, bool is_global = false, const std::string& _description = "");
        
    public:
        bool operator<(const FOI& other) const;
        bool operator==(const FOI& other) const;
        
        static const gui::Color& color(std::string_view name);
        ID_t id() const;
        
        static std::string_view name(ID_t id);
        static Info info(ID_t id);
        static ID_t to_id(std::string_view name);
        static void add(FOI&&);
        static std::optional<foi_type::mapped_type> foi(ID_t id);
        static std::set<ID_t> ids();
        static void remove_frames(Frame_t frameIndex);
        static void remove_frames(Frame_t frameIndex, ID_t id);
        static void remove_all_of(ID_t);
        static void replace_all_of(ID_t, std::set<FOI>&&);
        static void replace_all_of(std::unordered_map<ID_t, std::set<FOI>>&&);
        static void clear_tracking_fois();
        static void clear_all();
        static uint64_t last_change();
        static foi_type all_fois();
        std::string toStr() const;
        static consteval std::string_view class_name() { return "FOI"; }

    private:
        static void changed();
        
        static std::set<ID_t> global_ids();
        static void _remove_all_of(ID_t);
    };
}
