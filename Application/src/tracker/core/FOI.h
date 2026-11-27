#pragma once

#include <commons.pc.h>
#include <misc/colors.h>
#include <misc/ranges.h>
#include <misc/PBSettingsMacros.h>
#include <core/idx_t.h>
#include <misc/frame_t.h>

#define MAKE_FOINFO(variable_name, name, symbol, description) (variable_name, name, symbol, description)

#define FOI_INFOS \
MAKE_FOINFO(none, "none", {}, {}), \
MAKE_FOINFO(correcting, "correcting", "⚠", "Frames to review after a time jump, a manual match, a lost individual, or unusually fast movement."), \
MAKE_FOINFO(lost_1_fish, "lost >=1 fish", "⮟", "At least one individual tracked in the previous frame is missing in this frame."), \
MAKE_FOINFO(lost_2_fish, "lost >=2 fish", "⮟⮟", "The number of tracked individuals dropped by at least two since the previous frame."), \
MAKE_FOINFO(huge_time_jump, "huge time jump", "⏱", "A large time gap occurred between consecutive frame timestamps."), \
MAKE_FOINFO(manual_match, "manual match", "☞", "An individual was assigned to a detection using a manual match."), \
MAKE_FOINFO(long_tracklet, "long tracklet", "⛓", "Last frame before losing an individual whose tracklet reached the minimum output length."), \
MAKE_FOINFO(weird_distance, "weird distance", "↯", "An individual's speed reached the threshold for unusually fast movement."), \
MAKE_FOINFO(weird_distance_mm, "weird distance + mm", "↯☞", "Unusually fast movement and a manual match occurred in the same frame."), \
MAKE_FOINFO(weird_angle, "weird angle", "⮥", "An individual's orientation changed sharply between consecutive frames."), \
MAKE_FOINFO(tracklet_end, "tracklet end", "⏹", "Last frame of an individual tracklet before it ends or a new tracklet begins."), \
MAKE_FOINFO(global_tracklet, "global tracklet", "◎", "A continuous interval in which all expected individuals were tracked."), \
MAKE_FOINFO(failed_matches, "failed matches", "✖", "Manual matches could not be applied directly and required a recovery attempt."), \
MAKE_FOINFO(apprx_matching, "apprx matching", "⏳", "An interval marked for approximate matching after optimal matching failed."), \
MAKE_FOINFO(split_up, "split_up", "✂", "Start of a tracklet section created from changes in visual identity predictions."), \
MAKE_FOINFO(annotated, "annotated", "✎", "A frame with manual detection annotations or frame tags."), \
MAKE_FOINFO(uniqueness, "uniqueness", "✎", "When a visual identification model is loaded, this shows its local quality.")

#define FOI_DECLARE_NAME(entry) FOI_DECLARE_NAME_IMPL entry
#define FOI_DECLARE_NAME_IMPL(variable_name, name, symbol, description) \
static constexpr std::string_view variable_name = name;
#define FOI_MAKE_INFO(entry) FOI_MAKE_INFO_IMPL entry
#define FOI_MAKE_INFO_IMPL(variable_name, name, symbol, description) Info{variable_name, symbol, description}

namespace track {
    using namespace cmn;

    //! Frames of interest
    class FOI {
    public:
        using ID_t = long_t;
        
        struct Info {
            std::string_view name;
            std::string_view symbol;
            std::string_view tooltip;

            glz::generic to_json() const {
                return {
                    {"name", name},
                    {"symbol", symbol},
                    {"tooltip", tooltip}
                };
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

        _SIMPLE_APPLY(FOI_DECLARE_NAME, _SEP_EMPTY, FOI_INFOS)
        static inline constexpr std::array all_infos{SIMPLE_APPLY(FOI_MAKE_INFO, FOI_INFOS)};
        static inline constexpr auto all_names = [] {
            std::array<std::string_view, all_infos.size()> names{};
            for(size_t i = 0; i < names.size(); ++i)
                names[i] = all_infos[i].name;
            return names;
        }();

        template<const std::string_view& key>
        static consteval Info info() {
            for(const auto& entry : all_infos) {
                if(entry.name == key)
                    return entry;
            }
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
        static std::unordered_map<std::string_view, Properties> _string_to_id;
        static std::map<ID_t, std::string_view> _id_to_string;
        static std::set<ID_t> _ids;
        static foi_type _frames_of_interest;
        static std::recursive_mutex _mutex;
        static cmn::gui::ColorWheel _wheel;
        
        const Properties* _props{nullptr};
        
    protected:
        GETTER(Range<Frame_t>, frames);
        GETTER(std::set<fdx_t>, fdx);
        GETTER(std::set<bdx_t>, bdx);
        GETTER(std::string_view, name);
        
    public:
        bool valid() const { return _frames.start != _frames.end || _frames.start.valid(); }
        
        template<const std::string_view& name, typename... Args>
            requires (contains(all_names, name))
        static constexpr FOI Make(Args&&... args)
        {
            FOI foi;
            foi._name = name;
            foi.init(std::forward<Args>(args)...);
            return foi;
        }
        
        FOI() = default;

    private:
        void init(const Range<Frame_t>& frames, std::set<fdx_t> fdx);
        void init(const Range<Frame_t>& frames, std::set<bdx_t> bdx);
        
        void init(Frame_t frame, std::set<fdx_t> fdx);
        void init(Frame_t frame, std::set<bdx_t> bdx);
        
        void init(Frame_t frame, bool is_global = false);
        void init(const Range<Frame_t>& frames, bool is_global = false);
        
    public:
        bool operator<(const FOI& other) const;
        bool operator==(const FOI& other) const;
        
        static const gui::Color& color(std::string_view name);
        ID_t id() const;
        
        static std::string_view name(ID_t id);
        static constexpr Info dyn_info(ID_t id) {
            auto key = name(id);
            for(const auto& entry : all_infos) {
                if(entry.name == key)
                    return entry;
            }
            return Info{.name = key};
        }
        static constexpr Info dyn_info(std::string_view name) {
            for (const Info& info : all_infos) {
                if(info.name == name) {
                    return info;
                }
            }
            return Info{.name = name};
        }
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

#undef FOI_MAKE_INFO_IMPL
#undef FOI_MAKE_INFO
#undef FOI_DECLARE_NAME_IMPL
#undef FOI_DECLARE_NAME
#undef FOI_INFOS
#undef MAKE_FOINFO
