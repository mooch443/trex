#ifndef _OUTPUT_LIBRARY_H
#define _OUTPUT_LIBRARY_H

#include <commons.pc.h>
#include <file/Path.h>
#include <misc/ranges.h>
#include <tracking/OutputLibraryTypes.h>
#include <misc/SpriteMap.h>

namespace cmn::gui {
class Graph;
}

namespace Output {
    
    // , const std::function<float(float)>& options
#define LIBPARAM (Output::Library::LibInfo info, cmn::Frame_t frame, const track::MotionRecord* props, bool smooth)
#define MODIFIED(FNC, MODIFIER) (_output_modifiers.count(FNC) != 0 ? _output_modifiers.at(FNC).is(MODIFIER) : false)

    
    //! Calculates training data focussed on a fishs perspective.
    //  Training data will consist of:
    //  frame | x | y | angle | length(v) | length(a) | ..
    //  .. | neighbor[1...N].rel[x,y,angle,len(v),len(a)]
    bool save_focussed_on(const cmn::file::Path& file, const track::Individual* fish);
    
    class Library {
        static std::atomic<cmn::Vec2>& CENTER();
        static cmn::CallbackFuture _callback;
        
    public:
        
        struct LibInfo {
            size_t rec_depth;
            const track::Individual* fish;
            const Options_t modifiers;
            LibraryCache::Ptr _cache;
            
            LibInfo(const track::Individual* fish, const Options_t &modifiers, LibraryCache::Ptr cache = nullptr) : rec_depth(0), fish(fish), modifiers(modifiers), _cache(cache)
            {}
        };
        
        typedef OptionsList<std::string> OList;
        typedef std::function<double LIBPARAM> FunctionType;
        
        ~Library() {}
        static void Init();
        static void InitVariables();
        
        static void clear_cache();
        static void frame_changed(cmn::Frame_t frameIndex, LibraryCache::Ptr cache = nullptr);
        
        static double get(std::string_view name, LibInfo info, cmn::Frame_t frame);
        static double get_with_modifiers(const std::string& name, LibInfo info, cmn::Frame_t frame);
        static void add(const std::string& name, const FunctionType& func);
        
        static cached_output_fields_t get_cached_fields();
        static void init_graph(const cached_output_fields_t& output_fields, cmn::gui::Graph &graph, const track::Individual *fish, LibraryCache::Ptr cache = nullptr);
        static cached_output_fields_t parse_output_fields(const output_fields_t&);
        static bool has(const std::string& name);
        static std::vector<std::string_view> functions();
        
        static double pose(uint8_t index, uint8_t component, LibInfo info, cmn::Frame_t frame);
        
        static void remove_calculation_options();
        
        static bool is_global_function(std::string_view name);
        static std::set<Output::Modifiers::Class> possible_sources_for(std::string_view name);
        
    private:
        
        
        Library() {}
        
        static const track::MotionRecord* retrieve_props(std::string_view, 
            const track::Individual* fish, 
            cmn::Frame_t frame,
            const Options_t& modifiers);
        
    public:
        static const Calculation parse_calculation(const std::string& calculation);
        static bool parse_modifiers(const std::string_view& str, Options_t& modifiers);
        
    private:
        static float tailbeats(cmn::Frame_t frame, LibInfo info);
    };
}

namespace std
{
    template<> struct less<Output::Options_t>
    {
        bool operator() (const Output::Options_t& lhs, const Output::Options_t& rhs) const
        {
            uint32_t s0 = 0, s1 = 0;
			for (auto k : lhs.values())
				s0 += (uint32_t)k;
			for (auto k : rhs.values())
				s1 += (uint32_t)k;
			return s0 < s1;
        }
    };
}


#endif
