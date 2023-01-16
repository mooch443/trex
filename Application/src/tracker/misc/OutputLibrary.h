#ifndef _OUTPUT_LIBRARY_H
#define _OUTPUT_LIBRARY_H

#include <types.h>
#include <tracking/Individual.h>
#include <misc/GlobalSettings.h>
#include <gui/Graph.h>
#include <misc/ranges.h>
#include <misc/OptionsList.h>

namespace Output {
    using namespace track;
    
    ENUM_CLASS(Functions,
               X,Y,
               VX,VY,
               SPEED,
               ACCELERATION,
               ANGLE,
               ANGULAR_V,
               ANGULAR_A,
               MIDLINE_OFFSET,
               MIDLINE_DERIV,
               BINARY,
               BORDER_DISTANCE,
               NEIGHBOR_DISTANCE
            )
    
    // , const std::function<float(float)>& options
#define LIBPARAM (Output::Library::LibInfo info, Frame_t frame, const track::MotionRecord* props, bool smooth)
#define _LIBFNC(CONTENT) LIBPARAM -> float \
{ auto fish = info.fish; UNUSED(smooth); UNUSED(fish); UNUSED(frame); if(!props) return gui::Graph::invalid(); CONTENT }
#define LIBFNC(CONTENT) [] _LIBFNC(CONTENT)
    
#define _LIBGLFNC(CONTENT) LIBPARAM -> double \
{ (void)props; (void)smooth; CONTENT }
#define LIBGLFNC(CONTENT) [] _LIBGLFNC(CONTENT)

#define _LIBNCFNC(CONTENT) LIBPARAM -> double \
{ auto fish = info.fish; (void)props; (void)smooth; CONTENT }
#define LIB_NO_CHECK_FNC(CONTENT) [] _LIBNCFNC(CONTENT)
    
#define MODIFIED(FNC, MODIFIER) (_output_modifiers.count(FNC) != 0 ? _output_modifiers.at(FNC).is(MODIFIER) : false)
    
    //! Calculates training data focussed on a fishs perspective.
    //  Training data will consist of:
    //  frame | x | y | angle | length(v) | length(a) | ..
    //  .. | neighbor[1...N].rel[x,y,angle,len(v),len(a)]
    bool save_focussed_on(const file::Path& file, const Individual* fish);
    
    struct Calculation {
        float _factor;
        enum Operation {
            MUL,
            ADD,
            NONE
        } _operation;
        
        Calculation() : _operation(NONE) {}
        
        double apply(const double& val) const {
            // identity
            if(_operation == NONE)
                return val;
            
            // multiplication type
            if(_operation == MUL) {
                return _factor * val;
            }
            
            // add type
            return _factor + val;
        }
    };
    
    ENUM_CLASS( Modifiers,
        SMOOTH,
        CENTROID,
        POSTURE_CENTROID,
        WEIGHTED_CENTROID,
        POINTS,
        PLUSMINUS
    );

    using Options_t = OptionsList<Output::Modifiers::Class>;
    
    class Library;
    struct LibraryCache {
        typedef std::shared_ptr<LibraryCache> Ptr;
        
        std::recursive_mutex _cache_mutex;
        std::map<const Individual*, std::map<Frame_t, std::map<std::string, std::map<Options_t, double>>>> _cache;
        
        void clear();
        static LibraryCache::Ptr default_cache();
    };
    
    class Library {
        
        
    public:
        typedef std::vector<std::pair<std::string, std::vector<std::string>>> graphs_type;
        typedef std::unordered_map<std::string, std::vector<std::string>> default_options_type;
        
        struct LibInfo {
            size_t rec_depth;
            const Individual* fish;
            const Options_t modifiers;
            LibraryCache::Ptr _cache;
            
            LibInfo(const Individual* fish, const Options_t &modifiers, LibraryCache::Ptr cache = nullptr) : rec_depth(0), fish(fish), modifiers(modifiers), _cache(cache)
            {}
        };
        
        typedef OptionsList<std::string> OList;
        typedef std::function<double LIBPARAM> FunctionType;
        
        ~Library() {}
        static void Init();
        static void InitVariables();
        
        static void clear_cache();
        static void frame_changed(Frame_t frameIndex, LibraryCache::Ptr cache = nullptr);
        
        static double get(const std::string& name, LibInfo info, Frame_t frame);
        static double get_with_modifiers(const std::string& name, LibInfo info, Frame_t frame);
        static void add(const std::string& name, const FunctionType& func);
        static void init_graph(gui::Graph &graph, const Individual *fish, LibraryCache::Ptr cache = nullptr);
        static bool has(const std::string& name);
        static std::vector<std::string> functions();
        
        static void remove_calculation_options();
        
    private:
        
        
        Library() {}
        
        static const track::MotionRecord* retrieve_props(const std::string&, 
            const Individual* fish, 
            Frame_t frame,
            const Options_t& modifiers)
        {
            auto c = fish->centroid(frame);
            if(!c)
                return NULL;
            
            if (modifiers.is(Modifiers::CENTROID)) {
                return c;

            } else if(modifiers.is(Modifiers::POSTURE_CENTROID)) {
                return fish->centroid_posture(frame);

            } else if(modifiers.is(Modifiers::WEIGHTED_CENTROID)) {
                return fish->centroid_weighted(frame);

            } else if(fish->head(frame)) {
                return fish->head(frame);
            }
            
            return NULL;
        };
        
    public:
        static const Calculation parse_calculation(const std::string& calculation);
        static bool parse_modifiers(const std::string& str, Options_t& modifiers);
        
    private:
        static float tailbeats(Frame_t frame, LibInfo info);
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
