#ifndef _HIST_OPTIONS_H
#define _HIST_OPTIONS_H

namespace gui {
    namespace Hist {
        enum Options {
            NONE = 0,
            NORMED = 1
        };

        class Filter {
            GETTER(float, lower)
            GETTER(float, upper)
            GETTER(float, bins)
            
        private:
            constexpr Filter(float lower = -1, float upper = -1, float bins = -1)
                : _lower(lower), _upper(upper), _bins(bins)
            {}
        public:
            constexpr static Filter FixedBins(uint32_t bins) { return Filter(-1, -1, static_cast<float>(bins)); }
            constexpr static Filter FixedBins(float lower, float upper, uint32_t bins) { return Filter(lower, upper, static_cast<float>(bins)); }
            constexpr static Filter FixedRange(float lower, float upper) { return Filter(lower, upper); }
            constexpr static Filter Empty() { return Filter(); }
        };
        
        class Display {
            GETTER(float, max_y)
            GETTER(float, min_y)
            
        private:
            constexpr Display(float max_y = -1, float min_y = 0) : _max_y(max_y), _min_y(min_y) {}
            
        public:
            constexpr static Display Empty() { return Display(); }
            constexpr static Display LimitY(float max_y) { return Display(max_y); }
            constexpr static Display LimitY(float min_y, float max_y) { return Display(max_y, min_y); }
        };
    }
}

#endif
