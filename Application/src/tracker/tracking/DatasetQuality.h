#pragma once

#include <types.h>
#include <tracking/Tracker.h>
#include <limits>

namespace track {
    class DatasetQuality {
    public:
        struct Single {
            Idx_t id;
            
            float midline_len;
            float midline_std;
            
            float distance_travelled;
            float grid_cells_visited;
            
            float outline_len;
            float outline_std;
            
            Size2 median_blob_size;
            Size2 blob_size_std;
            
            float median_angle_var;
            
            long_t number_frames;
            
            Single(Idx_t id = Idx_t())
                : id(id), midline_len(0), midline_std(0), distance_travelled(0), grid_cells_visited(0), median_angle_var(0), number_frames(0)
            { }
            
            std::string toStr() const;
            static std::string class_name() {
                return "DatasetQuality::Single";
            }
        };
        
        struct Quality {
            Range<Frame_t> range;
            uint32_t min_cells;
            float average_samples;
            
            Quality(const Range<Frame_t>& range = Range<Frame_t>(Frame_t(), Frame_t()),
                    uint32_t min_cells = 0,
                    float average_samples = -1)
                : range(range), min_cells(min_cells), average_samples(average_samples)
            {}
            
            bool operator <(const Quality& other) const {
                return min_cells < other.min_cells || (min_cells == other.min_cells && average_samples < other.average_samples);
            }
            
            bool operator >(const Quality& other) const {
                return min_cells > other.min_cells || (min_cells == other.min_cells && average_samples > other.average_samples);
            }
            
            std::string toStr() const {
                return "Quality<"+Meta::toStr(range)+" min_cells:"+Meta::toStr(min_cells)+" samples:"+Meta::toStr(average_samples)+">";
            }
            static std::string class_name() {
                return "Quality";
            }
        };
        
    private:
        Range<Frame_t> _manually_selected;
        std::map<Range<Frame_t>, std::map<Idx_t, Single>> _cache;
        std::map<Range<Frame_t>, Quality> _quality;
        Range<Frame_t> _last_seen;
        std::set<Range<Frame_t>> _previous_selected;
        
        std::set<Quality, std::greater<>> _sorted;
        
    public:
        DatasetQuality();
        
        void remove_frames(Frame_t start);
        void update(const Tracker::LockGuard&);
        Quality quality(const Range<Frame_t>& range) const;
        //Quality quality(float frame) const;
        bool has(const Range<Frame_t>& range) const;
        Range<Frame_t> best_range() const;
        std::map<Idx_t, Single> per_fish(const Range<Frame_t>&) const;
        
        void print_info() const;
        
    private:
        void remove_segment(const Range<Frame_t>& range);
        bool calculate_segment(const Range<Frame_t>&, const uint64_t video_length, const Tracker::LockGuard&);
        Single evaluate_single(Idx_t id, Individual* fish, const Range<Frame_t>& consec, const Tracker::LockGuard& guard);
    };
}
