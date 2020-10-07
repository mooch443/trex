#pragma once

#include <types.h>
#include <tracking/Tracker.h>
#include <limits>

namespace track {
    class DatasetQuality {
    public:
        struct Single {
            idx_t id;
            
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
            
            Single(idx_t id = std::numeric_limits<idx_t>::max())
                : id(id), midline_len(0), midline_std(0), distance_travelled(0), grid_cells_visited(0), median_angle_var(0), number_frames(0)
            { }
            
            operator MetaObject() const;
            static std::string class_name() {
                return "DatasetQuality::Single";
            }
        };
        
        struct Quality {
            Rangel range;
            uint32_t min_cells;
            float average_samples;
            
            Quality(const Rangel& range = Rangel(-1, -1),
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
            
            operator MetaObject() const {
                return MetaObject("Quality<"+Meta::toStr(range)+" min_cells:"+Meta::toStr(min_cells)+" samples:"+Meta::toStr(average_samples)+">", "Quality");
            }
            static std::string class_name() {
                return "Quality";
            }
        };
        
    private:
        Rangel _manually_selected;
        std::map<Rangel, std::map<idx_t, Single>> _cache;
        std::map<Rangel, Quality> _quality;
        Rangel _last_seen;
        std::set<Rangel> _previous_selected;
        
        std::set<Quality, std::greater<>> _sorted;
        
    public:
        DatasetQuality();
        
        void remove_frames(long_t start);
        void update(const Tracker::LockGuard&);
        Quality quality(const Rangel& range) const;
        //Quality quality(float frame) const;
        bool has(const Rangel& range) const;
        Rangel best_range() const;
        std::map<idx_t, Single> per_fish(const Rangel&) const;
        
        void print_info() const;
        
    private:
        void remove_segment(const Rangel& range);
        bool calculate_segment(const Rangel&, const long_t video_length, const Tracker::LockGuard&);
        Single evaluate_single(idx_t id, Individual* fish, const Rangel& consec, const Tracker::LockGuard& guard);
    };
}
