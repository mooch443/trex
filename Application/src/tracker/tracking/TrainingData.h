#pragma once

#include <types.h>
#include <misc/metastring.h>
#include <misc/Image.h>
#include <pv.h>
#include <tracker/misc/default_config.h>

namespace track {

struct TrainingFilterConstraints {
    float median_midline_length_px;
    float median_number_outline_pts;
    float midline_length_px_std, outline_pts_std;
    float median_angle_diff;
    
    TrainingFilterConstraints()
        : median_midline_length_px(-1), median_number_outline_pts(-1), midline_length_px_std(-1), outline_pts_std(-1), median_angle_diff(-1)
    {}
    
    bool empty() const { return median_midline_length_px == -1; }
    bool has_std() const { return midline_length_px_std != -1; }
    
    operator MetaObject() const;
    static std::string class_name() {
        return "TrainingFilterConstraints";
    }
};

class TrainingData {
public:
    struct DataRange {
        struct PerIndividual {
            std::vector<Image::Ptr> images;
            std::vector<Vec2> positions;
            std::vector<size_t> num_pixels;
            std::vector<long_t> frame_indexes;
            //std::vector<size_t> global_array_indexes;
            std::set<FrameRange> ranges;
        };
        
        std::map<long_t, PerIndividual> mappings;
        std::map<long_t, long_t> applied_mapping;
        std::set<long_t> classes;
        std::vector<long_t> ids;
        std::vector<Image::Ptr> images;
        
        Rangel frames;
        bool salty;
        
        // merges the data of other with this and returns
        // a (potentially) changed frame range
        //Rangel merge(const DataRange& other);
        
        DataRange(bool salt = false) : frames(-1, -1), salty(salt) {}
        
        long_t map(long_t) const;
        long_t unmap(long_t) const;
        void reverse_mapping();
        
        operator MetaObject() const;
        static std::string class_name() {
            return "DataRange";
        }
    };
    
    struct MidlineFilters {
        std::map<long_t, std::map<FrameRange, TrainingFilterConstraints>> filters;
        
        MidlineFilters(const decltype(filters)& filters = {})
            : filters(filters)
        {}
        
        bool has(long_t ID, const FrameRange& range = FrameRange()) const {
            if(range.empty())
                return filters.count(ID) != 0;
            else {
                auto it = filters.find(ID);
                if(it == filters.end())
                    return false;
                
                for(auto && [r, f] : it->second) {
                    if(r == range || r.overlaps(range)) {
                        return true;
                    }
                }
                
                return false;
            }
        }
        
        bool has(long_t ID, long_t frame) const {
            auto it = filters.find(ID);
            if(it == filters.end())
                return false;
            
            for(auto && [r, f] : it->second) {
                if(r.contains(frame))
                    return true;
            }
            
            return false;
        }
        
        bool has_std() const {
            for(auto && [id, filter] : filters) {
                for(auto && [r, f] : filter) {
                    if(f.has_std())
                        return true;
                }
            }
            return false;
        }
        
        void set(long_t ID, const TrainingFilterConstraints& filter) {
            if(has(ID))
                Warning("[TrainingFilter] %d is already present. Replacing.", ID);
            
            if(!filters[ID].empty())
                U_EXCEPTION("[TrainingFilter] Cannot add both full-range filters, and range-specific filters at the same time.");
            filters[ID][FrameRange()] = filter;
        }
        
        void set(long_t ID, const FrameRange& range, const TrainingFilterConstraints& filter) {
            if(has(ID, range))
                Warning("[TrainingFilter] %d in range %d-%d is already present. Replacing.", ID, range.start(), range.end());
            if(filters[ID].find(FrameRange()) != filters[ID].end())
                U_EXCEPTION("[TrainingFilter] Cannot add both full-range filters, and range-specific filters at the same time.");
            
            filters[ID][range] = filter;
        }
        
        const TrainingFilterConstraints& get(long_t ID, long_t frame) const {
            auto fit = filters.find(ID);
            if(fit == filters.end())
                U_EXCEPTION("Cannot find ID %d in TrainingFilterConstraints.", ID);
            
            auto it = fit->second.find(FrameRange());
            if(it != fit->second.end())
                return it->second;
            
            for(auto && [r, f] : fit->second) {
                if(r.contains(frame))
                    return f;
            }
            
            U_EXCEPTION("Cannot find frame %d in TrainingFilterConstraints.", frame);
        }
    };
    
    
    struct TrainingAndValidation {
        std::vector<Image::Ptr> training_images, validation_images;
        std::vector<long_t> training_ids, validation_ids;
    };
    //std::map<long_t, TrainingFilterConstraints> custom_midline_lengths_no_std;
    //std::map<long_t, TrainingFilterConstraints> custom_midline_lengths_std;
    
private:
    GETTER_SETTER(default_config::recognition_normalization_t::Class, normalized)
    GETTER_SETTER(file::Path, save_path)
    GETTER_SETTER_PTR(std::shared_ptr<TrainingData>, data_source)
    
    using d_type = std::set<std::shared_ptr<DataRange>>;
    GETTER(d_type, data)
    GETTER(std::set<uint32_t>, all_classes)
    GETTER_NCONST(MidlineFilters, filters)
    
    using s_type = std::map<uint32_t, std::set<FrameRange>>;
    GETTER(s_type, included_segments)
    
    //FrameRanges frames;
    
public:
    TrainingData(const MidlineFilters& filters = MidlineFilters());
    ~TrainingData();
    
    void merge_with(std::shared_ptr<TrainingData>, bool unmap_everything = false);
    size_t size() const;
    bool empty() const;
    
    static void print_pointer_stats();
    
    enum class ImageClass {
        TRAINING,
        VALIDATION,
        NONE
    };
    
    class TrainingImageData : public Image::CustomData {
    public:
        ImageClass type;
        const long_t original_id;
        const std::string source;
        TrainingImageData(std::string source, long_t oid) : type(ImageClass::NONE), original_id(oid), source(source) {}
        ~TrainingImageData() {}
    };
    
    static inline ImageClass image_class(const Image::Ptr& image) {
        return image && image->custom_data() ? static_cast<TrainingImageData*>(image->custom_data())->type : ImageClass::NONE;
        //return image && image->index() < 0 ? ImageClass::VALIDATION : (image->index() > 0 ? ImageClass::TRAINING : ImageClass::NONE);
        
    }
    static inline bool image_is(const Image::Ptr& image, ImageClass c) {
        return image_class(image) == c;
    }
    static inline void set_image_class(Image::Ptr& image, ImageClass c) {
        if(image->index() == 0)
            return;
        
        auto isc = image_class(image);
        if(c != isc) {
            static_cast<TrainingImageData*>(image->custom_data())->type = c;
            
            /*if(c == ImageClass::NONE)
                image->set_index(0);
            else if(isc == ImageClass::NONE)
                image->set_index(c == ImageClass::TRAINING ? 1 : -1);
            else
                image->set_index(-image->index());
            
            assert((c == ImageClass::TRAINING && image->index() >= 0) || (c == ImageClass::VALIDATION && image->index() < 0));*/
        }
    }
    
    TrainingAndValidation join_split_data() const;
    std::tuple<std::vector<Image::Ptr>, std::vector<long_t>> join_arrays() const;
    std::tuple<std::vector<Image::Ptr>, std::vector<long_t>, std::vector<long_t>, std::map<long_t, Range<size_t>>> join_arrays_ordered() const;
    
    bool generate(const std::string& step_description, pv::File& video_file, std::map<long_t, std::set<long_t> > individuals_per_frame, const std::function<void(float)>& callback, const TrainingData* source);
    
    //bool generate(pv::File& video_file, const std::map<long_t, std::set<FrameRange>>&, const std::function<void(float)>& callback);
    
    std::shared_ptr<DataRange> add_salt(const std::shared_ptr<TrainingData>& source, const std::string& purpose);
    
    void add_frame(std::shared_ptr<DataRange> ptr, long_t frame_index, long_t id, long_t original_id, Image::Ptr image, const Vec2& pos, size_t px, const FrameRange& from_range);
    void apply_mapping(const std::map<long_t, long_t>&);
    operator MetaObject() const;
    static std::string class_name() {
        return "TrainingData";
    }
    
    //! used as an override for when data is just used to initialize the network and nothing more.
    void set_classes(const std::set<uint32_t>& classes);
    
    std::unique_ptr<Image> draw_coverage(const std::map<uint32_t, float>& uniquenesses = {}, const std::vector<Rangel>& = {}, const std::vector<Rangel>& added_ranges = {}, const std::map<uint32_t, float>& uniquenesses_temp = {}, std::shared_ptr<DataRange> current_salt = nullptr, const std::map<Rangel, std::tuple<double, FrameRange>>& assigned_unique_averages = {}) const;
};

}
