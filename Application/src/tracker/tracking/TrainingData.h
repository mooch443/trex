#pragma once

#include <commons.pc.h>

#include <misc/Image.h>
#include <pv.h>
#include <tracker/misc/default_config.h>
#include <tracker/misc/idx_t.h>
#include <misc/ranges.h>
#include <tracking/FilterCache.h>

namespace track {
using namespace constraints;
using namespace cmn;

class TrainingData {
public:
    struct DataRange {
        struct PerIndividual {
            std::vector<Image::SPtr> images;
            std::vector<Vec2> positions;
            std::vector<size_t> num_pixels;
            std::vector<Frame_t> frame_indexes;
            //std::vector<size_t> global_array_indexes;
            std::set<FrameRange> ranges;
        };
        
        std::map<Idx_t, PerIndividual> mappings;
        std::map<Idx_t, Idx_t> applied_mapping;
        std::set<Idx_t> classes;
        std::vector<Idx_t> ids;
        std::vector<Image::SPtr> images;
        
        Range<Frame_t> frames;
        bool salty;
        
        // merges the data of other with this and returns
        // a (potentially) changed frame range
        //Rangel merge(const DataRange& other);
        
        DataRange(bool salt = false) : frames({}, {}), salty(salt) {}
        
        Idx_t map(Idx_t) const;
        Idx_t unmap(Idx_t) const;
        void reverse_mapping();
        
        std::string toStr() const;
        static std::string class_name() {
            return "DataRange";
        }
    };
    
    struct MidlineFilters {
        std::map<Idx_t, std::map<FrameRange, FilterCache>> filters;
        
        MidlineFilters(const decltype(filters)& filters = {})
            : filters(filters)
        {}
        
        bool has(Idx_t ID, const FrameRange& range = FrameRange()) const {
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
        
        bool has(Idx_t ID, Frame_t frame) const {
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
        
        void set(Idx_t ID, const FilterCache& filter) {
            if(has(ID))
                Print("[TrainingFilter] ",ID," is already present. Replacing.");
            
            if(!filters[ID].empty())
                throw U_EXCEPTION("[TrainingFilter] Cannot add both full-range filters, and range-specific filters at the same time.");
            filters[ID][FrameRange()] = filter;
        }
        
        void set(Idx_t ID, const FrameRange& range, const FilterCache& filter) {
            if(has(ID, range))
                FormatWarning("[TrainingFilter] ", ID," in range ", range.start(), "-", range.end()," is already present. Replacing.");
            if(filters[ID].find(FrameRange()) != filters[ID].end())
                throw U_EXCEPTION("[TrainingFilter] Cannot add both full-range filters, and range-specific filters at the same time.");
            
            filters[ID][range] = filter;
        }
        
        const FilterCache& get(Idx_t ID, Frame_t frame) const {
            auto fit = filters.find(ID);
            if(fit == filters.end())
                throw U_EXCEPTION("Cannot find ID ",ID," in FilterCache.");
            
            auto it = fit->second.find(FrameRange());
            if(it != fit->second.end())
                return it->second;
            
            for(auto && [r, f] : fit->second) {
                if(r.contains(frame))
                    return f;
            }
            
            throw U_EXCEPTION("Cannot find frame ",frame," in FilterCache.");
        }
    };
    
    
    struct TrainingAndValidation {
        std::vector<Image::SPtr> training_images, validation_images;
        std::vector<Idx_t> training_ids, validation_ids;
    };
    
private:
    GETTER_SETTER_I(default_config::individual_image_normalization_t::Class, normalized, default_config::individual_image_normalization_t::none);
    GETTER_SETTER(file::Path, save_path);
    GETTER_SETTER_PTR(std::shared_ptr<TrainingData>, data_source)
    
    using d_type = std::set<std::shared_ptr<DataRange>>;
    GETTER(d_type, data);
    GETTER(std::set<Idx_t>, all_classes);
    GETTER_NCONST(MidlineFilters, filters);
    
    using s_type = std::map<Idx_t, std::set<FrameRange>>;
    GETTER(s_type, included_tracklets);
    
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
        const Idx_t original_id;
        const std::string source;
        TrainingImageData(std::string source, Idx_t oid, ImageClass type) : type(type), original_id(oid), source(source) {}
        ~TrainingImageData() {}
    };
    
    template<typename Ptr>
    static inline ImageClass image_class(const Ptr& image) {
        return image && image->custom_data() ? static_cast<TrainingImageData*>(image->custom_data())->type : ImageClass::NONE;
        //return image && image->index() < 0 ? ImageClass::VALIDATION : (image->index() > 0 ? ImageClass::TRAINING : ImageClass::NONE);
        
    }
    static inline bool image_is(const Image::SPtr& image, ImageClass c) {
        return image_class(image) == c;
    }
    template<typename Ptr>
    static inline void set_image_class(const Ptr& image, ImageClass c) {
        if(image->index() == 0)
            return;
        
        auto isc = image_class(image);
        if(c != isc) {
            static_cast<TrainingImageData*>(image->custom_data())->type = c;
            //assert(c != ImageClass::NONE);
            
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
    std::tuple<std::vector<Image::SPtr>, std::vector<Idx_t>> join_arrays() const;
    std::tuple<std::vector<Image::SPtr>, std::vector<Idx_t>, std::vector<Frame_t>, std::map<Frame_t, Range<size_t>>> join_arrays_ordered() const;
    
    bool generate(const std::string& step_description, pv::File& video_file, std::map<Frame_t, std::set<Idx_t> > individuals_per_frame, const std::function<void(float)>& callback, const TrainingData* source);
    
    //bool generate(pv::File& video_file, const std::map<long_t, std::set<FrameRange>>&, const std::function<void(float)>& callback);
    
    std::shared_ptr<DataRange> add_salt(const std::shared_ptr<TrainingData>& source, const std::string& purpose);
    
    void add_frame(std::shared_ptr<DataRange> ptr, Frame_t frame_index, Idx_t id, Idx_t original_id, const Image::SPtr& image, const Vec2& pos, size_t px, const FrameRange& from_range);
    void apply_mapping(const std::map<Idx_t, Idx_t>&);
    std::string toStr() const;
    static std::string class_name() {
        return "TrainingData";
    }
    
    //! used as an override for when data is just used to initialize the network and nothing more.
    void set_classes(const std::set<Idx_t>& classes);
    
    Image::Ptr draw_coverage(const std::map<Frame_t, float>& uniquenesses = {}, const std::vector<Range<Frame_t>>& = {}, const std::vector<Range<Frame_t>>& added_ranges = {}, const std::map<Frame_t, float>& uniquenesses_temp = {}, std::shared_ptr<DataRange> current_salt = nullptr, const std::map<Range<Frame_t>, std::tuple<double, FrameRange>>& assigned_unique_averages = {}) const;
};

}
