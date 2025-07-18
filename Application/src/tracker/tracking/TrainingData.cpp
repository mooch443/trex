#include "TrainingData.h"
#include <misc/GlobalSettings.h>
#include <tracking/Tracker.h>
#include <misc/PixelTree.h>
#include <tracking/VisualIdentification.h>
#include <tracking/FilterCache.h>
#include <tracking/IndividualManager.h>

//#undef NDEBUG

namespace py = Python;

namespace track {

//std::map<std::tuple<long_t, long_t>, Image*> did_image_already_exist;
//std::map<Image*, long_t> original_ids_check;

#ifndef NDEBUG
std::set<TrainingData*> _data_pointers;
std::mutex _data_pointer_mutex;
#endif

void TrainingData::print_pointer_stats() {
    /*PythonIntegration::async_python_function([]() -> bool {
        PythonIntegration::execute("print(locals(), globals())");
        return true;
    });*/
#ifndef NDEBUG
    std::lock_guard<std::mutex> guard(_data_pointer_mutex);
    Print("----");
    Print("Currently ", _data_pointers.size()," trainingdata are allocated.");
    size_t nimages{0};
    std::unordered_map<ImageClass, std::unordered_set<const Image*>> ptrs;
    for(auto &ptr : _data_pointers) {
        for(auto &data : ptr->data()) {
            nimages += data->images.size();
            for(auto &image : data->images)
                ptrs[image_class(image)].insert(image.get());
        }
        Print("\t", *ptr);
    }
    auto image_size = SETTING(individual_image_size).value<Size2>();
    const int channels = Background::meta_encoding() == meta_encoding_t::gray ? 1 : 3;
    
    size_t overall_unique_images{0};
    for(auto &[c, p] : ptrs) {
        const char* name;
        switch(c) {
            case ImageClass::TRAINING:
                name = "TRAINING";
                break;
            case ImageClass::VALIDATION:
                name = "VALIDATION";
                break;
            case ImageClass::NONE:
                name = "NONE";
                break;
            default:
                name = "unknown";
                break;
        }
        Print("\t ", name, " : ", p.size(), " images");
        overall_unique_images += p.size();
    }
    
    Print("Overall: ", FileSize{uint64_t(nimages * size_t(image_size.width) * size_t(image_size.height) * size_t(channels))}, " for ", nimages, " images at ",image_size," resolution (",overall_unique_images," unique images).");
    Print("----");
#endif
}

void add_pointer(TrainingData* data) {
#ifndef NDEBUG
    {
        std::lock_guard<std::mutex> guard(_data_pointer_mutex);
        _data_pointers.insert(data);
    }
    
    TrainingData::print_pointer_stats();
#else
    UNUSED(data);
#endif
}

void remove_pointer(TrainingData* data) {
#ifndef NDEBUG
    {
        std::lock_guard<std::mutex> guard(_data_pointer_mutex);
        auto str = Meta::toStr(*data);
        if(_data_pointers.count(data) == 0)
            throw U_EXCEPTION("Cannot find pointer to ",str);
        Print("Removing ", str.c_str());
        _data_pointers.erase(data);
    }
    
    TrainingData::print_pointer_stats();
#else
    UNUSED(data);
#endif
}

TrainingData::TrainingData(const MidlineFilters& filters)
    : _filters(filters)
{
    add_pointer(this);
    
    const auto normalize = default_config::valid_individual_image_normalization();
    set_normalized(normalize);
}

TrainingData::~TrainingData() {
    remove_pointer(this);
}

std::string TrainingData::toStr() const {
    return "TrainingData<norm:"+Meta::toStr(_normalized)+" path:"+save_path().str()+" size:"+Meta::toStr(size())+" ranges:"+Meta::toStr(_data)+">";
}

std::string TrainingData::DataRange::toStr() const {
    std::stringstream ss;
    ss << "(";
    if(salty) {
        ss << "SALT)";
    } else {
        bool first = true;
        for(auto && [id, data] : mappings) {
            std::set<Frame_t> frames(data.frame_indexes.begin(), data.frame_indexes.end());
            if(!first)
                ss << ", ";
            else
                first = false;
            
            ss << id.toStr() << "=[" << (frames.empty() ? "" : (Meta::toStr(*frames.begin()) + "-" + Meta::toStr(*frames.rbegin()))) << "]";
        }
        
        if(!applied_mapping.empty())
            ss << " map:" << Meta::toStr(applied_mapping);
        
        ss << ")";
    }
    
    return ss.str();
}

void TrainingData::add_frame(std::shared_ptr<TrainingData::DataRange> data, Frame_t frame_index, Idx_t id, Idx_t original_id, const Image::SPtr& image, const Vec2 & pos, size_t px, const FrameRange& from_range)
{
    assert(!image_is(image, ImageClass::NONE));
    /*auto it = original_ids_check.find(image.get());
    if(it  == original_ids_check.end())
        original_ids_check[image.get()] = id;
    else if(it->second != id) {
        FormatWarning("Changed identity of ",image->index()," from ",it->second," to ",id," without notice");
    }*/
    
    if(!image->custom_data() || static_cast<TrainingImageData*>(image->custom_data())->original_id != original_id) {
        auto str = Meta::toStr(data->applied_mapping);
        FormatExcept("mapping: ", str);
        throw U_EXCEPTION("individual ",id," frame ",frame_index," with original_id == ",image->custom_data() ? static_cast<TrainingImageData*>(image->custom_data())->original_id : Idx_t()," != ",original_id," (generated in '",image->custom_data() ? static_cast<TrainingImageData*>(image->custom_data())->source.c_str() : "","')");
    }
    
    //! debug
    for(auto &d : _data) {
        if(d == data)
            continue;
        
        auto it = d->mappings.find(id);
        if(it == d->mappings.end())
            continue;
        
        for(auto &range : it->second.ranges) {
            if(range.contains(frame_index)) {
                //FormatExcept("\tFound frame ",frame_index," already in range ",range.start(),"-",range.end()," (",from_range.start(),"-",from_range.end(),")");
                return;
            }
        }
    }
    
    if(_data.find(data) == _data.end()) {
        _data.insert(data);
    }
    
    auto &obj = data->mappings[id];
    //assert(obj.frame_indexes.empty() || obj.frame_indexes.back() < frame_index);
    
    if(_included_tracklets.find(id) == _included_tracklets.end())
        _included_tracklets[id] = {};
    if(_included_tracklets.at(id).find(from_range) == _included_tracklets.at(id).end()) {
        _included_tracklets.at(id).insert(from_range);
    }
    
    obj.images.push_back(image);
    obj.num_pixels.push_back(px);
    obj.frame_indexes.push_back(frame_index);
    obj.positions.push_back(pos);
    obj.ranges.insert(from_range);
    
    if(data->classes.find(id) == data->classes.end())
        data->classes.insert(id);
    
    data->images.push_back(image);
    data->ids.push_back(id);
    
    if(not data->frames.end.valid() || frame_index > data->frames.end)
        data->frames.end = frame_index;
    if(not data->frames.start.valid() || frame_index < data->frames.start)
        data->frames.start = frame_index;
    
    if(_all_classes.find(id) == _all_classes.end())
        _all_classes.insert(id);
}

void TrainingData::apply_mapping(const std::map<Idx_t, Idx_t>& mapping) {
    bool found = false;
    for(auto && [id, ID] : mapping) {
        if(id != ID) {
            found = true;
            break;
        }
    }
    
    if(!found)
        return; // mapping is exactly 1:1
    
    for(auto & data : _data) {
        if(data->salty)
            throw U_EXCEPTION("Cannot map salty data.");
        
        if(!data->applied_mapping.empty()) {
            auto str = Meta::toStr(_included_tracklets);
            throw U_EXCEPTION("Cannot apply two mappings to range ",str,".");
        }
        
        if(!data->salty) {
            auto str = Meta::toStr(mapping);
            Print("Changed mapping with ", str," for ", data->frames.start,"-",data->frames.end);
        }
        
        std::map<Idx_t, DataRange::PerIndividual> map;
        for(auto && [from, to] : mapping) {
            auto it = data->mappings.find(from);
            if(it != data->mappings.end())
                map[to] = it->second;
            /*for(size_t i=0; i < map[to].images.size(); ++i) {
                auto iit = original_ids_check.find(map[to].images[i].get());
                if(iit != original_ids_check.end()) {
                    // notify array of change
                    iit->second = to;
                }
            }*/
        }
        
        // also change ids of all images
        data->applied_mapping = mapping;
        
        for(auto & id : data->ids) {
            if(!data->applied_mapping.count(id)) {
                Print("\tCannot find what id ",id," maps to in applied mapping. Defaulting to same->same.");
                
                for(auto && [from, to] : data->applied_mapping) {
                    if(to == id) {
                        throw U_EXCEPTION("Cannot map ",id," -> ",id," and also ",from," -> ",to,".");
                    }
                }
                data->applied_mapping[id] = id;
                
            } else
                id = data->applied_mapping.at(id);
        }
        
        data->mappings = map;
    }
}

void TrainingData::set_classes(const std::set<Idx_t>& classes) {
    _all_classes = classes;
}

Image::Ptr TrainingData::draw_coverage(const std::map<Frame_t, float>& unique_percentages, const std::vector<Range<Frame_t>>& next_ranges, const std::vector<Range<Frame_t>>& added_ranges, const std::map<Frame_t, float>& uniquenesses_temp, std::shared_ptr<TrainingData::DataRange> current_salt, const std::map<Range<Frame_t>, std::tuple<double, FrameRange>>& assigned_unique_averages) const
{
    auto analysis_range = Tracker::analysis_range();
    auto image = Image::Make(500, 1800, 4);
    auto mat = image->get();
    mat = cv::Scalar(0, 0, 0, 0);
    
    std::map<Idx_t, gui::Color> colors;
    
    int rows = cmn::max(1, (long_t)Tracker::identities().size());
    
    float row_height = float(image->rows) / float(rows);
    float column_width = float(image->cols) / float(analysis_range.length().get());
    
    auto draw_range = [column_width, row_height, analysis_range, &colors](cv::Mat& mat, const Range<Frame_t>& range, Idx_t id, uint8_t alpha = 200){
        Vec2 topleft((range.start.try_sub(analysis_range.start())).get() * column_width, row_height * id.get());
        Vec2 bottomright((range.end.try_sub(analysis_range.start())).get() * column_width, row_height * (id.get() + 1));
        cv::rectangle(mat, topleft, bottomright, colors[id].alpha(alpha * 0.5), cv::FILLED);
    };
    
    
    cmn::gui::ColorWheel wheel;
    for(auto id : Tracker::identities()) {
        gui::Color color = wheel.next();
        colors[id] = color;
    }
    
    for(auto&data : _data) {
        if(data->salty)
            continue;
        
        for(auto && [id, per] : data->mappings) {
            for(auto range : per.ranges) {
                draw_range(mat, range.range, id, 25);
            }
        }
    }
    
    for(auto&data : _data) {
        if(data->salty)
            continue;
        
        for(auto && [id, per] : data->mappings) {
            std::set<Frame_t> frames(per.frame_indexes.begin(), per.frame_indexes.end());
            Range<Frame_t> range({}, {});
            
            for(auto f : frames) {
                if(!range.end.valid() || f - range.end > 1_f) {
                    if(range.end.valid()) {
                        draw_range(mat, range, id);
                    }
                    
                    range = Range<Frame_t>(f, f);
                } else
                    range.end = f;
            }
            
            if(range.end.valid()) {
                draw_range(mat, range, id);
            }
        }
    }
    
    std::vector<Vec2> unique_points;
    for(auto && [frame, p] : unique_percentages) {
        Vec2 topleft((frame.try_sub(analysis_range.start()).get()) * column_width, mat.rows * (1 - p) + 1);
        unique_points.push_back(topleft);
    }
    
    auto smooth = [&mat](auto& unique_points, gui::Color color) {
        std::vector<Vec2> smooth_points(unique_points);
        
        long_t L = (long_t)unique_points.size();
        for (long_t i=0; i<L; ++i) {
            long_t offset = 1;
            float factor = 0.5;
            
            unique_points[i].y = 0;
            float weight = 0;
            
            for(; offset < max(1, unique_points.size() * 0.15); ++offset) {
                long_t idx_1 = i-offset;
                long_t idx1 = i+offset;
                
                smooth_points[i].y +=
                    unique_points[idx_1 >= 0 ? idx_1 : 0].y * factor * 0.5
                  + unique_points[idx1 < L ? idx1 : L-1].y * factor * 0.5;
                weight += factor;
                factor *= factor;
                
                if(factor < 0.0001)
                    break;
            }
            
            if(weight > 0)
                smooth_points[i].y = smooth_points[i].y * 0.5 + unique_points[i].y / weight * 0.5;
            
            if(i > 0)
                DEBUG_CV(cv::line(mat, smooth_points[i], smooth_points[i-1], color.alpha(200), 2, cv::LINE_AA));
            DEBUG_CV(cv::circle(mat, unique_points[i], 2, color.alpha(200)));
        }
        unique_points = smooth_points;
    };
    
    smooth(unique_points, gui::Cyan);
    
    if(!uniquenesses_temp.empty()) {
        unique_points.clear();
        
        for(auto && [frame, p] : uniquenesses_temp) {
            Vec2 topleft((frame.try_sub(analysis_range.start()).get()) * column_width, mat.rows * (1 - p) + 1);
            unique_points.push_back(topleft);
        }
        
        smooth(unique_points, gui::White);
    }
    
    if(!added_ranges.empty()) {
        for(auto &range : added_ranges) {
            Vec2 topleft((range.start.try_sub(analysis_range.start()).get()) * column_width, 0);
            Vec2 bottomright((range.end.try_sub(analysis_range.start()).get() + 1) * column_width, 1);
            DEBUG_CV(cv::rectangle(mat, topleft, bottomright, gui::Green.alpha(100 + 100), cv::FILLED));
            DEBUG_CV(cv::putText(mat, Meta::toStr(range), (topleft + (bottomright - topleft) * 0.5) + Vec2(0,10), cv::FONT_HERSHEY_PLAIN, 0.5, gui::Green));
        }
    }
    
    if(!next_ranges.empty()) {
        for(auto it = next_ranges.rbegin(); it != next_ranges.rend(); ++it) {
            gui::Color color(0,200,255,255);
            if(it == --next_ranges.rend())
                color = gui::White;
            
            auto next_range = *it;
            
            Vec2 topleft((next_range.start.try_sub(analysis_range.start()).get()) * column_width, 0);
            Vec2 bottomright((next_range.end.try_sub(analysis_range.start()).get() + 1)* column_width, 1);
            DEBUG_CV(cv::rectangle(mat, topleft, bottomright, color.alpha(100 + 100), cv::FILLED));
            
            if(it == --next_ranges.rend())
                DEBUG_CV(cv::putText(mat, "next: "+Meta::toStr(next_range), (topleft + (bottomright - topleft) * 0.5) + Vec2(10), cv::FONT_HERSHEY_PLAIN, 0.5, color));
            
            if(assigned_unique_averages.count(next_range)) {
                auto && [distance, extended_range] = assigned_unique_averages.at(next_range);
                
                Vec2 rtl((extended_range.start().try_sub(analysis_range.start())).get() * column_width, (1 - distance / 110.0) * mat.rows + 5);
                Vec2 rbr((extended_range.end().try_sub(analysis_range.start()) + 1_f).get() * column_width, (1 - distance / 110.0) * mat.rows + 2 + 5);
                DEBUG_CV(cv::rectangle(mat, rtl, rbr, color));
                
                DEBUG_CV(cv::line(mat, Vec2(rtl.x, rtl.y), Vec2(topleft.x, bottomright.y), gui::Cyan));
                DEBUG_CV(cv::line(mat, Vec2(rbr.x, rtl.y), Vec2(bottomright.x, bottomright.y), gui::Cyan.alpha(50)));
            }
        }
    }
    
    if(current_salt) {
        for(auto && [id, per] : current_salt->mappings) {
            for(size_t i=0; i<per.frame_indexes.size(); ++i) {
                Vec2 topleft((per.frame_indexes[i].try_sub(analysis_range.start()).get()) * column_width, row_height * (id.get() + 0.2));
                Vec2 bottomright((per.frame_indexes[i].try_sub(analysis_range.start()).get() + 1)* column_width, row_height* (id.get() + 0.8));
                DEBUG_CV(cv::rectangle(mat, topleft, bottomright, gui::White.alpha(100 + 100), cv::FILLED));
            }
        }
    }
    
    for(auto id : all_classes())
        DEBUG_CV(cv::putText(mat, Meta::toStr(id), Vec2(10, row_height * (id.get() + 0.25)), cv::FONT_HERSHEY_PLAIN, 0.75, gui::White));
    
    return image;
}

void TrainingData::merge_with(std::shared_ptr<TrainingData> other, bool unmap_everything) {
    if(!other) {
        FormatWarning("Cannot merge with nullptr.");
        return;
    }
    
    Print("[TrainingData] Merging ",*this," with ",*other,".");
    
    // merge all_classes for both trainingdata and also merge filters
    for(auto id : other->all_classes()) {
        if(_all_classes.find(id) == _all_classes.end()) {
            _all_classes.insert(id);
        }
        
        // check for custom midline filters for this id
        for(auto && [id, map] : other->filters().filters) {
            for(auto && [range, filter] : map) {
                if(!filters().has(id, range)) {
                    filters().set(id, range, filter);
                }
            }
        }
    }
    
    std::map<Idx_t, std::set<FrameRange>> before_ranges;
    for(auto & mptr : data()) {
        for(auto && [id, per] : mptr->mappings)
            before_ranges[id].insert(per.ranges.begin(), per.ranges.end());
    }
    
    std::map<Idx_t, size_t> added_images;
    
    for(auto & ptr : other->data()) {
        // skip salt ranges
        if(ptr->salty)
            continue;
        
        auto new_ptr = std::make_shared<DataRange>(false);
        if(!unmap_everything)
            new_ptr->applied_mapping = ptr->applied_mapping;
        
        for(auto && [id, per] : ptr->mappings) {
            std::set<Frame_t> added_frame_indexes;
            auto ID = ptr->unmap(id);
            
            for(auto &mdata : data()) {
                if(mdata->salty)
                    continue;
                
                auto _id = mdata->map(ID);
                
                if(mdata->mappings.find(_id) == mdata->mappings.end())
                    continue;
                else {
                    auto &mper = mdata->mappings.at(_id);
                    added_frame_indexes.insert(mper.frame_indexes.begin(), mper.frame_indexes.end());
                }
                
                /*if(mdata->mappings.find(id) == mdata->mappings.end())
                    continue;
                else {
                    auto &mper = mdata->mappings.at(id);
                    added_frame_indexes.insert(mper.frame_indexes.begin(), mper.frame_indexes.end());
                }*/
            }
            
            size_t added = 0;
            
            for(size_t i=0; i<per.frame_indexes.size(); ++i) {
                auto frame = per.frame_indexes[i];
                if(added_frame_indexes.find(frame) == added_frame_indexes.end()) {
                    // frame has not been added yet, add it
                    FrameRange range;
                    for(auto &r : per.ranges) {
                        if(r.contains(frame)) {
                            range = r;
                            break;
                        }
                    }
                    
                    if(range.empty()) {
                        auto str = Meta::toStr(per.frame_indexes);
                        throw U_EXCEPTION("Cannot find a range that frame ",frame," belongs to in ",str,"");
                    }
                    
                    add_frame(new_ptr, frame, unmap_everything ? ID : id, ID, per.images[i], per.positions[i], per.num_pixels[i], range);
                    
                    ++added;
                    added_frame_indexes.insert(frame);
                    
                } //else
                    //Warning("The same image is already present in a different DataRange within the merged dataframe (%d, %d, %d).", id, ID, frame);
            }
            
            added_images[id] += added;
        }
        
        for(auto && [id, per] : new_ptr->mappings) {
            for(auto &range : per.ranges) {
                if(_included_tracklets[id].find(range) == _included_tracklets[id].end()) {
                    _included_tracklets.at(id).insert(range);
                }
            }
        }
    }
    
    Print("[TrainingData] Finished merging: ",*this," (added images: ",added_images,")");
    
    //if(unmap_everything) {
     //   auto image = draw_coverage();
     //   tf::imshow("generated", image->get());
    //}
}

size_t TrainingData::size() const {
    size_t n = 0;
    for(auto &d : _data) {
        n += d->images.size();
    }
    return n;
}

std::tuple<std::vector<Image::SPtr>, std::vector<Idx_t>> TrainingData::join_arrays() const {
    std::vector<Image::SPtr> images;
    std::vector<Idx_t> ids;
    
    const size_t L = size();
    ids.reserve(L);
    images.reserve(L);
    
    using fdx_t = long_t;
    using frame_t = long_t;
    
    // sanity checks
    std::map<fdx_t, std::set<frame_t>> added_data;
    
    if(_data.size() > 1)
        Print("Joining TrainingData, expecting ", L," images from ",_data.size()," arrays.");
    
    for(auto & d : _data) {
        // ignore salt
        //if(d->salty)
        //    continue;
        images.insert(images.end(), d->images.begin(), d->images.end());
        ids.insert(ids.end(), d->ids.begin(), d->ids.end());
    }
    
    if(L != images.size())
        FormatWarning("Only added ", images.size()," / ", L," possible images from ", _data.size()," arrays.");
    
    return { images, ids };
}

TrainingData::TrainingAndValidation TrainingData::join_split_data() const {
    TrainingAndValidation result;
    
    const size_t L = size();
    result.training_images.reserve(L * 0.75);
    result.training_ids.reserve(result.training_images.size());
    
    result.validation_images.reserve(L - result.training_images.size());
    result.validation_ids.reserve(result.training_images.size());
    
    using fdx_t = long_t;
    using frame_t = long_t;
    
    // sanity checks
    std::map<fdx_t, std::set<frame_t>> added_data;
    
    if(_data.size() > 1)
        Print("Joining TrainingData, expecting ", L," images from ",_data.size()," arrays.");
    
    for(auto &d : _data) {
        for(size_t i=0; i<d->images.size(); ++i) {
            auto c = image_class(d->images[i]);
            if(c == ImageClass::TRAINING) {
                // training data
                result.training_images.push_back(d->images[i]);
                result.training_ids.push_back(d->ids[i]);
                
            } else if(c == ImageClass::VALIDATION) {
                result.validation_images.push_back(d->images[i]);
                result.validation_ids.push_back(d->ids[i]);
            }
        }
    }
    
    if(L != result.training_images.size() + result.validation_images.size())
        FormatWarning("Only added ",result.training_images.size() + result.validation_images.size(), " / ", L," possible images from ", _data.size()," arrays.");
    
    return result;
}

Idx_t TrainingData::DataRange::map(Idx_t id) const {
    if(applied_mapping.empty()) return id;
    return applied_mapping.at(id);
}

Idx_t TrainingData::DataRange::unmap(Idx_t id) const {
    if(applied_mapping.empty()) return id;
    for (auto && [original, mapped] : applied_mapping) {
        if(mapped == id) {
            Print("\treversing applied mapping ", mapped," -> ",original);
            return original;
        }
    }
    throw U_EXCEPTION("Cannot find mapping for id == ",id,". Incomplete mapping.");
}

void TrainingData::DataRange::reverse_mapping() {
    if(salty)
        throw U_EXCEPTION("Cannot unmap salty data.");
    
    if(applied_mapping.empty())
        return;
    
    auto str = Meta::toStr(applied_mapping);
    Print("Reversing mapping with ", str," for ", frames.start,"-",frames.end);
    
    std::map<Idx_t, DataRange::PerIndividual> map;
    for(auto && [to, from] : applied_mapping) {
        auto it = mappings.find(from);
        if(it != mappings.end()) {
            map[to] = it->second;
        }
    }
    
    mappings = map;
    
    // also change ids of all images
    applied_mapping = {};
}

std::shared_ptr<TrainingData::DataRange> TrainingData::add_salt(const std::shared_ptr<TrainingData>& source, const std::string& purpose) {
    const size_t Nmax = 1000;
    
    std::map<Idx_t, std::set<FrameRange>> individual_ranges;
    std::map<Idx_t, std::tuple<size_t, size_t>> individual_samples;
    
    std::map<Idx_t, std::vector<FrameRange>> combined_ranges_per_individual;
    
    auto add_combined_range = [&combined_ranges_per_individual](const FrameRange& range, Idx_t id) {
        auto& combined_ranges = combined_ranges_per_individual[id];
        
        bool found = false;
        for(auto &c : combined_ranges) {
            if(c.overlaps(range)) {
                c.range.start = min(c.start(), range.start());
                c.range.end = max(c.end(), range.end());
                found = true;
                break;
            }
        }
        
        if(found) {
            auto it = combined_ranges.begin();
            while ( it != combined_ranges.end() ) {
                auto &A = *it;
                found = false;
                
                for (auto kit = combined_ranges.begin(); kit != combined_ranges.end(); ++kit) {
                    if(it == kit)
                        continue;
                    
                    if(A.overlaps(*kit)) {
                        A.range.start = min(A.start(), kit->start());
                        A.range.end = max(A.end(), kit->end());
                        combined_ranges.erase(kit);
                        it = combined_ranges.begin();
                        found = true;
                        break;
                    }
                }
                
                if(!found)
                    ++it;
            }
            
        } else {
            combined_ranges.push_back(range);
        }
    };
    
    for(auto & data : data()) {
        if(data->salty)
            throw U_EXCEPTION("Cannot add two salts.");
        
        for(auto && [id, per] : data->mappings) {
            for(auto &range : per.ranges) {
                if(individual_ranges[id].find(range) == individual_ranges[id].end()) {
                    individual_ranges[id].insert(range);
                    add_combined_range(range, id);
                }
            }
            for(size_t i=0; i<per.frame_indexes.size(); ++i)
                if(image_is(per.images[i], ImageClass::TRAINING))
                    ++std::get<0>(individual_samples[id]);
                else
                    ++std::get<1>(individual_samples[id]);
            //individual_samples[id] += per.frame_indexes.size(); // should probably use set
        }
    }
    
    for(auto && [id, combined] : combined_ranges_per_individual) {
        Frame_t N = 0_f;
        for (auto &range : combined) {
            N += range.length();
        }
        
        Print("\t(salt) ", id,": new salt N=",N);
    }
    
    size_t maximum_samples_per_individual = 0;
    for(auto && [id, samples] : individual_samples) {
        auto sum = std::get<0>(samples) + std::get<1>(samples);
        if(sum > maximum_samples_per_individual)
            maximum_samples_per_individual = sum;
    }
    
    std::map<Idx_t, std::set<std::tuple<FrameRange, const DataRange::PerIndividual*, const DataRange*, Idx_t>>> ranges_to_add;
    //std::map<Idx_t, std::set<Frame_t>> source_frames_per_individual;
    
    auto add_range = std::make_shared<DataRange>(true);
    for(auto &d : data()) {
        if(!d->applied_mapping.empty()) {
            add_range->applied_mapping = d->applied_mapping;
            Print("add_range->applied_mappig = ", add_range->applied_mapping);
            break;
        }
    }
    
    for(auto &data : source->data()) {
        for(auto && [id, per] : data->mappings) {
            // find original ID
            auto ID = data->unmap(id);
            
            for(auto & range : per.ranges) {
                if(individual_ranges[id].find(range) == individual_ranges[id].end())
                {
                    ranges_to_add[id].insert({range, &per, data.get(), ID});
                    add_combined_range(range, id);
                }
            }
            
            //source_frames_per_individual[id].insert(per.frame_indexes.begin(), per.frame_indexes.end());
        }
    }
    
    // add maximum of Nmax images per individual
    std::map<Idx_t, std::tuple<size_t, size_t, size_t, size_t>> individual_added_salt;
    std::map<Idx_t, std::tuple<size_t, size_t>> individual_samples_before_after;
    
    const double number_classes = SETTING(track_max_individuals).value<uint32_t>();
    const double gpu_max_sample_mb = double(SETTING(gpu_max_sample_gb).value<float>()) * 1000;
    const Size2 output_size = SETTING(individual_image_size);
    const double max_images_per_class = gpu_max_sample_mb * 1000 * 1000 / number_classes / output_size.width / output_size.height / 4;
    
    for(auto && [id, ranges] : ranges_to_add) {
        auto && [training_samples, validation_samples] = individual_samples[id];
        
        // the goal here is to sample all of the segments regularly, while also trying not to exceed a resource limit overall.
        // the percentage of frames for each range must in the end be representative of how many frames they represent of the whole video
        // so that means:
        //  - all ranges added together result in N < |video|
        //  - for all ranges R, stepsize_{R} = ceil( |S_R| / ( samples_max * |R| / N ) )
        //       (with S_R being the set of actually available frames <= all frames within R)
        
        // overall number of frames in global ranges
        Frame_t N = 0_f;
        for (auto &range : combined_ranges_per_individual[id]) {
            N += range.length();
        }
        
        
        size_t SR = 0;
        for(auto && [range, ptr, d, ID] : ranges)
            SR += ptr->frame_indexes.size();
        
        //auto id = add_range->applied_mapping.empty() ? ID : add_range->applied_mapping.at(ID);
        
        individual_samples_before_after[id] = {training_samples + validation_samples, 0};
        //auto sum = training_samples + validation_samples;
        individual_added_salt[id] = {0, 0, training_samples, validation_samples};
        
        size_t actually_added = 0;
        for(auto && [range, ptr, d, ID] : ranges) {
            size_t step_size = max(1u, (size_t)ceil(SR / max(1, (max_images_per_class * (double)(range.length() / N).get()))));
            
            std::vector<std::tuple<Frame_t, Image::SPtr, Vec2, size_t>> frames;
            for(size_t i=0; i<ptr->frame_indexes.size(); ++i) {
                if(range.contains(ptr->frame_indexes[i]))
                    frames.push_back({ptr->frame_indexes[i], ptr->images[i], ptr->positions[i], ptr->num_pixels[i]});
            }
            
            size_t L = frames.size();
            
            for(size_t i=0; i<L; i+=step_size) {
                auto && [frame, image, pos, px] = frames[i];
                
                if(!image->custom_data() || static_cast<TrainingImageData*>(image->custom_data())->original_id != ID) {
                    if(!image->custom_data()) {
                        throw U_EXCEPTION("No custom_data.");
                    } else {
                        auto str = Meta::toStr(d->applied_mapping);
                        auto str0 = Meta::toStr(add_range->applied_mapping);
                        FormatExcept("mapping: ", str);
                        FormatExcept("mapping_2: ", str0);
                        FormatExcept("individual ", id," frame ",frame," with original_id == ", static_cast<TrainingImageData*>(image->custom_data())->original_id," != ", ID," (generated in ", static_cast<TrainingImageData*>(image->custom_data())->source,", currently ", purpose,"), ", d->salty ? 1 : 0);
                    }
                    
                } else {
                    add_frame(add_range, frame, id, ID, image, pos, px, range);
                    if(image_is(image, ImageClass::TRAINING))
                        ++std::get<0>(individual_added_salt[id]);
                    else
                        ++std::get<1>(individual_added_salt[id]);
                    
                    ++actually_added;
                }
            }
        }
        
        Print("\t(salt) Individual ",id," (N=", N,"): added a total of ", actually_added," / ", int64_t(max_images_per_class)," frames (", std::get<0>(individual_added_salt[id])," training, ", std::get<1>(individual_added_salt[id])," validation)");
        
        std::get<1>(individual_samples_before_after[id]) = std::get<0>(individual_samples_before_after[id]) + std::get<0>(individual_added_salt[id]) + std::get<1>(individual_added_salt[id]);
    }
    
    auto str = Meta::toStr(individual_added_salt);
    auto after = Meta::toStr(individual_samples_before_after);
    Print("Added salt (maximum_samples_per_individual = ",maximum_samples_per_individual,", Nmax = ",Nmax,"): ",str.c_str()," -> ",after.c_str());
    
    return add_range;
}

bool TrainingData::generate(const std::string& step_description, pv::File & video_file, std::map<Frame_t, std::set<Idx_t> > individuals_per_frame, const std::function<void(float)>& callback, const TrainingData* source) {
    auto frames = extract_keys(individuals_per_frame);
    
    LockGuard guard(ro_t{}, "generate_training_data");
    PPFrame pp;
    pv::Frame video_frame;
    const Size2 output_size = SETTING(individual_image_size);
    const auto& custom_midline_lengths = filters();
    
    std::map<Idx_t, std::set<Frame_t>> illegal_frames;
    
    for(const auto & [frame, ids] : individuals_per_frame) {
        IndividualManager::transform_ids(ids, [frame=frame, &illegal_frames](auto id, auto fish){
            auto && [basic, posture] = fish->all_stuff(frame);
            
            if(!py::VINetwork::is_good(basic, posture)
               || !basic || basic->blob.split())
            {
                illegal_frames[id].insert(frame);
            }
        });
    }
    
    for(auto && [id, frames] : illegal_frames) {
        for(auto frame : frames) {
            individuals_per_frame.at(frame).erase(id);
            if(individuals_per_frame.at(frame).empty())
                individuals_per_frame.erase(frame);
        }
    }
    
    std::map<Idx_t, double> lengths;
    for(auto && [frame, ids] : individuals_per_frame) {
        for(auto id : ids)
            ++lengths[id];
    }
    
    double fewest_samples = std::numeric_limits<double>::max(), most_samples = 0;
    for(auto && [id, L] : lengths) {
        if(L < fewest_samples)
            fewest_samples = L;
        if(L > most_samples)
            most_samples = L;
    }
    
    if(fewest_samples == std::numeric_limits<double>::max())
        fewest_samples = most_samples = 0;
    
    const double number_classes = Tracker::identities().size();
    const double gpu_max_sample_gb = double(SETTING(gpu_max_sample_gb).value<float>());
    double percentile = ceil((most_samples - fewest_samples) * 0.65 + fewest_samples); // 0.65 percentile of #images/class
    const double current_filesize_per_class = percentile * (double)output_size.width * (double)output_size.height * 4;
    
    Print("Fewest samples for an individual is ", int64_t(fewest_samples)," samples, most are ", int64_t(most_samples),". 65% percentile is ", percentile);
    if(current_filesize_per_class * number_classes / 1000.0 / 1000.0 / 1000.0 >= gpu_max_sample_gb)
    {
        percentile = ceil(gpu_max_sample_gb * 1000 * 1000 * 1000 / (double)output_size.width / (double)output_size.height / 4.0 / double(number_classes));
        Print("\tsample size resource limit reached (with ", FileSize{uint64_t(current_filesize_per_class)}," / class in the 65 percentile, limit is ", dec<1>(gpu_max_sample_gb), "GB overall), limiting to ", int64_t(percentile)," images / class...");
    }
    
    // sub-sample any classes that need sub-sampling
    for(auto && [id, L] : lengths) {
        if(L > percentile) {
            auto step_size = percentile / L;
            //if(step_size == 1)
            //    continue;
            
            Print("\tsub-sampling class ",id," from ",L," to ",percentile," with step_size = ",step_size," (resulting in ",double(L) * step_size,")");
            
            // collect all frames where this individual is present
            
            std::set<Frame_t> empty_frames;
            size_t after = 0;
            double acc = 0;
            
            for(auto && [frame, ids] : individuals_per_frame) {
                if(ids.find(id) != ids.end()) {
                    if(acc < 1) {
                        ids.erase(id);
                        if(ids.empty())
                            empty_frames.insert(frame);
                    } else {
                        acc -= 1;
                        ++after;
                    }
                    acc += step_size;
                }
            }
            
            for(auto frame : empty_frames)
                individuals_per_frame.erase(frame);
            
            L = after;
        }
    }
    
    lengths.clear();
    for(auto && [frame, ids] : individuals_per_frame) {
        for(auto id : ids)
            ++lengths[id];
    }
    
    Print("L: ", lengths);
    
    auto data = std::make_shared<TrainingData::DataRange>();
    
    size_t i = 0;
    size_t counter = 0, failed = 0;
    Size2 minmum_size(FLT_MAX), maximum_size(0);
    Median<Float2_t> median_size_x, median_size_y;
    Frame_t inserted_start = Frame_t(std::numeric_limits<Frame_t::number_t>::max()), inserted_end{};
    
    // copy available images to map for easy access
    std::map<Idx_t, std::map<Frame_t, std::tuple<Idx_t, Image::SPtr>>> available_images;
    if(source) {
        for(auto & data : source->data()) {
            for(auto && [id, per] : data->mappings) {
                auto ID = data->unmap(id);
                auto &sub = available_images[ID];
                
                for(size_t i=0; i<per.images.size(); ++i) {
                    if(sub.find(per.frame_indexes[i]) != sub.end()) {
                        bool ptrs_equal = std::get<1>(sub[per.frame_indexes[i]]) == per.images[i];
                        if(!ptrs_equal || !data->salty) {
                            FormatExcept("Double image (",ptrs_equal ? 1 : 0,") frame ",per.frame_indexes[i]," for individual ",id," in training data (generated in ",static_cast<TrainingImageData*>(per.images[i]->custom_data())->source," with current purpose ",step_description,").");
                        } else if(data->salty) {
                            FormatWarning("Double image (",ptrs_equal ? 1 : 0,") frame ",per.frame_indexes[i]," for individual ",id," in training data (generated in ",static_cast<TrainingImageData*>(per.images[i]->custom_data())->source," with current purpose '",step_description,"').");
                        }
                    }
                    
                    if(per.images[i]->custom_data()) {
                        if( static_cast<TrainingImageData*>( per.images[i]->custom_data() )->original_id != ID )
                        {
                            FormatExcept(ID," != ", static_cast<TrainingImageData*>(per.images[i]->custom_data())->original_id," (generated in ", static_cast<TrainingImageData*>(per.images[i]->custom_data())->source," with current purpose ", step_description,")");
                        }
                        
                    } else
                        throw U_EXCEPTION("No labeling for image.");
                    
                    sub[per.frame_indexes[i]] = {id, per.images[i]};
                }
            }
        }
    }
    
    for(auto && [id, sub] : available_images) {
        Print("\t",id,": ",sub.size()," available images between ",sub.empty() ? Frame_t() : sub.begin()->first," and ",sub.empty() ? Frame_t() : sub.rbegin()->first);
    }
    
    size_t N_validation_images = 0, N_training_images = 0;
    size_t N_reused_images = 0;
    const bool calculate_posture = FAST_SETTING(calculate_posture);
    std::map<Idx_t, std::vector<std::tuple<Frame_t, Image::SPtr>>> individual_training_images;
    size_t failed_blobs = 0, found_blobs = 0;
    
    for(auto frame : frames) {
        if(individuals_per_frame.find(frame) == individuals_per_frame.end()) {
            ++i;
            continue;
        }
        
        if(not frame.valid()
           || frame >= video_file.length())
        {
            ++i;
            FormatExcept("Frame ", frame," out of range.");
            continue;
        }
        
        // check so that we do not generate images again, that we have generated before
        std::set<Idx_t> filtered_ids;
        
        for(auto id : Tracker::identities()) {
            if(individuals_per_frame.at(frame).find(id) != individuals_per_frame.at(frame).end())
                filtered_ids.insert(id);
        }
        
        if(frame.valid() and (not inserted_start.valid() or frame < inserted_start))
            inserted_start = frame;
        if(frame.valid() and (not inserted_end.valid() or frame > inserted_end))
            inserted_end = frame;
        
        IndividualManager::transform_ids(filtered_ids, [&](auto id, auto fish) {
            if(available_images.empty())
                return false;
            
            assert(individuals_per_frame.count(frame) && individuals_per_frame.at(frame).find(id) != individuals_per_frame.at(frame).end());
            
            auto fit = available_images[id].find(frame);
            if(fit != available_images[id].end()) {
                auto it = fish->iterator_for(frame);
                if(it == fish->tracklets().end())
                    return true;
                
                auto&& [ID, image] = fit->second;
                add_frame(data, frame, id, id, image, Vec2(), fish->thresholded_size(frame), *it->get());
                if(image_is(image, ImageClass::TRAINING))
                    ++N_training_images;
                else
                    ++N_validation_images;

                ++counter;
                ++N_reused_images;
                individuals_per_frame.at(frame).erase(id);
                if(individuals_per_frame.at(frame).empty()) {
                    individuals_per_frame.erase(frame);
                    return false;
                }
            }
            
            return true;
        });
        
        if(individuals_per_frame.find(frame) == individuals_per_frame.end()) {
            ++i;
            continue;
        }
        
        video_file.read_with_encoding(video_frame, frame, Background::meta_encoding());
        Tracker::preprocess_frame(std::move(video_frame), pp, nullptr, PPFrame::NeedGrid::NoNeed, video_file.header().resolution);
        
        IndividualManager::transform_ids(filtered_ids, [&](auto id, auto fish){
            /**
             * Check various conditions for whether the image is eligible for
             * training.
             *  - has to have a proper posture
             *  - it mustn't be a split blob
             *  - it must be within recognition bounds
             *  - size of the blob must fit within the given output_size
             */
            
            if(!individuals_per_frame.empty() && individuals_per_frame.at(frame).find(id) == individuals_per_frame.at(frame).end())
                return;
            
            auto filters = custom_midline_lengths.has(id)
                ? custom_midline_lengths.get(id, frame)
                : FilterCache();
            
            auto it = fish->iterator_for(frame);
            if(it == fish->tracklets().end())
                return;
            
            auto bidx = (*it)->basic_stuff(frame);
            auto pidx = (*it)->posture_stuff(frame);
            if(bidx == -1 || (pidx == -1 && calculate_posture))
                return;
            
            auto basic = fish->basic_stuff()[bidx].get();
            auto posture = pidx != -1 ? fish->posture_stuff()[pidx].get() : nullptr;
            
            if(!py::VINetwork::is_good(basic, posture))
                return;

            auto bid = basic->blob.blob_id();
            auto pid = basic->blob.parent_id;
            
            auto blob = Tracker::find_blob_noisy(pp, bid, pid, basic->blob.calculate_bounds());
            if(!blob)
                ++failed_blobs;
            else
                ++found_blobs;
            if(!blob || blob->split())
                return;
            
            ++counter;
            median_size_x.addNumber(blob->bounds().size().width);
            median_size_y.addNumber(blob->bounds().size().height);
            minmum_size = min(minmum_size, blob->bounds().size());
            maximum_size = max(maximum_size, blob->bounds().size());
            
            // try loading it all into a vector
            Image::SPtr image;
            
            /*auto iit = did_image_already_exist.find({id, frame});
            if(iit != did_image_already_exist.end()) {
                // this image was already created
                FormatWarning("Creating a second instance of id ", id," in frame ",frame);
            }*/
            
            using namespace default_config;
            auto midline = posture
                ? fish->calculate_midline_for(*posture)
                : nullptr;
            
            image = std::get<0>(constraints::diff_image(normalized(), blob.get(), midline ? midline->transform(normalized()) : gui::Transform(), filters.median_midline_length_px, output_size, Tracker::background()));
            
            if(blob->bounds().width > output_size.width
               || blob->bounds().height > output_size.height)
            {
                ++failed;
            }
            
            if(image != nullptr) {
                image->set_index(frame.get());
                
                assert(!image->custom_data());
                image->set_custom_data(new TrainingImageData("generate("+Meta::toStr(fish->identity().ID())+" "+step_description+")", id, ImageClass::TRAINING));
                set_image_class(image, ImageClass::TRAINING);
                
                ++N_training_images;
                
                if(frame.valid())
                    individual_training_images[id].push_back({frame, image});
                
                add_frame(data, frame, id, id, image, Vec2(), fish->thresholded_size(frame), *it->get());
            }
        });
        
        callback(++i / float(frames.size()));
    }
    
    Print("Failed blobs: ", failed_blobs," Found blobs: ",found_blobs);
    
    if(failed) {
        auto prefix = SETTING(individual_prefix).value<std::string>();
        FormatWarning("Some (", failed * 100 / counter,"%) ", prefix.c_str()," images are too big. Range: ", minmum_size," -> ", maximum_size," median ",median_size_x.empty() ? 0 : median_size_x.getValue(), "x", median_size_y.empty() ? 0 : median_size_y.getValue());
    }
    
    lengths.clear();
    std::map<Idx_t, std::map<ImageClass, size_t>> individual_image_types;
    for(auto &d : this->data()) {
        for(auto && [id, per] : d->mappings) {
            lengths[id] += per.images.size();
            for(auto & image : per.images)
                ++individual_image_types[id][image_class(image)];
        }
    }
    
    Print("[TrainingData] We collected ", N_training_images," training images and ", N_validation_images," validation images (",N_reused_images," reused). Checking individuals...");
    for(auto && [id, L] : lengths) {
        const size_t expected_number_validation = floor(0.25 * L);
        auto N_validation_images = individual_image_types[id][ImageClass::VALIDATION];
        if(N_validation_images < expected_number_validation) {
            auto &trainings = individual_training_images[id];
            auto available = individual_image_types[id][ImageClass::TRAINING];
            if(available < expected_number_validation - N_validation_images) {
               FormatError("\tCan only find ", available," of the ",expected_number_validation - N_validation_images," needed images.");
            } else {
                Print("\tFinding more (", expected_number_validation - N_validation_images,") validation images to reach ", expected_number_validation," samples from ",available," available images.");
                size_t step_size = max(1u, available / (expected_number_validation - N_validation_images));
                size_t N_selected = 0;
                for(size_t i=0; i<trainings.size(); i += step_size) {
                    assert(image_is(std::get<1>(trainings[i]), ImageClass::TRAINING));
                    set_image_class(std::get<1>(trainings[i]), ImageClass::VALIDATION);
                    ++N_selected;
                }
                Print("\tSelected ", N_selected," new images (", N_selected + N_validation_images," / ",expected_number_validation,")");
            }
        }
    }
    
    return N_training_images + N_validation_images > 0;
}

std::tuple<std::vector<Image::SPtr>, std::vector<Idx_t>, std::vector<Frame_t>, std::map<Frame_t, Range<size_t>>> TrainingData::join_arrays_ordered() const
{
    using fdx_t = Idx_t;
    
    std::vector<Image::SPtr> images;
    std::vector<fdx_t> ids;
    std::vector<Frame_t> frames;
    
    const size_t L = size();
    ids.reserve(L);
    images.reserve(L);
    frames.reserve(L);
    
    std::map<Frame_t, std::tuple<std::vector<fdx_t>, std::vector<Image::SPtr>>> collector;
    
    if(_data.size() > 1)
        Print("Joining TrainingData, expecting ", L," images from ",_data.size()," arrays.");
    
    for(auto & d : _data) {
        for(auto && [id, per] : d->mappings) {
            for(size_t i=0; i<per.frame_indexes.size(); ++i) {
                auto & [ids, images] = collector[per.frame_indexes[i]];
                auto it = insert_sorted(ids, id);
                auto offset = std::distance(ids.begin(), it);
                images.insert(images.begin() + offset, per.images[i]);
            }
        }
    }
    
    std::map<Frame_t, Range<size_t>> start_indexes;
    
    for(auto && [frame, data] : collector) {
        auto & [_ids, _images] = data;
        start_indexes[frame] = Range<size_t>(ids.size(), ids.size() + _ids.size());
        
        images.insert(images.end(), _images.begin(), _images.end());
        ids.insert(ids.end(), _ids.begin(), _ids.end());
        frames.insert(frames.end(), _ids.size(), frame);
    }
    
    return { images, ids, frames, start_indexes };
}
    
bool TrainingData::empty() const {
    return size() == 0;
}

}
