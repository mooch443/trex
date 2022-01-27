#include "TrainingData.h"
#include <misc/GlobalSettings.h>
#include <tracking/Tracker.h>
#include <tracking/Recognition.h>
#include <misc/PixelTree.h>

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
    Debug("----");
    Debug("Currently %d trainingdata are allocated.", _data_pointers.size());
    for(auto &ptr : _data_pointers) {
        auto str = Meta::toStr(*ptr);
        Debug("\t%S", &str);
    }
    Debug("----");
#endif
}

void add_pointer(TrainingData* data) {
#ifndef NDEBUG
    {
        std::lock_guard<std::mutex> guard(_data_pointer_mutex);
        _data_pointers.insert(data);
    }
    
    TrainingData::print_pointer_stats();
#endif
}

void remove_pointer(TrainingData* data) {
#ifndef NDEBUG
    {
        std::lock_guard<std::mutex> guard(_data_pointer_mutex);
        auto str = Meta::toStr(*data);
        if(_data_pointers.count(data) == 0)
            U_EXCEPTION("Cannot find pointer to %S", &str);
        Debug("Removing %S", &str);
        _data_pointers.erase(data);
    }
    
    TrainingData::print_pointer_stats();
#endif
}

std::string TrainingFilterConstraints::toStr() const {
    return "TFC<l:" + Meta::toStr(median_midline_length_px) + "+-" + Meta::toStr(midline_length_px_std) + " pts:" + Meta::toStr(median_number_outline_pts) + "+-" + Meta::toStr(outline_pts_std) + " angle:" + Meta::toStr(median_angle_diff) + ">";
}

TrainingData::TrainingData(const MidlineFilters& filters)
    : _filters(filters)
{
    add_pointer(this);
    
    auto normalize = SETTING(recognition_normalization).value<default_config::recognition_normalization_t::Class>();
    if(!FAST_SETTINGS(calculate_posture) && normalize == default_config::recognition_normalization_t::posture)
        normalize = default_config::recognition_normalization_t::moments;
    
    set_normalized(normalize);
}

TrainingData::~TrainingData() {
    remove_pointer(this);
}

std::string TrainingData::toStr() const {
    return "TrainingData<norm:"+Meta::toStr(_normalized)+"' path:"+save_path().str()+"' size:"+Meta::toStr(size())+" ranges:"+Meta::toStr(_data)+">";
}

std::string TrainingData::DataRange::toStr() const {
    std::stringstream ss;
    ss << "(";
    if(salty) {
        ss << "SALT)";
    } else {
        bool first = true;
        for(auto && [id, data] : mappings) {
            std::set<long_t> frames(data.frame_indexes.begin(), data.frame_indexes.end());
            if(!first)
                ss << ", ";
            else
                first = false;
            
            ss << id << "=[" << (frames.empty() ? "" : (Meta::toStr(*frames.begin()) + "-" + Meta::toStr(*frames.rbegin()))) << "]";
        }
        
        if(!applied_mapping.empty())
            ss << " map:" << Meta::toStr(applied_mapping);
        
        ss << ")";
    }
    
    return ss.str();
}

void TrainingData::add_frame(std::shared_ptr<TrainingData::DataRange> data, long_t frame_index, Idx_t id, int64_t original_id, Image::Ptr image, const Vec2 & pos, size_t px, const FrameRange& from_range)
{
    /*auto it = original_ids_check.find(image.get());
    if(it  == original_ids_check.end())
        original_ids_check[image.get()] = id;
    else if(it->second != id) {
        Warning("Changed identity of %d from %d to %d without notice", image->index(), it->second, id);
    }*/
    
    if(!image->custom_data() || static_cast<TrainingImageData*>(image->custom_data())->original_id != original_id) {
        auto str = Meta::toStr(data->applied_mapping);
        Except("mapping: %S", &str);
        U_EXCEPTION("individual %d frame %d with original_id == %d != %d (generated in '%s')", id, frame_index, image->custom_data() ? static_cast<TrainingImageData*>(image->custom_data())->original_id : -1, original_id, image->custom_data() ? static_cast<TrainingImageData*>(image->custom_data())->source.c_str() : "");
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
                //Except("\tFound frame %d already in range %d-%d (%d-%d)", frame_index, range.start(), range.end(), from_range.start(), from_range.end());
                return;
            }
        }
    }
    
    if(_data.find(data) == _data.end()) {
        //Debug("Adding new DataRange to TrainingData.");
        _data.insert(data);
    }
    
    auto &obj = data->mappings[id];
    //assert(obj.frame_indexes.empty() || obj.frame_indexes.back() < frame_index);
    
    if(_included_segments.find(id) == _included_segments.end())
        _included_segments[id] = {};
    if(_included_segments.at(id).find(from_range) == _included_segments.at(id).end()) {
        _included_segments.at(id).insert(from_range);
        //Debug("\t[TrainingData] Inserting range %d-%d for individual %d", from_range.start(), from_range.end(), id);
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
    
    if(frame_index > data->frames.end)
        data->frames.end = frame_index;
    if(data->frames.start == -1 || frame_index < data->frames.start)
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
            U_EXCEPTION("Cannot map salty data.");
        
        if(!data->applied_mapping.empty()) {
            auto str = Meta::toStr(_included_segments);
            U_EXCEPTION("Cannot apply two mappings to range %S.", &str);
        }
        
        if(!data->salty) {
            auto str = Meta::toStr(mapping);
            Debug("Changed mapping with %S for %d-%d", &str, data->frames.start, data->frames.end);
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
                Warning("\tCannot find what id %d maps to in applied mapping. Defaulting to same->same.", id);
                
                for(auto && [from, to] : data->applied_mapping) {
                    if(to == id) {
                        U_EXCEPTION("Cannot map %d -> %d and also %d -> %d.", id, id, from, to);
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

Image::UPtr TrainingData::draw_coverage(const std::map<Frame_t, float>& unique_percentages, const std::vector<Rangel>& next_ranges, const std::vector<Rangel>& added_ranges, const std::map<Frame_t, float>& uniquenesses_temp, std::shared_ptr<TrainingData::DataRange> current_salt, const std::map<Rangel, std::tuple<double, FrameRange>>& assigned_unique_averages) const
{
    auto analysis_range = Tracker::analysis_range();
    auto image = Image::Make(500, 1800, 4);
    auto mat = image->get();
    mat = cv::Scalar(0, 0, 0, 0);
    
    std::map<long_t, gui::Color> colors;
    
    int rows = cmn::max(1, (long_t)FAST_SETTINGS(manual_identities).size());
    
    float row_height = float(image->rows) / float(rows);
    float column_width = float(image->cols) / float(analysis_range.length());
    
    auto draw_range = [column_width, row_height, analysis_range, &colors](cv::Mat& mat, const Rangel& range, Idx_t id, uint8_t alpha = 200){
        Vec2 topleft((range.start - analysis_range.start) * column_width, row_height * id);
        Vec2 bottomright((range.end - analysis_range.start) * column_width, row_height * (id + 1));
        cv::rectangle(mat, topleft, bottomright, colors[id].alpha(alpha * 0.5), cv::FILLED);
    };
    
    
    ColorWheel wheel;
    for(auto id : FAST_SETTINGS(manual_identities)) {
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
            std::set<long_t> frames(per.frame_indexes.begin(), per.frame_indexes.end());
            Rangel range(-1, -1);
            
            for(auto f : frames) {
                if(range.end == -1 || f - range.end > 1) {
                    if(range.end != -1) {
                        draw_range(mat, range, id);
                    }
                    
                    range = Rangel(f, f);
                } else
                    range.end = f;
            }
            
            if(range.end != -1) {
                draw_range(mat, range, id);
            }
        }
    }
    
    std::vector<Vec2> unique_points;
    for(auto && [frame, p] : unique_percentages) {
        Vec2 topleft((frame - analysis_range.start) * column_width, mat.rows * (1 - p) + 1);
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
            Vec2 topleft((frame - analysis_range.start) * column_width, mat.rows * (1 - p) + 1);
            unique_points.push_back(topleft);
        }
        
        smooth(unique_points, gui::White);
    }
    
    if(!added_ranges.empty()) {
        for(auto &range : added_ranges) {
            Vec2 topleft((range.start - analysis_range.start) * column_width, 0);
            Vec2 bottomright((range.end - analysis_range.start + 1) * column_width, 1);
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
            
            Vec2 topleft((next_range.start - analysis_range.start) * column_width, 0);
            Vec2 bottomright((next_range.end - analysis_range.start + 1) * column_width, 1);
            DEBUG_CV(cv::rectangle(mat, topleft, bottomright, color.alpha(100 + 100), cv::FILLED));
            
            if(it == --next_ranges.rend())
                DEBUG_CV(cv::putText(mat, "next: "+Meta::toStr(next_range), (topleft + (bottomright - topleft) * 0.5) + Vec2(10), cv::FONT_HERSHEY_PLAIN, 0.5, color));
            
            if(assigned_unique_averages.count(next_range)) {
                auto && [distance, extended_range] = assigned_unique_averages.at(next_range);
                
                Vec2 rtl((extended_range.start() - analysis_range.start) * column_width, (1 - distance / 110.0) * mat.rows + 5);
                Vec2 rbr((extended_range.end() - analysis_range.start + 1) * column_width, (1 - distance / 110.0) * mat.rows + 2 + 5);
                DEBUG_CV(cv::rectangle(mat, rtl, rbr, color));
                
                DEBUG_CV(cv::line(mat, Vec2(rtl.x, rtl.y), Vec2(topleft.x, bottomright.y), gui::Cyan));
                DEBUG_CV(cv::line(mat, Vec2(rbr.x, rtl.y), Vec2(bottomright.x, bottomright.y), gui::Cyan.alpha(50)));
            }
        }
    }
    
    if(current_salt) {
        for(auto && [id, per] : current_salt->mappings) {
            for(size_t i=0; i<per.frame_indexes.size(); ++i) {
                Vec2 topleft((per.frame_indexes[i] - analysis_range.start) * column_width, row_height * (id + 0.2));
                Vec2 bottomright((per.frame_indexes[i] - analysis_range.start + 1) * column_width, row_height * (id + 0.8));
                DEBUG_CV(cv::rectangle(mat, topleft, bottomright, gui::White.alpha(100 + 100), cv::FILLED));
            }
        }
    }
    
    for(auto id : all_classes())
        DEBUG_CV(cv::putText(mat, Meta::toStr(id), Vec2(10, row_height * (id + 0.25)), cv::FONT_HERSHEY_PLAIN, 0.75, gui::White));
    
    return image;
}

void TrainingData::merge_with(std::shared_ptr<TrainingData> other, bool unmap_everything) {
    if(!other) {
        Warning("Cannot merge with nullptr.");
        return;
    }
    
    auto me = Meta::toStr(*this);
    auto he = Meta::toStr(*other);
    Debug("[TrainingData] Merging %S with %S.", &me, &he);
    
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
    
    std::map<long_t, std::set<FrameRange>> before_ranges;
    for(auto & mptr : data()) {
        for(auto && [id, per] : mptr->mappings)
            before_ranges[id].insert(per.ranges.begin(), per.ranges.end());
    }
    
    std::map<long_t, size_t> added_images;
    
    for(auto & ptr : other->data()) {
        // skip salt ranges
        if(ptr->salty)
            continue;
        
        auto new_ptr = std::make_shared<DataRange>(false);
        if(!unmap_everything)
            new_ptr->applied_mapping = ptr->applied_mapping;
        
        for(auto && [id, per] : ptr->mappings) {
            std::set<long_t> added_frame_indexes;
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
                        U_EXCEPTION("Cannot find a range that frame %d belongs to in %S", frame, &str);
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
                if(_included_segments[id].find(range) == _included_segments[id].end()) {
                    _included_segments.at(id).insert(range);
                }
            }
        }
    }
    
    auto str = Meta::toStr(added_images);
    
    me = Meta::toStr(*this);
    Debug("[TrainingData] Finished merging: %S (added images: %S)", &me, &str);
    
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

std::tuple<std::vector<Image::Ptr>, std::vector<Idx_t>> TrainingData::join_arrays() const {
    std::vector<Image::Ptr> images;
    std::vector<Idx_t> ids;
    
    const size_t L = size();
    ids.reserve(L);
    images.reserve(L);
    
    using fdx_t = long_t;
    using frame_t = long_t;
    
    // sanity checks
    std::map<fdx_t, std::set<frame_t>> added_data;
    
    if(_data.size() > 1)
        Debug("Joining TrainingData, expecting %d images from %d arrays.", L, _data.size());
    
    for(auto & d : _data) {
        // ignore salt
        //if(d->salty)
        //    continue;
        //Debug("\tadding range [%d-%d]...", r.start, r.end);
        images.insert(images.end(), d->images.begin(), d->images.end());
        ids.insert(ids.end(), d->ids.begin(), d->ids.end());
    }
    
    if(L != images.size())
        Warning("Only added %d / %d possible images from %d arrays.", images.size(), L, _data.size());
    
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
        Debug("Joining TrainingData, expecting %d images from %d arrays.", L, _data.size());
    
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
        Warning("Only added %d / %d possible images from %d arrays.", result.training_images.size() + result.validation_images.size(), L, _data.size());
    
    return result;
}

template< typename T >
typename std::vector<T>::iterator
   insert_sorted( std::vector<T> & vec, T const& item )
{
    return vec.insert
        (
            std::upper_bound( vec.begin(), vec.end(), item ),
            item
        );
}

Idx_t TrainingData::DataRange::map(Idx_t id) const {
    if(applied_mapping.empty()) return id;
    return applied_mapping.at(id);
}

Idx_t TrainingData::DataRange::unmap(Idx_t id) const {
    if(applied_mapping.empty()) return id;
    for (auto && [original, mapped] : applied_mapping) {
        if(mapped == id) {
            Debug("\treversing applied mapping %d -> %d", mapped, original);
            return original;
        }
    }
    U_EXCEPTION("Cannot find mapping for id == %d. Incomplete mapping.", id);
}

void TrainingData::DataRange::reverse_mapping() {
    if(salty)
        U_EXCEPTION("Cannot unmap salty data.");
    
    if(applied_mapping.empty())
        return;
    
    auto str = Meta::toStr(applied_mapping);
    Debug("Reversing mapping with %S for %d-%d", &str, frames.start, frames.end);
    
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
            U_EXCEPTION("Cannot add two salts.");
        
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
        long_t N = 0;
        for (auto &range : combined) {
            N += range.length();
        }
        
        Debug("\t(salt) %d: new salt N=%d", id, N);
    }
    
    size_t maximum_samples_per_individual = 0;
    for(auto && [id, samples] : individual_samples) {
        auto sum = std::get<0>(samples) + std::get<1>(samples);
        if(sum > maximum_samples_per_individual)
            maximum_samples_per_individual = sum;
    }
    
    std::map<Idx_t, std::set<std::tuple<FrameRange, const DataRange::PerIndividual*, const DataRange*, long_t>>> ranges_to_add;
    std::map<Idx_t, std::set<long_t>> source_frames_per_individual;
    
    auto add_range = std::make_shared<DataRange>(true);
    for(auto &d : data()) {
        if(!d->applied_mapping.empty()) {
            add_range->applied_mapping = d->applied_mapping;
            auto str = Meta::toStr(add_range->applied_mapping);
            Debug("add_range->applied_mappig = %S", &str);
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
            
            source_frames_per_individual[id].insert(per.frame_indexes.begin(), per.frame_indexes.end());
        }
    }
    
    // add maximum of Nmax images per individual
    std::map<Idx_t, std::tuple<size_t, size_t, size_t, size_t>> individual_added_salt;
    std::map<Idx_t, std::tuple<size_t, size_t>> individual_samples_before_after;
    
    const double number_classes = SETTING(track_max_individuals).value<uint32_t>();
    const double gpu_max_sample_mb = double(SETTING(gpu_max_sample_gb).value<float>()) * 1000;
    const Size2 output_size = SETTING(recognition_image_size);
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
        long_t N = 0;
        for (auto &range : combined_ranges_per_individual[id]) {
            N += range.length();
        }
        
        //Debug("\tNumber of frames in global ranges = %d", N);
        
        size_t SR = 0;
        for(auto && [range, ptr, d, ID] : ranges)
            SR += ptr->frame_indexes.size();
        
        //auto id = add_range->applied_mapping.empty() ? ID : add_range->applied_mapping.at(ID);
        
        individual_samples_before_after[id] = {training_samples + validation_samples, 0};
        //auto sum = training_samples + validation_samples;
        individual_added_salt[id] = {0, 0, training_samples, validation_samples};
        
        size_t actually_added = 0;
        for(auto && [range, ptr, d, ID] : ranges) {
            size_t step_size = max(1u, (size_t)ceil(SR / (max_images_per_class * double(range.length()) / double(N))));
            
            std::vector<std::tuple<long_t, Image::Ptr, Vec2, size_t>> frames;
            for(size_t i=0; i<ptr->frame_indexes.size(); ++i) {
                if(range.contains(ptr->frame_indexes[i]))
                    frames.push_back({ptr->frame_indexes[i], ptr->images[i], ptr->positions[i], ptr->num_pixels[i]});
            }
            
            size_t L = frames.size();
            
            for(size_t i=0; i<L; i+=step_size) {
                auto && [frame, image, pos, px] = frames[i];
                
                if(!image->custom_data() || static_cast<TrainingImageData*>(image->custom_data())->original_id != ID) {
                    if(!image->custom_data()) {
                        U_EXCEPTION("No custom_data.");
                    } else {
                        auto str = Meta::toStr(d->applied_mapping);
                        auto str0 = Meta::toStr(add_range->applied_mapping);
                        Except("mapping: %S", &str);
                        Except("mapping_2: %S", &str0);
                        Except("individual %d frame %d with original_id == %d != %d (generated in '%s', currently '%S'), %d", id, frame, static_cast<TrainingImageData*>(image->custom_data())->original_id, ID,  static_cast<TrainingImageData*>(image->custom_data())->source.c_str(), &purpose, d->salty ? 1 : 0);
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
        
        Debug("\t(salt) Individual %d (N=%d): added a total of %d / %.0f frames (%d training, %d validation)", id, N, actually_added, max_images_per_class, std::get<0>(individual_added_salt[id]), std::get<1>(individual_added_salt[id]));
        
        std::get<1>(individual_samples_before_after[id]) = std::get<0>(individual_samples_before_after[id]) + std::get<0>(individual_added_salt[id]) + std::get<1>(individual_added_salt[id]);
    }
    
    auto str = Meta::toStr(individual_added_salt);
    auto after = Meta::toStr(individual_samples_before_after);
    Debug("Added salt (maximum_samples_per_individual = %d, Nmax = %d): %S -> %S", maximum_samples_per_individual, Nmax, &str, &after);
    
    return add_range;
}

bool TrainingData::generate(const std::string& step_description, pv::File & video_file, std::map<Frame_t, std::set<Idx_t> > individuals_per_frame, const std::function<void(float)>& callback, const TrainingData* source) {
    auto frames = extract_keys(individuals_per_frame);
    
    Tracker::LockGuard guard("generate_training_data");
    PPFrame video_frame;
    const Size2 output_size = SETTING(recognition_image_size);
    const auto& custom_midline_lengths = filters();
    
    std::map<Idx_t, std::set<Frame_t>> illegal_frames;
    
    for(auto && [frame, ids] : individuals_per_frame) {
        for(auto id : ids) {
            auto filters = custom_midline_lengths.has(id, frame)
                ? custom_midline_lengths.get(id, frame)
                : TrainingFilterConstraints();
            
            auto fish = Tracker::individuals().at(id);
            auto && [basic, posture] = fish->all_stuff(frame);
            
            if(!Recognition::eligible_for_training(basic, posture, filters)
               || !basic || basic->blob.split())
            {
                illegal_frames[id].insert(frame);
            }
        }
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
    
    const double number_classes = FAST_SETTINGS(manual_identities).size();
    const double gpu_max_sample_gb = double(SETTING(gpu_max_sample_gb).value<float>());
    double percentile = ceil((most_samples - fewest_samples) * 0.65 + fewest_samples); // 0.65 percentile of #images/class
    const double current_filesize_per_class = percentile * (double)output_size.width * (double)output_size.height * 4;
    
    Debug("Fewest samples for an individual is %.0f samples, most are %.0f. 65%% percentile is %f", fewest_samples, most_samples, percentile);
    if(current_filesize_per_class * number_classes / 1000.0 / 1000.0 / 1000.0 >= gpu_max_sample_gb)
    {
        percentile = ceil(gpu_max_sample_gb * 1000 * 1000 * 1000 / (double)output_size.width / (double)output_size.height / 4.0 / double(number_classes));
    
        auto str = Meta::toStr(FileSize{uint64_t(current_filesize_per_class)});
        Debug("\tsample size resource limit reached (with %S / class in the 65 percentile, limit is %.1fGB overall), limiting to %.0f images / class...", &str, gpu_max_sample_gb, percentile);
    }
    
    // sub-sample any classes that need sub-sampling
    for(auto && [id, L] : lengths) {
        if(L > percentile) {
            auto step_size = percentile / L;
            //if(step_size == 1)
            //    continue;
            
            Debug("\tsub-sampling class %d from %f to %f with step_size = %f (resulting in %f)", id, L, percentile, step_size, double(L) * step_size);
            
            // collect all frames where this individual is present
            
            std::set<Frame_t> empty_frames;
            size_t count = 0, after = 0;
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
                    ++count;
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
    
    auto str = Meta::toStr(lengths);
    Debug("L: %S", &str);
    
    auto data = std::make_shared<TrainingData::DataRange>();
    
    size_t i = 0;
    size_t counter = 0, failed = 0;
    Size2 minmum_size(FLT_MAX), maximum_size(0);
    Median<Float2_t> median_size_x, median_size_y;
    long_t inserted_start = std::numeric_limits<long_t>::max(), inserted_end = -1;
    
    // copy available images to map for easy access
    std::map<Idx_t, std::map<long_t, std::tuple<long_t, Image::Ptr>>> available_images;
    if(source) {
        for(auto & data : source->data()) {
            for(auto && [id, per] : data->mappings) {
                auto ID = data->unmap(id);
                auto &sub = available_images[ID];
                
                for(size_t i=0; i<per.images.size(); ++i) {
                    if(sub.find(per.frame_indexes[i]) != sub.end()) {
                        bool ptrs_equal = std::get<1>(sub[per.frame_indexes[i]]) == per.images[i];
                        if(!ptrs_equal || !data->salty) {
                            Except("Double image (%d) frame %d for individual %d in training data (generated in '%S' with current purpose '%S').", ptrs_equal ? 1 : 0, per.frame_indexes[i], id, &static_cast<TrainingImageData*>(per.images[i]->custom_data())->source, &step_description);
                        } else if(data->salty) {
                            Warning("Double image (%d) frame %d for individual %d in training data (generated in '%S' with current purpose '%S').", ptrs_equal ? 1 : 0, per.frame_indexes[i], id, &static_cast<TrainingImageData*>(per.images[i]->custom_data())->source, &step_description);
                        }
                    }
                    
                    if(per.images[i]->custom_data()) {
                        if( static_cast<TrainingImageData*>( per.images[i]->custom_data() )->original_id != ID )
                        {
                            Except("%d != %d (generated in '%S' with current purpose '%S')",ID,  static_cast<TrainingImageData*>(per.images[i]->custom_data())->original_id, &static_cast<TrainingImageData*>(per.images[i]->custom_data())->source, &step_description);
                        }
                        
                    } else
                        U_EXCEPTION("No labeling for image.");
                    
                    sub[per.frame_indexes[i]] = {id, per.images[i]};
                }
            }
        }
    }
    
    for(auto && [id, sub] : available_images) {
        Debug("\t%d: %d available images between %d and %d", id, sub.size(), sub.empty() ? -1 : sub.begin()->first, sub.empty() ? -1 : sub.rbegin()->first);
    }
    
    size_t N_validation_images = 0, N_training_images = 0;
    size_t N_reused_images = 0;
    const bool calculate_posture = FAST_SETTINGS(calculate_posture);
    std::map<Idx_t, std::vector<std::tuple<long_t, Image::Ptr>>> individual_training_images;
    size_t failed_blobs = 0, found_blobs = 0;
    
    for(auto frame : frames) {
        if(individuals_per_frame.find(frame) == individuals_per_frame.end()) {
            ++i;
            continue;
        }
        
        if(frame < 0 || (size_t)frame >= video_file.length()) {
            ++i;
            Except("Frame %d out of range.", frame);
            continue;
        }
        
        // check so that we do not generate images again, that we have generated before
        std::set<Idx_t> filtered_ids;
        
        for(auto id : FAST_SETTINGS(manual_identities)) {
            if(individuals_per_frame.at(frame).find(Idx_t(id)) != individuals_per_frame.at(frame).end())
                filtered_ids.insert(Idx_t(id));
        }
        
        if(frame < inserted_start)
            inserted_start = frame;
        if(frame > inserted_end)
            inserted_end = frame;
        
        for (auto id : filtered_ids) {
            assert(individuals_per_frame.count(frame) && individuals_per_frame.at(frame).find(id) != individuals_per_frame.at(frame).end());
            
            if(!available_images.empty()) {
                auto fit = available_images[id].find(frame);
                if(fit != available_images[id].end()) {
                    auto fish = Tracker::individuals().at(id);
                    auto it = fish->iterator_for(frame);
                    if(it == fish->frame_segments().end())
                        continue;
                    
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
                        break;
                    }
                    
                    continue;
                }
            }
        }
        
        if(individuals_per_frame.find(frame) == individuals_per_frame.end()) {
            ++i;
            continue;
        }
        
        auto active =
            frame == Tracker::start_frame()
                ? std::unordered_set<Individual*>()
                : Tracker::active_individuals(frame-1);
        
        video_file.read_frame(video_frame.frame(), frame);
        Tracker::instance()->preprocess_frame(video_frame, active, NULL);
        
        for (auto id : filtered_ids) {
            /**
             * Check various conditions for whether the image is eligible for
             * training.
             *  - has to have a proper posture
             *  - it mustn't be a split blob
             *  - it must be within recognition bounds
             *  - size of the blob must fit within the given output_size
             */
            
            if(!individuals_per_frame.empty() && individuals_per_frame.at(frame).find(id) == individuals_per_frame.at(frame).end())
                continue;
            
            auto fish = Tracker::individuals().at(id);
            auto filters = custom_midline_lengths.has(id)
                ? custom_midline_lengths.get(id, frame)
                : TrainingFilterConstraints();
            
            auto it = fish->iterator_for(frame);
            if(it == fish->frame_segments().end())
                continue;
            
            auto bidx = (*it)->basic_stuff(frame);
            auto pidx = (*it)->posture_stuff(frame);
            if(bidx == -1 || (pidx == -1 && calculate_posture))
                continue;
            
            /*if(!available_images.empty()) {
                auto fit = available_images[id].find(frame);
                if(fit != available_images[id].end()) {
                    auto&& [ID, image] = fit->second;
                    add_frame(data, frame, id, id, image, Vec2(), fish->thresholded_size(frame), *it->second);
                    if(image_is(image, ImageClass::TRAINING))
                        ++N_training_images;
                    else
                        ++N_validation_images;
                    
                    ++N_reused_images;
                    continue;
                }
            }*/
            
            auto &basic = fish->basic_stuff()[bidx];
            auto posture = pidx != -1 ? fish->posture_stuff()[pidx] : nullptr;
            
            if(!Recognition::eligible_for_training(basic, posture, filters))
                continue;

            auto bid = basic->blob.blob_id();
            auto pid = basic->blob.parent_id;
            
            auto blob = Tracker::find_blob_noisy(video_frame, bid, pid, basic->blob.calculate_bounds());
            if(!blob)
                ++failed_blobs;
            else
                ++found_blobs;
            if(!blob || blob->split())
                continue;
            
            ++counter;
            median_size_x.addNumber(blob->bounds().size().width);
            median_size_y.addNumber(blob->bounds().size().height);
            minmum_size = min(minmum_size, blob->bounds().size());
            maximum_size = max(maximum_size, blob->bounds().size());
            
            // try loading it all into a vector
            Image::Ptr image;
            
            /*auto iit = did_image_already_exist.find({id, frame});
            if(iit != did_image_already_exist.end()) {
                // this image was already created
                Warning("Creating a second instance of id %d in frame %d", id, frame);
            }*/
            
            using namespace default_config;
            auto midline = posture ? fish->calculate_midline_for(basic, posture) : nullptr;
            Recognition::ImageData image_data(Recognition::ImageData::Blob{
                blob->num_pixels(), 
                blob->blob_id(), 
                pv::bid::invalid, 
                blob->parent_id(), 
                blob->bounds()
            }, frame, (FrameRange)*it->get(), fish, fish->identity().ID(), midline ? midline->transform(normalized()) : gui::Transform());
            image_data.filters = std::make_shared<TrainingFilterConstraints>(filters);
            
            image = std::get<0>(Recognition::calculate_diff_image_with_settings(normalized(), blob, image_data, output_size));
            
            if(blob->bounds().width > output_size.width
               || blob->bounds().height > output_size.height)
            {
                ++failed;
            }
            
            if(image != nullptr) {
                image->set_index(frame);
                
                assert(!image->custom_data());
                image->set_custom_data(new TrainingImageData("generate("+Meta::toStr(fish->identity().ID())+" "+step_description+")", id));
                
                set_image_class(image, ImageClass::TRAINING);
                ++N_training_images;
                
                if(frame > 0)
                    individual_training_images[id].push_back({frame, image});
                
                add_frame(data, frame, id, id, image, Vec2(), fish->thresholded_size(frame), *it->get());
            }
        }
        
        callback(++i / float(frames.size()));
    }
    
    Debug("Failed blobs: %d Found blobs: %d", failed_blobs, found_blobs);
    
    if(failed) {
        auto prefix = SETTING(individual_prefix).value<std::string>();
        Warning("Some (%d%%) %S images are too big. Range: [%.0fx%.0f, %.0fx%.0f] median %.0fx%.0f", failed * 100 / counter, &prefix, minmum_size.width, minmum_size.height, maximum_size.width, maximum_size.height, median_size_x.getValue(), median_size_y.getValue());
    }
    
    lengths.clear();
    std::map<long_t, std::map<ImageClass, size_t>> individual_image_types;
    for(auto &d : this->data()) {
        for(auto && [id, per] : d->mappings) {
            lengths[id] += per.images.size();
            for(auto & image : per.images)
                ++individual_image_types[id][image_class(image)];
        }
    }
    
    Debug("[TrainingData] We collected %d training images and %d validation images (%d reused). Checking individuals...", N_training_images, N_validation_images, N_reused_images);
    for(auto && [id, L] : lengths) {
        const size_t expected_number_validation = floor(0.25 * L);
        auto N_validation_images = individual_image_types[id][ImageClass::VALIDATION];
        if(N_validation_images < expected_number_validation) {
            auto &trainings = individual_training_images[id];
            auto available = individual_image_types[id][ImageClass::TRAINING];
            if(available < expected_number_validation - N_validation_images) {
                Error("\tCan only find %d of the %d needed images.", available, expected_number_validation - N_validation_images);
            } else {
                Debug("\tFinding more (%d) validation images to reach %d samples from %d available images.", expected_number_validation - N_validation_images, expected_number_validation, available);
                size_t step_size = max(1u, available / (expected_number_validation - N_validation_images));
                size_t N_selected = 0;
                for(size_t i=0; i<trainings.size(); i += step_size) {
                    assert(image_is(std::get<1>(trainings[i]), ImageClass::TRAINING));
                    set_image_class(std::get<1>(trainings[i]), ImageClass::VALIDATION);
                    ++N_selected;
                }
                Debug("\tSelected %d new images (%d / %d)", N_selected, N_selected + N_validation_images, expected_number_validation);
            }
        }
    }
    
    return N_training_images + N_validation_images > 0;
}

std::tuple<std::vector<Image::Ptr>, std::vector<Idx_t>, std::vector<long_t>, std::map<Frame_t, Range<size_t>>> TrainingData::join_arrays_ordered() const
{
    using fdx_t = Idx_t;
    using frame_t = long_t;
    
    std::vector<Image::Ptr> images;
    std::vector<fdx_t> ids;
    std::vector<frame_t> frames;
    
    const size_t L = size();
    ids.reserve(L);
    images.reserve(L);
    frames.reserve(L);
    
    std::map<frame_t, std::tuple<std::vector<fdx_t>, std::vector<Image::Ptr>>> collector;
    
    if(_data.size() > 1)
        Debug("Joining TrainingData, expecting %d images from %d arrays.", L, _data.size());
    
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
        start_indexes[Frame_t(frame)] = Range<size_t>(ids.size(), ids.size() + _ids.size());
        
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
