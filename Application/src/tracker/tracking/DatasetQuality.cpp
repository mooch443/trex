#include "DatasetQuality.h"
#include <tracking/Individual.h>
#include <gui/gui.h>
#include <tracking/Recognition.h>

namespace track {
std::string DatasetQuality::Single::toStr() const {
    return "{"+Meta::toStr(id)+","+Meta::toStr(distance_travelled)+" travelled,"+Meta::toStr(grid_cells_visited)+" cells visited}";
}

DatasetQuality::DatasetQuality()
:_manually_selected({},{})
{
    
}

void DatasetQuality::print_info() const {
    print("DatasetQuality: ", _sorted);
}

DatasetQuality::Quality DatasetQuality::quality(const Range<Frame_t> &range) const {
    auto it = _quality.find(range);
    if(it == _quality.end())
        return Quality();
    
    return it->second;
}

/*float DatasetQuality::quality(float frame) const {
    for(auto && [segment, value] : _quality) {
        if(segment.contains(frame) || segment.end == frame)
            return value;
    }
    return -1;
}*/

void DatasetQuality::remove_frames(Frame_t start) {
    for(auto it = _sorted.begin(); it != _sorted.end();) {
        if(it->range.end >= start)
            it = _sorted.erase(it);
        else
            ++it;
    }
    
    for(auto it = _cache.begin(); it != _cache.end();) {
        if(it->first.end >= start)
            it = _cache.erase(it);
        else
            ++it;
    }
    
    for(auto it = _quality.begin(); it != _quality.end();) {
        if(it->first.end >= start)
            it = _quality.erase(it);
        else
            ++it;
    }
    
    _manually_selected = Range<Frame_t>({},{});
    _last_seen = Range<Frame_t>({}, {});
}

bool DatasetQuality::calculate_segment(const Range<Frame_t> &consec, const uint64_t video_length, const Tracker::LockGuard& guard) {
    if(consec.length().get() < 5) {
        return true; // skipping range because its too short, but send "ok" signal
    }
    
    //auto str = Meta::toStr(consec);
    
    float max_cells = 0, min_cells = infinity<float>();
    float average_samples = 0;
    float num_average = 0;
    decltype(_cache)::mapped_type map;
    std::mutex thread_mutex;
    
    auto work = [&consec, &guard, &thread_mutex, this, &max_cells, &min_cells, &num_average, &average_samples, &map](Individual *fish) {
        // collect meta information for the currently selected best consecutive frames
        auto single = evaluate_single(fish->identity().ID(), fish, consec, guard);
        
        std::lock_guard<std::mutex> guard(thread_mutex);
        average_samples += single.number_frames;
        ++num_average;
        
        auto cells = single.grid_cells_visited;
        if(cells > max_cells)
            max_cells = cells;
        if(cells < min_cells)
            min_cells = cells;
        
        map[fish->identity().ID()] = single;
    };
    
    std::set<Individual*> found;
    
    for(auto id : FAST_SETTINGS(manual_identities)) {
        // does the given fish exist yet?
        auto it = Tracker::individuals().find(id);
        if(it == Tracker::individuals().end())
            continue;
        
        auto fish = it->second;
        if(!fish->frame_segments().empty()) {
            auto it = fish->frame_segments().rbegin();
            if((*it)->range.overlaps(consec)) {
                assert((*it)->range.end == fish->end_frame());
                                                                /* implicitly abusing uint64_t(-1) < video_length here */
                if(fish->end_frame() == Tracker::end_frame() && uint64_t(Tracker::end_frame().get()) < video_length)
                    return false;
            }
        }
        
        // if this fish is still found at the end of the analysis range
        /*if(consec.end < video_length && fish->end_frame() == Tracker::end_frame() && consec.end == fish->end_frame())
            return false; // send "not ok" signal
        if(consec.end == fish->end_frame() && fish->end_frame() == Tracker::end_frame() && Tracker::end_frame() < video_length)
            return false;*/
        
        found.insert(fish);
    }
    
    for(auto fish : found)
        Tracker::instance()->thread_pool().enqueue(work, fish);
    
    Tracker::instance()->thread_pool().wait();
    
    if(num_average != 0)
        average_samples /= num_average;
    
    /*std::set<float> values;
    float sum = 0;
    for(auto && [id, data] : map) {
        sum += data.grid_cells_visited;
        values.insert(max_cells > 0 ? (data.grid_cells_visited / max_cells) : 0);
    }*/
    
    _cache[consec] = map;
    
    if(map.empty()) {
        FormatExcept("Values in DatasetQuality::calculate_segment is empty.");
    } else {
        // calculate quality by multiplying the overall number of
        // cells visited by all fish inthe segment by the minimum ratio.
        // so if the segment is average long and has only same-size
        // elements, thats gonna be better than a really long segment
        // where one fish only moves 1 cell.
        _quality[consec] = Quality(consec, min_cells, average_samples);//(*values.begin()) * sum;
        _sorted.insert(_quality[consec]);
        
    }
    
    return true;
}

void DatasetQuality::remove_segment(const Range<Frame_t> &range) {
    auto it = _cache.find(range);
    if(it != _cache.end()) {
        _sorted.erase(range);
        _cache.erase(it);
        _quality.erase(range);
    }
}

void DatasetQuality::update(const Tracker::LockGuard& guard) {
    if(FAST_SETTINGS(manual_identities).empty() || Tracker::instance()->consecutive().empty())
        return;
    
    auto video_length = Tracker::analysis_range().end.get();
    auto end_frame = Tracker::end_frame();
    auto manual = FAST_SETTINGS(manually_approved);
    bool changed = false;
    //Rangel longest(-1, -1);
    
    // search the segments present in current iteration
    for(auto && [start, end] : manual) {
        auto it = _previous_selected.find(Range<Frame_t>(Frame_t(start), Frame_t(end)));
        if(it != _previous_selected.end())
            _previous_selected.erase(it);
    }
    
    // remove all the ones that have been deleted in the manually_approved segments
    for(auto & segment : _previous_selected) {
        if(has(segment)) {
            print("Removed previous manual segment ", segment.start,"-",segment.end);
            remove_segment(segment);
            changed = true;
        }
    }
    
    // calculate segments for current iteration
    for(auto && [start, end] : manual) {
        auto range = Range<Frame_t>(Frame_t(start), Frame_t(end));
        if(!has(range) && end_frame >= Frame_t(end) && range.length().get() >= 5) {
            if(calculate_segment(range, video_length, guard)) {
                print("Calculating manual segment ", start,"-",end);
                for(auto && [id, single] : _cache.at(range)) {
                    print("\t", id,": ",single.number_frames);
                }
                changed = true;
                
            } else
                print("Failed calculating ", start,"-",end);
        }
        _previous_selected.insert(range);
    }
    
    for(auto &consec : Tracker::instance()->consecutive()) {
        if(consec.end.get() != video_length && consec == Tracker::instance()->consecutive().back())
            break;
        
        if(_cache.find(consec) == _cache.end() && consec.length().get() > 5) {
            if(calculate_segment(consec, video_length, guard)) {
                //break; // if this fails, dont set last seen and try again next time
#ifndef NDEBUG
                print("Calculated segment ", consec.start,"-",consec.end);
#endif
                changed = true;
            }
        }
        
        _last_seen = consec;
    }
    
    if(changed)
        Tracker::global_segment_order_changed();
}

Range<Frame_t> DatasetQuality::best_range() const {
    if(!_sorted.empty())
        return _sorted.begin()->range;
    return Range<Frame_t>({},{});
}

std::map<Idx_t, DatasetQuality::Single> DatasetQuality::per_fish(const Range<Frame_t> &range) const {
    auto it = _cache.find(range);
    if(it == _cache.end())
        return {};
    return it->second;
}

bool DatasetQuality::has(const Range<Frame_t>& range) const {
    auto it = _cache.find(range);
    return it != _cache.end() && !it->second.empty();
}

DatasetQuality::Single DatasetQuality::evaluate_single(Idx_t id, Individual* fish, const Range<Frame_t> &_consec, const Tracker::LockGuard&)
{
    //assert(Tracker::individuals().find(id) != Tracker::individuals().end());
    
    constexpr size_t grid_res = 100;
    const Size2 grid_size = Tracker::average().bounds().size() / float(grid_res);
    std::map<std::tuple<uint16_t, uint16_t>, uint32_t> grid_cells;
    
    auto pos2grid = [&grid_size](const Vec2& pos) {
        return pos.div(grid_size).map<round>();
    };
    
    auto access = [&grid_cells](const Vec2& pos) -> uint32_t& {
        return grid_cells[{uint16_t(pos.x), uint16_t(pos.y)}];
    };
    
    std::vector<Vec2> positions;
    std::set<int> orientations;
    grid_cells.clear();
    uint32_t sum = 0;
    
    //std::set<float> midline_lengths, outline_stds;
    //Median<float> midline_median, outline_median;
    
    Vec2 prev(FLT_MAX);
    float travelled = 0;
    
    long_t number_frames = 0;
    bool debug = false;
    
    auto manually_approved = FAST_SETTINGS(manually_approved);
    if(manually_approved.find(_consec.start.get()) != manually_approved.end())
        debug = true;
    
    FrameRange consec(Range<Frame_t>({}, {}));
    auto it = std::lower_bound(fish->frame_segments().begin(), fish->frame_segments().end(), _consec.start, [](const auto& ptr, Frame_t frame) {
        return ptr->start() < frame;
    });
    /*if(debug && it != fish->frame_segments().end())
        print("\t... ", fish->identity().ID()," -> found before == ", it->second.range.start,"-",it->second.range.end);
    else
        print("\t... ", fish->identity().ID()," not found before first step");*/
    
    if(it != fish->frame_segments().end()
       && it != fish->frame_segments().begin()
       && (*it)->start() > _consec.end)
    {
        --it;
    } else if(it != fish->frame_segments().end()
              && (*it)->start() > _consec.start) {
        
    }
    
    /*if(debug && it != fish->frame_segments().end())
        print("\t... ", fish->identity().ID()," -> found it == ", it->second.range.start,"-",it->second.range.end);
    else
        print("\t... ", fish->identity().ID()," not found in first step");*/
    
    if(it == fish->frame_segments().end() && !fish->frame_segments().empty()
       && (*fish->frame_segments().rbegin())->overlaps(_consec))
    {
        it = fish->frame_segments().end();
        --it;
    }
    
    //if(debug && it != fish->frame_segments().end())
    
    while(it != fish->frame_segments().end()
          && it != fish->frame_segments().begin()
          && (*it)->overlaps(_consec))
    {
        auto copy = it;
        --copy;
        
        if((*copy)->overlaps(_consec)) {
            it = copy;
        } else
            break;
    }
    
    /*if(debug && it != fish->frame_segments().end())
        print("\t... ", fish->identity().ID()," -> starting with it == ", it->second.range.start,"-",it->second.range.end);*/
    
    // we found the segment where the start-frame is not smaller than _consec.start
    while(it != fish->frame_segments().end()
          && (*it)->overlaps(_consec))
    {
        if(!consec.range.start.valid())
            consec.range.start = (*it)->start();
        
        consec.range.end = (*it)->end();
        
        ++it;
    }
    
    if(consec.empty()) {
        FormatWarning("consec:[",consec.start(),"-",consec.end(),"] _consec:[",_consec.start,",",_consec.end,"] end()?",it == fish->frame_segments().end() ? 1 : 0," segments:", fish->frame_segments());
        if(it != fish->frame_segments().end()) {
            FormatWarning("\tit:[",(*it)->start(),"-",(*it)->end(),"] overlaps:",(*it)->overlaps(_consec) ? 1 : 0);
        }
        //auto it = fish->frame_segments().lower_bound(_consec.start);
        
        
        FormatExcept("Cannot find frame ", _consec.start," for fish ", fish->identity().ID(),"");
        return Single(id);
    }
    
    /*if(debug) {
        auto str = Meta::toStr(fish->frame_segments());
        std::stringstream ss;
        for(auto && [first, segment] : fish->frame_segments()) {
            if(segment.overlaps(_consec)) {
                ss << "\t\t"<< first <<": [" << segment.start() << "-" << segment.end() << "]\n";
            }
        }
        str = ss.str();
        
        print("\t... ",fish->identity().ID()," -> ",consec.range.start,"-",consec.range.end," (",str.c_str(),")");
    }*/
        
    //! TODO: Use local_midline_length function instead
    auto constraints = Tracker::recognition()->local_midline_length(fish, consec.range, true);
    if(consec.start() > Tracker::start_frame() && fish->centroid_weighted(consec.start() - 1_f)) {
        prev = fish->centroid_weighted(consec.start() - 1_f)->pos<Units::PX_AND_SECONDS>();
    }
    
    fish->iterate_frames(consec.range, [&](Frame_t i, const auto&, const Individual::BasicStuff* basic, auto posture) -> bool
    {
        if(!Recognition::eligible_for_training(basic, posture, constraints))
            return true;
        
        // go through all frames within the segment
        if(basic) {
            auto pos = basic->centroid.pos<Units::PX_AND_SECONDS>();
            auto grid = pos2grid(pos);
            
            ++access(grid);
            ++sum;
            
            ++number_frames;
            
            positions.push_back(pos);
            
            if(posture && posture->outline && posture->cached())
                orientations.insert(std::round(DEGREE(posture->midline_angle)));
            
            if(i > consec.start() && prev.x != FLT_MAX && pos != prev) {
                auto L = ((pos - prev) * FAST_SETTINGS(cm_per_pixel)).length();
                travelled += L;
            }
            
            prev = pos;
        }
        
        return true;
    });
    
    
    Single single(id);
    single.grid_cells_visited = grid_cells.size();
    single.midline_len = constraints.median_midline_length_px;
    single.midline_std = constraints.midline_length_px_std;
    single.distance_travelled = travelled;
    single.outline_len = constraints.median_number_outline_pts;
    single.outline_std = constraints.outline_pts_std;
    single.median_angle_var = constraints.median_angle_diff;
    single.number_frames = number_frames;
    return single;
}
}
