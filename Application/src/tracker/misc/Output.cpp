#include "Output.h"
#include <misc/Timer.h>
#include <tracking/FOI.h>
#include <misc/default_config.h>
#include <tracking/Recognition.h>
#include <lzo/minilzo.h>
#include <gui/gui.h>
#include <gui/WorkProgress.h>
#include <misc/checked_casts.h>
#include <tracking/Categorize.h>

using namespace track;
typedef int64_t data_long_t;

/*IMPLEMENT(Output::ResultsFormat::_blob_pool)(cmn::hardware_concurrency(), [](Individual*obj){
    Timer timer;
    
    
});*/

Output::ResultsFormat::ResultsFormat(const file::Path& filename, std::function<void(const std::string&, float, const std::string&)> update_progress)
    : DataFormat(filename.str()), _update_progress(update_progress), last_callback(0), estimated_size(0),
    _post_pool(cmn::hardware_concurrency(), [this](Individual* obj) {
    // post processing for individuals
    Timer timer;
    //pv::Frame frame;
    //PPFrame output;

    /*for(auto &stuff : obj->_basic_stuff) {
        stuff->blob->calculate_properties();
        stuff->blob->calculate_moments();
    }*/

    if (timer.elapsed() > 1)
        Debug("Blobs took %fs", timer.elapsed());

    timer.reset();

    //Debug("Generate midlines for %d...", obj->identity().ID());
    obj->update_midlines(_property_cache.get());
    //Debug("Done with midlines for %d.", obj->identity().ID());

    //data_long_t previous = obj->start_frame();
    //for(auto && [i, c] : obj->_centroid) {
        /*auto outline = obj->outline(i);
        auto midline = obj->midline(i);

        if(outline && midline) {

            // calculate midline centroid
            Vec2 centroid_point(0, 0);
            auto points = outline->uncompress();

            for (auto &p : points) {
                centroid_point += p;
            }
            centroid_point /= float(points.size());
            centroid_point += obj->_blobs.at(i)->bounds().pos();

            auto enhanced = new PhysicalProperties(prev_enhanced, c->time(), centroid_point, midline->angle());
            obj->_centroid_posture[i] = enhanced;
            prev_enhanced = enhanced;
        }*/

        // save frame segments
        /*obj->push_to_segments(i, previous);
        previous = i;
    }*/

    obj->_local_cache.regenerate(obj);

    if (timer.elapsed() >= 1) {
        auto us = timer.elapsed() * 1000 * 1000;

        auto str = DurationUS{ (uint64_t)us }.to_string();
        auto name = obj->identity().name();

        if (!SETTING(quiet))
            Debug("%S post-processing took %S", &name, &str);
    }
}), _generic_pool(min(4u, cmn::hardware_concurrency()), [this](std::exception_ptr e) {
    _exception_ptr = e; // send it to main thread
}), _expected_individuals(0), _N_written(0)
{}

Output::ResultsFormat::~ResultsFormat() {
    _generic_pool.wait();
    _post_pool.wait();
}

template<> void Data::read(track::FrameProperties& p) {
    read<uint64_t>(p.org_timestamp);
    p.time = p.org_timestamp / double(1000*1000);
    auto *ptr = static_cast<Output::ResultsFormat*>(this);
    if(ptr->header().version >= Output::ResultsFormat::V_31) {
        read_convert<data_long_t>(p.active_individuals);
    } else
        p.active_individuals = -1;
}

template<>
uint64_t Data::write(const track::FrameProperties& val) {
    write<data_long_t>(val.frame);
    write<uint64_t>(val.org_timestamp);
    return write<data_long_t>(val.active_individuals);
}

MinimalOutline::Ptr Output::ResultsFormat::read_outline(Data& ref, Midline::Ptr midline) const {
    MinimalOutline::Ptr ptr = std::make_shared<MinimalOutline>();
    static_assert(MinimalOutline::factor == 10, "MinimalOutline::factor was 10 last time I checked.");
    
    uint64_t L;
    ref.read<uint64_t>(L);
    ptr->_points.resize(L);
    /*if(_header.version > Output::ResultsFormat::Versions::V_9) {
        ptr->_tail_index = ref.read<data_long_t>();
    } else
        ptr->_tail_index = ptr->_points.size() * 0.5;*/
    if(_header.version > Output::ResultsFormat::Versions::V_9 && _header.version < Output::ResultsFormat::Versions::V_24) {
        data_long_t index;
        ref.read<data_long_t>(index);
        midline->tail_index() = (long_t)index;
    } /*else if(_header.version >= Output::ResultsFormat::Versions::V_24) {
        midline->tail_index() = ref.read<data_long_t>();
        midline->head_index() = ref.read<data_long_t>();
    }*/
    
    if(_header.version >= Output::ResultsFormat::Versions::V_17) {
        ref.read_convert<float>(ptr->_first.x);
        ref.read_convert<float>(ptr->_first.y);
        
        ref.read_data(ptr->_points.size() * sizeof(decltype(ptr->_points)::value_type), (char*)ptr->_points.data());
    } else {
        struct Point {
            float x, y;
        };
        
        std::vector<Point> points;
        points.resize(ptr->_points.size());
        ref.read_data(ptr->_points.size() * sizeof(Point), (char*)points.data());
        std::vector<Vec2> vecs;
        for(auto &p : points)
            vecs.push_back(Vec2(p.x, p.y));
        
        ptr->convert_from(vecs);
    }
    
    return ptr;
}

template<>
uint64_t Data::write(const track::MinimalOutline& val) {
    auto p = write<uint64_t>(val._points.size());
    //write<data_long_t>(val.tail_index());
    write<float>(val._first.x);
    write<float>(val._first.y);
    
    static_assert(std::is_same<uint16_t, decltype(MinimalOutline::_points)::value_type>::value, "Assuming that MinimalOutline::_points is an array of uint16_t");
    write_data(val._points.size() * sizeof(decltype(val._points)::value_type), (char*)val._points.data());
    //pack.write<uint64_t>(outline->points.size());
    //pack.write<data_long_t>(outline->tail_index);
    //pack.write_data(outline->points.size() * sizeof(Vec2), (char*)outline->points.data());
    return p;
}

template<>
uint64_t Data::write(const PhysicalProperties& val) {
    /**
     * Format of binary representation:
     *  - POSITION (2x4 bytes) in pixels
     *  - ANGLE (4 bytes)
     * both are float.
     *
     * Derivates etc. can be calculated after loading.
     */
    
    uint64_t p = write<Vec2>(val.pos(Units::PX_AND_SECONDS));
    write<float>(val.angle());
    //write<double>(val.frame());
    
    return p;
}

void Output::ResultsFormat::read_blob(Data& ref, pv::CompressedBlob& blob) const {
    const uint64_t elem_size = sizeof(pv::ShortHorizontalLine);
    
    uint16_t id = UINT16_MAX;
    if(_header.version >= Output::ResultsFormat::Versions::V_4
       && _header.version <= Output::ResultsFormat::Versions::V_11)
    {
        ref.read<uint16_t>(id);
    }
    
    bool split = false, tried_to_split = false;
    uint8_t byte = 0;
    if(_header.version >= Output::ResultsFormat::Versions::V_20) {
        ref.read<uint8_t>(byte);
        split = (byte & 0x1) != 0;
    }
    
    data_long_t parent_id = -1;
    if(_header.version >= Output::ResultsFormat::Versions::V_26) {
        if((byte & 0x2) != 0)
            ref.read<data_long_t>(parent_id);
        if((byte & 0x4) != 0)
            tried_to_split = true;
        
    } else if(split
              && _header.version >= Output::ResultsFormat::Versions::V_22
              && _header.version <= Output::ResultsFormat::Versions::V_25)
        ref.read<data_long_t>(parent_id);
    
    uint16_t start_y, len;
    ref.read<uint16_t>(start_y);
    ref.read<uint16_t>(len);
    
    if(_header.version < Output::ResultsFormat::Versions::V_32) {
        std::vector<pv::LegacyShortHorizontalLine> legacy(len);
        blob.lines.clear();
        blob.lines.reserve(len);
        
        ref.read_data(sizeof(pv::LegacyShortHorizontalLine) * len, (char*)legacy.data());
        std::copy(legacy.begin(), legacy.end(), std::back_inserter(blob.lines));
        
    } else {
        blob.lines.resize(len);
        ref.read_data(elem_size * len, (char*)blob.lines.data());
    }
    
    blob.status_byte = byte;
    blob.start_y = start_y;
    blob.parent_id = (long_t)parent_id;
}

template<>
uint64_t Data::write(const pv::BlobPtr& val) {
    auto &mask = val->hor_lines();
    auto compressed = pv::ShortHorizontalLine::compress(mask);
    
    const uint64_t elem_size = sizeof(pv::ShortHorizontalLine);
    
    // this will turn
    uint8_t byte = (val->parent_id() != -1 ? 0x2 : 0x0)
                   | uint8_t(val->split() ? 0x1 : 0)
                   | uint8_t(val->tried_to_split() ? 0x4 : 0x0);
    uint64_t p = write<uint8_t>(byte);
    if(val->parent_id() != -1)
        write<data_long_t>(val->parent_id());
    write<uint16_t>(uint16_t(mask.empty() ? 0 : mask.front().y));
    write<uint16_t>(uint16_t(compressed.size()));
    write_data(compressed.size() * elem_size, (char*)compressed.data());
    
    return p;
}

Midline::Ptr Output::ResultsFormat::read_midline(Data& ref) {
    auto midline = std::make_shared<Midline>();
    ref.read<float>(midline->len());
    ref.read<float>(midline->angle());
    ref.read<Vec2>(midline->offset());
    ref.read<Vec2>(midline->front());
    if(_header.version >= Versions::V_24) {
        ref.read_convert<data_long_t>(midline->tail_index());
        ref.read_convert<data_long_t>(midline->head_index());
    }
    midline->is_normalized() = _header.version < Versions::V_25;
    
    uint64_t L;
    ref.read<uint64_t>(L);
    midline->segments().resize(L);
    if(_header.version >= Output::ResultsFormat::Versions::V_10) {
        std::vector<Output::V20MidlineSegment> segments;
        segments.resize(midline->segments().size());
        ref.read_data(midline->segments().size() * sizeof(Output::V20MidlineSegment), (char*)segments.data());
        for(uint64_t i=0; i<segments.size(); i++)
            midline->segments()[i] = segments[i];
    } else {
        std::vector<Output::V9MidlineSegment> segments;
        segments.resize(midline->segments().size());
        
        ref.read_data(midline->segments().size() * sizeof(Output::V9MidlineSegment), (char*)segments.data());
        for(uint64_t i=0; i<segments.size(); i++)
            midline->segments()[i] = segments[i];
    }
    
    return midline;
}

template<>
uint64_t Data::write(const Midline& val) {
    auto p = write<float>(val.len());
    write<float>(val.angle());
    write<Vec2>(val.offset());
    write<Vec2>(val.front());
    write<data_long_t>(val.tail_index());
    write<data_long_t>(val.head_index());
    
    write<uint64_t>(val.segments().size());
    
    std::vector<Output::V20MidlineSegment> segments;
    segments.resize(val.segments().size());
    for(uint64_t i=0; i<segments.size(); i++) {
        segments[i].height = val.segments()[i].height;
        segments[i].l_length = val.segments()[i].l_length;
        segments[i].x = (float)val.segments()[i].pos.x;
        segments[i].y = (float)val.segments()[i].pos.y;
    }
    write_data(segments.size() * sizeof(Output::V20MidlineSegment), (char*)segments.data());
    
    return p;
}

Individual* Output::ResultsFormat::read_individual(cmn::Data &ref, const CacheHints* cache) {
    Timer timer;
    
    uint32_t ID;
    
    if(_header.version >= Output::ResultsFormat::Versions::V_5)
        ref.read<uint32_t>(ID);
    else {
        uint16_t sid;
        ref.read<uint16_t>(sid);
        ID = (uint32_t)sid;
    }
    
    Individual *fish = new Individual(ID);
    
    if(_header.version <= Output::ResultsFormat::Versions::V_15) {
        ref.seek(ref.tell() + sizeof(data_long_t) * 2);
        //ref.read<data_long_t>(); // pixel_samples
        //ref.read<data_long_t>();
    }
    
    if(_header.version <= Output::ResultsFormat::Versions::V_13) {
        ref.seek(ref.tell() + sizeof(uint8_t) * 3);
        //ref.read<uchar>(); // jump over colors
        //ref.read<uchar>();
        //ref.read<uchar>();
    }
    
    if(_header.version >= Output::ResultsFormat::Versions::V_7) {
        std::string name;
        ref.read<std::string>(name);
        
        Identity id(ID);
        if(name != id.raw_name() && !name.empty()) {
            auto map = FAST_SETTINGS(individual_names);
            map[ID] = name;
            SETTING(individual_names) = map;
        }
    }
    
    fish->_manually_matched.clear();
    if(_header.version >= Output::ResultsFormat::Versions::V_15) {
        data_long_t tmp;
        uint64_t N;
        ref.read<uint64_t>(N);
        
        for (uint64_t i=0; i<N; ++i) {
            ref.read<data_long_t>(tmp);
            fish->_manually_matched.insert((long_t)tmp);
        }
    }
    
    fish->identity().set_ID(ID);
    
    //PhysicalProperties *prev = NULL;
    //PhysicalProperties *prev_weighted = NULL;
    std::future<void> last_future;
    
    uint64_t N;
    ref.read<uint64_t>(N);
    //double psize = 0;
    //uint64_t pos_before = this->tell();
    data_long_t prev_frame = -1;
    data_long_t frameIndex;
    
    auto analysis_range = Tracker::analysis_range();
    bool check_analysis_range = SETTING(analysis_range).value<std::pair<long_t, long_t>>().first != -1 || SETTING(analysis_range).value<std::pair<long_t, long_t>>().second != -1;
    
    struct TemporaryData {
        std::shared_ptr<Individual::BasicStuff> stuff;
        data_long_t prev_frame;
        Vec2 pos;
        float angle;
    };
    
    std::mutex mutex;
    std::condition_variable variable;
    std::deque<TemporaryData> stuffs;
    std::atomic_bool stop = false;
    
    std::thread worker([&mutex, &variable, &stuffs, &stop, fish, check_analysis_range, analysis_range, cache_ptr = cache]()
    {
        cmn::set_thread_name("Output::ResultsFormat::worker");
        
        std::unique_lock<std::mutex> guard(mutex);
        auto _no_cache = (const CacheHints*)0x1;
        
        while(!stop || !stuffs.empty()) {
            variable.wait_for(guard, std::chrono::milliseconds(250));
            
            while(!stuffs.empty()) {
                auto & data = stuffs.front();
                
                guard.unlock();
                {
                    const long_t& frameIndex = data.stuff->frame;
                    
                    if(fish->_startFrame == -1)
                        fish->_startFrame = frameIndex;
                    fish->_endFrame = frameIndex;
                    
                    auto prop = new PhysicalProperties(fish, frameIndex, data.pos, data.angle, cache_ptr);
                    data.stuff->centroid = prop;
                    
                    auto label = FAST_SETTINGS(track_consistent_categories)/* || !FAST_SETTINGS(track_only_categories).empty()*/ ? Categorize::DataStore::ranged_label(Frame_t(frameIndex), data.stuff->blob) : nullptr;
                    auto cache = fish->cache_for_frame(frameIndex, Tracker::properties(frameIndex, cache_ptr)->time, cache_ptr);
                    auto p = fish->probability(label ? label->id : -1, cache, frameIndex, data.stuff->blob).p;
                    
                    auto segment = fish->update_add_segment(
                        frameIndex,
                        data.stuff->centroid,
                        (long_t)data.prev_frame,
                        &data.stuff->blob,
                        p
                    );
                    
                    segment->add_basic_at(frameIndex, (long_t)fish->_basic_stuff.size());
                    fish->_basic_stuff.push_back(data.stuff);
                }
                guard.lock();
                
                stuffs.pop_front();
            }
        }
    });
    
    TemporaryData data;
    double time;
    
    for (uint64_t i=0; i<N; i++) {
        ref.read<data_long_t>(frameIndex);
        if(prev_frame == -1 && (!check_analysis_range || frameIndex >= analysis_range.start))
            prev_frame = frameIndex;
        
        {
            ref.read<Vec2>(data.pos);
            ref.read<float>(data.angle);
                
            if(_header.version < Output::ResultsFormat::Versions::V_27) {
                if(_header.version >= Output::ResultsFormat::Versions::V_8)
                    ref.read<double>(time);
                else
                    ref.read_convert<float>(time);
            }
        }
        
        //fish->_blob_indices[frameIndex] = ref.read<uint32_t>();
        if(_header.version < Output::ResultsFormat::Versions::V_7)
            ref.seek(ref.tell() + sizeof(uint32_t)); // blob index no longer used
            //ref.read<uint32_t>();
        
        data.stuff = std::make_shared<Individual::BasicStuff>();
        data.stuff->frame = (long_t)frameIndex;
        data.prev_frame = prev_frame;
        read_blob(ref, data.stuff->blob);
        
        if(_header.version >= Output::ResultsFormat::Versions::V_7 && _header.version < Output::ResultsFormat::Versions::V_29)
        {
            static Vec2 tmp;
            ref.read<Vec2>(tmp);
        }
        
        if(check_analysis_range && (frameIndex > analysis_range.end || frameIndex < analysis_range.start)) {
            continue;
        }
        
        {
            std::lock_guard<std::mutex> guard(mutex);
            stuffs.push_back(data);
        }
        variable.notify_one();
        
        prev_frame = frameIndex;
        
        if(i%100000 == 0 && i)
            Debug("Blob %d/%d", i, N);
    }
    
    stop = true;
    variable.notify_all();
    worker.join();
    
    //Output::ResultsFormat::_blob_pool.enqueue(fish);
    
    // read pixel information
    
    if(_header.version >= Versions::V_19) {
        ref.read<uint64_t>(N);
        data_long_t frame;
        uint64_t value;
        
        for(uint64_t i=0; i<N; ++i) {
            ref.read<data_long_t>(frame);
            ref.read<uint64_t>(value);
            
            if(check_analysis_range && (frame > analysis_range.end || frame < analysis_range.start))
                continue;
            
            auto stuff = fish->basic_stuff((long_t)frame);
            if(!stuff) {
                Except("(%d) Cannot find basic_stuff for frame %d.", fish->identity().ID(), frame);
            } else
                stuff->thresholded_size = value;
        }
    }
    
    //uint64_t delta = this->tell() - pos_before;
    //Debug("PSize %f", delta / double(1000*1000));
    //pos_before = this->tell();
    
    // number of head positions
    if(_header.version <= Versions::V_24) {
        ref.read<uint64_t>(N);
        
        for (uint64_t i=0; i<N; i++) {
            ref.read<data_long_t>(frameIndex);
            
            PhysicalProperties *prop;
            {
                Vec2 pos;
                float angle;
                double time;
                
                ref.read<Vec2>(pos);
                ref.read<float>(angle);
                
                if(_header.version < Output::ResultsFormat::Versions::V_27) {
                    if(_header.version >= Output::ResultsFormat::Versions::V_8)
                        ref.read<double>(time);
                    else
                        ref.read_convert<float>(time);
                }
                
                prop = new PhysicalProperties(fish, frameIndex, pos, angle);
            }
            
            auto midline = read_midline(ref);
            auto outline = read_outline(ref, midline);
        
            if(check_analysis_range && (frameIndex > analysis_range.end || frameIndex < analysis_range.start))
                continue;
            
            if(FAST_SETTINGS(calculate_posture)) {
                // save values
                auto stuff = std::make_shared<Individual::PostureStuff>();
                
                stuff->frame = (long_t)frameIndex;
                stuff->cached_pp_midline = midline;
                stuff->head = prop;
                stuff->outline = outline;
                
                auto segment = fish->segment_for((long_t)frameIndex);
                if(!segment) U_EXCEPTION("(%d) Have to add basic stuff before adding posture stuff (frame %d).", fish->identity().ID(), frameIndex);
                segment->add_posture_at(stuff, fish);
                
            } else {
                delete prop;
            }
        }
        
    } else if(_header.version >= Versions::V_25) {
        // now read outlines and midlines
        ref.read<uint64_t>(N);
        
        std::map<long_t, std::shared_ptr<Individual::PostureStuff>> sorted;
        data_long_t previous_frame = -1;
        
        for (uint64_t i=0; i<N; i++) {
            ref.read<data_long_t>(frameIndex);
            if(frameIndex < previous_frame) {
                Warning("Unordered frames (%ld vs %d)", frameIndex, previous_frame);
                return fish;
            }
            previous_frame = frameIndex;
            
            auto midline = read_midline(ref);
            
            if(check_analysis_range && (frameIndex > analysis_range.end || frameIndex < analysis_range.start))
                continue;
            
            if(FAST_SETTINGS(calculate_posture)) {
                // save values
                auto stuff = std::make_shared<Individual::PostureStuff>();
                
                stuff->frame = (long_t)frameIndex;
                stuff->cached_pp_midline = midline;
                
                auto segment = fish->segment_for((long_t)frameIndex);
                if(!segment) U_EXCEPTION("(%d) Have to add basic stuff before adding posture stuff (frame %d).", fish->identity().ID(), frameIndex);
                
                sorted[stuff->frame] = stuff;
            }
        }
        
        ref.read<uint64_t>(N);
        for (uint64_t i=0; i<N; i++) {
            ref.read<data_long_t>(frameIndex);
            auto outline = read_outline(ref, nullptr);
            
            if(check_analysis_range && (frameIndex > analysis_range.end || frameIndex < analysis_range.start))
                continue;
            
            if(FAST_SETTINGS(calculate_posture)) {
                //fish->posture_stuff(frameIndex);
                std::shared_ptr<Individual::PostureStuff> stuff;
                auto it = sorted.find((long_t)frameIndex);
                
                if(it == sorted.end()) {
                    stuff = std::make_shared<Individual::PostureStuff>();
                    stuff->frame = (long_t)frameIndex;
                    
                    //
                    //fish->_posture_stuff.push_back(stuff);
                    sorted[(long_t)frameIndex] = stuff;
                    
                } else
                    stuff = it->second;
                
                assert(stuff);
                stuff->outline = outline;
            }
        }
        
        std::shared_ptr<Individual::SegmentInformation> segment = nullptr;
        for(auto && [frame, stuff] : sorted) {
            if(!segment || !segment->contains(frame))
                segment = fish->segment_for(frame);
            if(!segment) U_EXCEPTION("(%d) Have to add basic stuff before adding posture stuff (frame %d).", fish->identity().ID(), frame);
            
            segment->add_posture_at(stuff, fish);
        }
    }
    
    //delta = this->tell() - pos_before;
    _post_pool.enqueue(fish);
    
    //if(N > 1000)
    //    Debug("Time for individual %d: %f", fish->identity().ID(), timer.elapsed());
    
    return fish;
}

template<> void Data::read(Individual*& out_ptr) {
    struct Callback {
        std::function<void()> _fn;
        Callback(std::function<void()> fn) : _fn(fn) {}
        ~Callback() {
            try {
                _fn();
            } catch(const std::exception& e) {
                Except("Caught an exception inside ~Callback(): %s",e.what());
            }
        }
    };
    
    auto results = dynamic_cast<Output::ResultsFormat*>(this);
    auto callback = new Callback([results](){
        ++results->_N_written;
        
        if(!SETTING(quiet)) {
            auto N = results->_expected_individuals.load();
            double N_written = results->_N_written.load();
            if(N <= 100 || results->_N_written % max(100u, uint64_t(N * 0.01)) == 0) {
                Debug("Read individual %.0f/%lu (%.0f%%)...", N_written, N, N_written / float(N) * 100);
                results->_update_progress("", narrow_cast<float>(N_written / double(N)), results->filename().str()+"\n<ref>loading individual</ref> <number>"+Meta::toStr(results->_N_written)+"</number> <ref>of</ref> <number>"+Meta::toStr(N)+"</number>");
            }
        }
    });
    
    if(results && results->_header.version >= Output::ResultsFormat::V_18) {
        uint64_t size, uncompressed_size;
        read<uint64_t>(size);
        read<uint64_t>(uncompressed_size);
        
        auto in = Meta::toStr(FileSize(size));
        auto out = Meta::toStr(FileSize(uncompressed_size));
        
        //Debug("Reading compressed block of size %S.", &in, &out);
        
        std::vector<char>* cache = nullptr;
        auto ptr = read_data_fast(size);
        
        results->_generic_pool.enqueue([ptr, uncompressed_size, out_ptr = &out_ptr, results, size, callback, cache]()
        {
            DataPackage /*compressed_block, */uncompressed_block;
            uncompressed_block.resize(uncompressed_size, false);
            
            lzo_uint new_len;
            if(lzo1x_decompress((uchar*)ptr,size,(uchar*)uncompressed_block.data(),&new_len,NULL) == LZO_E_OK)
            {
                if(new_len != uncompressed_size)
                    Warning("Uncompressed size %lu is different than expected %lu", new_len, uncompressed_size);
                ReadonlyMemoryWrapper compressed((uchar*)uncompressed_block.data(), new_len);
                (*out_ptr) = results->read_individual(compressed, results->_property_cache.get());
                
            } else {
                U_EXCEPTION("Failed to decode individual from file %S", &results->filename());
            }
            
            if(cache)
                delete cache;
            delete callback;
        });
        
        return;
    }
    
    if(!results)
        U_EXCEPTION("This is not ResultsFormat.");
    
    out_ptr = results->read_individual(*this, results->_property_cache.get());
    delete callback;
}

#define OUT_LEN(L)     (L + L / 16 + 64 + 3)

#define HEAP_ALLOC(var,size) \
lzo_align_t __LZO_MMODEL var [ ((size) + (sizeof(lzo_align_t) - 1)) / sizeof(lzo_align_t) ]

static HEAP_ALLOC(wrkmem, LZO1X_1_MEM_COMPRESS);

template<>
uint64_t Data::write(const Individual& val) {
    /**
     * Structure of exported binary:
     *  4 bytes ID
     *
     *  N number of frames for centroid
     *  (Nx (8 bytes frameIndex + 12 bytes)) PhysicalProperties for centroid
     *  (Nx k bytes) Blob
     *  (Nx (8 bytes M + Mx 1 byte grey values))
     
     *  N number of frames for head
     *  (Nx (8 bytes frameIndex + 12 bytes)) PhysicalProperties for head
     *  (Nx Midline)
     *  (Nx Outline)
     */
    
    const uint64_t pack_size = Output::ResultsFormat::estimate_individual_size(val);
    auto *ptr = static_cast<Output::ResultsFormat*>(this);
    const std::function<void(uint64_t)> callback = [ptr](uint64_t pos) {
        pos += ptr->current_offset();
        
        if(pos - ptr->last_callback > ptr->estimated_size * 0.01) {
            ptr->_update_progress("", narrow_cast<float>(min(1.0, double(pos)/double(ptr->estimated_size))), "");
            ptr->last_callback = pos;
        }
    };
    
    DataPackage pack(pack_size, &callback);
    
    // header
    assert(val.identity().ID() >= 0);
    uint32_t ID = (uint32_t)val.identity().ID();
    pack.write<uint32_t>(ID);
    pack.write<std::string>(val.identity().raw_name());
    
    pack.write<uint64_t>(val._manually_matched.size());
    for (auto m : val._manually_matched)
        pack.write<data_long_t>(m);
    
    // centroid based information
    pack.write<uint64_t>(val._basic_stuff.size());
    
    std::set<std::tuple<long_t, const Individual::BasicStuff*>> sorted;
    for(auto& stuff : val._basic_stuff) {
        sorted.insert({stuff->frame, stuff.get()});
    }
    
    for(auto&& [frame, stuff] : sorted) {
        // write frame number
        pack.write<data_long_t>(stuff->frame);
        
        // write centroid PhysicalProperties
        pack.write<PhysicalProperties>(*stuff->centroid);
        
        // assume we have a blob and grey values as well
        pack.write<pv::BlobPtr>(stuff->blob.unpack());
        
        //pack.write<Vec2>(stuff->centroid->pos(Units::PX_AND_SECONDS));
    }
    
    // write pixel size information
    pack.write<uint64_t>(val._basic_stuff.size());
    for(auto & stuff : val._basic_stuff) {
        pack.write<data_long_t>(stuff->frame);
        pack.write<uint64_t>(stuff->thresholded_size);
    }
    
    //auto str = Meta::toStr(FileSize(pack.size()));
    //Debug("Individual %d is %S after centroid", val.identity().ID(), &str);
    
    // head based information
    /*pack.write<uint64_t>(val._head.size());
    
    for (auto &c : val._head) {
        // write frame number
        pack.write<data_long_t>(c.first);
        
        // write head PhysicalProperties
        pack.write(*c.second);
    }*/
    
    // write N, and then write all midlines and outlines (unprocessed)
    std::map<data_long_t, Midline::Ptr> cached_pp_midlines;
    std::map<data_long_t, MinimalOutline::Ptr> outlines;
    for(auto& stuff : val._posture_stuff) {
        if(stuff->cached_pp_midline)
            cached_pp_midlines[stuff->frame] = stuff->cached_pp_midline;
        if(stuff->outline)
            outlines[stuff->frame] = stuff->outline;
    }
    
    pack.write<uint64_t>(cached_pp_midlines.size());
    for(auto && [frame, midline] : cached_pp_midlines) {
        pack.write<data_long_t>(frame);
        pack.write<Midline>(*midline);
    }
    
    pack.write<uint64_t>(outlines.size());
    
    for(auto && [frame, outline] : outlines) {
        pack.write<data_long_t>(frame);
        pack.write<MinimalOutline>(*outline);
    }
    
    //str = Meta::toStr(FileSize(pack.size()));
    //auto estimate = Meta::toStr(FileSize(pack_size));
    //auto per_frame = Meta::toStr(FileSize(pack.size() / double(val.frame_count())));
    
    //Debug("Individual %d is %S (estimated %S) thats %S / frame", val.identity().ID(), &str, &estimate, &per_frame);
    
    lzo_uint out_len = 0;
    assert(pack.size() < LZO_UINT_MAX);
    uint64_t in_len = pack.size();
    uint64_t reserved_size = OUT_LEN(in_len);
    
    DataPackage out;
    out.resize(reserved_size);
    
    // lock for wrkmem
    if(lzo1x_1_compress((uchar*)pack.data(), in_len, (uchar*)out.data(), &out_len, wrkmem) == LZO_E_OK)
    {
        uint64_t size = out_len + sizeof(uint32_t)*2;
        
        pack.reset_offset();
        
        assert(out_len < UINT32_MAX);
        pack.write<uint64_t>(out_len);
        pack.write<uint64_t>(in_len);
        pack.write_data(out_len, out.data());
        
        auto ptr = static_cast<const Output::ResultsFormat*>(this);
        if(ptr->_expected_individuals.load() < 1000 || ptr->_N_written.load() % 100 == 0 || ptr->_N_written.load() + 1 == ptr->_expected_individuals.load()) {
            auto before = Meta::toStr(FileSize(in_len));
            auto after = Meta::toStr(FileSize(size));
            
            Debug("Saved %.2f%%... (individual %d compressed from %S to %S).", double(ptr->_N_written.load() + 1) / double(ptr->_expected_individuals.load()) * 100, val.identity().ID(), &before, &after);
        }
    
    } else {
        U_EXCEPTION("Compression of %lu bytes failed (individual %d).", pack.size(), val.identity().ID());
    }
    
    return write(pack);
}

namespace Output {
    Timer reading_timer;
    float bytes_per_second = 0, samples = 0;
    std::string speed = "";
    float percent_read;
    
    /*const char* ResultsFormat::read_data_fast(uint64_t num_bytes) {
        if(reading_timer.elapsed() >= 1) {
            if(samples > 0) {
                speed = Meta::toStr(FileSize(uint64_t(bytes_per_second / samples / reading_timer.elapsed())));
                Debug("Reading @ %S/s", &speed);
            }
            reading_timer.reset();
            bytes_per_second = 0;
            samples = 0;
        }
        
        if(current_offset() - last_callback > read_size() * 0.01) {
            last_callback = current_offset();
            percent_read = float((current_offset() + num_bytes) / double(read_size()));
            
            //if(int(percent_read*100) % 10 == 0)
            {
                _update_progress("", percent_read, ""+filename().str()+"\n<ref>reading @ "+speed+"/s</ref>");
                Debug("Reading %.0f%% (@ %S/s)", percent_read*100, &speed);
            }
        }
        
        Timer timer;
        auto ptr = DataFormat::read_data_fast(num_bytes);
        auto time = timer.elapsed();
        
        bytes_per_second += num_bytes / time;
        ++samples;
        
        return ptr;
    }*/
    
    void ResultsFormat::_read_header() {
        std::string version_string;
        read<std::string>(version_string);
        if(!utils::beginsWith(version_string, "TRACK"))
            U_EXCEPTION("Illegal file format for tracking results.");
        
        if (version_string == "TRACK") {
            _header.version = Versions::V_1;
            
        } else {
            std::string str = version_string.substr(5);
            _header.version = (Versions)Meta::fromStr<int>(str);
        }
        
        if(_header.version >= V_3) {
            read<uint64_t>(_header.gui_frame);
        } else _header.gui_frame = 0;
        if(_header.version >= V_11 && _header.version < V_15) {
            seek(tell() + sizeof(data_long_t));
        }
        
        _header.consecutive_segments.clear();
        _header.video_resolution = Size2(-1);
        _header.video_length = 0;
        _header.average.clear();
        
        if(_header.version >= ResultsFormat::V_28) {
            uint32_t N;
            read<uint32_t>(N);
            
            // read N pairs of numbers to convert to ranges
            uint32_t start, end;
            for (uint32_t i=0; i<N; ++i) {
                read<uint32_t>(start);
                read<uint32_t>(end);
                
                _header.consecutive_segments.push_back(Rangel(narrow_cast<long_t>(start), narrow_cast<long_t>(end)));
            }
            
            read<Size2>(_header.video_resolution);
            read<uint64_t>(_header.video_length);
            
            _header.average.create((uint)_header.video_resolution.height, (uint)_header.video_resolution.width, 1);
            read_data(_header.average.size(), (char*)_header.average.data());
        }
        
        if(_header.version >= ResultsFormat::V_30) {
            read<int64_t>(_header.analysis_range.start);
            read<int64_t>(_header.analysis_range.end);
        } else
            _header.analysis_range = Range<int64_t>(-1, -1);
        
        if(_header.version >= ResultsFormat::V_14) {
            read<std::string>(_header.settings);
            //Debug("Read settings map: %S", &_header.settings);
        }
        
        if(_header.version >= ResultsFormat::V_23) {
            read<std::string>(_header.cmd_line);
        }
        
        if(!SETTING(quiet)) {
            DebugHeader("READING PROGRAM STATE");
            Debug("Read head of '%S' (version:V_%d gui_frame:%lu analysis_range:%ld-%ld)", &filename().str(), (int)_header.version+1, _header.gui_frame, _header.analysis_range.start, _header.analysis_range.end);
            Debug("Generated with command-line: '%S'", &_header.cmd_line);
        }
    }
    
    void ResultsFormat::_write_header() {
        std::string version_string = "TRACK"+std::to_string((int)Versions::current);
        write<std::string>(version_string);
        if(!SETTING(quiet)) {
            Debug("Writing version string '%S'", &version_string);
            Debug("Writing frame %lu", _header.gui_frame);
        }
        write<uint64_t>(_header.gui_frame);
        auto consecutive = Tracker::instance()->consecutive();
        write<uint32_t>((uint32_t)consecutive.size());
        for (auto &c : consecutive) {
            write<uint32_t>(sign_cast<uint32_t>(c.start));
            write<uint32_t>(sign_cast<uint32_t>(c.end));
        }
        
        write<Size2>(Tracker::average().bounds().size());
        write<uint64_t>(FAST_SETTINGS(video_length));
        
        uint64_t bytes = Tracker::average().cols * Tracker::average().rows;
        write_data(bytes, (const char*)Tracker::average().data());
        
        auto [start, end] = FAST_SETTINGS(analysis_range);
        write<int64_t>(start);
        write<int64_t>(end);
    }
    
    uint64_t ResultsFormat::write_data(uint64_t num_bytes, const char *buffer) {
        if(current_offset() - last_callback > estimated_size * 0.01) {
            _update_progress("", narrow_cast<float>(min(1.0, double(current_offset()+num_bytes)/double(estimated_size))), "");
            last_callback = current_offset();
        }
        
        return DataFormat::write_data(num_bytes, buffer);
    }
    
    uint64_t ResultsFormat::estimate_individual_size(const Individual& val) {
        static const uint64_t physical_properties_size = (2+1+1)*sizeof(float);
        const uint64_t pack_size =
        4 + sizeof(data_long_t)*2
        + sizeof(uchar)*3
        + val._basic_stuff.size() * (sizeof(data_long_t)+physical_properties_size+sizeof(uint32_t)+(val._basic_stuff.empty() ? 100 : (*val._basic_stuff.begin())->blob.lines.size())*1.1*sizeof(pv::ShortHorizontalLine))
        + val._posture_stuff.size() * (sizeof(data_long_t)+sizeof(uint64_t)+((val._posture_stuff.empty() || !val._posture_stuff.front()->cached_pp_midline ?SETTING(midline_resolution).value<uint32_t>() : (*val._posture_stuff.begin())->cached_pp_midline->size()) * sizeof(float) * 2 + sizeof(float) * 5 + sizeof(uint64_t))+physical_properties_size+((val._posture_stuff.empty() || !val._posture_stuff.front()->outline ? 0 : val._posture_stuff.front()->outline->size()*1.1)*sizeof(uint16_t) + sizeof(float)*2+sizeof(uint64_t)))
        + val._basic_stuff.size() * sizeof(decltype(Individual::BasicStuff::thresholded_size)) + sizeof(uint64_t);
        
        return pack_size;
    }
    
    void ResultsFormat::write_file(const std::vector<track::FrameProperties> &frames, const std::unordered_map<long_t, Tracker::set_of_individuals_t > &active_individuals_frame, const std::unordered_map<Idx_t, Individual *> &individuals, const std::vector<std::string>& exclude_settings)
    {
        estimated_size = sizeof(uint64_t)*3 + frames.size() * (sizeof(data_long_t)+sizeof(CompatibilityFrameProperties)) + active_individuals_frame.size() * (sizeof(data_long_t)+sizeof(uint64_t)+(active_individuals_frame.empty() ? individuals.size() : active_individuals_frame.begin()->second.size())*sizeof(data_long_t));
        
        _expected_individuals = individuals.size();
        for (auto& fish: individuals) {
            estimated_size += estimate_individual_size(*fish.second);
        }
        
        if(!SETTING(quiet)) {
            auto str = Meta::toStr(FileSize(estimated_size));
            Debug("Estimating %S for the whole file.", &str);
        }
        std::string text = default_config::generate_delta_config(true, exclude_settings);
        write<std::string>(text);
        write<std::string>(SETTING(cmd_line).value<std::string>());
        
        // write recognition data
        const auto &recognition = *Tracker::recognition();
        write<uint64_t>(recognition.data().size());
        for(auto && [frame, map] : recognition.data()) {
            write<data_long_t>(frame);
            write<uint64_t>(map.size());
            
            for(auto && [bid, vector] : map) {
                write<uint32_t>(bid);
                write<uint64_t>(vector.size());
                write_data(sizeof(float) * vector.size(), (const char*)vector.data());
            }
        }
        
        // write categorization data, if it exists
        Categorize::DataStore::write(*this, header().version);
        
        // write frame properties
        write<uint64_t>(frames.size());
        if(!SETTING(quiet))
            Debug("Writing %ld frames", frames.size());
        for (auto &p : frames)
            write<track::FrameProperties>(p);
        
        // write number of individuals
        write<uint64_t>(_expected_individuals);
        
        _N_written = 0;
        for (auto& fish: individuals) {
            write<Individual>(*fish.second);
            ++_N_written;
        }
        
        // write active individuals per frame
        write<uint64_t>(active_individuals_frame.size());
        for (auto &p : active_individuals_frame) {
            write<data_long_t>(p.first);
            write<uint64_t>(p.second.size());
            for(auto &fish : p.second) {
                write<data_long_t>(fish->identity().ID());
            }
        }
    }
    
    Path TrackingResults::expected_filename() {        
        file::Path filename = SETTING(filename).value<Path>().filename();
        filename = filename.extension() == "pv" ?
        filename.replace_extension("results") : filename.add_extension("results");
        return pv::DataLocation::parse("output", filename);
    }
    
    void TrackingResults::save(std::function<void (const std::string &, float, const std::string &)> update_progress, Path filename, const std::vector<std::string>& exclude_settings) const {
        if(filename.empty())
            filename = expected_filename();
        
        if(!filename.remove_filename().empty() && !filename.remove_filename().exists())
            filename.remove_filename().create_folder();
        
        // add a temporary extension, so that we dont overwrite initially
        // (until we're certain its done)
        filename = filename.add_extension("tmp01");
        
        ResultsFormat file(filename.str(), update_progress);
        file.header().gui_frame = sign_cast<uint64_t>(SETTING(gui_frame).value<long_t>());
        file.start_writing(true);
        file.write_file(_tracker._added_frames, _tracker._active_individuals_frame, _tracker._individuals, exclude_settings);
        file.close();
        
        // go back from .tmp01 to .results
        if(filename.move_to(filename.remove_extension())) {
            filename = filename.remove_extension();
            if(!SETTING(quiet)) {
                DebugHeader("Finished writing '%S'.", &filename.str());
                DebugCallback("Finished writing '%S'.", &filename.str());
            }
        } else
            U_EXCEPTION("Cannot move '%S' to '%S' (but results have been saved, you just have to rename the file).", &filename.str(), &filename.remove_extension().str());
    }
    
    void TrackingResults::clean_up() {
        _tracker._individuals.clear();
        _tracker._added_frames.clear();
        _tracker.clear_properties();
        _tracker._active_individuals.clear();
        _tracker._active_individuals_frame.clear();
        _tracker._startFrame = -1;
        _tracker._endFrame = -1;
        _tracker._max_individuals = 0;
        _tracker._consecutive.clear();
        FOI::clear();
    }
    
    ResultsFormat::Header TrackingResults::load_header(const file::Path &filename) {
        bytes_per_second = samples = percent_read = 0;
        
        if(!SETTING(quiet))
            Debug("Trying to open results '%S' (retrieve header only)", &filename.str());
        ResultsFormat file(filename.str(), [](const auto&, auto, const auto&){});
        file.start_reading();
        return file.header();
    }

void TrackingResults::update_fois(const std::function<void(const std::string&, float, const std::string&)>& update_progress) {
    const auto number_fish = FAST_SETTINGS(track_max_individuals);
    data_long_t prev = 0;
    data_long_t n = 0;
    double prev_time = _tracker.start_frame() == -1 ? 0:  _tracker.properties(_tracker.start_frame())->time;
    
    //auto it = _tracker._active_individuals_frame.begin();
    if(_tracker._active_individuals_frame.size() != _tracker._added_frames.size()) {
        U_EXCEPTION("This is unexpected (%d != %d).", _tracker._active_individuals_frame.size(), _tracker._added_frames.size());
    }
    
    const track::FrameProperties* prev_props = nullptr;
    data_long_t prev_frame = -1;
    
    std::unordered_map<Idx_t, Individual::segment_map::const_iterator> iterator_map;
    
    for(const auto &props : _tracker._added_frames) {
        prev_time = props.time;
        
        // number of individuals actually assigned in this frame
        /*n = 0;
        for(const auto &fish : it->second) {
            n += fish->has(props.frame) ? 1 : 0;
        }*/
        n = props.active_individuals;
        
        // update tracker with the numbers
        //assert(it->first == props.frame);
        auto &active = _tracker._active_individuals_frame.at(props.frame);
        if(prev_props && prev_frame - props.frame > 1)
            prev_props = nullptr;
        
        _tracker.update_consecutive(active, props.frame, false);
        _tracker.update_warnings(props.frame, props.time, (long_t)number_fish, n, prev, &props, prev_props, active, iterator_map);
        
        prev = n;
        prev_props = &props;
        prev_frame = props.frame;
        
        if((uint)props.frame % max(1u, uint64_t(_tracker._added_frames.size() / 10u)) == 0) {
            update_progress("FOIs...", props.frame / float(_tracker.end_frame()), Meta::toStr(props.frame)+" / "+Meta::toStr(_tracker.end_frame()));
            if(!SETTING(quiet))
                Debug("\tupdate_fois %d / %d\r", props.frame, _tracker.end_frame());
        }
    }
    
    if(_tracker.recognition() && FAST_SETTINGS(recognition_enable)) {
        update_progress("Finding segments...", -1, "");
        _tracker.recognition()->update_dataset_quality();
    }
}
    
    void TrackingResults::load(std::function<void(const std::string&, float, const std::string&)> update_progress, Path filename) {
        Timer loading_timer;
        
        if (filename.empty())
            filename = expected_filename();
        else if (!filename.exists())
            Error("Cannot find '%S' as requested. Trying standard paths.", &filename.str());

        if(!filename.exists()) {
            file::Path file = SETTING(filename).value<Path>();
            file = file.extension() == "pv" ?
            file.replace_extension("results") : file.add_extension("results");
            
            //file = pv::DataLocation::parse("input", filename.filename());
            if(file.exists()) {
                Warning("Not loading from the output folder, but from the input folder because '%S' could not be found, but '%S' could.", &filename.str(), &file.str());
                filename = file;
            } else
                Warning("Searched at '%S', but also couldnt be found.", &file.str());
        }
        
        bytes_per_second = samples = percent_read = 0;
        
        if(!SETTING(quiet))
            Debug("Trying to open results '%S'", &filename.str());
        ResultsFormat file(filename.str(), update_progress);
        file.start_reading();
        
        auto tmp = _tracker.individuals();
        clean_up();
        
        data_long_t biggest_id = -1;
        
        Tracker::instance()->_individual_add_iterator_map.clear();
        Tracker::instance()->_segment_map_known_capacity.clear();

        for (auto& p : tmp)
            delete p.second;

        // read recognition data
        auto &recognition = *Tracker::recognition();
        recognition.data().clear();
        
        if(file._header.version >= ResultsFormat::V_13) {
            std::vector<float> tmp;
            uint64_t L;
            file.read<uint64_t>(L);
            data_long_t frame;
            uint64_t size, vsize;
            uint32_t bid;
            
            if(L > 0) {
                file._header.has_recognition_data = true;
            }
            
            for(uint64_t i=0; i<L; ++i) {
                file.read<data_long_t>(frame);
                file.read<uint64_t>(size);
                
                for(uint64_t j=0; j<size; ++j) {
                    file.read<uint32_t>(bid);
                    file.read<uint64_t>(vsize);
                    tmp.resize(vsize);
                    file.read_data(vsize * sizeof(float), (char*)tmp.data());
                    
                    recognition.data()[(long_t)frame][bid] = tmp;
                }
            }
        }
        
        if(file.header().version >= ResultsFormat::Versions::V_33) {
            // read category data
            Categorize::DataStore::read(file, file.header().version);
        }
        
        // read frame properties
        uint64_t L;
        file.read<uint64_t>(L);
        
        track::FrameProperties props;
        CompatibilityFrameProperties comp;
        data_long_t frameIndex;
        bool check_analysis_range = SETTING(analysis_range).value<std::pair<long_t, long_t>>().first != -1 || SETTING(analysis_range).value<std::pair<long_t, long_t>>().second != -1;
        
        auto analysis_range = Tracker::analysis_range();
        _tracker.clear_properties();
        file._property_cache = std::make_shared<CacheHints>(L);
        
        for (uint64_t i=0; i<L; i++) {
            file.read<data_long_t>(frameIndex);
            
            if(file._header.version >= ResultsFormat::Versions::V_2)
                file.read<track::FrameProperties>(props);
            else {
                file.read<CompatibilityFrameProperties>(comp);
                props.org_timestamp = comp.timestamp;
                props.time = comp.time;
            }
            
            props.frame = frameIndex;
            
            if(check_analysis_range && (frameIndex > analysis_range.end || frameIndex < analysis_range.start))
                continue;
            
            if(_tracker._startFrame == -1)
                _tracker._startFrame = frameIndex;
            _tracker._endFrame = frameIndex;
            
            _tracker.add_next_frame(props);
        }
        
        for(auto &prop : _tracker.frames())
            file._property_cache->push(prop.frame, &prop);
        
        // read the individuals
        std::map<uint32_t, Individual*> map_id_ptr;
        std::vector<Individual*> fishes;
        
        file.read<uint64_t>(L);
        fishes.resize(L);
        file._expected_individuals = L;
        file._N_written = 0;
        
        for (uint64_t i=0; i<L; i++) {
            fishes[i] = nullptr;
            
            file.read<Individual*>(fishes[i]);
            
            if(SETTING(terminate)) {
                clean_up();
                return;
            }
        }
        
        file._generic_pool.wait();
        if(file._exception_ptr) //! an exception happened inside the generic pool
            std::rethrow_exception(file._exception_ptr);
        
        for(auto &fish : fishes) {
            if(fish) {
                if(biggest_id < fish->identity().ID())
                    biggest_id = fish->identity().ID();
                map_id_ptr[fish->identity().ID()] = fish;
                _tracker._individuals[fish->identity().ID()] = fish;
            }
        }
        
        track::Identity::set_running_id(biggest_id+1);
        
        uint64_t n;
        data_long_t ID;
        file.read<uint64_t>(L);
        for (uint64_t i=0; i<L; i++) {
            Tracker::set_of_individuals_t active;
            
            file.read<data_long_t>(frameIndex);
            file.read<uint64_t>(n);
            
            for (uint64_t j=0; j<n; j++) {
                file.read<data_long_t>(ID);
                
                if(check_analysis_range && (frameIndex > analysis_range.end || frameIndex < analysis_range.start))
                    continue;
                
                auto it = map_id_ptr.find(ID);
                if (it == map_id_ptr.end())
                    U_EXCEPTION("Cannot find individual with ID %ld in map.", ID);
                active.insert(it->second);
            }
            
            if(check_analysis_range && (frameIndex > analysis_range.end || frameIndex < analysis_range.start))
                continue;
            
            _tracker._active_individuals_frame[(long_t)frameIndex] = active;
            _tracker._active_individuals = active;
            
            _tracker._max_individuals = max(_tracker._max_individuals, active.size());
        }
        
        _tracker._inactive_individuals.clear();
        for(auto&& [id, fish] : _tracker._individuals) {
            if(_tracker._active_individuals.find(fish) == _tracker._active_individuals.end()) {
                _tracker._inactive_individuals.insert(id);
            }
        }
        
        file.close();
        
        update_progress("post processing...", -1, "");
        file._post_pool.wait();
        
        if(file._header.version < ResultsFormat::V_31) {
            //! have to regenerate the number of individuals / frame
            long_t n = 0;
            
            for(auto &props : _tracker._added_frames) {
                // number of individuals actually assigned in this frame
                n = 0;
                for(const auto &fish : _tracker._active_individuals_frame.at(props.frame)) {
                    n += fish->has(props.frame);
                }
                props.active_individuals = n;
            }
        }
        
        update_fois(update_progress);
        
        /*for(long_t i=_tracker.start_frame(); i<=_tracker.end_frame(); i++) {
            long_t n = _tracker.found_individuals_frame(i);
            
            auto props = _tracker.properties(i);
            prev_time = props->time;
            _tracker.update_consecutive(_tracker.active_individuals(i), i, true);
            _tracker.update_warnings(i, props->time, number_fish, n, prev);
            
            prev = n;
            
            //_tracker.generate_pairdistances(i);
        }*/
        
        {
            sprite::Map config;
            GlobalSettings::docs_map_t docs;
            config.set_do_print(false);
            
            default_config::get(config, docs, NULL);
            try {
                default_config::load_string_with_deprecations(filename, file.header().settings, config, AccessLevelType::STARTUP, true);
                
            } catch(const cmn::illegal_syntax& e) {
                Error("Illegal syntax in .results settings.");
            }
            
            if(config.has("gui_focus_group")) {
                SETTING(gui_focus_group) = config["gui_focus_group"].value<std::vector<Idx_t>>();
            } else
                SETTING(gui_focus_group) = std::vector<Idx_t>{};
            
            SETTING(gui_frame).value<long_t>() = (long_t)file.header().gui_frame;
        }
        
        if((file.header().analysis_range.start != -1 || file.header().analysis_range.end != -1) && SETTING(analysis_range).value<std::pair<long_t, long_t>>() == std::pair<long_t,long_t>{-1,-1})
        {
            SETTING(analysis_range) = std::pair<long_t, long_t>(file.header().analysis_range.start, file.header().analysis_range.end);
        }
        
        if(Recognition::recognition_enabled()) {
            GUI::instance()->work().add_queue("", [](){
                Tracker::instance()->check_segments_identities(false, [](float x) { },
                [](const std::string&t, const std::function<void()>& fn, const std::string&b)
                {
                    if(GUI::instance())
                        GUI::instance()->work().add_queue(t, fn, b);
                });
            });
            
            if(GUI::instance()) {
                /*update_progress("apply network...", -1, "");
                
                Tracker::instance()->check_segments_identities(false, [](float){}, [](const std::string&t, const std::function<void()>& fn, const std::string&b) {
                    if(GUI::instance())
                        GUI::work().add_queue(t, fn, b);
                });
                
                Tracker::instance()->thread_pool().enqueue([](){
                    Tracker::recognition()->update_dataset_quality();
                });*/
            }
        }
        
        if(!SETTING(quiet)) {
            Debug("Successfully read file '%S' (version:V_%d gui_frame:%lu start:%lu end:%lu)", &file.filename().str(), (int)file._header.version+1, file.header().gui_frame, Tracker::start_frame(), Tracker::end_frame());
        
            DurationUS duration{uint64_t(loading_timer.elapsed() * 1000 * 1000)};
            auto str = Meta::toStr(duration);
            DebugHeader("FINISHED READING PROGRAM STATE IN %S", &str);
        }
    }
}
