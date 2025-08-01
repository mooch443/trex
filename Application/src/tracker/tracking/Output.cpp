#include "Output.h"
#include <misc/Timer.h>
#include <misc/FOI.h>
#include <misc/default_config.h>
#include <lzo/minilzo.h>
#include <tracking/CategorizeDatastore.h>
#include <misc/frame_t.h>
#include <misc/IdentifiedTag.h>
#include <tracking/Tracker.h>
#include <file/DataLocation.h>
#include <tracking/IndividualManager.h>
#include <tracking/DatasetQuality.h>
#include <misc/SettingsInitializer.h>
#include <tracking/TrackingHelper.h>
#include <tracking/AutomaticMatches.h>

using namespace track;
typedef int64_t data_long_t;

/*IMPLEMENT(Output::ResultsFormat::_blob_pool)(cmn::hardware_concurrency(), [](Individual*obj){
    Timer timer;
    
    
});*/

void post_process(Output::ResultsFormat* _self, Individual* obj) {
    CachedSettings settings;
    obj->update_midlines(settings, _self->property_cache().get());
    obj->local_cache().regenerate(obj);
}

Output::ResultsFormat::ResultsFormat(const file::Path& filename, std::function<void(const std::string&, double, const std::string&)> update_progress)
 :
DataFormat(filename.str()),
_update_progress(update_progress), 
last_callback(0), 
estimated_size(0),
_post_pool(cmn::hardware_concurrency(), [this](Individual* obj) 
{
    // post processing for individuals
    Timer timer;
    auto name = get_thread_name();
    set_thread_name(obj->identity().name()+"_post");
    post_process(this, obj);
    
    if (timer.elapsed() >= 1) {
        auto us = timer.elapsed() * 1000 * 1000;
        if (!GlobalSettings::is_runtime_quiet())
            Print(obj->identity()," post-processing took ", DurationUS{ (uint64_t)us });
    }
    set_thread_name(name);
        
}, "Output::post_pool"),
_generic_pool(cmn::hardware_concurrency(), "Output::GenericPool", [this](std::exception_ptr e) {
    _exception_ptr = e; // send it to main thread
}),
_load_pool(cmn::hardware_concurrency(), "Output::loadPool", [this](std::exception_ptr e) {
    _exception_ptr = e; // send it to main thread
}),
_expected_individuals(0), _N_written(0)
{
    //if(!filename.exists())
    //    throw U_EXCEPTION("File ", filename," cannot be found.");
}

Output::ResultsFormat::~ResultsFormat() {
    _generic_pool.wait();
    _post_pool.wait();
}

void Output::ResultsFormat::read_prediction(Data& ref, blob::Prediction& pred) const {
    ref.read<uint8_t>(pred.clid);
    if(pred.clid == 255u) {
        /// invalid
        assert(not pred.valid());
        return;
    }
    
    ref.read<uint8_t>(pred.p);
    
    uint8_t N;
    
    /// read pose
    pred.pose = {};
    ref.read<uint8_t>(N);
    for(uint8_t i = 0; i<N; ++i) {
        uint16_t x, y;
        ref.read<uint16_t>(x);
        ref.read<uint16_t>(y);
        pred.pose.points.emplace_back(x, y);
    }
    
    /// read outlines
    pred.outlines = {};
    ref.read<uint8_t>(N);
    for(uint8_t i = 0; i<N; ++i) {
        uint16_t Npoints;
        ref.read<uint16_t>(Npoints);
        
        blob::SegmentedOutlines::Outline outline;
        for(uint16_t j=0; j<Npoints; ++j) {
            int32_t pt;
            ref.read<int32_t>(pt);
            outline._points.emplace_back(pt);
        }
        pred.outlines.add(std::move(outline));
    }
    
    /// read the original outline
    if(_header.version >= ResultsFormat::Versions::V_37) {
        uint32_t M;
        ref.read<uint32_t>(M);
        if(M > 0) {
            blob::SegmentedOutlines::Outline outline;
            outline._points.resize(M);
            ref.read_data(M * sizeof(int32_t), (char*)outline._points.data());
            pred.outlines.original_outline = std::move(outline);
        }
    }
}

uint64_t write_prediction(Data& ref, const blob::Prediction& pred) {
    if(not pred.valid()) {
        return ref.write<uint8_t>(255u);
    }
    
    auto position = ref.write<uint8_t>(pred.clid);
    ref.write<uint8_t>(pred.p);
    
    ref.write<uint8_t>(narrow_cast<uint8_t>(pred.pose.points.size(), tag::fail_on_error{}));
    for(auto &pt : pred.pose.points) {
        ref.write<uint16_t>(pt.x);
        ref.write<uint16_t>(pt.y);
    }
    
    if(pred.outlines.lines.size() >= 255) {
#ifndef NDEBUG
        Size2 dim(5120, 2700);
        cv::Mat mat = cv::Mat::zeros(dim.height, dim.width, CV_8UC3);
        gui::ColorWheel wheel;
        for(auto &outline : pred.outlines.lines) {
            auto clr = wheel.next();
            Vec2 previous = outline[outline.size() - 1u];
            for(size_t i = 0; i < outline.size(); ++i) {
                auto pt = outline[i];
                cv::line(mat, previous, pt, clr);
                previous = pt;
            }
        }
        tf::imshow("too many sub-outlines", mat);
#endif
        
        static constexpr auto limit = std::numeric_limits<uint8_t>::max();
        FormatWarning("Prediction ", pred, " has too many outlines: ", pred.outlines.lines.size(), ". Only saving the first ",limit,".");
        
        const uint8_t N = static_cast<uint8_t>(min(pred.outlines.lines.size(), limit));
        ref.write<uint8_t>(N);
        
        for(uint8_t i = 0; i < N; ++i) {
            const blob::SegmentedOutlines::Outline &line = pred.outlines.lines[i];
            ref.write<uint16_t>(narrow_cast<uint16_t>(line._points.size(), tag::fail_on_error{}));
            for(auto &pt : line._points)
                ref.write<int32_t>(pt);
        }
        
    } else {
        ref.write<uint8_t>(narrow_cast<uint8_t>(pred.outlines.lines.size(), tag::fail_on_error{}));
        for(const blob::SegmentedOutlines::Outline &line : pred.outlines.lines) {
            ref.write<uint16_t>(narrow_cast<uint16_t>(line._points.size(), tag::fail_on_error{}));
            for(auto &pt : line._points)
                ref.write<int32_t>(pt);
        }
    }
    
    if(pred.outlines.has_original_outline()) {
        auto &line = *pred.outlines.original_outline;
        ref.write<uint32_t>(narrow_cast<uint32_t>(line._points.size(), tag::fail_on_error{}));
        ref.write_data(sizeof(int32_t) * line._points.size(), (char*)line._points.data());
    } else {
        ref.write<uint32_t>(0);
    }
    
    return position;
}

template<> void Data::read(track::FrameProperties& p) {
    uint64_t ts;
    read<uint64_t>(ts);
    p.set_timestamp(ts);
    
    long_t active_individuals;
    
    auto *ptr = static_cast<Output::ResultsFormat*>(this);
    if(ptr->header().version >= Output::ResultsFormat::V_31) {
        read_convert<data_long_t>(active_individuals);
    } else
        active_individuals = -1;
    
    p.set_active_individuals(active_individuals);
}

template<>
uint64_t Data::write(const track::FrameProperties& val) {
    write<data_long_t>(val.frame().get());
    write<uint64_t>(val.timestamp().get());
    return write<data_long_t>(val.active_individuals());
}

// Function to multiply x and y with a factor
uint16_t multiply_xy(uint16_t value, float factor) {
    // Extract x and y
    int8_t x = (value >> 8) & 0xFF; // Top 8 bits
    int8_t y = value & 0xFF;        // Bottom 8 bits

    // Multiply x and y by the factor
    x = saturate(roundf(float(x) * factor), CHAR_MIN, CHAR_MAX);
    y = saturate(roundf(float(y) * factor), CHAR_MIN, CHAR_MAX);

    // Recombine x and y into a single uint16_t
    return (uint16_t(x) << 8) | (uint16_t(y) & 0xFF);
}

MinimalOutline Output::ResultsFormat::read_outline(Data& ref, Midline* midline) const {
    MinimalOutline ptr;
    //static_assert(MinimalOutline::factor == 10, "MinimalOutline::factor was 10 last time I checked.");
    const float conversion_factor = 1.f / float(_header.midline_factor);
    
    uint64_t L;
    ref.read<uint64_t>(L);
    ptr._points.resize(narrow_cast<uint32_t>(L)); // prevent malformed files from filling the ram
    /*if(_header.version > Output::ResultsFormat::Versions::V_9) {
        ptr->_tail_index = ref.read<data_long_t>();
    } else
        ptr->_tail_index = ptr->_points.size() * 0.5;*/
    if(_header.version > Output::ResultsFormat::Versions::V_9 && _header.version < Output::ResultsFormat::Versions::V_24) {
        data_long_t index;
        ref.read<data_long_t>(index);
        if(midline)
            midline->tail_index() = (long_t)index;
    } /*else if(_header.version >= Output::ResultsFormat::Versions::V_24) {
        midline->tail_index() = ref.read<data_long_t>();
        midline->head_index() = ref.read<data_long_t>();
    }*/
    
    if(_header.version >= Output::ResultsFormat::Versions::V_17) {
        ref.read_convert<float>(ptr._first.x);
        ref.read_convert<float>(ptr._first.y);
        
        ref.read_data(ptr._points.size() * sizeof(decltype(ptr._points)::value_type), (char*)ptr._points.data());
        
        if(_header.version >= Output::ResultsFormat::Versions::V_38) {
            ref.read_convert<float>(ptr.scale);
            
        } else {
            // we need to change the conversion factor
            if(conversion_factor != 1)
                ptr.convert_from(ptr.uncompress(_header.midline_factor));
        }
        
    } else {
        struct Point {
            float x, y;
        };
        
        std::vector<Point> points;
        points.resize(ptr._points.size());
        ref.read_data(ptr._points.size() * sizeof(Point), (char*)points.data());
        std::vector<Vec2> vecs;
        for(auto &p : points)
            vecs.push_back(Vec2(p.x, p.y));
        
        ptr.convert_from(vecs);
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
    write<float>(val.scale);
    
    return p;
}

template<>
uint64_t Data::write(const MotionRecord& val) {
    /**
     * Format of binary representation:
     *  - POSITION (2x4 bytes) in pixels
     *  - ANGLE (4 bytes)
     * both are float.
     *
     * Derivates etc. can be calculated after loading.
     */
    
    uint64_t p = write<Vec2>(val.pos<Units::PX_AND_SECONDS>());
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
    
    bool split = false;
    uint8_t byte = 0;
    if(_header.version >= Output::ResultsFormat::Versions::V_20) {
        ref.read<uint8_t>(byte);
        split = (byte & 0x1) != 0;
    }
    
    data_long_t parent_id = -1;
    if(_header.version >= Output::ResultsFormat::Versions::V_26) {
        if((byte & 0x2) != 0)
            ref.read<data_long_t>(parent_id);
        
    } else if(split
              && _header.version >= Output::ResultsFormat::Versions::V_22
              && _header.version <= Output::ResultsFormat::Versions::V_25)
        ref.read<data_long_t>(parent_id);
    
    uint16_t start_y, len;
    ref.read<uint16_t>(start_y);
    ref.read<uint16_t>(len);
    
    if(_header.version < Output::ResultsFormat::Versions::V_32) {
        std::vector<pv::LegacyShortHorizontalLine> legacy(len);
        blob._lines.clear();
        blob._lines.reserve(len);
        
        ref.read_data(sizeof(pv::LegacyShortHorizontalLine) * len, (char*)legacy.data());
        std::copy(legacy.begin(), legacy.end(), std::back_inserter(blob._lines));
        
    } else {
        blob._lines.resize(len);
        ref.read_data(elem_size * len, (char*)blob._lines.data());
    }
    
    blob.status_byte = byte;
    blob.start_y = start_y;
    blob.reset_id();
    
    if(parent_id < 0)
        blob.parent_id = pv::bid::invalid;
    else
        blob.parent_id = pv::bid(uint32_t(parent_id));
    
    if(_header.version >= Output::ResultsFormat::Versions::V_36) {
        read_prediction(ref, blob.pred);
    }
}

template<>
uint64_t Data::write(const pv::BlobPtr& val) {
    auto &mask = val->hor_lines();
    auto compressed = pv::ShortHorizontalLine::compress(mask);
    
    const uint64_t elem_size = sizeof(pv::ShortHorizontalLine);
    
    // this will turn
    uint8_t byte = val->flags();  /* uint8_t(val->split()                    ? (1 << 0) : 0)
                   | uint8_t(val->parent_id().valid()        ? (1 << 1) : 0)
                   | uint8_t(val->tried_to_split()           ? (1 << 2) : 0)
                   | uint8_t(val->is_tag()                   ? (1 << 3) : 0)
                   | uint8_t(val->is_instance_segmentation() ? (1 << 4) : 0)
                   | pv::Blob::get_only_flag(pv::Blob::Flags::is_rgb, val->is_rgb());*/
    uint64_t p = write<uint8_t>(byte);
    if((byte & uint8_t(1 << 1)) != 0) {
        assert(val->parent_id().valid());
        write<data_long_t>((int64_t)val->parent_id());
    }
    write<uint16_t>(narrow_cast<uint16_t>(mask.empty() ? 0 : mask.front().y, tag::fail_on_error{}));
    
    uint16_t L = narrow_cast<uint16_t>(compressed.size(), tag::fail_on_error{});
    write<uint16_t>(L);
    write_data(L * elem_size, (char*)compressed.data());
    
    write_prediction(*this, val->prediction());
    
    return p;
}

Midline::Ptr Output::ResultsFormat::read_midline(Data& ref) {
    auto midline = std::make_unique<Midline>();
    ref.read_convert<float>(midline->len());
    ref.read_convert<float>(midline->angle());
    ref.read<Vec2>(midline->offset());
    ref.read<Vec2>(midline->front());
    if(_header.version >= Versions::V_24) {
        ref.read_convert<data_long_t>(midline->tail_index());
        ref.read_convert<data_long_t>(midline->head_index());
    }
    midline->is_normalized() = _header.version < Versions::V_25;
    
    uint64_t L;
    ref.read<uint64_t>(L);
    midline->segments().resize(narrow_cast<uint16_t>(L)); // prevent malformed files from filling the RAM
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

void Output::ResultsFormat::process_frame(
          const CachedSettings& settings,
          const CacheHints* cache_ptr,
          Individual* fish,
          TemporaryData&& data)
{
    [[maybe_unused]] track::TrackingThreadG g{};
    const auto& frameIndex = data.stuff->frame;
    
    const Match::prob_t p_threshold = FAST_SETTING(match_min_probability);
    
#if !COMMONS_NO_PYTHON
    auto label =
        FAST_SETTING(track_consistent_categories)
            ? Categorize::DataStore::ranged_label(frameIndex, data.stuff->blob)
            : nullptr;
#else
    Categorize::Label::Ptr label = nullptr;
#endif
    Match::prob_t p = p_threshold;
    if(!fish->empty()) {
        auto cache = fish->cache_for_frame(Tracker::properties(frameIndex - 1_f), frameIndex, data.time, cache_ptr);
        if(cache) {
            assert(frameIndex > fish->start_frame());
            p = Individual::probability(settings, label ? label->id : MaybeLabel{}, cache.value(), frameIndex, data.stuff->blob);//.p;
        } else {
            throw U_EXCEPTION("Cannot calculate cache_for_frame for ", fish->identity(), " in ", frameIndex, " because: ", cache.error());
        }
    }
    
    if(fish->empty())
        fish->_startFrame = frameIndex;
    assert(not fish->_endFrame.valid() || fish->_endFrame < frameIndex);
    fish->_endFrame = frameIndex;
    
    auto tracklet = fish->update_add_tracklet(
        frameIndex, Tracker::properties(data.stuff->frame), Tracker::properties(data.stuff->frame - 1_f),
        data.stuff->centroid,
        data.prev_frame,
        &data.stuff->blob,
        p
    );
    
    tracklet->add_basic_at(frameIndex, (long_t)data.index);
    fish->_basic_stuff[data.index] = std::move(data.stuff);
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
    
    Individual *fish = IndividualManager::make_individual(Idx_t{ID});
    auto thread_name = get_thread_name();
    set_thread_name(fish->identity().name()+"_read");
    
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
        
        auto id = Identity::Temporary( Idx_t{ID} );
        if(name != id.raw_name() && !name.empty()) {
            auto map = FAST_SETTING(individual_names);
            map[Idx_t{ID}] = name;
            SETTING(individual_names) = map;
        }
    }
    
    //fish->_manually_matched.clear();
    if(_header.version >= Output::ResultsFormat::Versions::V_15) {
        data_long_t tmp;
        uint64_t N;
        ref.read<uint64_t>(N);
        
        for (uint64_t i=0; i<N; ++i) {
            ref.read<data_long_t>(tmp);
            //fish->_manually_matched.insert(Frame_t(tmp));
        }
    }
    
    if(fish->identity().ID() != Idx_t{ID})
        throw U_EXCEPTION("Failed to load ID ", ID," from file ", filename());
    //fish->identity().set_ID(Idx_t(ID));
    
    //MotionRecord *prev = NULL;
    //MotionRecord *prev_weighted = NULL;
    std::future<void> last_future;
    
    uint64_t N;
    ref.read<uint64_t>(N);
    
    auto analysis_range = Tracker::analysis_range();
    bool check_analysis_range = SETTING(analysis_range).value<Range<long_t>>().start != -1 || SETTING(analysis_range).value<Range<long_t>>().end != -1;
    
    std::mutex mutex;
    std::condition_variable variable;
    std::deque<TemporaryData> stuffs;
    std::atomic_bool stop{false};
    
    //! looping through all data points

    size_t index = 0;// start with basic_stuff == zero
    
    double time;
    
    Frame_t prev_frame;
    data_long_t frameIndex;
    
    //!TODO: too much resize.
    fish->_basic_stuff.resize(N);
    fish->_matched_using.resize(N);
    std::fill(fish->_matched_using.begin(), fish->_matched_using.end(), default_config::matching_mode_t::benchmark);
    
    const MotionRecord* prev = nullptr;
    std::future<void> ended;

    try {
        //! start worker that iterates the frames / fills in
        //! additional info that was not read directly from the file
        //! per frame.
        ended = _load_pool.enqueue([&stop, &stuffs, &variable, cache, fish, &mutex]() mutable {
            auto thread_name = get_thread_name();
            set_thread_name("read_individual_"+fish->identity().name()+"_worker");
            [[maybe_unused]] track::TrackingThreadG g{};
            CachedSettings settings;
            
            std::unique_lock<std::mutex> guard(mutex);
            while(!stop || !stuffs.empty()) {
                variable.wait_for(guard, std::chrono::milliseconds(1));
                
                while(!stuffs.empty()) {
                    auto data = std::move(stuffs.front());
                    stuffs.pop_front();
                    
                    guard.unlock();
                    auto frame = data.index;
                    try {
                        process_frame(settings, cache, fish, std::move(data));
                    } catch(const std::exception& ex) {
                        FormatExcept("Exception when processing frame ",frame," for fish ", fish, ": ", ex.what());
                    } catch(...) {
                        FormatExcept("Unknown exception when processing frame ",frame," for fish ", fish);
                    }
                    guard.lock();
                }
            }
                
            set_thread_name(thread_name);
        });
        
    } catch(const UtilsException& e) {
        FormatExcept("Exception when starting worker threads on _load_pool: ", e.what());
        throw;
    }

    try {
        //! read the actual frame data, pushing to worker thread each time
        for (uint64_t i=0; i<N; i++) {
            ref.read<data_long_t>(frameIndex);
            //if(!prev_frame.valid()
            //   && (!check_analysis_range || Frame_t(frameIndex) >= analysis_range.start))
            //    prev_frame = frameIndex;
            
            TemporaryData data;
            {
                ref.read<Vec2>(data.pos);
                ref.read<float>(data.angle);
                
                if(_header.version < Output::ResultsFormat::Versions::V_27) {
                    if(_header.version >= Output::ResultsFormat::Versions::V_8)
                        ref.read<double>(time);
                    else
                        ref.read_convert<float>(time);
                } else {
                    auto p = Tracker::properties(Frame_t(frameIndex), cache);
                    if(p) time = p->time();
                    else {
                        FormatWarning("Frame ", frameIndex, " seems to be outside the range of the video file.");
                        time = -1;
                    }
                }
            }
            
            //fish->_blob_indices[frameIndex] = ref.read<uint32_t>();
            if (_header.version < Output::ResultsFormat::Versions::V_7) {
                // blob index no longer used
                uint32_t bdx;
                ref.read<uint32_t>(bdx);
            }
            
            data.time = time;
            data.index = index;
            data.stuff = std::make_unique<BasicStuff>();
            data.stuff->frame = Frame_t(frameIndex);
            
            read_blob(ref, data.stuff->blob);
            
            if(_header.version >= Output::ResultsFormat::Versions::V_7 && _header.version < Output::ResultsFormat::Versions::V_29)
            {
                static Vec2 tmp;
                ref.read<Vec2>(tmp);
            }
            
            if(check_analysis_range && not analysis_range.contains(Frame_t(frameIndex))) {
                continue;
            }
            
            assert(frameIndex <= analysis_range.end().get()
                   && frameIndex >= analysis_range.start().get());
            
            data.prev_frame = prev_frame;
            prev_frame = Frame_t(frameIndex);
            
            data.stuff->centroid.init(prev, time, data.pos, data.angle);
            prev = &data.stuff->centroid;
            
            {
                std::unique_lock guard(mutex);
                stuffs.push_back(std::move(data));
            }
            variable.notify_one();
            
            ++index;
            
            if(i%100000 == 0 && i)
                Print("Blob ", i,"/",N);
        }
        
    } catch(...) {
        FormatExcept("Exception reading ", N, " blobs from ", filename(),".");
        stop = true;
        variable.notify_all();
        ended.get();
        throw;
    }
    
    stop = true;
    variable.notify_all();
    ended.get();
    
    //!TODO: resize back to intended size
    if(index != fish->_basic_stuff.size()) {
        fish->_basic_stuff.resize(index);
        fish->_matched_using.resize(index);
    }
    
#ifndef NDEBUG
    if(fish->empty()) {
        FormatWarning("Individual ", fish->identity(), " is empty (index=", index, " basic_stuff=", fish->_basic_stuff.size()," N=",N,")");
    }
#endif
    //assert(!fish->empty());
    
    // read pixel information
    if(_header.version >= Versions::V_19) {
        ref.read<uint64_t>(N);
        if(fish->_basic_stuff.empty()) {
            assert(N == 0);
        }
        
        data_long_t frameIndex;
        uint64_t value;
        Frame_t frame;
        
        for(uint64_t i=0; i<N; ++i) {
            ref.read<data_long_t>(frameIndex);
            ref.read<uint64_t>(value);
            
            frame = Frame_t( frameIndex );
            
            if(check_analysis_range && not analysis_range.contains(frame))
                continue;
            
            auto stuff = fish->basic_stuff(frame);
            if(!stuff) {
                FormatExcept("(", fish->identity().ID(),") Cannot find basic_stuff for frame ", frame,".");
            } else
                stuff->thresholded_size = value;
        }
    }
    
    /// Read the outlines / midlines as well as
    /// the typical PostureStuff:
    if(_header.version <= Versions::V_24) {
        ref.read<uint64_t>(N);
        Frame_t frame;
        
        /// prev is pointing to something in prop
        /// so we need to make sure its not used after prop
        /// is freed.
        const MotionRecord *prev = nullptr;
        std::unique_ptr<MotionRecord> prop;
        
        for (uint64_t i=0; i<N; i++) {
            ref.read<data_long_t>(frameIndex);
            frame = Frame_t( frameIndex );
            
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
                } else {
                    time = Tracker::properties(frame)->time();
                }
                
                auto p = std::make_unique<MotionRecord>();
                p->init(prev, time, pos, angle);
                prop = std::move(p);
            }
            
            auto midline = read_midline(ref);
            auto outline = read_outline(ref, midline.get());
        
            prev = prop.get();
            
            if(check_analysis_range && not analysis_range.contains(frame)) {
                continue;
            }
            
            if(FAST_SETTING(calculate_posture)) {
                // save values
                auto stuff = std::make_unique<PostureStuff>();
                
                stuff->frame = frame;
                stuff->cached_pp_midline = std::move(midline);
                stuff->head = std::move(prop);
                stuff->outline = std::move(outline);
                
                if(auto tracklet = fish->tracklet_for(frame);
                   tracklet)
                {
                    tracklet->add_posture_at(std::move(stuff), fish);
                    
                } else {
                    throw U_EXCEPTION("(",fish->identity().ID(),") Have to add basic stuff before adding posture stuff (frame ",frameIndex,").");
                }
            }
        }
        
    } else if(_header.version >= Versions::V_25) {
        // now read outlines and midlines
        ref.read<uint64_t>(N);
        
        std::map<Frame_t, std::unique_ptr<PostureStuff>> sorted;
        data_long_t previous_frame = -1;
        Frame_t frame;
        
        for (uint64_t i=0; i<N; i++) {
            ref.read<data_long_t>(frameIndex);
            if(frameIndex < previous_frame) {
                FormatWarning("Unordered frames (", frameIndex," vs ",previous_frame,")");
                return fish;
            }
            previous_frame = frameIndex;
            frame = Frame_t( frameIndex );
            
            auto midline = read_midline(ref);
            
            if(check_analysis_range && not analysis_range.contains(frame))
                continue;
            
            if(FAST_SETTING(calculate_posture)) {
                // save values
                auto stuff = std::make_unique<PostureStuff>();
                
                stuff->frame = frame;
                stuff->cached_pp_midline = std::move(midline);
                
                auto tracklet = fish->tracklet_for(frame);
                if(!tracklet) throw U_EXCEPTION("(",fish->identity().ID(),") Have to add basic stuff before adding posture stuff (frame ",frameIndex,").");
                
                sorted[stuff->frame] = std::move(stuff);
            }
        }
        
        ref.read<uint64_t>(N);
        for (uint64_t i=0; i<N; i++) {
            ref.read<data_long_t>(frameIndex);
            auto outline = read_outline(ref, nullptr);
            frame = Frame_t(frameIndex);
            
            if(check_analysis_range && not analysis_range.contains(frame))
                continue;
            
            if(FAST_SETTING(calculate_posture)) {
                //fish->posture_stuff(frameIndex);
                PostureStuff* stuff = nullptr;
                auto it = sorted.find(frame);
                
                if(it == sorted.end()) {
                    auto ptr = std::make_unique<PostureStuff>();
                    ptr->frame = frame;
                    stuff = ptr.get();
                    sorted[frame] = std::move(ptr);
                    
                } else
                    stuff = it->second.get();
                
                assert(stuff);
                stuff->outline = std::move(outline);
            }
        }
        
        std::shared_ptr<TrackletInformation> tracklet = nullptr;
        for(auto && [frame, stuff] : sorted) {
            if(!tracklet || !tracklet->contains(frame))
                tracklet = fish->tracklet_for(frame);
            if(!tracklet) throw U_EXCEPTION("(",fish->identity().ID(),") Have to add basic stuff before adding posture stuff (frame ",frame,").");
            
            tracklet->add_posture_at(std::move(stuff), fish);
        }
    }

    /// Deal with QRCode tags
    if (_header.version >= Versions::V_34) {
        uint64_t N;
        ref.read<uint64_t>(N);

        ska::bytell_hash_map<Frame_t, IDaverage> qrcode_identities;
        ska::bytell_hash_map<Frame_t, std::vector<tags::Detection>> identifiers;

        for (uint64_t i = 0; i < N; ++i) {
            data_long_t frame;
            ref.read<data_long_t>(frame);

            int32_t id;
            ref.read<int32_t>(id);

            float prob;
            ref.read<float>(prob);

            uint32_t samples;
            ref.read<uint32_t>(samples);

#if !COMMONS_NO_PYTHON
            auto seg = fish->tracklet_for(Frame_t(frame));
            if(seg) {
                for(auto i : seg->basic_index) {
                    if(i == -1) continue;
                    auto center = fish->_basic_stuff[i]->blob.calculate_bounds().center();

                    identifiers[fish->_basic_stuff[i]->frame].push_back(tags::Detection{
                        .id = Idx_t(id),
                        .pos = center,
                        .bid = fish->_basic_stuff[i]->blob.blob_id(),
                        .p = prob
                    });
                }
            }

            qrcode_identities[Frame_t(frame)] = { id, prob, samples };
#endif
        }

        tags::detected(std::move(identifiers));
        fish->_qrcode_identities = qrcode_identities;
    }
    
    /// Read auto match property
    fish->automatically_matched.clear();
    if(_header.version >= Versions::V_39)
    {
        uint64_t N;
        ref.read<uint64_t>(N);
        
        for(uint64_t i = 0; i < N; ++i) {
            uint32_t index;
            ref.read<uint32_t>(index);
            
            if(not Frame_t(index).valid()) {
                FormatWarning("[AutoAssign] Read invalid frame ", index, " from ", filename(), " for individual ", fish->identity());
                continue;
            }
            fish->automatically_matched.insert(Frame_t(index));
        }
    }
    
    //delta = this->tell() - pos_before;
    try {
        //post_process(this, fish);
        _post_pool.enqueue(fish);
    } catch(const UtilsException& e) {
        FormatExcept("Exception when starting worker threads on _post_pool: ", e.what());
        throw;
    }
    
    //if(N > 1000)
    set_thread_name(thread_name);
    return fish;
}

void Output::ResultsFormat::read_single_individual(Individual** out_ptr) {
    struct Callback {
        std::function<void()> _fn;
        Callback(std::function<void()> fn) : _fn(fn) {}
        ~Callback() {
            try {
                _fn();
            } catch(const std::exception& e) {
                FormatExcept("Caught an exception inside ~Callback(): ",e.what());
            }
        }
    };
    
    auto results = dynamic_cast<Output::ResultsFormat*>(this);
    auto callback = std::make_unique<Callback>([results, quiet = GlobalSettings::is_runtime_quiet()](){
        ++results->_N_written;
        
        if(!quiet) {
            auto N = results->_expected_individuals.load();
            double N_written = results->_N_written.load();
            if(N <= 100 || results->_N_written % max(100u, uint64_t(N * 0.01)) == 0) {
                Print("Read individual ", int64_t(N_written),"/", N," (",dec<2>(double(N_written) / double(N) * 100),"%)...");
                results->_update_progress("", N_written / double(N), results->filename().str()+"\n<ref>loading individual</ref> <number>"+Meta::toStr(results->_N_written)+"</number> <ref>of</ref> <number>"+Meta::toStr(N)+"</number>");
            }
        }
    });
    
    if(results && results->_header.version >= Output::ResultsFormat::V_18) {
        uint64_t size, uncompressed_size;
        read<uint64_t>(size);
        read<uint64_t>(uncompressed_size);
        
        auto in = Meta::toStr(FileSize(size));
        auto out = Meta::toStr(FileSize(uncompressed_size));
        
        auto ptr = read_data_fast(size);
        
        results->_generic_pool.enqueue([ptr, uncompressed_size, out_ptr, results, size, callback = std::move(callback)]()
        {
            std::vector<char> cache;
            DataPackage /*compressed_block, */uncompressed_block;
            uncompressed_block.resize(uncompressed_size, false);
            
            lzo_uint new_len;
            if(lzo1x_decompress((uchar*)ptr,size,(uchar*)uncompressed_block.data(),&new_len,NULL) == LZO_E_OK)
            {
                if(new_len != uncompressed_size)
                    FormatWarning("Uncompressed size ", new_len," is different than expected ",uncompressed_size);
                ReadonlyMemoryWrapper compressed((uchar*)uncompressed_block.data(), new_len);
                (*out_ptr) = results->read_individual(compressed, results->_property_cache.get());
                
            } else {
                throw U_EXCEPTION("Failed to decode individual from file ",results->filename());
            }
        });
        
        return;
    }
    
    if(!results)
        throw U_EXCEPTION("This is not ResultsFormat.");
    
    *out_ptr = results->read_individual(*this, results->_property_cache.get());
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
     *  (Nx (8 bytes frameIndex + 12 bytes)) MotionRecord for centroid
     *  (Nx k bytes) Blob
     *  (Nx (8 bytes M + Mx 1 byte grey values))
     
     *  N number of frames for head
     *  (Nx (8 bytes frameIndex + 12 bytes)) MotionRecord for head
     *  (Nx Midline)
     *  (Nx Outline)
     */
    
    const uint64_t pack_size = Output::ResultsFormat::estimate_individual_size(val);
    auto *ptr = static_cast<Output::ResultsFormat*>(this);
    const std::function<void(uint64_t)> callback = [ptr](uint64_t pos) {
        pos += ptr->current_offset();
        
        if(pos - ptr->last_callback > ptr->estimated_size * 0.01) {
            ptr->_update_progress("", min(1.0, double(pos)/double(ptr->estimated_size)), "");
            ptr->last_callback = pos;
        }
    };
    
    DataPackage pack(pack_size, &callback);
    
    // header
    assert(val.identity().ID().valid());
    uint32_t ID = val.identity().ID().get();
    pack.write<uint32_t>(ID);
    pack.write<std::string>(val.identity().raw_name());
    
    pack.write<uint64_t>(0);
    //pack.write<uint64_t>(val._manually_matched.size());
    //for (auto m : val._manually_matched)
    //    pack.write<data_long_t>(m.get());
    
    // centroid based information
    pack.write<uint64_t>(val._basic_stuff.size());
    
    {
        // sorting by frame index std::less style (0 < N)
        static constexpr auto compare = [](const auto& A, const auto& B) -> bool {
            return A->frame.get() < B->frame.get();
        };
        
        // construct sorted set
        std::set<const BasicStuff*, decltype(compare)> sorted{ compare };
        for (auto& ptr : val._basic_stuff)
            sorted.insert(ptr.get());
    
        // write it to file
        for(auto &stuff : sorted) {
            // write frame number
            pack.write<data_long_t>(stuff->frame.get());
            
            // write centroid MotionRecord
            pack.write<MotionRecord>(stuff->centroid);
            
            // assume we have a blob and grey values as well
            pack.write<pv::BlobPtr>(stuff->blob.unpack());
        }
    }
    
    // write pixel size information
    pack.write<uint64_t>(val._basic_stuff.size());
    for(auto & stuff : val._basic_stuff) {
        pack.write<data_long_t>(stuff->frame.get());
        pack.write<uint64_t>(stuff->thresholded_size);
    }
    
    //auto str = Meta::toStr(FileSize(pack.size()));
    
    // head based information
    /*pack.write<uint64_t>(val._head.size());
    
    for (auto &c : val._head) {
        // write frame number
        pack.write<data_long_t>(c.first);
        
        // write head MotionRecord
        pack.write(*c.second);
    }*/
    
    // write N, and then write all midlines and outlines (unprocessed)
    std::map<data_long_t, const Midline*> cached_pp_midlines;
    std::map<data_long_t, const MinimalOutline*> outlines;
    for(auto& stuff : val._posture_stuff) {
        if(stuff->cached_pp_midline)
            cached_pp_midlines[stuff->frame.get()] = stuff->cached_pp_midline.get();
        if(stuff->outline)
            outlines[stuff->frame.get()] = &stuff->outline;
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

#if !COMMONS_NO_PYTHON
    pack.write<uint64_t>(val._qrcode_identities.size());
    for (auto& [frame, match] : val._qrcode_identities) {
        pack.write<data_long_t>(frame.get());

        auto& [id, prob, n] = match;
        pack.write<int32_t>(narrow_cast<int32_t>(id));
        pack.write<float>(prob);
        pack.write<uint32_t>(n);
    }
#else
    pack.write<uint64_t>(0u);
#endif
    
    pack.write<uint64_t>(val.automatically_matched.size());
    for(auto &frame : val.automatically_matched) {
        pack.write<uint32_t>(frame.valid() ? frame.get() : static_cast<uint32_t>(-1));
    }
    
    //str = Meta::toStr(FileSize(pack.size()));
    //auto estimate = Meta::toStr(FileSize(pack_size));
    //auto per_frame = Meta::toStr(FileSize(pack.size() / double(val.frame_count())));
    
    
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
            
            Print("Saved ", dec<2>(double(ptr->_N_written.load() + 1) / double(ptr->_expected_individuals.load()) * 100),"%... (individual ", val.identity().ID()," compressed from ", before.c_str()," to ", after.c_str(),").");
        }
    
    } else {
        throw U_EXCEPTION("Compression of ",pack.size()," bytes failed (individual ",val.identity().ID(),").");
    }
    
    return write(pack);
}

namespace Output {
    Timer reading_timer;
    float bytes_per_second = 0, samples = 0;
    std::string speed = "";
    float percent_read;
    
    void ResultsFormat::_read_header() {
        std::string version_string;
        read<std::string>(version_string);
        if(!utils::beginsWith(version_string, "TRACK"))
            throw U_EXCEPTION("Illegal file format for tracking results.");
        
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
        
        _header.tracklets.clear();
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
                
                _header.tracklets.push_back(Range<Frame_t>{
                    Frame_t(narrow_cast<Frame_t::number_t>(start)),
                    Frame_t(narrow_cast<Frame_t::number_t>(end))
                });
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

        if (_header.version >= ResultsFormat::V_34) {
            uint64_t stamp;
            read<uint64_t>(stamp);
            _header.creation_time = timestamp_t{ stamp };
        }
        
        if(_header.version >= ResultsFormat::V_38) {
            _header.midline_factor = 1;
        } else {
            _header.midline_factor = 10; // it used to be 10 always
        }
        
        if(_header.version >= ResultsFormat::V_14) {
            read<std::string>(_header.settings);
        }
        
        if(_header.version >= ResultsFormat::V_23) {
            read<std::string>(_header.cmd_line);
        }

        // read recognition data
        if(_header.version >= ResultsFormat::V_13) {
            std::vector<float> tmp;
            uint64_t L;
            read<uint64_t>(L);
            data_long_t frame;
            uint64_t size, vsize;
            uint32_t bid;
            _header.rec_data.clear();

            if(L > 0) {
                _header.has_recognition_data = true;
            }

            for(uint64_t i = 0; i < L; ++i) {
                read<data_long_t>(frame);
                read<uint64_t>(size);
                auto smaller = narrow_cast<uint16_t>(size);

                for(uint16_t j = 0; j < smaller; ++j) {
                    read<uint32_t>(bid);
                    read<uint64_t>(vsize);
                    tmp.resize(narrow_cast<uint16_t>(vsize)); // more than 2^16 identities? hardly. this also prevents malformed files from crashing the program
                    read_data(vsize * sizeof(float), (char*)tmp.data());

                    _header.rec_data[Frame_t(frame)][pv::bid(bid)] = tmp;
                }
            }
        }

        _header.has_categories = false;
        if(header().version >= ResultsFormat::Versions::V_33) {
            // read category data
            if(Categorize::DataStore::wants_to_read(*this, header().version))
                _header.has_categories = true;
        }
        
        if(!GlobalSettings::is_runtime_quiet()) {
            DebugHeader("READING PROGRAM STATE");
            Print("Read head of ",filename()," (version:V_",(int)_header.version+1," gui_frame:",_header.gui_frame," analysis_range:",_header.analysis_range.start,"-",_header.analysis_range.end," created at ",_header.creation_time.to_date_string()," has_categories:", _header.has_categories, " recognition:", _header.rec_data.size(), ")");
            Print("Generated with command-line: ",_header.cmd_line);
        }
    }
    
    void ResultsFormat::_write_header() {
        std::string version_string = "TRACK"+std::to_string((int)Versions::current);
        write<std::string>(version_string);
        if(!GlobalSettings::is_runtime_quiet()) {
            Print("Writing version string ",version_string);
            Print("Writing frame ", _header.gui_frame);
        }
        write<uint64_t>(_header.gui_frame);
        auto consecutive = Tracker::instance()->consecutive();
        write<uint32_t>((uint32_t)consecutive.size());
        for (auto &c : consecutive) {
            write<uint32_t>(sign_cast<uint32_t>(c.start.get()));
            write<uint32_t>(sign_cast<uint32_t>(c.end.get()));
        }
        
        write<Size2>(Tracker::average().bounds().size());
        write<uint64_t>(FAST_SETTING(video_length));
        
        uint64_t bytes = Tracker::average().cols * Tracker::average().rows;
        write_data(bytes, (const char*)Tracker::average().data());
        
        auto range = FAST_SETTING(analysis_range);
        write<int64_t>(range.start);
        write<int64_t>(range.end);
        
        write<uint64_t>(_header.creation_time.get());
        
        //write<int32_t>(MinimalOutline::factor);

        std::string text = default_config::generate_delta_config(AccessLevelType::LOAD, nullptr, true, _header.exclude_settings).to_settings();
        write<std::string>(text);
        write<std::string>(SETTING(cmd_line).value<std::string>());

        // write recognition data
        if(not Tracker::instance()->has_vi_predictions())
            write<uint64_t>(0);
        else {
            write<uint64_t>(Tracker::instance()->number_vi_predictions());
            Tracker::instance()->transform_vi_predictions([&](auto& frame, auto& map) {
                write<data_long_t>(frame.get());
                write<uint64_t>(map.size());

                for(auto&& [bid, vector] : map) {
                    write<uint32_t>((uint32_t)bid);
                    write<uint64_t>(vector.size());
                    write_data(sizeof(float) * vector.size(), (const char*)vector.data());
                }
            });
        }

        // write categorization data, if it exists
        Categorize::DataStore::write(*this, header().version);
        
        //! write other tag representation
        tags::write(*this);
        
        //! write automatic assignments, if we have any
        AutoAssign::write(*this);
    }
    
    uint64_t ResultsFormat::write_data(uint64_t num_bytes, const char *buffer) {
        if(current_offset() - last_callback > estimated_size * 0.01) {
            _update_progress("", min(1.0, double(current_offset()+num_bytes)/double(estimated_size)), "");
            last_callback = current_offset();
        }
        
        return DataFormat::write_data(num_bytes, buffer);
    }
    
    uint64_t ResultsFormat::estimate_individual_size(const Individual& val) {
        static const uint64_t physical_properties_size = (2+1+1)*sizeof(float);
        const uint64_t pack_size =
        4 + sizeof(data_long_t)*2
        + sizeof(uchar)*3
        + val._basic_stuff.size() * (sizeof(data_long_t)+physical_properties_size+sizeof(uint32_t)+(val._basic_stuff.empty() ? 100 : (*val._basic_stuff.begin())->blob._lines.size())*1.1*sizeof(pv::ShortHorizontalLine))
        + val._posture_stuff.size() * (sizeof(data_long_t)+sizeof(uint64_t)+((val._posture_stuff.empty() || !val._posture_stuff.front()->cached_pp_midline ?SETTING(midline_resolution).value<uint32_t>() : (*val._posture_stuff.begin())->cached_pp_midline->size()) * sizeof(float) * 2 + sizeof(float) * 5 + sizeof(uint64_t))+physical_properties_size+((val._posture_stuff.empty() || !val._posture_stuff.front()->outline ? 0 : val._posture_stuff.front()->outline.size()*1.1)*sizeof(uint16_t) + sizeof(float)*2+sizeof(uint64_t)))
        + val._basic_stuff.size() * sizeof(decltype(BasicStuff::thresholded_size)) + sizeof(uint64_t);
        
        return pack_size;
    }
    
    void ResultsFormat::write_file(
        const std::vector<track::FrameProperties::Ptr> &frames,
        const active_individuals_map_t &active_individuals_frame,
        const individuals_map_t &individuals)
    {
        estimated_size = sizeof(uint64_t)*3
            + frames.size() * (sizeof(data_long_t)+sizeof(CompatibilityFrameProperties))
            + active_individuals_frame.size() * (sizeof(data_long_t)
            + sizeof(uint64_t)
            + (active_individuals_frame.empty()
               ? individuals.size()
               : active_individuals_frame.begin()->second->size())
             * sizeof(data_long_t));
        
        _expected_individuals = individuals.size();
        for (auto& fish: individuals) {
            estimated_size += estimate_individual_size(*fish.second);
        }
        
        const bool quiet = GlobalSettings::is_runtime_quiet();
        if(!quiet) {
            Print("Estimating ", FileSize(estimated_size)," for the whole file.");
        }
        
        start_writing(true);
        
        // write frame properties
        write<uint64_t>(frames.size());
        if(!quiet)
            Print("Writing ", frames.size()," frames");
        for (auto &p : frames)
            write<track::FrameProperties>(*p);
        
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
            write<data_long_t>(p.first.get());
            write<uint64_t>(p.second->size());
            for(auto &fish : *p.second) {
                data_long_t ID = fish->identity().ID().get();
                write<data_long_t>(ID);
            }
        }
    }
    
    Path TrackingResults::expected_filename() {        
        file::Path filename = GlobalSettings::read([](const Configuration& config) {
            return settings::find_output_name(config.values);
        });
        filename = filename.has_extension("pv")
                    ? filename.replace_extension("results")
                    : filename.add_extension("results");
        return file::DataLocation::parse("output", filename);
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
        auto gui_frame = SETTING(gui_frame).value<Frame_t>();
        file.header().gui_frame = sign_cast<uint64_t>(gui_frame.valid() ? gui_frame.get() : 0);
        file.header().creation_time = Image::now();
        file.header().exclude_settings = exclude_settings;
        file.write_file(_tracker._added_frames, IndividualManager::_all_frames(), IndividualManager::individuals());
        file.close();
        
        if(update_progress)
            update_progress("", 1.f, "");
        
        // go back from .tmp01 to .results
        if(filename.move_to(filename.remove_extension())) {
            filename = filename.remove_extension();
            if(!GlobalSettings::is_runtime_quiet()) {
                DebugHeader("Finished writing ", filename, ".");
                DebugCallback("Finished writing ", filename, ".");
            }
        } else
            throw U_EXCEPTION("Cannot move ",filename," to ",filename.remove_extension()," (but results have been saved, you just have to rename the file).");
    }
    
    void TrackingResults::clean_up() {
        _tracker._added_frames.clear();
        _tracker.clear_properties();
        IndividualManager::clear();
        _tracker._startFrame = Frame_t();
        _tracker._endFrame = Frame_t();
        _tracker._max_individuals = 0;
        _tracker._consecutive.clear();
        FOI::clear();
    }
    
    ResultsFormat::Header TrackingResults::load_header(const file::Path &filename) {
        bytes_per_second = samples = percent_read = 0;
        
        if(!GlobalSettings::is_runtime_quiet())
            Print("Trying to open results ",filename.str()," (retrieve header only)");
        ResultsFormat file(filename.str(), [](const auto&, auto, const auto&){});
        file.start_reading();
        /// we will for sure read this sequentially
        file.hint_access_pattern(DataFormat::AccessPattern::Sequential);
        return file.header();
    }

void TrackingResults::update_fois(const std::function<void(const std::string&, float, const std::string&)>& update_progress) {
    const auto number_fish = FAST_SETTING(track_max_individuals);
    data_long_t prev = 0;
    data_long_t n = 0;
    
    //auto it = _tracker._active_individuals_frame.begin();
    if(IndividualManager::_all_frames().size() != _tracker._added_frames.size()) {
        throw U_EXCEPTION("This is unexpected (",IndividualManager::_all_frames().size()," != ",_tracker._added_frames.size(),").");
    }
    
    const track::FrameProperties* prev_props = nullptr;
    Frame_t prev_frame;
    
    ska::bytell_hash_map<Idx_t, Individual::tracklet_map::const_iterator> iterator_map;
    
    track::CachedSettings cached;
    
    for(const auto &props : _tracker._added_frames) {
        // number of individuals actually assigned in this frame
        /*n = 0;
        for(const auto &fish : it->second) {
            n += fish->has(props.frame) ? 1 : 0;
        }*/
        n = props->active_individuals();
        
        // update tracker with the numbers
        //assert(it->first == props.frame);
        auto &active = *IndividualManager::active_individuals(props->frame()).value();
        assert(props->frame().valid());
        if(prev_props && prev_frame > props->frame() + 1_f)
            prev_props = nullptr;
        
        _tracker.update_consecutive(cached, active, props->frame(), false);
        _tracker.update_warnings(cached, props->frame(), props->time(), (long_t)number_fish, (long_t)n, (long_t)prev, props.get(), prev_props, active, iterator_map);
        
        prev = n;
        prev_props = props.get();
        prev_frame = props->frame();
        
        if(props->frame().get() % max(1u, uint64_t(_tracker._added_frames.size() / 10u)) == 0) {
            update_progress("FOIs...", props->frame().get() / float(_tracker.end_frame().get()), Meta::toStr(props->frame())+" / "+Meta::toStr(_tracker.end_frame()));
            if(!GlobalSettings::is_runtime_quiet())
                Print("\tupdate_fois ", props->frame()," / ",_tracker.end_frame(),"\r");
        }
    }
    
    {
        update_progress("Finding segments...", -1, "");
        DatasetQuality::update();
    }
}

FrameProperties CompatibilityFrameProperties::convert(Frame_t frame) const {
    return FrameProperties(frame, time, timestamp);
}
    
    ResultsFormat::Header TrackingResults::load(std::function<void(const std::string&, float, const std::string&)> update_progress, Path filename) {
        Timer loading_timer;
        
        if (filename.empty())
            filename = expected_filename();
        else if (!filename.exists())
            FormatError("Cannot find ",filename.str()," as requested. Trying standard paths.");

        if(!filename.exists()) {
            file::Path file = SETTING(filename).value<Path>();
            file = file.has_extension("pv")
                    ? file.replace_extension("results")
                    : file.add_extension("results");
            
            //file = file::DataLocation::parse("input", filename.filename());
            if(file.is_regular()) {
                FormatWarning("Not loading from the output folder, but from the input folder because ", filename," could not be found, but ",file," could.");
                filename = file;
            } else
                Print("Searched at ",file,", but also couldnt be found.");
        }
        
        bytes_per_second = samples = percent_read = 0;
        
        if(!GlobalSettings::is_runtime_quiet())
            Print("Trying to open results ",filename.str());
        ResultsFormat file(filename.str(), update_progress);
        
        clean_up();
        
        data_long_t biggest_id = -1;
        
        Tracker::instance()->_individual_add_iterator_map.clear();
        Tracker::instance()->_tracklet_map_known_capacity.clear();

        file.start_reading();

        //if(!file.header().rec_data.empty())
            Tracker::instance()->set_vi_data(file.header().rec_data);
        //else if(!file.header().rec_data.empty())
        //    FormatWarning("Throwing away ", file.header().rec_data.size(), " entries in recognition data from the .results file, since recognition was disabled.");

        // read the actual categorization data first
        if(file.header().has_categories)
            Categorize::DataStore::read(file, file.header().version);
        
        if(file.header().version >= ResultsFormat::Versions::V_35) {
            tags::read(file);
        }
        
        //! see if we need automatic assignments
        if(file.header().version >= ResultsFormat::Versions::V_39) {
            AutoAssign::read(file);
        }

        // read frame properties
        uint64_t L;
        file.read<uint64_t>(L);
        
        track::FrameProperties props;
        CompatibilityFrameProperties comp;
        data_long_t frameIndex;
        bool check_analysis_range = SETTING(analysis_range).value<Range<long_t>>().start != -1 || SETTING(analysis_range).value<Range<long_t>>().end != -1;
        
        auto analysis_range = Tracker::analysis_range();
        _tracker.clear_properties();
        file._property_cache = std::make_shared<CacheHints>(L);
        
        for (uint64_t i=0; i<L; i++) {
            file.read<data_long_t>(frameIndex);
            
            if(file._header.version >= ResultsFormat::Versions::V_2) {
                file.read<track::FrameProperties>(props);
                props = FrameProperties(Frame_t(frameIndex), props.time(), props.timestamp());
            } else {
                file.read<CompatibilityFrameProperties>(comp);
                props = comp.convert(Frame_t(frameIndex));
            }
            
            if(check_analysis_range && not analysis_range.contains(props.frame()))
                continue;
            
            if(!_tracker._startFrame.load().valid())
                _tracker._startFrame = props.frame();
            _tracker._endFrame = props.frame();
            
            _tracker.add_next_frame(props);
        }
        
        // read the individuals
        std::map<Idx_t, Individual*> map_id_ptr;
        std::vector<Individual*> fishes;
        
        file.read<uint64_t>(L);
        fishes.resize(narrow_cast<uint16_t>(L)); // prevent crashes caused by malformed files
        file._expected_individuals = L;
        file._N_written = 0;
        
        for (uint64_t i=0; i<L; i++) {
            fishes[i] = nullptr;
            
            file.read_single_individual(&fishes[i]);
            
            if(BOOL_SETTING(terminate)) {
                file._generic_pool.wait();
                file._post_pool.wait();
                clean_up();
                return {};
            }
        }
        
        file._generic_pool.wait();
        file._post_pool.wait();

        if(file._exception_ptr) //! an exception happened inside the generic pool
            std::rethrow_exception(file._exception_ptr);
        
        for(auto &fish : fishes) {
            if(fish) {
                if(biggest_id < (data_long_t)fish->identity().ID().get())
                    biggest_id = fish->identity().ID().get();
                map_id_ptr[fish->identity().ID()] = fish;
            }
        }
        
        track::Identity::Reset(Idx_t(biggest_id+1));
        
        uint64_t n;
        data_long_t ID;
        Frame_t frame;
        file.read<uint64_t>(L);
        for (uint64_t i=0; i<L; i++) {
            auto active = std::make_unique<set_of_individuals_t>();
            
            file.read<data_long_t>(frameIndex);
            file.read<uint64_t>(n);
            
            frame = Frame_t(frameIndex);
            
            for (uint64_t j=0; j<n; j++) {
                file.read<data_long_t>(ID);
                
                if(check_analysis_range && not analysis_range.contains(frame))
                    continue;
                
                auto it = map_id_ptr.find(Idx_t(ID));
                if (it == map_id_ptr.end()) {
                   throw U_EXCEPTION("Cannot find individual with ID ", ID," in map.");
                } else if(not (*it).second->start_frame().valid()
                          || (*it).second->start_frame() > frame) {
                    //FormatExcept("Individual ", ID," start frame = ", map_id_ptr.at(Idx_t(ID))->start_frame(),", not ",frameIndex);
                    continue;
                }
                auto r = active->insert(it->second);
                if(!std::get<1>(r))
                    throw U_EXCEPTION("Did not insert ID ",ID," for frame ",frameIndex,".");
            }
            
            if(check_analysis_range && not analysis_range.contains(frame))
                continue;
            
            _tracker._max_individuals = max(_tracker._max_individuals.load(), active->size());
            
            IndividualManager::_last_active() = active.get();
            IndividualManager::_all_frames()[frame] = std::move(active);
        }
        
        IndividualManager::_inactive().clear();
        if(IndividualManager::_last_active()) {
            IndividualManager::transform_all([](auto, auto fish){
                if(IndividualManager::_last_active()->find(fish) == IndividualManager::_last_active()->end()) {
                    IndividualManager::_inactive()[fish->identity().ID()] = (fish);
                }
            });
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
                for(const auto &fish : Tracker::active_individuals(props->frame())) {
                    n += fish->has(props->frame());
                }
                props->set_active_individuals(n);
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
        
        /// update the tracklet order cache
        _tracker.global_tracklet_order_changed();
        _tracker.global_tracklet_order();
        
        if(_tracker.end_frame().valid())
            _tracker._add_frame_callbacks.callAll(_tracker.end_frame());
        
        if(!GlobalSettings::is_runtime_quiet()) {
            Print("Successfully read file ",file.filename()," (version:V_",(int)file._header.version+1," gui_frame:",file.header().gui_frame,"u start:",Tracker::start_frame(),"u end:",Tracker::end_frame(),"u)");
        
            DurationUS duration{uint64_t(loading_timer.elapsed() * 1000 * 1000)};
            DebugHeader("FINISHED READING PROGRAM STATE IN ", duration);
        }
        
        return file._header;
    }
}
