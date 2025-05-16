#include "Outline.h"
#include "Posture.h"
#include "DebugDrawing.h"
#include <misc/GlobalSettings.h>
#include "Tracker.h"
#include <misc/curve_discussion.h>
#include <misc/Timer.h>
#include <misc/stacktrace.h>
#include <misc/CircularGraph.h>
#include <gui/DrawCVBase.h>
#include <misc/default_config.h>
#include <misc/create_struct.h>

using namespace track;
//#define _DEBUG_MEMORY

#ifdef _DEBUG_MEMORY
std::map<void*, std::tuple<int, std::shared_ptr<void*>>> all_objects;
std::mutex all_mutex;
#endif

//ENUM_CLASS(PEAK_MODE, pointy, broad)

namespace outline {
CREATE_STRUCT(Settings,
    (float, outline_curvature_range_ratio),
    (bool, outline_use_dft),
    (default_config::peak_mode_t::Class, peak_mode),
    (float, midline_walk_offset),
    (uint8_t, outline_approximate),
    (uint8_t, outline_smooth_samples),
    (bool, midline_start_with_head),
    (float, midline_stiff_percentage),
    (uint32_t, midline_resolution),
    (uint8_t, posture_closing_steps),
    (uint8_t, posture_closing_size),
    (float, outline_resample)
)
}
// saved global settings
/*float outline_curvature_range_ratio = 0;
bool outline_use_dft = false;
auto peak_mode = default_config::peak_mode_t::pointy.value();
float midline_walk_offset = 3;
uint8_t outline_approximate = 3;
uint8_t outline_smooth_samples = 0;
bool midline_start_with_head = false;*/

// some debug information
Float2_t _max_curvature = 0;
std::mutex _max_curvature_lock;
Float2_t _average_curvature = 0;
long_t _curvature_samples  = 0;

//uint32_t midline_resolution = 0;
//float midline_stiff_percentage = 0;

//Midline _empty_midline;
//bool _callback_registered = false;

#define OUTLINE_SETTING(NAME) outline::Settings::copy<outline::Settings:: NAME >()

Float2_t Outline::get_curvature_range_ratio() {
    return OUTLINE_SETTING(outline_curvature_range_ratio);
}

uint8_t Outline::get_outline_approximate() {
    return OUTLINE_SETTING(outline_approximate);
}

void Outline::check_constants() {
    outline::Settings::init();
}

Float2_t Outline::average_curvature() {
    std::unique_lock<std::mutex> guard(_max_curvature_lock);
    return _curvature_samples ? _average_curvature / Float2_t(_curvature_samples) : 0;
}

Float2_t Outline::max_curvature() {
    std::unique_lock<std::mutex> guard(_max_curvature_lock);
    return _max_curvature;
}

//#undef assert
//#define assert(e) ((void)0)

std::vector<Vec2> MinimalOutline::uncompress(Float2_t factor) const {
    std::vector<Vec2> vector;
    if(_points.empty())
        return vector;
    
    vector.resize(_points.size());
    
    Vec2 previous = _first;
    Vec2 vec;
    
    vector[0] = _first;
    
    for(size_t i=1; i<_points.size(); ++i) {
        vec.x = char(_points[i] >> 8);
        vec.y = char(_points[i] & 0xff);
        
        vec /= Float2_t(factor);
        
        vector[i] = previous + vec;
        previous = vector[i];
    }
    
    return vector;
}

std::vector<Vec2> MinimalOutline::uncompress() const {
    std::vector<Vec2> vector;
    if(_points.empty())
        return vector;
    
    vector.resize(_points.size());
    
    Vec2 previous = _first;
    Vec2 vec;
    
    vector[0] = _first;
    
    for(size_t i=1; i<_points.size(); ++i) {
        vec.x = char(_points[i] >> 8);
        vec.y = char(_points[i] & 0xff);
        
        vec /= Float2_t(scale);
        
        vector[i] = previous + vec;
        previous = vector[i];
    }
    
    return vector;
}
MinimalOutline::MinimalOutline() : _first(0) {
#ifdef _DEBUG_MEMORY
    std::lock_guard<std::mutex> guard(all_mutex);
    all_objects[this] = retrieve_stacktrace();
#endif
}

MinimalOutline::MinimalOutline(const Outline& outline) {
    if(outline.points().empty())
        FormatWarning("Converting from empty outline.");
    convert_from(outline.points());
#ifdef _DEBUG_MEMORY
    std::lock_guard<std::mutex> guard(all_mutex);
    all_objects[this] = retrieve_stacktrace();
#endif
}

MinimalOutline::~MinimalOutline() {
#ifdef _DEBUG_MEMORY
    std::lock_guard<std::mutex> guard(all_mutex);
    auto it = all_objects.find(this);
    if(it == all_objects.end())
        FormatError("Double delete?");
    else
        all_objects.erase(it);
#endif
}

void MinimalOutline::convert_from(const std::vector<Vec2>& array) {
    _first = array.empty() ? Vec2(0) : array.front();
    
    //float m = 0, lost = 0;
    Vec2 previous = _first;
    
    _points.resize(array.size());
    
    char ux, uy;
    Float2_t x,y;
    Vec2 relative;
    
    const auto N = array.size();
    const auto step = max(1u, N / 10u);
    Float2_t maximum = 0;
    for(size_t i=1; i<N; i+=step) {
        auto m = (array[i] - array[i - 1]).max();
        if(m > maximum)
            maximum = m;
    }
    
    scale = Float2_t(CHAR_MAX) / (maximum * 10 + 1);
    //Print("\t-> ", scale, " scaling factor");
    
    for(size_t i=1; i<N; ++i) {
        relative = array[i] - previous;
        
        x = round(relative.x * scale);
        y = round(relative.y * scale);
        
        if(x >= Float2_t(CHAR_MAX) || y >= Float2_t(CHAR_MAX) || x <= Float2_t(CHAR_MIN) || y <= Float2_t(CHAR_MIN))
            FormatWarning("Cannot compress ",x,",",y," to char (",arange(CHAR_MIN, CHAR_MAX),"). This is an unresolvable error and is likely related to a too large outline_resample value. You will generate invalid outlines with this - instead, you could try resetting `outline_resample` and using `outline_compression`.");
        
        ux = x;
        uy = y;
        
        _points[i] = ((uint16_t(ux) << 8) | (uint16_t(uy) & 0xff));
        previous = previous + Vec2(ux,uy) / scale;
    }
}

size_t Midline::memory_size() const {
    return sizeof(Midline) + sizeof(decltype(_segments)::value_type) * _segments.size();
}

Outline::Outline(std::unique_ptr<std::vector<Vec2>>&& points, Frame_t f)
    : frameIndex(f), _points(std::move(points)), _original_angle(0),
      _concluded(false)//, _needs_invert(false)
{
    check_constants();
    
#ifdef _DEBUG_MEMORY
    std::lock_guard<std::mutex> guard(all_mutex);
    all_objects[this] = retrieve_stacktrace();
#endif
}

Outline::~Outline() {
#ifdef _DEBUG_MEMORY
    std::lock_guard<std::mutex> guard(all_mutex);
    auto it = all_objects.find(this);
    if(it == all_objects.end())
        FormatError("Double delete?");
    else
        all_objects.erase(it);
#endif
}

size_t Outline::memory_size() const {
    return sizeof(Outline) + sizeof(decltype(_points)::element_type::value_type) * (_points ? _points->size() : 0);
}

Vec2& Outline::operator[](size_t index) {
    assert(_points && index >= 0 && index < _points->size());
    return (*_points)[index];
}

const Vec2& Outline::operator[](size_t index) const {
    assert(_points && index >= 0 && index < _points->size());
    return (*_points)[index];
}

void Outline::push_back(const Vec2 &pt) {
    insert(size(), pt);
}

void Outline::push_front(const Vec2 &pt) {
    insert(0, pt);
}

void Outline::remove(size_t index) {
    assert(!_concluded);
    assert(_points);
    _points->erase(_points->begin() + index);
}

void Outline::insert(size_t index, const Vec2 &pt) {
    assert(!_concluded);
    assert(_points);
    _points->insert(_points->begin() + index, pt);
}

/*float Outline::slope(size_t index) const {
    assert(!empty() && index <= size()-1);
    
    size_t idx0 = index ? index-1 : size()-1;
    size_t idx1 = index;
    
    auto &pt0 = at(idx0), &pt1 = at(idx1);
    
    //_slope[index] = cmn::abs((pt1.y - pt0.y) / (pt1.x - pt0.x));
    auto line = pt1 - pt0;
    return (cmn::atan2(line.y, line.x));
}*/

void Outline::finish() {
    _concluded = true;
    
    if(size() <= 3)
        return;
    
    /*curvature_range = calculate_curvature_range(_points->size());
    _curvature.resize(_points->size());
    for(size_t i = 0; i < size(); i++)
        calculate_curvature(i);*/
}

Float2_t Outline::calculate_curvature(const int curvature_range, const std::vector<Vec2>& points, size_t index, Float2_t scale) {
    const auto L = narrow_cast<long_t>(points.size());
    assert(L > 0 && index <= size_t(L)-1);
    
    if(L < 3)
        throw U_EXCEPTION("Cannot calculate curvature with less than 3 values.");
    
    long_t offset = curvature_range * 2 * scale;
    if(L < offset)
        offset = 1;
    
    size_t idx01= long_t(index) < L-offset ? index+offset : long_t(index)+offset-L;
    size_t idx0 = long_t(index) >= offset  ? index-offset : long_t(index)-offset+L;
    //size_t idx1 = index;
    
    const auto &p3 = points[idx01];
    const auto &p2 = points[idx0];
    //const auto &p1 = points[idx1];
    
    return - euclidean_distance(p2, p3);
    
    /*if(p1 == p2 || p1 == p3 || p2 == p3)
        return 0.0f;
    else
        // TODO: this calculation is slow
        // 2 + ((x_2 - x_1)*(y_3 - y_2) - (y_2 - y_1)*(x_3 - x_2)) /
        return 2.f * ((p2.x-p1.x)*(p3.y-p2.y) - (p2.y-p1.y)*(p3.x-p2.x))
        / sqrt(sqdistance(p1, p2) * sqdistance(p2, p3) * sqdistance(p1, p3));*/
}

void Outline::calculate_curvature(size_t index) {
    assert(index < _curvature.size());
    if(not _points || _points->size() <= 3)
        _curvature[index] = 0;
    else
        _curvature[index] = calculate_curvature(curvature_range, *_points, index);
}

periodic::points_t smooth_outline(const periodic::points_t& points, Float2_t range, long_t step) {
    const auto L = narrow_cast<long_t>(points->size());
    
    if(L > range) {
        periodic::points_t smoothed = std::make_unique<std::vector<Vec2>>();
        smoothed->reserve(L);
        
        const Float2_t step_row = range * step;
        
        std::vector<Float2_t> weights;
        weights.reserve(L);
        {
            Float2_t sum = 0;
            for (int i=-step_row; i<=step_row; i+=step) {
                Float2_t val = (step_row-cmn::abs(i))/step_row;
                sum += val;
                
                weights.push_back(val);
            }
            for (auto &v : weights) {
                v /= sum;
            }
        }
        
        Vec2 pt;
        for (long_t i=0; i<L; i++) {
            long_t samples = 0;
            pt.x = pt.y = 0;
            
            for (long_t j=i-step_row; j<=i+step_row; j+=step)
            {
                auto idx = j;
                while (idx < 0)
                    idx += L;
                while (idx >= L)
                    idx -= L;
                
                pt += points->operator[](idx) * weights[samples++];
            }
            
            if(samples)
                smoothed->push_back(pt);
        }
        
        return smoothed;
    }
    
    return nullptr;
}

void Outline::smooth() {
    assert(!_concluded);
    const auto L = narrow_cast<long_t>(size());
    const Float2_t range = OUTLINE_SETTING(outline_smooth_samples);
    
    if(L > range) {
        const long_t step = FAST_SETTING(outline_smooth_step);
        auto smooth = smooth_outline(_points, OUTLINE_SETTING(outline_smooth_samples), step);
        if(smooth)
            _points = std::move(smooth);
        
        /*std::vector<Vec2> smoothed;
        smoothed.reserve(_points->size());
        
        
        const long_t step = FAST_SETTING(outline_smooth_step);
        const float step_row = range * step;
        
        std::vector<float> weights;
        weights.reserve(L);
        {
            float sum = 0;
            for (int i=-step_row; i<=step_row; i+=step) {
                float val = (step_row-abs(i))/step_row;
                sum += val;
                
                weights.push_back(val);
            }
            for (auto &v : weights) {
                v /= sum;
            }
        }

        Vec2 pt;
        for (long_t i=0; i<L; i++) {
            long_t samples = 0;
            pt.x = pt.y = 0;
            
            for (long_t j=i-step_row; j<=i+step_row; j+=step)
            {
                auto idx = j;
                while (idx < 0)
                    idx += L;
                while (idx >= L)
                    idx -= L;
                
                pt += at(idx) * weights[samples++];
            }
            
            if(samples)
                smoothed.push_back(pt);
        }
        
        _points->swap(smoothed);*/
        
        _curvature.clear();
    }
}

inline std::tuple<bool, periodic::scalar_t> is_in_periodic_range(size_t period, const periodic::range_t& range, periodic::scalar_t x)
{
    if(range.start < 0) {
        if(x - period >= range.start)
            return {true, x - period};
        return {x <= range.end, x};
    } else if(range.end >= period) {
        if(x + period <= range.end) {
            return {true, x + period};
        }
        return {x >= range.start, x};
    }
    return {range.contains(x), x};
}

tl::expected<std::tuple<long_t, long_t>, const char*> Outline::offset_to_middle(const DebugInfo& info) {
    assert(_concluded);
    static constexpr bool use_old_method = false;
    if(!_points || _points->empty()) {
        return tl::unexpected("[offset_to_middle] Points were empty.");
    }
    
    if(use_old_method) {
        auto idx = find_tail(info);
        
        // did not find a curvature to use as indication
        // of where the tail is
        if(not idx)
            return tl::unexpected(idx.error());
        
        //assert(!_curvature.empty());
        //assert(_slope.empty());
        assert(_points->size() > size_t(idx.value()));
        
        /*_tail_index -= idx;
        if(_tail_index < 0)
            _tail_index += _points->size();
        
        if(_head_index != -1) {
            _head_index -= idx;
            if(_head_index < 0)
                _head_index += _points->size();
        }*/
        
        _curvature.clear();
        std::rotate(_points->begin(), _points->begin()+idx.value(), _points->end());
        
    } else {
        assert(_curvature.empty());
        
        using namespace periodic;
        
        assert(_points);
        periodic::points_t::element_type* ptr = _points.get();
        auto && [sum, _] = differentiate_and_test_clockwise(*ptr);
        std::vector<scalars_t> diffs;
        scalars_t curv;
        
        if(sum < 0)
            // we have to invert the outline, in order to make it clockwise
            std::reverse(ptr->begin(), ptr->end());
        
        if(OUTLINE_SETTING(outline_approximate) > 0 && _points && !_points->empty()) {
            Vec2 center;
            for(auto &pt : *_points)
                center += pt;
            center /= _points->size();

            auto coeffs = periodic::eft(*_points, OUTLINE_SETTING(outline_approximate));
            if(coeffs) {
                auto pts = std::move(periodic::ieft(*coeffs, coeffs->size(), _points->size(), center, false).front());
                _points = std::move(pts);
                ptr = _points.get();
            }
        }
        
        curv = periodic::curvature(*ptr, max(1, OUTLINE_SETTING(outline_curvature_range_ratio) * ptr->size()), OUTLINE_SETTING(outline_approximate) > 0);
        diffs = periodic::differentiate(*curv, 2);
           
        if(!curv) {
            return tl::unexpected("Cannot calculate the curvature of the points array.");
        }
           
        
        if(info.debug) {
            Print("Smoothed curvature: ", *_points);
        }
        
        auto mode = OUTLINE_SETTING(peak_mode) == default_config::peak_mode_t::broad ? PeakMode::FIND_BROAD : PeakMode::FIND_POINTY;
        auto && [maxima_ptr, minima_ptr] = periodic::find_peaks(curv, 0, diffs, mode);
        
        if(!maxima_ptr || !minima_ptr) {
            return tl::unexpected("Cannot find both a minimum and a maximum peak");
        }
        
        std::vector<Vec2> maxi;
        Float2_t max_h = -1, max_int = -1, max_y = -1;
        scalar_t max_y_idx = 0;
        std::vector<Vec2> max_int_pts;
        for (auto &peak : *maxima_ptr) {
            auto h = cmn::abs(peak.range.end - peak.range.start);
            max_h = max(max_h, h);
            if(peak.integral > max_int) {
                //max_int_index = peak.position.x;
                max_int = peak.integral;
                max_int_pts = peak.points;
            }
            if(peak.position.y > max_y) {
                max_y = peak.position.y;
                max_y_idx = peak.position.x;
            }
        }
        
        std::vector<Peak> high_peaks;
        
        for (auto &peak : *maxima_ptr) {
            maxi.push_back(peak.position);
            
            /*auto h = cmn::abs(peak.range.end - peak.range.start) / max_h;
            auto percent = peak.integral / max_int;
            auto y = h / M_PI;
            y = percent / M_PI;
            h = percent;*/
            
            if(cmn::abs(peak.integral - max_int) <= 1e-5) {
                high_peaks.push_back(peak);
            }
        }
        
        //static std::mutex find_tail_mutex;
        //std::lock_guard<std::mutex> guard(find_tail_mutex);
        
        if(info.debug) {
           auto str = Meta::toStr(high_peaks);
        
           Print("\n", info.frameIndex,"(", info.fdx,"): Finding tail. ",str);
            std::vector<Vec2> maximums, highmax;
            for(auto peak : *maxima_ptr) {
                maximums.push_back(peak.position);
            }
            
            printf("\nstd::vector<float> curv{");
            for(auto c : *curv)
                printf("%f,",c);
            printf("}\n");
            
            scalar_t mi = FLT_MAX, ma = 0;
            for(auto c : *curv) {
                mi = min(c, mi);
                ma = max(c, ma);
            }
            
            for(auto peak : high_peaks)
                highmax.push_back(Vec2(peak.position.x, (ma - mi) * 0.5 + mi));
            
           /* using namespace gui;
            gui::Graph graph(Bounds(0, 0, 800, 400), "curvature", Rangef(0, size()), Rangef(mi, ma));
            graph.add_function(Graph::Function("curv", Graph::Type::DISCRETE, [&](float x) -> float {
                if(x>=0 && x<curv->size())
                    return curv->at(x);
                return GlobalSettings::invalid();
            }));
            graph.add_points("max", maximums);
            graph.add_points("high", highmax);
            
            for (auto &peak : *maxima_ptr) {
                if(peak.range.start < 0)
                    graph.add_line(Vec2(size() + peak.range.start, peak.position.y), Vec2(size(), peak.position.y), Blue);
                if(peak.range.end >= size())
                    graph.add_line(Vec2(peak.range.end - size(), peak.position.y), Vec2(0, peak.position.y), Blue);
                graph.add_line(Vec2(peak.range.start, peak.position.y), Vec2(peak.range.end, peak.position.y), Blue);
            }
            graph.add_line(Vec2(0, max_int), Vec2(size(), max_int), Cyan);
            
            DrawStructure g(800, 400);
            g.wrap_object(graph);
            
            cv::Mat mat = cv::Mat::zeros(400, 800, CV_8UC4);
            CVBase base(mat);
            base.paint(g);
            base.display();*/
        }
        
        scalar_t idx = -1;
        //
        if(mode == PeakMode::FIND_POINTY) {
            idx = max_y_idx;//max_int_index;
            //pts.insert(max_int_pts.begin(), max_int_pts.end());
        } else {
            range_t merged;
            scalar_t max_y = 0;
            std::set<point_t> pts;
            
            for(size_t i=0; i<high_peaks.size(); ++i) {
                if(!i)
                    merged = high_peaks.at(i).range;
                else
                    merged = range_t(min(high_peaks.at(i).range.start, merged.start), max(high_peaks.at(i).range.end, merged.end));
                if(high_peaks.at(i).max_y > max_y)
                    max_y = high_peaks.at(i).max_y;
                
                pts.insert(high_peaks.at(i).points.begin(), high_peaks.at(i).points.end());
            }
            
            scalar_t start = merged.end, end = merged.start;
            for(auto & pt : pts) {
                auto && [in_range, cx] = is_in_periodic_range(ptr->size(), merged, pt.x);
                if(pt.y >= max_y * 0.9 && in_range) {
                    if(start > cx)
                        start = cx;
                    if(end < cx)
                        end = cx;
                }
            }
            
            if(info.debug)
                Print("peak has range ",merged.start,"-",merged.end," (",merged.length(),") - ",start,"-",end);
            idx = round(start + (end - start)*0.5);
            if(idx < 0)
                idx += ptr->size();
            if(idx >= ptr->size())
                idx -= ptr->size();
        }
        
        //auto pos = (ptr->at(idx) - center) * scale + zero;
        //butt.set_pos(pos);
        long_t _tail_index = idx;
        long_t _head_index = -1;
        
        scalar_t max_d = 0;
        scalar_t broadest = 0;
        //scalar_t broadest_idx = -1;
        for(auto &peak : *maxima_ptr) {
            scalar_t d;
            if(peak.position.x >= idx) {
                d = min(cmn::abs(peak.position.x - idx), cmn::abs(peak.position.x - idx - size()));
            } else
                d = min(cmn::abs(idx - peak.position.x), cmn::abs(idx - peak.position.x - size()));
            
            if(d > max_d) {
                max_d = d;
                _head_index = peak.position.x;
            }
            
            if(peak.range.length() > broadest) {
                broadest = peak.range.length();
                //broadest_idx = peak.position.x;
            }
            
        }
        
        if(OUTLINE_SETTING(midline_start_with_head) && _points && _head_index != -1) {
            if(_tail_index != -1) {
                _tail_index -= _head_index;
                if(_tail_index < 0)
                    _tail_index += _points->size();
            }
            
            std::rotate(_points->begin(), _points->begin()+_head_index, _points->end());
            _head_index = 0;
            
        } else if(_points) {
            if(_head_index != -1) {
                _head_index -= _tail_index;
                if(_head_index < 0)
                    _head_index += _points->size();
            }
            
            std::rotate(_points->begin(), _points->begin()+_tail_index, _points->end());
            _tail_index = 0;
        }
        
        if(FAST_SETTING(midline_invert))
            std::swap(_tail_index, _head_index);
        
        return std::make_tuple(_tail_index, _head_index);
    }
    
    return tl::unexpected("Unknown error.");
}

int Outline::calculate_curvature_range(size_t number_points) {
    return max(1, number_points * OUTLINE_SETTING(outline_curvature_range_ratio));
}

void Outline::resample(const Float2_t resampling_distance) {
    assert(!_concluded);
    assert(_curvature.empty());
    
    if(resampling_distance <= 0)
        return;
    
    std::vector<Vec2> resampled;
    Float2_t walked_distance = 0.0;
    const auto L = size();
    if(L <= 1)
        return;
    
    for (uint32_t i=0; i<L; i++) {
        uint32_t idx1 = i + 1, idx0 = i;
        if (idx1 >= L)
            idx1 -= L;
        
        auto &pt0 = (*_points)[idx0];
        auto &pt1 = (*_points)[idx1];
        
        auto line = pt1 - pt0;
        
        const Float2_t len = length(line);
        walked_distance += len;
        
        const Float2_t percent = len / resampling_distance;
        Float2_t walked_percent = walked_distance / resampling_distance;
        
        int offset = 0;
        while (walked_percent >= 1.0) {
            auto pt = pt0 + line * (offset * 1.0 / percent);
            
            resampled.push_back(pt);
            offset++;
            
            walked_distance -= resampling_distance;
            walked_percent -= 1.0;
        }
    }
    
    *_points = resampled;
}

tl::expected<Midline::Ptr, const char*> Outline::calculate_midline(const DebugInfo& info) {
    if(OUTLINE_SETTING(outline_smooth_samples) > 0)
        smooth();
    
    finish(); // conclude the outline and calculate curvature
    
    // offset all points so that the first point is the one with the highest curvature
    auto offset = offset_to_middle(info);
    if(not offset) {
        return tl::unexpected(offset.error());
    }
    
    auto midline = std::make_unique<Midline>();
    
    auto& [_tail_index, _head_index] = offset.value();
    midline->tail_index() = _tail_index;
    midline->head_index() = _head_index;
    
    if(size() <= 1) {
        //Print("Empty outline (",frameIndex,").");
        return tl::unexpected("Empty outline was given, cannot calculate midline.");
    }
    
    // now find pairs for all the points
    #define INV_IDX(IDX) (L+IDX)
    const auto L = narrow_cast<long_t>(size());
    int idx_r = 1, idx_l = -1;
    
    MidlineSegment segment;
    const int max_offset = max(3_F, (Float2_t)OUTLINE_SETTING(midline_walk_offset) * (Float2_t)L);
    
    while (idx_r < INV_IDX(idx_l)) {
        Vec2 pt_r;
        Vec2 pt_l = at(INV_IDX(idx_l));
        
        Float2_t min_d = FLT_MAX;
        int min_idx = -1;
        
        for (int i=0; i<max_offset; i++) {
            if (idx_r + i >= L)
                break;
            
            auto &pt = at(idx_r + i);
            auto line = pt - pt_l;
            
            auto len = length(line);
            if (len < min_d) {
                min_d = len;
                min_idx = idx_r + i;
            }
        }
        
        if (min_idx != -1) {
            pt_r = at(min_idx);
            idx_r = min_idx;
        }
        
        min_d = FLT_MAX;
        min_idx = 1;
        
        for (int i=0; i<max_offset; i++) {
            if (idx_l - i <= -L)
                break;
            
            auto &pt = at(INV_IDX(idx_l - i));
            auto line = pt_r - pt;
            
            auto len = length(line);
            if (len < min_d) {
                min_d = len;
                min_idx = idx_l - i;
            }
        }
        
        if (min_idx != 1) {
            pt_l = at(INV_IDX(min_idx));
            idx_l = min_idx;
        }
        
        auto line = pt_r - pt_l;
        auto m = pt_l + line * 0.5;
        
        segment.pos = m;
        segment.height = euclidean_distance(pt_l, pt_r);
        segment.l_length = euclidean_distance(m, pt_l);
        midline->segments().emplace_back(std::move(segment));
        
        idx_r++;
        idx_l--;
    }
    
    // it may be inverted for some reason
    _inverted_because_previous = false;
    
    if(midline->segments().size() <= 2) {
        //_confidence = 0;
        return tl::unexpected("Too few midline segments calculated.");
    }
    
    return midline;
}

Vec2 Midline::midline_direction() const {
    const long_t samples = max(1, segments().size() * OUTLINE_SETTING(midline_stiff_percentage));
    Vec2 direction{0};
    
    long_t counted{0};
    for (long_t i=0; i<samples && i + 1 < (long_t)segments().size(); i++, counted++) {
        direction += segments().at(i+1).pos - segments().at(i).pos;
    }
    
    if(counted > 0) {
        direction /= Float2_t(counted);
        direction = direction.normalize();
    } else {
        FormatWarning("No samples for midline smoothing. Expected ", samples, " counted ", counted);
    }
    
    return direction;
}

Float2_t Midline::original_angle() const {
    auto direction = midline_direction();
    auto _needs_invert = !FAST_SETTING(midline_invert);
    return atan2(_needs_invert ? direction : -direction);
}

void Midline::post_process(const MovementInformation &movement, DebugInfo info) {
    if(segments().size() <= 2) {
        static bool printed_warning = false;
        if(!printed_warning) {
            FormatWarning("Fewer midline segments in ", info.fdx,"@", info.frameIndex," (", segments().size(),"). This means that your parameters might not be properly adjusted for this video, or this is not an individual. Check `track_posture_threshold` for example.");
            printed_warning = true;
        }
        return;
    }
    
    auto direction = midline_direction();
    auto _needs_invert = !FAST_SETTING(midline_invert);
    direction = _needs_invert ? direction : -direction;
    //auto current_direction = direction[current_index];
    //outline.original_angle() = atan2(current_direction);
    
    if(movement.direction != Vec2(0)) {
        /*auto next_position = movement.position + movement.velocity.x * movement.direction.normalize() * 10;
        Vec2 dpos[2] {
            movement.position + direction[0].normalize() * 10,//movement.velocity.x,
            movement.position + direction[1].normalize() * 10//movement.velocity.x
        };*/
        
        /*float sums[2] = {0,0};
        float samples = 0;
        for(auto d : movement.directions) {
            if(d == Vec2(0))
                continue;
            sums[0] += euclidean_distance(d, dpos[0]);
            sums[1] += euclidean_distance(d, dpos[1]);
            ++samples;
        }
        sums[0] /= samples;
        sums[1] /= samples;
        
        std::vector<float> angles;
        for(auto d : movement.directions) {
            if(d != Vec2(0))
                angles.push_back(atan2(d) * 180 / M_PI);
        }
        
        auto str = Meta::toStr(angles);
        Print("Array is ",str,", direction is ",atan2(direction[0]) * 180 / M_PI,";",atan2(direction[1]) * 180 / M_PI," sums are [",sums[0],",",sums[1],"] ",current_index," => ",sums[1 - current_index] < sums[current_index]);
        
        if(sums[1 - current_index] < sums[current_index]) {*/
        
        if(acos((-direction).dot(movement.direction)) < acos(direction.dot(movement.direction))) {
        //if(euclidean_distance(next_position, dpos[1 - current_index]) < euclidean_distance(next_position, dpos[current_index])) {
            
            /*if(angle_between_vectors(direction[0], direction[1]) >= RADIANS(90)) {
             float adiff0 = angle_between_vectors(direction[1 - current_index], movement.direction);
             float adiff1 = angle_between_vectors(direction[current_index], movement.direction);
             
             if(adiff0 < adiff1) {*/
            _needs_invert = !_needs_invert;
            _inverted_because_previous = true;
            std::swap(head_index(), tail_index());
            //}
        } else {
        }
    }
    
    if(_needs_invert) {
        if(info.debug)
            Print(info.frameIndex,"(",info.fdx,"): inverting Tail: ",tail_index(),", Head: ",head_index());
        
        if(!OUTLINE_SETTING(midline_start_with_head))
            std::reverse(segments().begin(), segments().end());
        
    } else if(OUTLINE_SETTING(midline_start_with_head))
        std::reverse(segments().begin(), segments().end());
    
    if(OUTLINE_SETTING(midline_stiff_percentage) > 0) {
        std::vector<MidlineSegment> old_copy(segments());
        
        size_t center = min(Float2_t(segments().size())-1, // ensure the following i+1 are < size
                            round(Float2_t(segments().size()) * OUTLINE_SETTING(midline_stiff_percentage)) + 1);
        
        const Vec2 center_point = segments().at(center).pos;
        
        // calculate axis to align the body to
        Vec2 axis(0);
        uint32_t count = 0;
        size_t _size = segments().size();
        const size_t extra_axis_offset = min((int)_size, center + max(0.0, (double)_size * 0.1));
        
        for(size_t i=center; i<extra_axis_offset; ++i) {
           axis += (segments().at(i).pos - segments().at(i + 1).pos).normalize();
            ++count;
        }

        if(count > 0)
            axis /= Float2_t(count);

        std::vector<MidlineSegment> copy(segments());
        for(size_t i = center; i > 0; --i) {
            auto &p0 = segments().at(i-1).pos;
            const auto &p1 = segments().at(i).pos;
            
            auto L = (copy.at(i).pos - copy.at(i-1).pos).length();
            
            auto direction_to_center = (p0 - center_point).normalize();
            Vec2 test = ((direction_to_center + axis) * 0.5).normalize();
            
            p0 = p1 + L * test;
        }
        
        /*auto old_code = [center, center_point](std::vector<MidlineSegment> &segments){
            Vec2 axis(0);
            float count = 0;
            const size_t range = min(segments.size(),
                                     size_t(center + segments.size() * 0.1));
            
            for(size_t i=max(1u, center); i<range; i++) {
                axis += (segments.at(i-1).pos - segments.at(i).pos).normalize();
                count++;
            }
            if(count)
                axis /= count;
            //assert(count == extra_axis_offset);
            
            std::vector<float> lengths;
            lengths.resize(center);
            for(size_t i=0; i<center; i++) {
                Vec2 p0 = segments.at(i).pos;
                Vec2 p1 = segments.at(i+1).pos;
                lengths[i] = (p1 - p0).length();
            }
            
            for(long_t i=center-1; i>=0; i--) {
                Vec2 p1 = segments.at(i).pos;
                Vec2 direction = (p1 - center_point).normalize();
                
                Vec2 test = ((direction + axis) * 0.5).normalize();
                
                assert(!cmn::isnan(direction));
                segments.at(i).pos = segments.at(i+1).pos + lengths[i] * test;
            }
            
            return axis;
        };
        
        auto aa = old_code(old_copy);
        auto p = euclidean_distance(aa, axis) / max(aa.abs().max(), axis.abs().max()) * 100;
        if(p > 0)
            FormatWarning(p,"%%");
        if(info.debug) {
            segments() = old_copy;
            Print("Old");
        }
        
        *for (size_t i=0; i<segments().size(); ++i) {
            auto d = euclidean_distance(old_copy[i].pos, segments()[i].pos);
            auto m = max(old_copy[i].pos.abs().max(), segments()[i].pos.abs().max());
            
            if(d > std::numeric_limits<Float2_t>::epsilon() * m)
            {
                FormatWarning(i,": ",old_copy[i].pos.x,",",old_copy[i].pos.y," != ",segments()[i].pos.x,",",segments()[i].pos.y," by ",d / m * 100,"%%");
                //break;
            }
        }*/
    }
    //if(!midline_start_with_head)
    std::reverse(segments().begin(), segments().end());
    //std::swap(_tail_index, _head_index);
}

Midline::Midline()
{
    Outline::check_constants();
    
#ifdef _DEBUG_MEMORY
    std::lock_guard<std::mutex> guard(all_mutex);
    all_objects[this] = retrieve_stacktrace();
#endif
}

Midline::~Midline() {
#ifdef _DEBUG_MEMORY
    std::lock_guard<std::mutex> guard(all_mutex);
    auto it = all_objects.find(this);
    if(it == all_objects.end())
        FormatError("Double delete?");
    else
        all_objects.erase(it);
#endif
}

Midline::Midline(const Midline& other)
: _len(other._len), _angle(other._angle), _offset(other._offset), _front(other._front), _segments(other._segments), _head_index(other._head_index), _tail_index(other._tail_index), _inverted_because_previous(other._inverted_because_previous), _is_normalized(other._is_normalized)
{
#ifdef _DEBUG_MEMORY
    std::lock_guard<std::mutex> guard(all_mutex);
    all_objects[this] = retrieve_stacktrace();
#endif
}

size_t Midline::saved_midlines() {
#ifdef _DEBUG_MEMORY
    std::lock_guard<std::mutex> guard(all_mutex);
    
    std::set<std::string> resolved;
    for(auto && [ptr, tuple] : all_objects) {
        resolved.insert(resolve_stacktrace(tuple));
    }
    auto str = Meta::toStr(resolved);
    auto f = fopen("posture.log", "wb");
    if(f) {
        fwrite(str.data(), sizeof(char), str.length(), f);
        fclose(f);
    } else
        FormatError("Cannot write 'posture.log'");
    
    return all_objects.size();
#else
    return 0;
#endif
}

Float2_t Midline::calculate_angle(const std::vector<MidlineSegment>& segments) {
    if(segments.size() < 2)
        return 0;
    
    const Float2_t center(max(0, segments.size() - 2 - segments.size() * OUTLINE_SETTING(midline_stiff_percentage)));
    const size_t start = center;
    const Float2_t rest = center - start;
    
    auto line = segments.back().pos - (segments.at(start).pos * (1 - rest) + segments.at(start + 1).pos * (rest));
    return cmn::atan2(line.y, line.x);
}

void Midline::fix_length(Float2_t len, std::vector<MidlineSegment>& pts, bool debug) {
    const auto resolution = OUTLINE_SETTING(midline_resolution);
    const Float2_t step = len / Float2_t(resolution);
    
    MidlineSegment seg;
    std::vector<MidlineSegment> midline_points;
    seg = pts.front();
    midline_points.push_back(seg);
    
    uint32_t j=1;
    int last_real = 0;
    Float2_t last_t = -1;
    Float2_t travelled_len = step;
    Float2_t original_len = 0;
    
    for (uint32_t i=1; i<resolution; i++) {
        original_len += (pts.at(i).pos - pts.at(i-1).pos).length();
    }
    
    for (uint32_t i=1; i<resolution; i++) {
        Float2_t md = FLT_MAX;
        Vec2 mdv;
        
        for (; j<resolution && j<pts.size(); j++) {
            auto v0 = pts.at(j-1).pos;
            auto v1 = pts.at(j).pos;
            auto h0 = pts.at(j-1).height;
            auto h1 = pts.at(j).height;
            
            auto ts = (t_circle_line(v0, v1, seg.pos, step));
            assert(!cmn::isnan(ts.first) && !cmn::isnan(ts.second));
            
            Float2_t t0 = ts.first;
            Float2_t t1 = ts.second;
            
            if(t0 >= 0 && t0 <= 1 && t0 > last_t) {
                md = j;
                mdv = v0 + (v1 - v0) * t0;
                seg.height = t0 * h1 + (1-t0) * h0;
                last_t = t0;
                
                break;
                
            } else if(t1 >= 0 && t1 <= 1 && t1 > last_t) {
                md = j;
                mdv = v0 + (v1 - v0) * t1;
                seg.height = t1 * h1 + (1-t1) * h0;
                last_t = t1;
                
                break;
            }
            
            last_t = -1;
        }
        
        if(md != FLT_MAX) {
            seg.pos = mdv;
            travelled_len += step;
            
            midline_points.push_back(seg);
            last_real = i;
            
        } else if(j >= resolution) {
           if(debug)
               Print("Cannot find anything for ",i,", extrapolating ",i - last_real," (",travelled_len," > ",original_len,")");

           if(pts.size()>=3) {
                auto v1 = pts.back().pos;
                auto v0 = pts.at(pts.size()-2).pos;
                auto v2 = pts.at(pts.size()-3).pos;

                auto line = v1 - v0;
                auto line1 = v0 - v2;

               Float2_t angle0 = atan2(line.y, line.x);
               Float2_t angle1 = atan2(line1.y, line1.x);
               assert(!cmn::isnan(angle0));
               assert(!cmn::isnan(angle1));

               Float2_t change = angle0-angle1;
               Float2_t angle = angle0;

                while (midline_points.size() < resolution) {
                    seg.pos += Vec2(cos(angle), sin(angle)) * step;
                    assert(!cmn::isnan(seg.pos));
                    travelled_len += step;

                    //auto d = travelled_len - original_len; UNUSED(d);
                    //assert(d >= 0);
                    angle += change;//change * max(0, (1 - min(1, (d / (original_len * 0.5)))));
                    seg.height *= 0.5;

                    if(debug)
                        Print("Added ", midline_points.size(),"/",resolution);

                    midline_points.push_back(seg);
                    i++;
                }
           }
           
           break;
        }
    }
    
    pts = midline_points;
    //this->len() = len;
    
    //if(!is_normalized())
    //    this->angle() = calculate_angle();
}

gui::Transform Midline::transform(const default_config::individual_image_normalization_t::Class &type, bool to_real_world) const {
    gui::Transform tr;
    
    if(empty())
        return tr;
    
    if(to_real_world) {
        Float2_t angle = this->angle() + M_PI;
        tr.translate(offset());
        tr.rotate(DEGREE(angle));
        return tr;
    }
    
    Float2_t angle = -this->angle() + (type == default_config::individual_image_normalization_t::legacy ? (M_PI) : (M_PI * 0.25_F));
    tr.translate(-front());
    tr.rotate(DEGREE(angle));
    tr.translate(-offset());
    return tr;
}

Vec2 Midline::real_point(const Bounds& bounds, size_t index) const {
    auto pt = segments().at(index).pos;
    return real_point(bounds, pt);
}

Vec2 Midline::real_point(const Bounds& bounds, const Vec2& pt) const {
    auto angle = this->angle() + M_PI;
    auto x = (pt.x * cmn::cos(angle) - pt.y * cmn::sin(angle));
    auto y = (pt.x * cmn::sin(angle) + pt.y * cmn::cos(angle));
    
    return Vec2(x, y) + bounds.pos() + offset();
}

Midline::Ptr Midline::normalize(Float2_t fix_length, bool debug) const {
    assert(!std::isinf(fix_length));
    
    if(is_normalized())
        FormatWarning("Normalizing a normalized midline.");
    
    if(empty() || size()<2)
        return nullptr;
    
    double len = 0.0f;
    for (uint32_t i=1; i<segments().size(); i++) {
        if(!segments().empty())
            len += length(segments()[i].pos - segments()[i-1].pos);
    }
    
    if(len == 0.0) {
        // Midline is empty
        FormatWarning("Midline is empty.");
        return nullptr;
    }
    
    // resample midline to x segments
    const auto resolution = OUTLINE_SETTING(midline_resolution);
    const int max_segments = resolution - 1;
    double step = (double(len) / double(max_segments));
    if(step < 0)
        throw U_EXCEPTION("Step length is negative (",step,") with a length of ",len," and max_segments of ",max_segments,".");
    
    size_t index = 0;
    std::vector<MidlineSegment> reduced;
    reduced.reserve(_segments.size());
    reduced.push_back(_segments.front());
    /*long_t tail_index = -1, head_index = -1;
    if(this->head_index() == 0)
        head_index = 0;
    if(this->tail_index() == 0)
        tail_index = 0;*/
    
    double last_pt_distance = 0.0, distance;
    for(distance = 0.0; distance <= len && index < _segments.size()-1;) {
        MidlineSegment segment;
        assert(!cmn::isnan(distance));
        
        while(distance-last_pt_distance < step && index < _segments.size()-1) {
            auto s0 = _segments.at(index);
            auto s1 = _segments.at(index+1);
            
            auto line = s1.pos - s0.pos;
            auto local_d = length(line);
            
            distance += local_d;
            
            /*if((long_t)index >= this->tail_index() && tail_index == -1)
                tail_index = reduced.size();
            if((long_t)index >= this->head_index() && head_index == -1)
                head_index = reduced.size();*/
            
            index++;
        }
        
        Float2_t off = (distance - last_pt_distance);
        if(off < step) {
            break;
        }
        
        while (off >= step) {
            off -= step;
            
            if(index > 0) {
                auto s0 = _segments.at(index - 1);
                auto s1 = _segments.at(index);
                
                auto line = s1.pos - s0.pos;
                auto local_d = length(line);
                
                Float2_t percent = off;
                if(local_d > 0)
                    percent /= local_d;
                percent = 1.f - percent;
                
                segment.pos = s0.pos + line * percent;
                segment.height = s0.height * (percent) + s1.height * (1.0 - percent);
                segment.l_length = max(s0.l_length, s1.l_length);
                reduced.emplace_back(std::move(segment));
                
                last_pt_distance = distance - length(line * (1.0 - percent));
                
            } else {
                auto s0 = _segments.at(index);
                
                segment.pos = s0.pos;
                segment.height = s0.height;
                
                reduced.emplace_back(std::move(segment));
                
                last_pt_distance = distance;
            }
            
        }
    }
           
    //if(tail_index == -1)
    //    tail_index = reduced.size()-1;
    
    if(length(reduced.back().pos - _segments.back().pos) >= 0.01) {
        reduced.emplace_back(_segments.back());
    }
    
    if(reduced.size() != resolution) {
        return nullptr;
    }
    
    auto line = reduced.at(1).pos - reduced.front().pos;
    Float2_t percent;
    {
        percent = length(line);
        if(len > 0)
            percent /= len;
    }
    
    reduced.front().height = reduced.at(1).height * (percent) + reduced.front().height * (1.0 - percent);
    
    /*if(tail_index >= (long_t)reduced.size())
        tail_index = (long_t)reduced.size() - 1;
    if(head_index >= (long_t)reduced.size())
        head_index = (long_t)reduced.size() - 1;*/
    
    if(fix_length > 0) {
        std::reverse(reduced.begin(), reduced.end());
        this->fix_length(fix_length, reduced, debug);
        std::reverse(reduced.begin(), reduced.end());
    }
    
    len = 0.0f;
    for (uint32_t i=1; i<reduced.size(); i++) {
        assert(!cmn::isnan(reduced[i].pos));
        if(!reduced.empty())
            len += length(reduced[i].pos - reduced[i-1].pos);
    }
    
    // rotate points according to angle
    auto &A = reduced.back().pos;
    Float2_t angle = -calculate_angle(reduced) + M_PI;
    Float2_t offx = A.x,
    offy = A.y;
    
    assert(!cmn::isnan(angle));
    
    gui::Transform tf;
    tf.rotate(DEGREE(angle));
    tf.translate(-offx, -offy);
    
    std::vector<MidlineSegment> rotated;
    long_t L = narrow_cast<long_t>(reduced.size());
    rotated.reserve(L);
    
    for (long_t i=L-1; i>=0; i--) {
        long_t idx0 = i+1;
        if (idx0>=L)
            idx0 -= L;
        
        auto &seg = reduced[i];
        auto pt = tf.transformPoint(seg.pos);
        rotated.emplace_back(MidlineSegment{ seg.height, seg.l_length, pt });
    }
    
    auto front = rotated.front().pos;
    if(front.x || front.y) {
        for (auto &pt : rotated) {
            pt.pos -= front;
        }
    }
    
    auto midline = std::make_unique<Midline>();
    midline->segments() = std::move(rotated);
    midline->len() = len;
    midline->angle() = calculate_angle(reduced);
    midline->offset() = Vec2(offx, offy);
    midline->is_normalized() = true;
    midline->inverted_because_previous() = _inverted_because_previous;
    midline->tail_index() = _tail_index;
    midline->head_index() = _head_index;
    
    return midline;
}

std::vector<Float2_t> Outline::smoothed_curvature_array(Float2_t& max_curvature) const {
    assert(_concluded);
    if(_curvature.empty())
        return {};
    
    std::vector<Float2_t> smoothed_curvature;
    smooth_array(_curvature, smoothed_curvature, &max_curvature);
    return smoothed_curvature;
}

void Outline::smooth_array(const std::vector<Float2_t>& input, std::vector<Float2_t>& output, Float2_t *max_curvature) {
    const auto L = input.size();
    output.resize(L);
    if(L == 0)
        return;
    
    for (uint32_t i=0; i<L; i++) {
        Float2_t y = input[i];
        Float2_t prev = input[i >= 1 ? i-1 : L-1];
        Float2_t next = input[i < L-1 ? i+1 : 0];
        
        Float2_t c = (8/16_F * prev + 8/16_F * next + y) * 0.5_F;
        if(max_curvature && cmn::abs(c) > *max_curvature)
            *max_curvature = cmn::abs(c);
        output[i] = c;
    }
}

namespace Smaller {
    
    struct Area {
        Float2_t idx;
        Float2_t area;
        
        Area(Float2_t idx, Float2_t area)
            : idx(idx), area(area)
        {}
        
        virtual ~Area() {}
        
        bool operator<(const Area&) const {
            throw U_EXCEPTION("Calling wrong operator.");
        }
    };
    
    template<bool inverted>
    bool compare(const typename std::enable_if<inverted, Area>::type & obj, const typename std::enable_if<inverted, Area>::type & other) {
        return obj.area < other.area;
    }
    
    template<bool inverted>
    bool compare(const typename std::enable_if<!inverted, Area>::type & obj, const typename std::enable_if<!inverted, Area>::type & other) {
        return obj.area > other.area;
    }
}

#include <misc/CircularGraph.h>

tl::expected<long_t, const char*> Outline::find_tail(const DebugInfo&) {
    //static Timing timing("find_tail", 0.001);
    //TakeTiming take(timing);
    
    //long_t _tail_index = -1, _head_index = -1;
    
    //if(!FAST_SETTING(midline_invert)) {
    if(not _points || _points->empty() || size() <= 3)
        return tl::unexpected("The number of points is too small (<=3) to calculate a tail position.");
    
    assert(_concluded);
    assert(!_curvature.empty());
    
    // curvature not needed for new method
    if(size() <= 3)
        return -1;
    
    curvature_range = calculate_curvature_range(_points->size());
    _curvature.resize(_points->size());
    for(size_t i = 0; i < size(); i++)
        calculate_curvature(i);
    
    Float2_t max_curvature;
    
    auto corrected = smoothed_curvature_array(max_curvature);
    _max_curvature_lock.lock();
    _max_curvature = max(max_curvature, _max_curvature);
    if(_curvature_samples < 10000) {
        _average_curvature += max_curvature;
        _curvature_samples++;
    }
    _max_curvature_lock.unlock();
    
    curves::Extrema extrema;
    std::vector<Float2_t> io;
    std::vector<Float2_t> *used = &corrected;
    
    if(OUTLINE_SETTING(outline_use_dft)) {
        used = &io;
        io.resize(corrected.size());
        
        cv::Mat output(1, (int)corrected.size(), CV_32FC1);
        cv::dft(cv::Mat(1, (int)corrected.size(), CV_32FC1, corrected.data()), output);
        
        const size_t start = min(corrected.size()-1, max(5u, size_t(corrected.size()*0.03)));
        auto ptr = output.ptr<Float2_t>(0, 0);
        std::fill(ptr + start, ptr + corrected.size(), 0.f);
        
        cv::Mat tmp(1, (int)io.size(), CV_32FC1, io.data());
        cv::dft(output, tmp, cv::DFT_INVERSE + cv::DFT_SCALE);
        
        auto derivative = curves::derive(io);
        extrema = curves::find_extreme_points(io, derivative);
        
    } else {
        auto derivative = curves::derive(corrected);
        extrema = curves::find_extreme_points(corrected, derivative);
    }
    
    Float2_t idx = FLT_MAX;
    
    auto compare = [invert = FAST_SETTING(midline_invert)](const Smaller::Area& A, const Smaller::Area& B) {
        if(invert)
            return Smaller::compare<true>(A, B);
        return Smaller::compare<false>(A, B);
    };
    
    std::set<Smaller::Area, decltype(compare)> areas(compare);
    for (auto m : extrema.maxima) {
        if(m < 0)
            m += corrected.size();
        
        if(curves::interpolate(*used, m) <= 0) {
            Float2_t c = curves::interpolate(corrected, m);
            areas.insert(Smaller::Area(m, c));
        }
    }
    
    if(!areas.empty()) {
        auto idx0 = areas.begin()->idx; // probably the tail
        Float2_t max_d = -FLT_MAX;
        auto max_i = areas.end();
        for (auto it = areas.begin(); it != areas.end(); ++it) {
            auto &area = *it;
            if(area.idx == idx0)
                continue;
            
            Float2_t d = cmn::abs(idx0 - area.idx);
            if(d > max_d) {
                max_d = d;
                max_i = it;
            }
        }
        
        //_tail_index = round(idx0);
        
        if(max_i != areas.end()) {
            idx = max_i->idx; // head?
            //_head_index = round(idx);
        } else {
            idx = idx0;
            //_needs_invert = true;
        }
    }
    
   // idx = 0;
    
    if(idx == FLT_MAX) {
        return tl::unexpected("Cannot find the tail since we cannot find a distinct curvature peak, given current settings.");
    }
    
    return idx;
}

void Outline::clear() {
    minimize_memory();
    if(_points)
        _points->clear();
    //_tail_index = -1;
    //_head_index = -1;
    _concluded = false;
}

void Outline::minimize_memory() {
#define CLEAR_VECTOR(X) decltype(X)().swap(X)
    
    CLEAR_VECTOR(_curvature);
}
