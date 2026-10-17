#include "TrackingSettings.h"

namespace track {

#if defined(DEBUG_TRACKING_THREADS)
TrackingThreadG::TrackingThreadG() {
    track::add_tracking_thread_id(std::this_thread::get_id());
}
TrackingThreadG::~TrackingThreadG() {
    track::remove_tracking_thread_id(std::this_thread::get_id());
}

void assert_tracking_thread() {
    std::shared_lock guard(tracking_thread_mutex);
    if(!tracking_thread_ids.contains(std::this_thread::get_id())) {
        FormatWarning("Wrong thread ", get_thread_name(), " to read from settings.");
    }
}
#endif

std::map<Idx_t, float> prediction2map(const std::vector<float>& pred) {
    std::map<Idx_t, float> map;
    for (size_t i=0; i<pred.size(); i++) {
        map[Idx_t(i)] = pred[i];
    }
    return map;
}

std::string DetailProbability::toStr() const {
    return "{p="+dec<2>(p * 100).toStr()+" p_time="+dec<2>(p_time * 100).toStr()+" p_pos="+dec<2>(p_pos * 100).toStr()+" p_angle="+dec<2>(p_angle * 100).toStr()+"}";
}

std::string PoseMidlineIndexes::toStr() const {
    return Meta::toStr(indexes);
}

glz::json_t PoseMidlineIndexes::to_json() const {
    std::vector<glz::json_t> a;
    for(auto i : indexes)
        a.push_back(i);
    return a;
}

ShapeList::ShapeList(std::vector<std::vector<Vec2>> shape)
{
    update_value(std::move(shape));
}

std::string ShapeList::toStr() const {
    return Meta::toStr(_value);
}

glz::json_t ShapeList::to_json() const {
    return cvt2json(_value);
}

void ShapeList::push_back(std::vector<Vec2> points) {
    fix_value(points);
    _value.push_back(std::move(points));
}

void ShapeList::update_value(std::vector<std::vector<Vec2>>&& tmp)
{
    for(auto &vec : tmp) {
        fix_value(vec);
    }
    
    _value = std::move(tmp);
}

void ShapeList::fix_value(std::vector<Vec2>& vec) {
    if(vec.size() <= 2)
        return;
    
    auto ptr = poly_convex_hull(&vec);
    if(ptr && vec != *ptr)
        vec = *ptr;
}

bool ShapeList::empty() const {
    return _value.empty();
}

}
