#include "GPURecognitionTypes.h"

namespace track::detect {

KeypointData::KeypointData(std::vector<float>&& data, size_t bones)
    : _num_bones(bones), _xy_conf(std::move(data))
{
    if(data.size() % (sizeof(Bone) / sizeof(decltype(Bone::x))) != 0u)
        throw InvalidArgumentException("Invalid size for KeypointData constructor. Please use a size that is divisible by ", sizeof(Bone) / sizeof(decltype(Bone::x)), " and is a flat ", Meta::name<decltype(Bone::x)>(), " array.");
    assert(data.size() % (sizeof(Bone) / sizeof(decltype(Bone::x))) == 0u);
    assert(data.size() % _num_bones == 0);
}

Keypoint KeypointData::operator[](size_t index) const {
    if(index * num_bones() * 2u >= xy_conf().size())
        throw OutOfRangeException("The index ", index, " is outside the keypoints arrays dimensions of ", size());
    return Keypoint{
        .bones = std::vector<Bone>{
            reinterpret_cast<const Bone*>(xy_conf().data()) + num_bones() * index,
            reinterpret_cast<const Bone*>(xy_conf().data()) + num_bones() * (index + 1)
        }
    };
}

ICXYWHR ObbData::operator[](size_t index) const {
    if(index * 7u >= icxywhr().size())
        throw OutOfRangeException("The index ", index, " is outside the OBB arrays dimensions of ", size());
    return reinterpret_cast<const ICXYWHR*>(icxywhr().data())[index];
}

ObbData::ObbData(std::vector<float>&& data)
    : _icxywhr(std::move(data))
{
    if(!_icxywhr.empty() && _icxywhr.size() % 7u != 0u)
        throw InvalidArgumentException("Invalid size for ObbData constructor. Please use a size that is divisible by 7 and is a flat ICXYWHR array.");
    assert(_icxywhr.size() % 7u == 0u);
}

std::array<cmn::Vec2, 4> ICXYWHR::corners() const {
    const float cos_r = std::cos(r);
    const float sin_r = std::sin(r);
    const float dx = w / 2.f;
    const float dy = h / 2.f;
    std::array<cmn::Vec2, 4> out;
    out[0] = {x + (-dx) * cos_r - (-dy) * sin_r, y + (-dx) * sin_r + (-dy) * cos_r};
    out[1] = {x + dx * cos_r - (-dy) * sin_r, y + dx * sin_r + (-dy) * cos_r};
    out[2] = {x + dx * cos_r - dy * sin_r, y + dx * sin_r + dy * cos_r};
    out[3] = {x + (-dx) * cos_r - dy * sin_r, y + (-dx) * sin_r + dy * cos_r};
    return out;
}

Bounds ICXYWHR::bounding_box() const {
    return bounding_box(corners());
}

Bounds ICXYWHR::bounding_box(const std::array<cmn::Vec2, 4>& pts) {
    float min_x = pts[0].x;
    float min_y = pts[0].y;
    float max_x = pts[0].x;
    float max_y = pts[0].y;
    for(int i = 1; i < 4; ++i) {
        min_x = std::min(min_x, pts[i].x);
        min_y = std::min(min_y, pts[i].y);
        max_x = std::max(max_x, pts[i].x);
        max_y = std::max(max_y, pts[i].y);
    }
    return Bounds(min_x, min_y, max_x - min_x, max_y - min_y);
}

std::array<cmn::Vec2, 4> ICXYR::corners() const {
    return std::array{
        Vec2(x - r, y - r),
        Vec2(x + r, y - r),
        Vec2(x + r, y + r),
        Vec2(x - r, y + r)
    };
}

Bounds ICXYR::bounding_box() const {
    return bounding_box(corners());
}

Bounds ICXYR::bounding_box(const std::array<cmn::Vec2, 4>& pts) {
    return ICXYWHR::bounding_box(pts);
}

ICXYR PointData::operator[](size_t index) const {
    if(index * 5u >= icxyr().size())
        throw OutOfRangeException("The index ", index, " is outside the PointData arrays dimensions of ", size());
    return reinterpret_cast<const ICXYR*>(icxyr().data())[index];
}

PointData::PointData(std::vector<float>&& data)
    : _icxyr(std::move(data))
{
    if(!_icxyr.empty() && _icxyr.size() % 5u != 0u)
        throw InvalidArgumentException("Invalid size for PointData constructor. Please use a size that is divisible by 5 and is a flat ICXYR array.");
    assert(_icxyr.size() % 5u == 0u);
}

std::string Sam3PromptPayload::toStr() const {
    switch(type()) {
        case Sam3PromptType::text:
            if(std::holds_alternative<std::monostate>(value))
                return "null";
            return text();
        case Sam3PromptType::boxes:
        {
            std::vector<std::array<float, 4>> xyxy_boxes;
            xyxy_boxes.reserve(boxes().size());
            for(const auto& box : boxes()) {
                xyxy_boxes.push_back({
                    box.x,
                    box.y,
                    box.width,
                    box.height
                });
            }
            return Meta::toStr(xyxy_boxes);
        }
        case Sam3PromptType::points:
            return Meta::toStr(points());
        default:
            throw InvalidArgumentException("Not implemented: ", type());
    }
}

glz::json_t Sam3PromptPayload::to_json() const {
    switch(type()) {
        case Sam3PromptType::text:
            if(std::holds_alternative<std::monostate>(value))
                return "null";
            return text();
        case Sam3PromptType::boxes:
        {
            glz::json_t::array_t xyxy_boxes;
            xyxy_boxes.reserve(boxes().size());
            for(const auto& box : boxes()) {
                xyxy_boxes.push_back(glz::json_t::array_t{
                    box.x,
                    box.y,
                    box.width,
                    box.height
                });
            }
            return xyxy_boxes;
        }
        case Sam3PromptType::points:
            return cvt2json(points());
        default:
            throw InvalidArgumentException("Not implemented: ", type());
    }
}

glz::json_t Sam3Prompts::to_json() const {
    if(empty())
        return glz::json_t::object_t();
    return cvt2json(map);
}

std::string Sam3Prompts::toStr() const {
    if(empty()) {
        return "{}";
    }
    
    if(size() == 1u
       && map.begin()->first == Frame_t{})
    {
        /// we only have one global prompt
        return Meta::toStr(map.begin()->second);
    }
    
    return Meta::toStr(map);
}

glz::json_t Sam3PromptList::to_json() const {
    return cvt2json(static_cast<const base_t&>(*this));
}

std::string Sam3PromptList::toStr() const {
    if(size() == 1u) {
        return Meta::toStr(front());
    }
    return Meta::toStr(static_cast<const base_t&>(*this));
}

} // namespace track::detect
