#include "TrackingSettings.h"

namespace track {

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

PoseMidlineIndexes PoseMidlineIndexes::fromStr(const std::string& str) {
    return PoseMidlineIndexes{
        .indexes = Meta::fromStr<std::vector<uint8_t>>(str)
    };
}

std::string PoseMidlineIndexes::toStr() const {
    return Meta::toStr(indexes);
}

nlohmann::json PoseMidlineIndexes::to_json() const {
    auto a = nlohmann::json::array();
    for(auto i : indexes)
        a.push_back(i);
    return a;
}

}
