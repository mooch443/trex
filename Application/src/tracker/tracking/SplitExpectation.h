#pragma once

#include <commons.pc.h>

namespace track {

struct split_expectation {
    size_t number;
    bool allow_less_than;
    std::vector<std::vector<cmn::Vec2>> centers;

    split_expectation(size_t number = 0, bool allow_less_than = false)
        : number(number), allow_less_than(allow_less_than)
    { }

    std::string toStr() const {
        return "{" + std::to_string(number) + ","
             + (allow_less_than ? "true" : "false") + ","
             + cmn::Meta::toStr(centers) + "}";
    }

    static consteval std::string_view class_name() {
        return "split_expectation";
    }
};

}
