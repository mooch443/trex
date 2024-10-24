#pragma once

#include <commons.pc.h>
#include <misc/bid.h>
#include <tracking/Stuffs.h>
#include <tracking/Outline.h>

namespace cmn::gui {

struct BdxAndPred {
    pv::bid bdx;
    std::optional<track::BasicStuff> basic_stuff;
    std::optional<track::PostureStuff> posture_stuff;
    std::optional<std::vector<float>> pred;
    track::Midline::Ptr midline;
    bool automatic_match;
    Range<Frame_t> segment;
    
    BdxAndPred clone() const {
        BdxAndPred copy;
        copy.bdx = bdx;
        copy.basic_stuff = basic_stuff;
        if(posture_stuff)
            copy.posture_stuff = posture_stuff->clone();
        else
            copy.posture_stuff = std::nullopt;
        copy.pred = pred;
        copy.midline = midline ? std::make_unique<track::Midline>(*midline) : nullptr;
        copy.automatic_match = automatic_match;
        copy.segment = segment;
        return copy;
    }
};

}
