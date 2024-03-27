#include "SegmentInformation.h"
#include <tracking/Individual.h>

namespace track {
using namespace cmn;

void SegmentInformation::add_basic_at(Frame_t frame, long_t gdx) {
    UNUSED(frame);
    assert(end() == frame);
    basic_index.push_back(gdx);
}

#ifndef NDEBUG
#define SEGMENT_ACCESS(INDEXARRAY, INDEX) INDEXARRAY . at( INDEX )
#define FRAME_SEGMENT_ACCESS(INDEXARRAY, INDEX) INDEXARRAY . at( ( INDEX ).get() )
#else
#define SEGMENT_ACCESS(INDEXARRAY, INDEX) INDEXARRAY [ INDEX ]
#define FRAME_SEGMENT_ACCESS(INDEXARRAY, INDEX) INDEXARRAY [ ( INDEX ).get() ]
#endif

void SegmentInformation::add_posture_at(std::unique_ptr<PostureStuff>&& stuff, Individual* fish) {//long_t gdx) {
    size_t L = sign_cast<size_t>(length().get());
    if(posture_index.size() != size_t(L)) {
        auto prev = posture_index.size();
        posture_index.resize(L);
        for (size_t i=prev; i<L; ++i) {
            SEGMENT_ACCESS(posture_index, i) = -1;
        }
    }
    
    auto gdx = fish->_posture_stuff.size();
    
    if(fish->added_postures.find(stuff->frame) == fish->added_postures.end()) {
        fish->added_postures.insert(stuff->frame);
    } else {
        print(fish->added_postures);
        throw SoftException("(", fish->identity(),") Posture for frame ",stuff->frame," already added.");
    }
    
    if(!fish->_posture_stuff.empty() && stuff->frame < fish->_posture_stuff.back()->frame)
        throw SoftException("(", fish->identity().ID(),") Adding frame ", stuff->frame," after frame ", fish->_last_posture_added);
    
    fish->_last_posture_added = stuff->frame;
    FRAME_SEGMENT_ACCESS(posture_index, stuff->frame - start()) = sign_cast<int>(gdx);

    fish->_posture_stuff.push_back(std::move(stuff));
}

long_t SegmentInformation::basic_stuff(Frame_t frame) const {
    //assert(frame >= start() && frame <= end() && size_t(frame-start()) < basic_index.size());
    if(frame < start() || frame > end() || (size_t)(frame-start()).get() >= basic_index.size())
        return -1;
    return FRAME_SEGMENT_ACCESS(basic_index, frame - start());
}

long_t SegmentInformation::posture_stuff(Frame_t frame) const {
    if(posture_index.empty() || !contains(frame)
       || (posture_index.size() < basic_index.size() && (size_t)(frame - start()).get() >= posture_index.size() ))
        return -1;
    assert(frame >= start() && frame <= end() && (size_t)(frame-start()).get() < posture_index.size());
    return FRAME_SEGMENT_ACCESS(posture_index, frame - start()); //posture_index .at( frame - start() );
}

}

