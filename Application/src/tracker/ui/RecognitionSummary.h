#pragma once

#include <commons.pc.h>
#include <gui/types/Entangled.h>
#include <gui/DrawStructure.h>
#include <data/FrameRepository.h>

namespace cmn::gui {
    class RecognitionSummary {
        gui::Entangled obj;
    public:
        void update(const data::FrameRepository& frames, gui::DrawStructure&);
    };
}
