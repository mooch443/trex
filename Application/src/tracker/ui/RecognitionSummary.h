#pragma once

#include <commons.pc.h>
#include <gui/types/Entangled.h>
#include <gui/DrawStructure.h>

namespace cmn::gui {
    class RecognitionSummary {
        gui::Entangled obj;
    public:
        void update(gui::DrawStructure&);
    };
}
