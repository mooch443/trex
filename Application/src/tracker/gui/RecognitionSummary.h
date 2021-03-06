#pragma once

#include <types.h>
#include <gui/types/Entangled.h>
#include <gui/DrawStructure.h>

namespace gui {
    class RecognitionSummary {
        gui::Entangled obj;
    public:
        void update(gui::DrawStructure&);
    };
}
