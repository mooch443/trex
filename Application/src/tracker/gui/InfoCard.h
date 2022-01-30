#pragma once

#include <gui/DrawStructure.h>
#include <gui/types/Button.h>

namespace track {
    class Individual;
}

namespace gui {
    class InfoCard : public Entangled {
        std::shared_ptr<Button> prev, next, detail_button, automatic_button;
        Frame_t _frameNr;
        track::Individual *_fish;
        //Button detail_button;
        std::vector<std::tuple<Text*, std::string>> segment_texts;
        Text * previous = nullptr;
    public:
        InfoCard();
        void update(gui::DrawStructure&, Frame_t);
        void update() override;
    };
}
