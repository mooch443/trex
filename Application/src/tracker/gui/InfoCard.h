#pragma once

#include <gui/DrawStructure.h>
#include <gui/types/Button.h>

namespace track {
    class Individual;
}

namespace gui {

    class InfoCard : public Entangled {
        struct ShadowIndividual;
        ShadowIndividual *_shadow{nullptr};
        std::shared_ptr<Button> prev, next, detail_button, automatic_button;
        //Button detail_button;
        std::vector<std::tuple<Text*, std::string>> segment_texts;
        Text * previous = nullptr;
        std::function<void(Frame_t)> _reanalyse;

    public:
        InfoCard(std::function<void(Frame_t)> reanalyse);
        ~InfoCard();
        void update(gui::DrawStructure&, Frame_t);
        void update() override;
    };
}
