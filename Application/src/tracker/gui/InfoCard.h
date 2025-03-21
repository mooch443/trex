#pragma once

#include <commons.pc.h>
#include <misc/idx_t.h>
#include <misc/frame_t.h>
#include <gui/ShadowTracklet.h>
#include <gui/types/Entangled.h>
#include <misc/derived_ptr.h>

namespace track {
    class Individual;
}

namespace cmn::gui {
    class DrawStructure;
    class Text;
    class Button;
    class HorizontalLayout;
    class Tooltip;
    class Text;
    class Rect;

    class InfoCard : public Entangled {
        struct ShadowIndividual;
        ShadowIndividual *_shadow{nullptr};
        derived_ptr<Button> prev, next, detail_button, automatic_button;
        //Button detail_button;
        std::vector<std::tuple<derived_ptr<Text>, std::string>> tracklet_texts;
        std::weak_ptr<Text> previous;
        std::function<void(Frame_t)> _reanalyse;

    public:
        InfoCard(std::function<void(Frame_t)> reanalyse);
        ~InfoCard();
        void update(gui::DrawStructure&, Frame_t);
        void update() override;
    };

    class DrawSegments : public Entangled {
        std::vector<ShadowTracklet> _tracklets;
        std::vector<ShadowTracklet> _displayed_tracklets;
        std::vector<std::tuple<std::shared_ptr<Text>, std::string>> tracklet_texts;
        std::unique_ptr<Tooltip> _tooltip;
        
        GETTER(track::Idx_t, fdx);
        GETTER(Frame_t, frame);
        Font _font{0.6};
        Margins _margins{0,0,0,0};
        SizeLimit _limits{300,0};
        std::weak_ptr<Text> _selected{};
        std::unique_ptr<Rect> _highlight;
        Bounds _previous_bounds;
        Bounds _target_bounds;
        
    public:
        DrawSegments();
        ~DrawSegments();
        
        using Entangled::set;
        void set(track::Idx_t fdx, Frame_t frame, const std::vector<ShadowTracklet>& tracklets);
        void set(Font);
        void set(Margins);
        void set(SizeLimit);
        
        Float2_t add_segments(bool display_hints, float offx);
        
        void update();
        void update_box();
    };

}
