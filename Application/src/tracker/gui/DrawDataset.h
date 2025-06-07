#pragma once

#include <gui/DrawStructure.h>
#include <gui/types/Entangled.h>
#include <tracking/DatasetQuality.h>
#include <misc/idx_t.h>

namespace cmn::gui {
    class StaticText;
    class GUICache;

    class DrawDataset : public Entangled {
        Frame_t _last_frame;
        Range<Frame_t> _last_tracklet, _last_current_frames;
        
        double _index_percentage{0.0};
        Color _color{Black.alpha(150)};
        track::DatasetQuality::Quality _quality, _current_quality;
        
        std::vector<std::unique_ptr<StaticText>> _texts;
        
        bool _initial_pos_set;
        
        Range<Frame_t> current_consec;
        Range<Frame_t> consec;
        Frame_t frame;
        std::vector<Range<Frame_t>> tracklet_order;
        
        struct Data;
        std::unique_ptr<Data> _data;
        
    public:
        DrawDataset();
        virtual ~DrawDataset();
        
        void set_data(Frame_t frameIndex, const GUICache& tracker);
        void update() override;
        void clear_cache();
        
    private:
        void update_background_color(bool hovered);
    };
}
