#pragma once

#include <gui/DrawStructure.h>
#include <gui/types/Entangled.h>
#include <tracking/DatasetQuality.h>
#include <misc/idx_t.h>

namespace gui {
    class DrawDataset : public Entangled {
        Frame_t _last_frame;
        Range<Frame_t> _last_consecutive_frames, _last_current_frames;
        track::DatasetQuality::Quality _quality, _current_quality;
        
        std::map<track::Idx_t, std::tuple<size_t, std::map<track::Idx_t, float>>> _cache;
        std::map<track::Idx_t, std::string> _names;
        std::vector<std::shared_ptr<StaticText>> _texts;
        
        std::map<track::Idx_t, track::DatasetQuality::Single> _meta;
        std::map<track::Idx_t, track::DatasetQuality::Single> _meta_current;
        bool _initial_pos_set;
        
    public:
        DrawDataset();
        virtual ~DrawDataset() {}
        
        void update() override;
        void clear_cache();
        
    private:
        
    };
}
