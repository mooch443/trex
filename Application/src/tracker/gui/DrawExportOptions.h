#pragma once

namespace cmn::gui {
class DrawStructure;
struct TrackingState;

struct DrawExportOptions {
    struct Data;
    Data *_data;
    
public:
    DrawExportOptions();
    ~DrawExportOptions();
    void draw(DrawStructure&, TrackingState*);
};
}
