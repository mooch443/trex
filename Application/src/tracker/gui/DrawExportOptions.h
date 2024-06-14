#pragma once

namespace cmn::gui {
class DrawStructure;

struct DrawExportOptions {
    struct Data;
    Data *_data;
    
public:
    DrawExportOptions();
    ~DrawExportOptions();
    void draw(DrawStructure&);
};
}
