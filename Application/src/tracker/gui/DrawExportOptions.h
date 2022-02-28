#pragma once

namespace gui {
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
