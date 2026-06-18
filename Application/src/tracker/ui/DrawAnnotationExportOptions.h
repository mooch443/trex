#pragma once

namespace cmn::gui {
class DrawStructure;

struct DrawAnnotationExportOptions {
    struct Data;
    Data* _data;

public:
    DrawAnnotationExportOptions();
    ~DrawAnnotationExportOptions();
    void draw(DrawStructure&);
};

}
