#pragma once

namespace cmn::gui {
class DrawStructure;

struct DrawAnnotationImportOptions {
    struct Data;
    Data* _data;

public:
    DrawAnnotationImportOptions();
    ~DrawAnnotationImportOptions();
    void draw(DrawStructure&);
};

}
