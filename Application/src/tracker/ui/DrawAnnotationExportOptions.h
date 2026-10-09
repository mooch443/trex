#pragma once

#include <pv.h>

namespace cmn::gui {
class DrawStructure;

struct DrawAnnotationExportOptions {
    struct Data;
    Data* _data;

public:
    DrawAnnotationExportOptions(std::shared_ptr<pv::File>);
    ~DrawAnnotationExportOptions();
    void draw(DrawStructure&);
};

}
