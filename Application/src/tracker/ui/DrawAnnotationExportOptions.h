#pragma once

#include <pv.h>

namespace track {
class Tracker;
}

namespace cmn::gui {
class DrawStructure;

struct DrawAnnotationExportOptions {
    struct Data;
    Data* _data;

public:
    DrawAnnotationExportOptions(std::shared_ptr<pv::File>, std::shared_ptr<track::Tracker>);
    ~DrawAnnotationExportOptions();
    void draw(DrawStructure&);
};

}
