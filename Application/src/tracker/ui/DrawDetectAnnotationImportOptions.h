#pragma once

namespace cmn::gui {
class DrawStructure;

/// DynamicGUI-backed options and preview widget for detect-annotation imports.
struct DrawDetectAnnotationImportOptions {
    struct Data;
    Data* _data;

public:
    DrawDetectAnnotationImportOptions();
    ~DrawDetectAnnotationImportOptions();
    void draw(DrawStructure&);
};

}
