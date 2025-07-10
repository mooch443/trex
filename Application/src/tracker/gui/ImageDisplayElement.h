#pragma once

#include <commons.pc.h>
#include <gui/dyn/State.h>
#include <gui/dyn/ParseText.h>
#include <gui/GuiTypes.h>
#include <gui/ImageGeneratorRegistry.h>

namespace cmn::gui {

/// Custom element that displays an ExternalImage generated
/// by a named generator.
class ImageDisplayElement : public dyn::CustomElement {
public:
    explicit ImageDisplayElement(ImageGeneratorRegistry* registry)
        : _registry(registry)
    {
        name = "image_generator";
        create = [this](dyn::LayoutContext& ctx) {
            return _create(ctx);
        };
        update = [this](Layout::Ptr& o,
                        const dyn::Context& ctx,
                        dyn::State& state,
                        dyn::PatternMapType& patterns)
        {
            return _update(o, ctx, state, patterns);
        };
    }

    virtual ~ImageDisplayElement() = default;

private:
    ImageGeneratorRegistry* _registry;
    ImageGeneratorRegistry::Generator _generator;

    Layout::Ptr _create(dyn::LayoutContext& context);

    bool _update(Layout::Ptr& o,
                 const dyn::Context& context,
                 dyn::State& state,
                 dyn::PatternMapType& patterns);
};

} // namespace cmn::gui
