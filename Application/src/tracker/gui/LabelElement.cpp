#include "LabelElement.h"
#include <gui/dyn/ParseText.h>
#include <gui/ParseLayoutTypes.h>
#include <misc/Coordinates.h>
#include <gui/Label.h>
#include <gui/LabelWrapper.h>
#include <misc/CTCollection.h>

namespace cmn::gui {

using namespace track;
using namespace dyn;

LabelElement::LabelElement(LabelCache_t* labelCache,
                           std::unordered_map<Idx_t, std::shared_ptr<Label>>* labelsMap,
                           double* dt)
    : _labelCache(labelCache), _labelsMap(labelsMap), _dt(dt)
{
    name = "label";
    // Set up the create callback.
    create = [this](dyn::LayoutContext& layout) -> Layout::Ptr {
        return _create(layout);
    };
    // Set up the update callback.
    update = [this](Layout::Ptr& o,
                    const dyn::Context& context,
                    dyn::State& state,
                    const dyn::PatternMapType& patterns) -> bool {
        return _update(o, context, state, patterns);
    };
}

LabelElement::~LabelElement() {
    // Clean up if necessary.
}

Layout::Ptr LabelElement::_create(dyn::LayoutContext& layout) {
    std::shared_ptr<Label> ptr;
    
    auto text = layout.get<std::string>("", "text");
    auto center = layout.get<Vec2>(Vec2(), "center");
    auto line_length = layout.get<float>(float(60), "length");
    auto id = layout.get<Idx_t>(Idx_t(), "id");
    auto color = layout.textClr;
    auto line = layout.line;
    auto fill = layout.fill;
    
    if (id.valid()) {
        auto it = _labelsMap->find(id);
        if (it != _labelsMap->end()) {
            ptr = it->second;
        } else {
            (*_labelsMap)[id] = ptr = _labelCache->getObject();
        }
    } else {
        ptr = _labelCache->getObject();
    }
    
    if (!ptr)
        throw RuntimeError("Apparently out of memory generating label ", text, ".");
    
    ptr->set_line_length(line_length);
    ptr->set_data(0_f, text, Bounds(layout.pos, layout.size), center);
    auto font = parse_font(layout.obj, layout._defaults.font);
    ptr->text()->set(font);
    ptr->text()->set(color);
    ptr->set(FillClr{ fill });
    ptr->set_line_color(line);
    
    if (!id.valid())
        ptr->set_uninitialized();
    
    return Layout::Ptr(std::make_shared<LabelWrapper>(*_labelCache, std::move(ptr)));
}

bool LabelElement::_update(Layout::Ptr& o,
                           const dyn::Context& context,
                           dyn::State& state,
                           const dyn::PatternMapType& patterns)
{
    Idx_t id;
    if (patterns.contains("id"))
        id = Meta::fromStr<Idx_t>(parse_text(patterns.at("id").original, context, state));
    
    if (id.valid()) {
        if (auto it = _labelsMap->find(id); it != _labelsMap->end()) {
            if (it->second.get() != o.get())
                o = Layout::Ptr(it->second);
        }
    }
    
    Label* p;
    if (o.is<LabelWrapper>()) {
        p = o.to<LabelWrapper>()->label();
    } else {
        p = o.to<Label>();
    }
    
    if (!id.valid()) {
        p->set_uninitialized();
    }
    
    auto source = p->source();
    using namespace cmn::ct;
    
    CTCollection map{
        Key<"text", "pos", "size", "center", "line", "fill", "color">{},
        p->text()->text(),
        Vec2(source.pos()),
        source.size(),
        source.pos() + Vec2(source.width, source.height) * 0.5_F,
        p->line_color(),
        p->fill_color(),
        TextClr{p->text()->text_color()}
    };
    
    map.apply([&](std::string_view key, auto& value) {
        value = parse_value_with_default(value, key, patterns, context, state);
    });
    
    source = Bounds{ map.get<"pos">(), map.get<"size">() };
    p->set_line_color(map.get<"line">());
    p->set_fill_color(map.get<"fill">());
    p->text()->set(map.get<"color">());
    
    p->set_data(0_f, map.get<"text">(), source, map.get<"center">());
    
    // Use dt from the provided pointer.
    p->update(FindCoord::get(), 1, 1, false, *_dt, Scale{1});
    
    return true;
}

} // namespace cmn::gui
