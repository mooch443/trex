#include <gui/ImageDisplayElement.h>
#include <gui/ParseLayoutTypes.h>
// Nothing else to implement; all logic is header-only.

namespace cmn::gui {
Layout::Ptr ImageDisplayElement::_create(dyn::LayoutContext& context) {
    // register generator variable
    [[maybe_unused]] auto generator = context.get(std::string(), "generator");
    if(not generator.empty()) {
        _generator = _registry->get_generator(generator);
        if(_generator.reset)
            _generator.reset();
    }
    auto scale = context.get(Vec2(1), "scale");
    return Layout::Make<ExternalImage>(Image::Make(1,1,4), Vec2(0,0), scale, Color(255,255,255,255));
}

bool ImageDisplayElement::_update(Layout::Ptr& o,
             const dyn::Context& context,
             dyn::State& state,
             const dyn::PatternMapType& patterns)
{
    // Expect a pattern named "generator" holding the name
    if (patterns.contains("generator")) {
        // Evaluate the generator name
        const auto& pat = patterns.at("generator").original;
        std::string genName = parse_text(pat, context, state);
        
        // Fetch the lambda and generate an Image::Ptr
        _generator = _registry->get_generator(genName);
    }
    
    dyn::VarProps props{/* you may need to construct appropriate VarProps */};
    // e.g. props.parameters = { ... };
    // here we pass only the context/state if needed
    
    auto imgPtr = _generator.generate(props);
    if(imgPtr) {
        // Replace the ExternalImage inside our layout
        auto widget = o.to<ExternalImage>();
        widget->update_with(std::move(imgPtr));
    }

    return false; // no layout change needed
}
}
