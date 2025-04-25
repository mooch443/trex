#include <gui/ImageDisplayElement.h>
#include <gui/ParseLayoutTypes.h>
#include <gui/GUICache.h>

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
    auto color = context.get(Color(255,255,255,255), "color");
    return Layout::Make<ExternalImage>(Image::Make(1,1,4), Vec2(0,0), scale, color);
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
    auto widget = o.to<ExternalImage>();
    if(imgPtr) {
        // Replace the ExternalImage inside our layout
        widget->update_with(std::move(imgPtr));
    }
    
    if (patterns.contains("color")) {
        // parse the desired target color (may include alpha)
        const Color target = Meta::fromStr<Color>(
            parse_text(patterns.at("color").original, context, state)
        );
        const Color current = widget->color();

        // get delta time in seconds (fictive dt())
        auto delta = saturate(GUICache::instance().dt(), 0.001, 0.1);

        // define animation speeds (channel units per second)
        constexpr float rgbSpeed   = 700; // how fast RGB moves (0-255)
        constexpr float alphaSpeed = 700;  // how fast alpha moves (0-255)

        // compute per-frame step amounts, clamped 0..255
        uint8_t rgbStep   = static_cast<uint8_t>(std::clamp(rgbSpeed * delta, 0.0, 255.0));
        uint8_t alphaStep = static_cast<uint8_t>(std::clamp(alphaSpeed * delta, 0.0, 255.0));

        // animate RGB by blending a low-alpha copy of target
        Color fadeRgb  = target.alpha(rgbStep);
        Color blended  = Color::blend(current, fadeRgb);

        // animate alpha by stepping toward the target
        int da = target.a - current.a;
        int stepA = std::clamp(da, -int(alphaStep), int(alphaStep));
        uint8_t nextA = static_cast<uint8_t>(std::clamp(current.a + stepA, 0, 255));

        // combine new RGB with animated alpha
        Color nextColor{ blended.r, blended.g, blended.b, nextA };
        widget->set_color(nextColor);
    }
    
    return false; // no layout change needed
}
}
