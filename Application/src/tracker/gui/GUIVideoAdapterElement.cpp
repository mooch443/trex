#include "GUIVideoAdapterElement.h"

#include <gui/IMGUIBase.h>
#include <gui/dyn/ParseText.h>
#include <gui/ParseLayoutTypes.h>
#include <gui/GUIVideoAdapter.h>

namespace cmn::gui {

using namespace dyn;

GUIVideoAdapterElement::GUIVideoAdapterElement(
   IMGUIBase* window,
   std::function<Size2()> size_function,
   std::function<void(VideoInfo)> open_callback,
   std::function<Layout::Ptr(const file::PathArray&, IMGUIBase*, std::function<void(VideoInfo)>)> create_object)
    :   _size_function(size_function),
        _open_callback(open_callback),
        _create_object(create_object),
        _window(window)
{
    name = "video";
    create = [this](LayoutContext& context){
        return _create(context);
    };
    update = [this](Layout::Ptr& o,
                    const Context& context,
                    State& state,
                    auto& patterns)
    {
        return _update(o, context, state, patterns);
    };
}

GUIVideoAdapterElement::GUIVideoAdapterElement(const GUIVideoAdapterElement& other) 
{
    name = other.name;
    _open_callback = other._open_callback;
    _create_object = other._create_object;
    
    create = [this](LayoutContext& context){
        return _create(context);
    };
    update = [this](Layout::Ptr& o,
                    const Context& context,
                    State& state,
                    auto& patterns)
    {
        return _update(o, context, state, patterns);
    };
}

GUIVideoAdapterElement::~GUIVideoAdapterElement() {
    
}

Layout::Ptr GUIVideoAdapterElement::_create(LayoutContext& context) {
    auto path = context.get(std::string(), "path");
    auto blur = context.get(float(0), "blur");
    Size2 size;
    if(_size_function)
        size = _size_function();
    auto max_size = context.get(size, "max_size");
    auto frame_time = context.get(double(0.1), "frame_seconds");
    auto margins = context.get(Margins{}, "pad");
    auto alpha = context.get(Alpha{1.0}, "alpha");
    
    Layout::Ptr ptr;
    if(_create_object)
        ptr = Layout::Ptr(_create_object(file::PathArray(path), _window, _open_callback));
    else
        ptr = Layout::Make<GUIVideoAdapter>(file::PathArray(path), _window, _open_callback);
    
    auto p = ptr.to<GUIVideoAdapter>();
    p->set(GUIVideoAdapter::Blur{blur});
    p->set(SizeLimit{max_size});
    p->set(GUIVideoAdapter::FrameTime{frame_time});
    p->set(margins);
    p->set(alpha);
    return ptr;
}

template<typename SourceType, typename TargetType>
void check_field(std::string_view name, const Layout::Ptr& ptr, PatternMapType& patterns, const Context& context, State& state) {
    auto it = patterns.find(name);
    if(it == patterns.end())
        return;
    try {
        auto text = it->second.realize(context, state);
        auto fill = Meta::fromStr<SourceType>(text);
        LabeledField::delegate_to_proper_type(TargetType{fill}, ptr);
        
    } catch(const std::exception& e) {
        FormatError("Error parsing context; ", patterns, ": ", e.what());
    }
};

bool GUIVideoAdapterElement::_update(Layout::Ptr& o,
            const Context& context,
            State& state,
            PatternMapType& patterns)
{
    auto p = o.to<GUIVideoAdapter>();
    std::string path;
    if (auto it = patterns.find("path");
        it != patterns.end())
    {
        path = Meta::fromStr<std::string>(it->second.realize(context, state));
        //path = Meta::fromStr<std::string>(parse_text(patterns.at("path").original, context, state));
        
        if(not _last_path_str || path != *_last_path_str) {
            _last_path_str = path;
            _last_path = file::PathArray{path};
        }
        
        if(_last_path)
            p->set(*_last_path);
        else
            p->set(file::PathArray{});
    }
    
    check_field<double, GUIVideoAdapter::FrameTime>("frame_seconds", o, patterns, context, state);
    check_field<double, Alpha>("alpha", o, patterns, context, state);
    check_field<float, GUIVideoAdapter::Blur>("blur", o, patterns, context, state);
    check_field<Bounds, Margins>("pad", o, patterns, context, state);
    
    Size2 max_size = _size_function ? _size_function() : Size2();
    if(auto it = patterns.find("max_size");
       it != patterns.end())
    {
        max_size = Meta::fromStr<Size2>(it->second.realize(context, state));
        //max_size = Meta::fromStr<Size2>(parse_text(patterns.at("max_size").original, context, state));
    }
    p->set(SizeLimit{max_size});
    
    return false;
}

}
