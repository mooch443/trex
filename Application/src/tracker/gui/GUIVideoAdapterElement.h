#pragma once
#include <commons.pc.h>
#include <gui/ParseLayoutTypes.h>
#include <gui/dyn/State.h>
#include <misc/VideoInfo.h>

namespace cmn::gui {

class IMGUIBase;

struct GUIVideoAdapterElement : public dyn::CustomElement {
    std::function<Size2()> _size_function;
    std::function<void(VideoInfo)> _open_callback;
    std::function<Layout::Ptr(const file::PathArray&, IMGUIBase*, std::function<void(VideoInfo)>)> _create_object;
    IMGUIBase *_window;
    
    std::optional<std::string> _last_path_str;
    std::optional<file::PathArray> _last_path;
    
    GUIVideoAdapterElement(IMGUIBase* window,
                           std::function<Size2()> size_function,
                           std::function<void(VideoInfo)> open_callback = nullptr,
                           std::function<Layout::Ptr(const file::PathArray&, IMGUIBase*, std::function<void(VideoInfo)>)> = nullptr);
    GUIVideoAdapterElement(GUIVideoAdapterElement&&) = delete;
    GUIVideoAdapterElement(const GUIVideoAdapterElement&);
    GUIVideoAdapterElement& operator=(GUIVideoAdapterElement&&) = delete;
    GUIVideoAdapterElement& operator=(const GUIVideoAdapterElement&) = delete;
    virtual ~GUIVideoAdapterElement();
    
    Layout::Ptr _create(dyn::LayoutContext& context);
    
    bool _update(Layout::Ptr& o,
                 const dyn::Context& context,
                 dyn::State& state,
                 const dyn::PatternMapType& patterns);
};

}
