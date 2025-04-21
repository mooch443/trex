#include "PreviewAdapterElement.h"
#include <gui/dyn/ParseText.h>
#include <gui/ParseLayoutTypes.h>
#include <misc/idx_t.h>
#include <tracking/FilterCache.h>
#include <tracking/Outline.h>
#include <gui/DrawPreviewImage.h>
#include <tracking/PPFrame.h>
#include <tracking/Tracker.h>

namespace cmn::gui {

using namespace track;

class IndividualImage : public Entangled {
    GETTER(Idx_t, fdx);
    Image::Ptr ptr;
    GETTER(Frame_t, frame);
    ExternalImage _display;

    static constexpr inline std::array<std::string_view, 10> _setting_names {
        "individual_image_normalization",
        "individual_image_size",
        "individual_image_scale",
        
        "track_background_subtraction",
        "meta_encoding",
        "track_threshold",
        "track_posture_threshold",
        "track_size_filter",
        "track_include", "track_ignore"
    };
    std::unordered_map<std::string_view, std::string> _settings;
    
public:
    using Entangled::set;
    void set_data(Idx_t fdx, Frame_t frame, pv::BlobWeakPtr blob, const track::Background* background, const constraints::FilterCache* filters, const Midline* midline) {
        // already set
        if(fdx == _fdx && _frame == frame && not settings_changed())
            return;
        
        this->_fdx = fdx;
        this->_frame = frame;
        
        auto &&[image, pos] = DrawPreviewImage::make_image(blob, midline, filters, background);
        _display.set_source(std::move(image));
        update_settings();
        update();
    }

    bool settings_changed() const {
        if(_settings.empty())
            return true;
        for(auto&[key, value] : _settings) {
            if(GlobalSettings::map().at(key).get().valueString() != value) {
                return true;
            }
        }
        return false;
    }
    void update_settings() {
        for(auto key : _setting_names) {
            _settings[key] = GlobalSettings::map().at(key).get().valueString();
        }
    }
    
    void update() override {
        OpenContext([this]{
            advance_wrap(_display);
        });
        
        auto_size({});
    }
};

using namespace dyn;

PreviewAdapterElement::PreviewAdapterElement(decltype(get_current_frame)&& fn, decltype(get_filter_cache)&& fc)
    : get_current_frame(std::move(fn)), get_filter_cache(std::move(fc))
{
    name = "preview";
    
    create = [this](LayoutContext& context){
        return _create(context);
    };
    update = [this](Layout::Ptr& o,
                    const Context& context,
                    State& state,
                    const auto& patterns)
    {
        return _update(o, context, state, patterns);
    };
}

PreviewAdapterElement::~PreviewAdapterElement() {
    //Print("Freeing preview adapter element @ ", hex(this));
}

Layout::Ptr PreviewAdapterElement::_create(LayoutContext& context) {
    [[maybe_unused]] auto fdx = context.get(Idx_t(), "fdx");
    return Layout::Make<IndividualImage>();
}

bool PreviewAdapterElement::_update(Layout::Ptr& o,
            const Context& context,
            State& state,
            const PatternMapType& patterns)
{
    auto ptr = o.to<IndividualImage>();
    //auto &cache = GUICache::instance();
    
    Idx_t fdx;
    const PPFrame* ppframe = get_current_frame();
    if(not ppframe) {
        throw RuntimeError("Cannot find current frame (in preview adapter).");
    }
    
    Frame_t frame = ppframe->index();
    
    if(patterns.contains("fdx")) {
        fdx = Meta::fromStr<Idx_t>(parse_text(patterns.at("fdx").original, context, state));
    }
    /*if(patterns.contains("frame")) {
        frame = Meta::fromStr<Frame_t>(parse_text(patterns.at("frame").original, context, state));
    }*/
    
    if(fdx != ptr->fdx()
       || frame != ptr->frame()
       || ptr->settings_changed())
    {
        auto [filters, bdxnpred] = get_filter_cache(fdx);
        pv::BlobWeakPtr blob_ptr{nullptr};
        
        if(bdxnpred.has_value()) {
            ppframe->transform_blobs_by_bid(std::array{bdxnpred->bdx}, [&blob_ptr](pv::Blob& blob) {
                blob_ptr = &blob;
            });
            
            if(blob_ptr) {
                if(blob_ptr->encoding() == Background::meta_encoding())
                    ptr->set_data(fdx, frame, blob_ptr, track::Tracker::background(), filters, bdxnpred->midline.get());
#ifndef NDEBUG
                else
                    FormatWarning("Not displaying image yet because of the wrong encoding: ", blob_ptr->encoding(), " vs. ", Background::meta_encoding());
#endif
            }
#ifndef NDEBUG
              else
                 throw InvalidArgumentException("Cannot find pixels for ", fdx, " and ", bdxnpred->bdx);
#endif
        }//else
         //  throw InvalidArgumentException("Cannot find individual ", fdx, " in cache.");
        
        /*if(auto it = cache.fish_selected_blobs.find(fdx);
           it != cache.fish_selected_blobs.end())
        {
            cache.processed_frame().transform_blobs_by_bid(std::array{it->second.bdx}, [&blob_ptr](pv::Blob& blob) {
                blob_ptr = &blob;
            });
            
            if(blob_ptr) {
                if(blob_ptr->encoding() == Background::meta_encoding())
                    ptr->set_data(fdx, frame, blob_ptr, cache.background(), filters, it->second.midline.get());
                else
                    FormatWarning("Not displaying image yet because of the wrong encoding: ", blob_ptr->encoding(), " vs. ", Background::meta_encoding());
            }
            //else
            //    throw InvalidArgumentException("Cannot find pixels for ", fdx, " and ", it->second.bdx);
        } //else
          //  throw InvalidArgumentException("Cannot find individual ", fdx, " in cache.");*/
    }
    
    return false;
}

}

