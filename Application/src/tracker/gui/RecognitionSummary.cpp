#include "RecognitionSummary.h"
#include <tracking/Tracker.h>
#include <gui/gui.h>
#include <tracking/VisualIdentification.h>
#include <gui/GUICache.h>
#include <gui/DrawBase.h>

using namespace track;
namespace py = Python;

namespace gui {
    void RecognitionSummary::update(gui::DrawStructure& base) {
        auto & cache = GUI::instance()->cache();
        
        const float interface_scale = gui::interface_scale();
        
        Font title_font(0.9f / interface_scale, Style::Bold, Align::Center);
        Font font(0.8f / interface_scale);
        Font side_font(0.8f / interface_scale, Align::Right);
        Font bottom_font(0.8f / interface_scale, Align::Center);
        
        auto manual_identities = Tracker::identities();
        auto sorted = manual_identities;
        for(auto id : manual_identities) {
            if(cache.individuals.find(id) == cache.individuals.end())
                sorted.erase(id);
        }

        size_t output_size = py::VINetwork::number_classes();
        obj.set_scale(base.scale().reciprocal().mul(interface_scale));

        const float margin = 5 / interface_scale,
        bar_width = 80 / (interface_scale * 1.25f),
        title_height = Base::default_line_spacing(title_font) + margin;

        float sidebar_width = 0;
        for(auto id : sorted) {
            auto fish = cache.individuals.at(id);
            auto bds = Base::default_text_bounds(fish->identity().name(), &obj, side_font);
            sidebar_width = max(sidebar_width, bds.width);
        }
        sidebar_width += 3 * margin;

        obj.set_origin(Vec2(0.5));
        obj.set_bounds(Bounds(Vec2(Tracker::average().cols, Tracker::average().rows) * 0.5f,
                              Size2(sidebar_width * 1.5f, Base::default_line_spacing(font) + margin + title_height) + Size2(margin * 2) + bar_width * Size2(output_size, sorted.size())));
        obj.set_background(Black.alpha(150));

        if(!cache.recognition_updated) {
            obj.update([&] (Entangled& base) {
                std::vector<float> outputs;
                base.add<Text>(Str("recognition summary"), Loc(obj.width() * 0.5f, margin + (title_height - margin) * 0.5f), TextClr(White), title_font);
                
                size_t counter = 0, j = 0;
                std::map<Idx_t, size_t> fdx_to_idx;
                std::map<size_t, Idx_t> idx_to_fdx;
                
                outputs.resize(output_size * sorted.size());
                
                for(auto id : sorted) {
                    auto fish = cache.individuals.at(id);
                    
                    float maxp = 0;
                    
                    for(auto && [fdx, p] : fish->average_recognition()) {
                        if(p > maxp) {
                            maxp = p;
                        }
                        
                        auto it = fdx_to_idx.find(fdx);
                        if(it == fdx_to_idx.end()) {
                            idx_to_fdx[counter] = fdx;
                            fdx_to_idx[fdx] = counter++;
                        }
                        auto idx = fdx_to_idx.at(fdx);
                        outputs.at(j * output_size + idx) = p;
                    }
                    
                    ++j;
                }
                
                auto image = Image::Make(bar_width * sorted.size(), bar_width * output_size, 4);
                image->set_to(0);
                auto mat = image->get();
                
                for(size_t row = 0; row < sorted.size(); ++row) {
                    for (size_t j=0; j<output_size; ++j) {
                        auto p = idx_to_fdx.count(j) ? outputs.at(row * output_size + idx_to_fdx.at(j).get()) : 0;
                        
                        Color interp = Viridis::value(p).alpha(200);
                        cv::rectangle(mat,
                                      Vec2(j,row) * bar_width + Vec2(1),
                                      Vec2(j+1,row+1) * bar_width - Vec2(2),
                                      interp,
                                      -1);
                    }
                }
                
                auto pos = Vec2(margin + sidebar_width, margin + title_height);
                auto bounds = Box(pos, image->bounds().size());
                base.add<ExternalImage>(std::move(image), pos);
                base.add<Rect>(Box(bounds), FillClr{Transparent}, LineClr{White.alpha(200)});
                
                // draw vertical bar (active fish)
                pos = Vec2(margin) + Vec2(sidebar_width - 10 / interface_scale, bar_width * 0.5f - Base::default_line_spacing(font) * 0.5f + title_height);
                
                size_t row = 0;
                for(auto id : sorted) {
                    auto fish = cache.individuals.at(id);
                    base.add<Text>(Str(fish->identity().name()), Loc(pos + Vec2(0, bar_width) * row), TextClr(White), side_font);
                    ++row;
                }
                
                // draw horizontal bar (matched fish from network)
                pos = Vec2(margin) + Vec2(sidebar_width + bar_width * 0.5f, bounds.height + margin + Base::default_line_spacing(font) * 0.5f + title_height);
                for(size_t idx = 0; idx < output_size; ++idx) {
                    base.add<Text>(Str(Meta::toStr(idx)), Loc(pos), TextClr(White), bottom_font);
                    pos += Vec2(bar_width, 0);
                }
            });
            
            cache.recognition_updated = true;
        }
        
        base.wrap_object(obj);
    }
}
