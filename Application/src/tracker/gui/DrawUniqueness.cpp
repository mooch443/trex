#include "DrawUniqueness.h"
#include <misc/GlobalSettings.h>
#include <gui/Graph.h>
#include <misc/vec2.h>
#include <tracking/VisualIdentification.h>
#include <gui/GUICache.h>
#include <gui/WorkProgress.h>
#include <ml/Accumulation.h>
#include <misc/Coordinates.h>
#include <gui/types/Button.h>

namespace py = Python;

namespace cmn::gui {

struct DrawUniqueness::Data {
    Graph graph{Bounds(50, 100, 800, 400), "uniqueness"};
    Rect hover_rect{attr::Box{0,0,0,0}, FillClr{White.alpha(15)}};
    StaticText _title{attr::Margins{0, 0, 0, 0}};
    Button _close{
        attr::Size{30,30},
        Str{"<sym>✕</sym>"},
        FillClr{100,50,50,150},
        TextClr{White}, Font{0.55}, Margins{-5,0,0,0}, Origin{1,0}
    };
    std::mutex mutex;
    std::map<Frame_t, float> estimated_uniqueness;
    std::vector<Vec2> uniquenesses;
    std::optional<std::future<void>> running;
    std::optional<track::vi::VIWeights> last_origin;
    std::optional<tl::expected<track::vi::VIWeights, std::string>> uniqueness_origin;
    Frame_t frameNr;

    std::map<long_t, float> smooth_points;
    GUICache* _cache{nullptr};
    std::weak_ptr<pv::File> _video_source;
    
    struct Samples {
        std::shared_ptr<TrainingData> data;
        std::vector<Image::SPtr> images;
        std::map<Frame_t, Range<size_t>> map;
    };
    
    std::mutex _samples_mutex;
    std::optional<Samples> _samples;

    void update(Entangled& base);
    bool should_update_uniquenesses();
};

DrawUniqueness::DrawUniqueness(GUICache* cache, std::weak_ptr<pv::File> video_source)
    : _data(std::make_unique<Data>())
{
    assert(cache);
    _data->_cache = cache;
    _data->_video_source = video_source;
    _data->graph.set_clickable(false);
    _data->graph.set_background(Transparent);
    on_click([this](Event e){
        auto frames = max(0_F, (Float2_t)_data->_cache->tracked_frames.end.get() - (Float2_t)_data->_cache->tracked_frames.start.get());
        if(_data && _data->graph.size().width > 0) {
            auto frameIndex = saturate(e.mbutton.x / _data->graph.size().width * frames, 0_F, (Float2_t)frames);
            if(euclidean_distance(absolute_drag_start(), global_bounds().pos()) < 5)
            {
                //Print("click = ", e.mbutton.x, ", ", e.mbutton.y, " = ", frameIndex);
                SETTING(gui_frame) = Frame_t((uint32_t)frameIndex);
            } else {
                //Print(_data->graph.absolute_drag_start(), " vs. ", _data->graph.pos(), " ", _data->graph.global_bounds());
            }
        }
    });
    on_hover([this](Event e){
        if(not _data)
            return;
        
        _data->hover_rect.set(Box{
            e.hover.x + _data->graph.pos().x,
            _data->graph.pos().y,
            1,
            _data->graph.height()});
    });
    
    _data->_close.on_click([](auto){
        SETTING(gui_show_uniqueness) = false;
    });
    
    set_draggable();
}

DrawUniqueness::~DrawUniqueness() {
    _data->hover_rect.set_parent(nullptr);
    _data->graph.set_parent(nullptr);
    _data->_close.set_parent(nullptr);
    _data = nullptr;
}

void DrawUniqueness::set(Frame_t frame) {
    if(_data->frameNr == frame)
        return;
    _data->frameNr = frame;
    set_content_changed(true);
}

void DrawUniqueness::update() {
    assert(_data);

    OpenContext([this]() {
        try {
            _data->update(*this);
        } catch(const std::exception& e) {
            FormatExcept("Caught exception in DrawUniqueness: ", e.what());
        }
    });
}

void DrawUniqueness::reset() {
    if(_data) {
        std::unique_lock guard{_data->_samples_mutex};
        _data->_samples.reset();
    }
}

bool DrawUniqueness::Data::should_update_uniquenesses() {
    std::lock_guard guard(mutex);
    if(not estimated_uniqueness.empty()
       && last_origin.has_value()
       && last_origin.value() == Python::VINetwork::status().weights)
    {
        return false;
    }
    
    if(not uniqueness_origin.has_value())
        return true;
    
    if(auto origin = uniqueness_origin.value();
       (origin.has_value()
        && origin.value() != Python::VINetwork::status().weights)
       || (not origin.has_value()
            && last_origin != Python::VINetwork::status().weights))
    {
        /*Print("Reasoning: ", uniqueness_origin.has_value()
              ? (origin != Python::VINetwork::status().weights
                 ? "updated network: "+origin->toStr() + " vs. " + Python::VINetwork::status().weights.toStr()
                 : "there was an error and the networks are the same")
              : "no origin");*/
        return true;
    }
    
    return false;
}

void DrawUniqueness::Data::update(Entangled& base) {
#if !COMMONS_NO_PYTHON
    if(not SETTING(gui_show_uniqueness)) {
        return;
    }
    
    if(std::lock_guard guard(mutex);
       running
       && (not running->valid()
           || running->wait_for(std::chrono::milliseconds(0)) == std::future_status::ready))
    {
        try {
            if(running->valid()) {
                running->get();
                running.reset();
                
                graph.clear();
            } else
                running.reset();
        } catch(const SoftExceptionImpl&) {
            /// nothing
#ifndef NDEBUG
            FormatWarning("Caught exception.");
#endif
        } catch(const std::exception& ex) {
            FormatExcept("Exception when loading uniqueness: ", ex.what());
        }
    }
    
    if(should_update_uniquenesses()) {
        if(not running) {
            running = WorkProgress::add_queue("generate images", [&]() {
                try {
                    Accumulation::setup();
                    auto w = Python::VINetwork::status().weights;
                    if(std::lock_guard guard(mutex);
                       not w.loaded())
                    {
                        uniqueness_origin = tl::unexpected<std::string>("No weights are loaded.");
                        last_origin = w;
                        estimated_uniqueness.clear();
                        uniquenesses.clear();
                        return;
                    } else {
                        last_origin = w;
                    }
                    
                    std::tuple<float, hash_map<Frame_t, float>, float> unique;
                    
                    {
                        std::unique_lock guard{_samples_mutex};
                        if(not _samples) {
                            if(auto lock = _video_source.lock();
                               lock)
                            {
                                auto && [data, images, image_map] = Accumulation::generate_discrimination_data(*lock);
                                
                                _samples = Samples{
                                    .data = std::move(data),
                                    .images = std::move(images),
                                    .map = std::move(image_map)
                                };
                            }
                        }
                        
                        unique = Accumulation::calculate_uniqueness(false, _samples->images, _samples->map);
                    }
                    
                    std::lock_guard guard(mutex);
                    estimated_uniqueness.clear();
                    
                    auto && [u, umap, uq] = unique;
                    
                    for(auto &[k,v] : umap)
                        estimated_uniqueness[k] = v;
                    
                    uniquenesses.clear();
                    for(auto && [frame, q] :umap) {
                        uniquenesses.push_back(Vec2(frame.get(), q));
                    }
                    
                    uniqueness_origin = last_origin.value();
                    
                } catch(const SoftExceptionImpl& e) {
#ifndef NDEBUG
                    FormatExcept("Caught exception: ", e.what());
#endif
                    std::lock_guard guard(mutex);
                    uniqueness_origin = tl::unexpected<std::string>(e.what());
                }
            });
        }
    }
    
    auto coords = FindCoord::get();
    auto size = Size2(max(500_F, coords.screen_size().width - 300_F), min(coords.screen_size().height - 100_F, 400_F));
    
    {
        auto p = base.pos();
        if(p.x < 0 || p.y < 0) {
            base.set_pos(Vec2(max(p.x, 0), max(p.y, 0)));
        }
        if(auto screen = coords.screen_size();
           p.x + size.width > screen.width
           || p.y + size.height > screen.height)
        {
            Vec2 min_pos = screen - size;
            base.set_pos(min(min_pos, p));
        }
    }
    
    std::lock_guard guard(mutex);
    if(not estimated_uniqueness.empty()) {
        if(graph.empty()
           || graph.x_range().end == FLT_MAX
           || graph.x_range().end != _cache->tracked_frames.end.get()
           || not graph.size().Equals(size))
        {
            graph.clear();
            _title.set_txt("<h3>Uniqueness</h3>\n<c>"
                    + (last_origin.has_value() && not last_origin->_path.empty()
                        ? utils::ShortenText(last_origin->_path.filename(), 100)
                        : (last_origin->loaded() ? "" : "No VI weights loaded"))
                            + ((last_origin.has_value() && last_origin->_uniqueness.has_value() && last_origin->_uniqueness.value() > 0)
                            ? " (<nr>" + dec<1>(last_origin->_uniqueness.value()*100).toStr()+"</nr><i>%</i>)"
                            : "")
                    + "</c>");
            
            long_t L = (long_t)uniquenesses.size();
            for (long_t i=0; i<L; ++i) {
                long_t offset = 1;
                float factor = 0.5;
                
                smooth_points[i] = 0;
                
                for(; offset < max(1, uniquenesses.size() * 0.15); ++offset) {
                    long_t idx_1 = i-offset;
                    long_t idx1 = i+offset;
                    
                    smooth_points[i] += uniquenesses[idx_1 >= 0 ? idx_1 : 0].y * factor + uniquenesses[idx1 < L ? idx1 : L-1].y * factor;
                    factor *= factor;
                }
                
                smooth_points[i] = (smooth_points[i] + uniquenesses[i].y) * 0.5;
            }
            
            graph.set_ranges(Rangef(_cache->tracked_frames.start.get(), _cache->tracked_frames.end.get()), Rangef(0, 1));
            graph.add_function(Graph::Function("", Graph::Type::DISCRETE, [this, uq = &estimated_uniqueness](float x) -> float {
                std::lock_guard guard(mutex);
                auto it = uq->upper_bound(Frame_t(sign_cast<uint32_t>(x)));
                if(!uq->empty() && it != uq->begin())
                    --it;
                if(it != uq->end() && it->second <= x) {
                    return it->second;
                }
                return gui::GlobalSettings::invalid();
            }, Cyan));
            graph.add_points("", uniquenesses);
            graph.set_bounds(Bounds(Vec2(10,_title.height() + _title.pos().y + 10), size - Size2(10 * 2,_title.height() + _title.pos().y + 15)));
        }
        
        graph.set_zero(frameNr.get());
        base.advance_wrap(graph);
        graph.set_scale(base.scale().reciprocal());
        
        _title.set(Loc{15, 15});
        base.advance_wrap(_title);
        base.advance_wrap(_close);
        _close.set(Loc{size.width - 15, 15});
        base.set_size(size);
        
        if(base.hovered()) {
            base.set_background(Black.alpha(200), Color::blend(Green.alpha(100), White.alpha(200)).alpha(200));
            base.advance_wrap(hover_rect);
        } else {
            base.set_background(Black.alpha(125), White.alpha(200));
        }
    }
#endif
}

}
