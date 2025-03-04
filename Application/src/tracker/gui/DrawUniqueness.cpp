#include "DrawUniqueness.h"
#include <misc/GlobalSettings.h>
#include <gui/Graph.h>
#include <misc/vec2.h>
#include <tracking/VisualIdentification.h>
#include <gui/GUICache.h>
#include <gui/WorkProgress.h>
#include <ml/Accumulation.h>
#include <misc/Coordinates.h>

namespace py = Python;

namespace cmn::gui {

struct DrawUniqueness::Data {
    Graph graph{Bounds(50, 100, 800, 400), "uniqueness"};
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

    void update(Entangled& base);
    bool should_update_uniquenesses();
};

DrawUniqueness::DrawUniqueness(GUICache* cache, std::weak_ptr<pv::File> video_source)
    : _data(std::make_unique<Data>())
{
    assert(cache);
    _data->_cache = cache;
    _data->_video_source = video_source;
    _data->graph.on_click([this](Event e){
        auto frames = max(0_F, (Float2_t)_data->_cache->tracked_frames.end.get() - (Float2_t)_data->_cache->tracked_frames.start.get());
        if(_data && _data->graph.size().width > 0) {
            auto frameIndex = saturate(e.mbutton.x / _data->graph.size().width * frames, 0_F, (Float2_t)frames);
            if(euclidean_distance(_data->graph.absolute_drag_start(), _data->graph.global_bounds().pos()) < 5)
            {
                //Print("click = ", e.mbutton.x, ", ", e.mbutton.y, " = ", frameIndex);
                SETTING(gui_frame) = Frame_t((uint32_t)frameIndex);
            } else {
                //Print(_data->graph.absolute_drag_start(), " vs. ", _data->graph.pos(), " ", _data->graph.global_bounds());
            }
        }
    });
}

DrawUniqueness::~DrawUniqueness() { }

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

bool DrawUniqueness::Data::should_update_uniquenesses() {
    std::lock_guard guard(mutex);
    if(not estimated_uniqueness.empty()
       && last_origin == Python::VINetwork::status().weights)
    {
        return false;
    }
    
    if(not uniqueness_origin.has_value())
        return true;
    
    if(auto origin = uniqueness_origin.value();
       (origin.has_value()
         && origin.value() != Python::VINetwork::status().weights)
       || Python::VINetwork::status().weights.valid())
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
    
    {
        std::lock_guard guard(mutex);
        if(running
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
    }
    
    if(should_update_uniquenesses()) {
        if(not running) {
            running = WorkProgress::add_queue("generate images", [&]() {
                if(auto lock = _video_source.lock();
                   lock)
                {
                    try {
                        Accumulation::setup();
                        auto && [data, images, image_map] = Accumulation::generate_discrimination_data(*lock);
                        auto && [u, umap, uq] = Accumulation::calculate_uniqueness(false, images, image_map);
                        
                        std::lock_guard guard(mutex);
                        last_origin.reset();
                        estimated_uniqueness.clear();
                        
                        for(auto &[k,v] : umap)
                            estimated_uniqueness[k] = v;
                        
                        uniquenesses.clear();
                        for(auto && [frame, q] :umap) {
                            uniquenesses.push_back(Vec2(frame.get(), q));
                        }
                        
                        last_origin = Python::VINetwork::status().weights;
                        uniqueness_origin = last_origin.value();
                        
                    } catch(const SoftExceptionImpl& e) {
#ifndef NDEBUG
                        FormatExcept("Caught exception: ", e.what());
#endif
                        std::lock_guard guard(mutex);
                        uniqueness_origin = tl::unexpected<std::string>(e.what());
                    }
                }
            });
        }
    }
    
    auto coords = FindCoord::get();
    auto size = Size2(max(500_F, coords.screen_size().width - 300_F), min(coords.screen_size().height - 100_F, 400_F));
    
    if(auto p = base.pos();
       p.x < 0 || p.y < 0)
    {
        base.set_pos(Vec2());
        
    } else if(auto screen = coords.screen_size();
              p.x >= screen.width
           || p.y >= screen.height)
    {
        base.set_pos(screen - size);
    }
    
    std::lock_guard guard(mutex);
    if(not estimated_uniqueness.empty()) {
        if(graph.empty()
           || graph.x_range().end == FLT_MAX
           || graph.x_range().end != _cache->tracked_frames.end.get()
           || not graph.size().Equals(size))
        {
            graph.clear();
            graph.set_title("Uniqueness"
                    + (last_origin.has_value() && not last_origin->_path.empty()
                        ? ": "+utils::ShortenText(last_origin->_path.filename(), 35)
                        : ": No VI weights loaded")
                            + ((last_origin.has_value() && last_origin->_uniqueness.has_value() && last_origin->_uniqueness.value() > 0)
                            ? " (" + dec<1>(last_origin->_uniqueness.value()*100).toStr()+"%)"
                            : ""));
            
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
            graph.add_function(Graph::Function("raw", Graph::Type::DISCRETE, [this, uq = &estimated_uniqueness](float x) -> float {
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
            graph.set_draggable();
            graph.set_size(size);
        }
        
        graph.set_zero(frameNr.get());
        base.advance_wrap(graph);
        graph.set_scale(base.scale().reciprocal());
    }
#endif
}

}
