#include "DrawUniqueness.h"
#include <misc/GlobalSettings.h>
#include <gui/Graph.h>
#include <misc/vec2.h>
#include <tracking/VisualIdentification.h>
#include <gui/GUICache.h>
#include <gui/WorkProgress.h>
#include <ml/Accumulation.h>

namespace py = Python;

namespace cmn::gui {

struct DrawUniqueness::Data {
    Graph graph{Bounds(50, 100, 800, 400), "uniqueness"};
    std::mutex mutex;
    std::map<Frame_t, float> estimated_uniqueness;
    std::vector<Vec2> uniquenesses;
    std::optional<std::future<void>> running;
    Frame_t frameNr;

    std::map<long_t, float> smooth_points;
    GUICache* _cache{nullptr};
    std::weak_ptr<pv::File> _video_source;

    void update(Entangled& base);
};

DrawUniqueness::DrawUniqueness(GUICache* cache, std::weak_ptr<pv::File> video_source)
    : _data(std::make_unique<Data>())
{
    assert(cache);
    _data->_cache = cache;
    _data->_video_source = video_source;
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

void DrawUniqueness::Data::update(Entangled& base) {
#if !COMMONS_NO_PYTHON
    if(not SETTING(gui_show_uniqueness)) {
        return;
    }
        
    if(estimated_uniqueness.empty()
        && py::VINetwork::status().weights_valid)
    {
        std::lock_guard guard(mutex);
        if(running
           && running->wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
        {
            try {
                running->get();
            } catch(const std::exception& ex) {
                FormatExcept("Exception when loading uniqueness: ", ex.what());
            }
        }
        
        if(not running) {
            running = WorkProgress::add_queue("generate images", [&]() {
                if(auto lock = _video_source.lock();
                   lock)
                {
                    auto && [data, images, image_map] = Accumulation::generate_discrimination_data(*lock);
                    auto && [u, umap, uq] = Accumulation::calculate_uniqueness(false, images, image_map);
                    
                    estimated_uniqueness.clear();
                    
                    std::lock_guard guard(mutex);
                    for(auto &[k,v] : umap)
                        estimated_uniqueness[k] = v;
                    
                    uniquenesses.clear();
                    for(auto && [frame, q] :umap) {
                        uniquenesses.push_back(Vec2(frame.get(), q));
                    }
                }
            });
        }
    }
    
    std::lock_guard guard(mutex);
    if(!estimated_uniqueness.empty()) {
        if(graph.x_range().end == FLT_MAX || graph.x_range().end != _cache->tracked_frames.end.get()) {
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
            if(graph.empty()) {
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
            }
            graph.set_draggable();
        }
        
        graph.set_zero(frameNr.get());
        base.advance_wrap(graph);
        graph.set_scale(base.scale().reciprocal());
    }
#endif
}

}
