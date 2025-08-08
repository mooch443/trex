#include "UniquenessProvider.h"
#include <tracking/VisualIdentification.h>
#include <gui/WorkProgress.h>
#include <ml/Accumulation.h>
#include <tracking/VisualIdentification.h>

using namespace cmn;

namespace track {

UniquenessProvider::UniquenessProvider(std::weak_ptr<pv::File> vs) noexcept
    : _video_source(vs)
{}

bool UniquenessProvider::busy() const noexcept
{
    std::lock_guard lg{_mutex};
    return _busy();
}

bool UniquenessProvider::_busy() const noexcept
{
    return _running && _running->valid()
           && _running->wait_for(std::chrono::milliseconds(0)) != std::future_status::ready;
}

bool UniquenessProvider::ready() const noexcept
{
    std::lock_guard lg{_mutex};
    // original behaviour: ready when we either have a result or captured an error
    return _result_origin.has_value() || _last_error.has_value();
}

const UniquenessProvider::Map&
UniquenessProvider::estimated_uniqueness() const
{
    std::lock_guard lg{_mutex};
    return _map;
}

const std::vector<Vec2>&
UniquenessProvider::uniqueness_points() const
{
    std::lock_guard lg{_mutex};
    return _points;
}

const std::optional<std::string>&
UniquenessProvider::last_error() const noexcept
{
    return _last_error;
}

const std::optional<std::expected<track::vi::VIWeights, std::string>>&
UniquenessProvider::uniqueness_origin() const noexcept
{
    std::lock_guard lg{_mutex};
    return _uniqueness_origin;
}

// --------------------------------------------------------------------------
// consolidated safe accessor
// --------------------------------------------------------------------------
std::expected<std::vector<cmn::Vec2>, std::string>
UniquenessProvider::points_if_ready() const noexcept
{
    std::lock_guard lg{_mutex};

    if (_busy())
        return std::unexpected<std::string>("busy");
    if (_last_error)
        return std::unexpected<std::string>(*_last_error);
    if (!_result_origin)
        return std::unexpected<std::string>("not ready");

    return _points;   // OK
}

std::optional<track::vi::VIWeights>
UniquenessProvider::origin() const noexcept
{
    std::lock_guard lg{_mutex};
    return _result_origin;
}

void UniquenessProvider::request_update()
{
#if COMMONS_NO_PYTHON
    return;
#else
    std::lock_guard guard(_mutex);

    const auto current_w = Python::VINetwork::status().weights;

    /* ---------------------------------------------------------------------
       1.  If we already have a valid map produced with the same weights and
           there is no pending error, we can skip immediately.
    --------------------------------------------------------------------- */
    if (!_map.empty()
        && _last_origin && (*_last_origin == current_w)
        && !_last_error)
    {
        return;
    }

    /* ---------------------------------------------------------------------
       2.  Decide whether we *need* a new run.
           – No previous run     → need
           – Previous run success with different weights → need
           – Previous run error  and weights changed     → need
    --------------------------------------------------------------------- */
    bool needs = false;

    if (!_uniqueness_origin) {
        needs = true;                               // never ran
    } else {
        // unwrap expected without UB
        const auto& exp = *_uniqueness_origin;
        if ((exp && (*exp != current_w))            // success, weights changed
            || (!exp && _last_origin && (*_last_origin != current_w))) // error + change
        {
            needs = true;
        }
    }

    if (!needs || (_running && _busy()))
        return;

    launch_worker_locked_();
#endif
}

void UniquenessProvider::launch_worker_locked_()
{
    _running = cmn::gui::WorkProgress::add_queue("uniqueness calc", [this]() {
        try {
            Accumulation::setup();
            auto w = Python::VINetwork::status().weights;
            
            // 1.  make sure weights are valid
            if (std::lock_guard lg{_mutex};
                !w.loaded())
            {
                _uniqueness_origin = std::unexpected<std::string>("No weights are loaded.");
                _last_origin = w;
                return;
            } else {
                _last_origin = w;
            }

            // 2.  reuse / build samples once
            {
                std::unique_lock sm{_samples_mutex};
                if (!_samples)
                {
                    if (auto vs = _video_source.lock())
                    {
                        auto && [data, images, map]
                            = Accumulation::generate_discrimination_data(*vs);
                        _samples = Samples{ std::move(data),
                                            std::move(images),
                                            std::move(map) };
                    }
                }
            }

            // 3.  heavy lifting
            auto [u, umap, uq]
                = Accumulation::calculate_uniqueness(false,
                                                     _samples->images,
                                                     _samples->map);

            // 4.  commit results
            {
                std::lock_guard lg{_mutex};
                _map.clear();
                _points.clear();

                for (auto &[k,v] : umap)
                {
                    _map.emplace(k, v);
                    _points.emplace_back(Vec2(k.get(), v));
                }

                _result_origin = w;
                _last_error.reset();
                _uniqueness_origin = std::expected<track::vi::VIWeights, std::string>(w);
            }
        }
        catch (const SoftExceptionImpl& e)
        {
            std::lock_guard lg{_mutex};
            _map.clear(); _points.clear();
            _result_origin.reset();
            _last_error        = e.what();
            _uniqueness_origin = std::unexpected<std::string>(e.what());
        }
    });
}

void UniquenessProvider::reset()
{
    std::lock_guard lg{_mutex};
    _running.reset();
    _map.clear();
    _points.clear();
    _result_origin.reset();
    _last_error.reset();
    _uniqueness_origin.reset();
}

} // namespace track
