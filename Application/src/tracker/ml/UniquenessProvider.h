#pragma once

#include <commons.pc.h>
#include <misc/frame_t.h>
#include <misc/vec2.h>
#include <pv.h>
#include <misc/DetectionTypes.h>
#include <tracking/TrainingData.h>

namespace track {

/**
 * @brief Thread-safe provider that keeps the *latest* uniqueness values.
 *
 * The first `request_update()` call (or any call after the underlying
 * VI-weights change) launches an asynchronous worker that fills:
 *  * `estimated_uniqueness()`   – map<Frame_t,float>     (fast access)
 *  * `uniqueness_points()`      – vector<Vec2>           (ready for graphs)
 *
 * The worker re-uses the heavy *Accumulation* pipeline already present
 * in `DrawUniqueness.cpp`; the code has simply moved here.
 *
 * The class is *header-only* on purpose (all short inline accessors) while
 * the implementation of the async worker lives in `UniquenessProvider.cpp`.
 */
class UniquenessProvider
{
public:
    using Map  = std::unordered_map<cmn::Frame_t, float>;

    explicit UniquenessProvider(std::weak_ptr<pv::File> video_source) noexcept;

    /** If necessary, launches / relaunches the async calculation.            */
    void request_update();

    /** @return `true` while the worker thread is still crunching numbers.    */
    [[nodiscard]] bool busy()   const noexcept;

    /** @return `true` if a valid result (or error) is already available.     */
    [[nodiscard]] bool ready()  const noexcept;

    /** latest VI weights that produced the result. `std::nullopt` on error.  */
    [[nodiscard]] std::optional<track::vi::VIWeights> origin() const noexcept;

    /** Map for direct look-ups (undefined if !ready())                       */
    [[nodiscard]] const Map& estimated_uniqueness() const;

    /** Pre-packed Vec2 list for graphs (undefined if !ready())               */
    [[nodiscard]] const std::vector<cmn::Vec2>& uniqueness_points() const;

    /** If the worker finished with an error, this holds the message.         */
    [[nodiscard]] const std::optional<std::string>& last_error() const noexcept;

    /** Summary of the last run: `std::nullopt` while busy / before first run.
     *  • `std::expected<VIWeights,std::string>{weights}`  on success
     *  • `std::unexpected<std::string>{msg}`              on error
     */
    [[nodiscard]] const std::optional<std::expected<track::vi::VIWeights,
                                                   std::string>>&
    uniqueness_origin() const noexcept;

    /**
     * @brief Convenience: fetch point vector only if a *valid* result is ready.
     *
     * @return `std::expected<const std::vector<Vec2>&, std::string>`
     *         • **value**   – reference to internal `_points` when a result is present,
     *                          the provider is *not* busy, and no error occurred.
     *         • **error**   – explanatory message otherwise.
     *
     * Thread‑safe and zero‑copy; the caller must *not* modify the returned vector.
     */
    [[nodiscard]]
    std::expected<std::vector<cmn::Vec2>, std::string>
    points_if_ready() const noexcept;

    /** Forget everything – next call to `request_update()` recomputes.       */
    void reset();

private:
    void launch_worker_locked_();   // called with _mutex held

    // ------------------------------------------------------------
    mutable std::mutex                        _mutex;
    std::weak_ptr<pv::File>                   _video_source;

    std::optional<std::future<void>>          _running;
    std::optional<track::vi::VIWeights>       _last_origin;      // what UI saw last
    std::optional<track::vi::VIWeights>       _result_origin;    // weights used for current result
    std::optional<std::string>                _last_error;

    Map                                       _map;
    std::vector<cmn::Vec2>                         _points;

    // buffered TrainingData/images (moved here from DrawUniqueness)
    struct Samples {
        std::shared_ptr<TrainingData> data;
        std::vector<Image::SPtr>      images;
        std::map<cmn::Frame_t, cmn::Range<size_t>> map;
    };
    std::mutex                     _samples_mutex;
    std::optional<Samples>         _samples;

    std::optional<std::expected<track::vi::VIWeights, std::string>> _uniqueness_origin;
    
    
    /** @return `true` while the worker thread is still crunching numbers.    */
    [[nodiscard]] bool _busy()   const noexcept;
};

} // namespace track
