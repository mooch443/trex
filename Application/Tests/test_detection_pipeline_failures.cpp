#include <commons.pc.h>

#include "gtest/gtest.h"

#include "segmenter_test_utils.h"

#include <ui/Segmenter.h>
#include <ui/WorkProgress.h>
#include <python/PipelineRegistry.h>
#include <processing/Background.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>

using namespace cmn;
using namespace cmn::file;
using namespace track;
using namespace track::detect;

namespace {

namespace fs = std::filesystem;
using namespace trex_test;

/// Upper bound for any single blocking wait in these tests. A wedged
/// pipeline must fail the test, not stall CI until the ctest TIMEOUT.
constexpr auto kWaitDeadline = std::chrono::seconds(60);

/// Terminates the whole binary if a test overruns even its failed-wait
/// asserts (e.g. a destructor blocking forever after the wait timed out).
struct TestWatchdog {
    std::mutex mutex;
    std::condition_variable cv;
    bool stopped{false};
    std::thread thread;

    explicit TestWatchdog(std::chrono::seconds deadline)
        : thread([this, deadline]() {
            std::unique_lock guard(mutex);
            if (!cv.wait_for(guard, deadline, [this]() { return stopped; })) {
                fprintf(stderr, "[watchdog] Test exceeded its hard deadline - terminating binary.\n");
                fflush(stderr);
                std::_Exit(2);
            }
        })
    { }

    ~TestWatchdog() {
        {
            std::unique_lock guard(mutex);
            stopped = true;
        }
        cv.notify_all();
        thread.join();
    }
};

enum class FaultMode {
    succeed,             //!< behave like NoDetection::apply
    fail_with_exception, //!< set a SoftException on every promise (a raised CUDA OOM looks like this)
    drop_promises,       //!< destroy the promises unfulfilled -> broken_promise at the consumer
    throw_from_callback  //!< throw out of the pipeline callback without touching the promises
};

std::atomic<FaultMode> g_fault_mode{FaultMode::succeed};
std::atomic<size_t> g_succeed_first_n{0};
std::atomic<size_t> g_processed{0};

void release_tile_images(TileImage& tile) {
    for (auto& image : tile.images) {
        buffers::TileBuffers::get().move_back(std::move(image));
    }
    tile.images.clear();
}

void invoke_tile_callback(TileImage& tile) {
    try {
        if (tile.callback)
            tile.callback();
    } catch (...) {
        FormatExcept("Exception in fake-pipeline tile callback.");
    }
}

void fulfill_like_no_detection(TileImage& tile) {
    SegmentationData data = std::move(tile.data);
    data.frame.set_encoding(Background::meta_encoding());

    if (tile.promise) {
        tile.promise->set_value(std::move(data));
        tile.promise = nullptr;
    }

    invoke_tile_callback(tile);
    release_tile_images(tile);
}

/// Replacement for the registered detection pipeline: succeeds for the
/// first `g_succeed_first_n` tiles, then misbehaves per `g_fault_mode`.
void fake_pipeline(std::vector<TileImage>&& tiled) {
    for (auto&& tile : tiled) {
        const auto index = g_processed.fetch_add(1);
        const auto mode = index < g_succeed_first_n.load()
            ? FaultMode::succeed
            : g_fault_mode.load();

        switch (mode) {
        case FaultMode::succeed:
            fulfill_like_no_detection(tile);
            break;

        case FaultMode::fail_with_exception:
            if (tile.promise) {
                try {
                    throw SoftException("CUDA out of memory (fake)");
                } catch (...) {
                    tile.promise->set_exception(std::current_exception());
                }
                tile.promise = nullptr;
            }
            invoke_tile_callback(tile);
            release_tile_images(tile);
            break;

        case FaultMode::drop_promises:
            // destroys the promise without fulfilling it
            tile.promise = nullptr;
            invoke_tile_callback(tile);
            release_tile_images(tile);
            break;

        case FaultMode::throw_from_callback:
            // the remaining tiles (incl. this one) die with the packet
            throw SoftException("fake pipeline callback exploded");
        }
    }
}

void set_fault_mode(FaultMode mode, size_t succeed_first_n = 0) {
    g_processed = 0;
    g_succeed_first_n = succeed_first_n;
    g_fault_mode = mode;
}

void configure_conversion_settings(const TempWorkspace& ws, const std::vector<std::string>& source_paths) {
    const auto output_dir = ws.root / "output";

    SETTING(output_dir) = Path(output_dir.string());
    SETTING(filename) = Path("synthetic_segment");
    SETTING(source) = PathArray(source_paths);
    SETTING(detect_type) = ObjectDetectionType_t{ObjectDetectionType::none};
    SETTING(track_background_subtraction) = false;
    SETTING(calculate_posture) = false;
    SETTING(meta_encoding) = meta_encoding_t::gray;
    SETTING(nowindow) = true;
    SETTING(auto_quit) = false;
    SETTING(save_raw_movie) = false;
    SETTING(frame_rate) = uint32_t(25);
    SETTING(track_threshold) = int(15);
    SETTING(meta_real_width) = Float2_t(1);
    SETTING(cm_per_pixel) = Float2_t(1);
    SETTING(average_samples) = uint32_t(4);
    SETTING(video_conversion_range) = Range<long_t>(-1, -1);
}

/// Runs a headless conversion over `frame_count` synthetic frames with the
/// fault-injecting pipeline and returns the completion status ("eof" or
/// "error:<message>"). Callbacks are wired like the headless
/// `start_converting()` in main.cpp when `set_terminate_flags` is true.
std::string run_conversion(size_t frame_count, bool set_terminate_flags = false) {
    register_data_locations_once();
    reset_global_settings();

    const TempWorkspace ws = make_workspace();
    const auto source_paths = create_synthetic_sequence(ws.root / "source", frame_count);
    configure_conversion_settings(ws, source_paths);

    SETTING(terminate) = false;
    SETTING(error_terminate) = false;

    auto completion = make_completion_state();
    auto future = completion->result.get_future();

    {
        Segmenter segmenter(
            [completion, set_terminate_flags]() {
                if (set_terminate_flags)
                    SETTING(terminate) = true;
                signal_completion(completion, "eof");
            },
            [completion, set_terminate_flags](std::string error) {
                if (set_terminate_flags) {
                    SETTING(error_terminate) = true;
                    SETTING(terminate) = true;
                }
                signal_completion(completion, "error:" + error);
            }
        );

        segmenter.open_video();

        /// open_video registered the real `none` pipeline via Detection{};
        /// replace it with the fault-injecting one. Batch size 1 keeps the
        /// per-tile fault sequence deterministic.
        detect::register_pipeline(
            ObjectDetectionType::none,
            1,
            /*start_paused=*/false,
            [](std::vector<TileImage>&& tiled) {
                fake_pipeline(std::move(tiled));
            });

        segmenter.start();

        if (future.wait_for(kWaitDeadline) != std::future_status::ready) {
            ADD_FAILURE() << "Timed out waiting for the conversion to finish or fail.";
            return "timeout";
        }
    }

    detect::unregister_pipeline(ObjectDetectionType::none);
    return future.get();
}

} // namespace

TEST(PipelineFaults, SucceedingPipelineReachesEof) {
    TestWatchdog watchdog(kWaitDeadline * 3);
    set_fault_mode(FaultMode::succeed);

    const auto status = run_conversion(12);
    EXPECT_EQ(status, "eof") << status;
}

TEST(PipelineFaults, PredictFailsMidConversionStopsWithError) {
    TestWatchdog watchdog(kWaitDeadline * 3);
    set_fault_mode(FaultMode::fail_with_exception, 4);

    const auto status = run_conversion(12);
    ASSERT_TRUE(status.starts_with("error:")) << status;
    EXPECT_NE(status.find("CUDA out of memory (fake)"), std::string::npos) << status;
}

TEST(PipelineFaults, CallbackDropsPromisesStillTerminates) {
    TestWatchdog watchdog(kWaitDeadline * 3);
    set_fault_mode(FaultMode::drop_promises);

    /// a dropped promise surfaces as std::future_error (broken_promise) at
    /// the consumer; the conversion must end with an error, never hang.
    const auto status = run_conversion(12);
    ASSERT_TRUE(status.starts_with("error:")) << status;
}

TEST(PipelineFaults, CallbackThrowsStillTerminates) {
    TestWatchdog watchdog(kWaitDeadline * 3);
    set_fault_mode(FaultMode::throw_from_callback);

    /// the exception is parked in the pipeline's async future and rethrown
    /// on the next enqueue; it must surface as a conversion error.
    const auto status = run_conversion(12);
    ASSERT_TRUE(status.starts_with("error:")) << status;
}

TEST(HeadlessExit, ErrorCallbackSetsTerminateFlags) {
    TestWatchdog watchdog(kWaitDeadline * 3);
    set_fault_mode(FaultMode::fail_with_exception);

    const auto status = run_conversion(12, /*set_terminate_flags=*/true);
    ASSERT_TRUE(status.starts_with("error:")) << status;

    /// the headless wait loop in main.cpp spins on `terminate`; it must be
    /// released promptly once the error callback fired.
    const auto deadline = std::chrono::steady_clock::now() + kWaitDeadline;
    while (!BOOL_SETTING(terminate) && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

    EXPECT_TRUE(BOOL_SETTING(terminate));
    EXPECT_TRUE(BOOL_SETTING(error_terminate));
}

namespace {

void run_work_queue_failure_case(std::function<void()> throwing_task) {
    auto future = gui::WorkProgress::add_queue("failing task", std::move(throwing_task));

    ASSERT_EQ(future.wait_for(kWaitDeadline), std::future_status::ready)
        << "The work queue never finished the throwing task.";
    future.get();

    EXPECT_TRUE(BOOL_SETTING(error_terminate))
        << "A failed work-queue task in headless mode must set error_terminate.";
    EXPECT_TRUE(BOOL_SETTING(terminate));
    EXPECT_FALSE(BOOL_SETTING(auto_quit))
        << "auto_quit must be cleared, otherwise the headless wait loop spins forever.";

    gui::WorkProgress::stop();
}

} // namespace

TEST(WorkProgressFailures, HeadlessQueueExceptionSetsErrorTerminate) {
    TestWatchdog watchdog(kWaitDeadline * 2);
    register_data_locations_once();
    reset_global_settings();

    SETTING(nowindow) = true;
    SETTING(auto_quit) = true;
    SETTING(auto_train_on_startup) = false;
    SETTING(terminate) = false;
    SETTING(error_terminate) = false;

    run_work_queue_failure_case([]() {
        throw SoftException("fake ML failure in work queue");
    });
}

TEST(WorkProgressFailures, AutoTrainStartupFailureTerminates) {
    TestWatchdog watchdog(kWaitDeadline * 2);
    register_data_locations_once();
    reset_global_settings();

    SETTING(nowindow) = false;
    SETTING(auto_quit) = true;
    SETTING(auto_train_on_startup) = true;
    SETTING(terminate) = false;
    SETTING(error_terminate) = false;

    run_work_queue_failure_case([]() {
        throw U_EXCEPTION("auto_train failed on startup (fake)");
    });
}
