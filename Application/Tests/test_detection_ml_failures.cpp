#include <commons.pc.h>

#include "gtest/gtest.h"

#include "segmenter_test_utils.h"

#include <python/Detection.h>
#include <python/PipelineRegistry.h>
#include <python/PythonWrapper.h>
#include <python/YOLO.h>
#include <ui/Segmenter.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <future>
#include <optional>
#include <string_view>
#include <thread>

using namespace cmn;
using namespace cmn::file;
using namespace track;
using namespace track::detect;

namespace {

namespace fs = std::filesystem;
using namespace trex_test;

constexpr auto kWaitDeadline = std::chrono::seconds(60);

class BinaryWatchdog {
    std::shared_ptr<std::atomic_bool> stopped = std::make_shared<std::atomic_bool>(false);

public:
    BinaryWatchdog() {
        std::thread([state = stopped]() {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(4);
            while(!state->load() && std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::seconds(1));

            if(!state->load()) {
                fprintf(stderr, "[watchdog] ML failure tests exceeded their process deadline.\n");
                fflush(stderr);
                std::_Exit(2);
            }
        }).detach();
    }

    ~BinaryWatchdog() {
        stopped->store(true);
    }
};

BinaryWatchdog binary_watchdog;

void write_text_file(const fs::path& path, std::string_view contents) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if(!stream)
        throw std::runtime_error("Cannot write test Python module: " + path.string());
    stream << contents;
    if(!stream)
        throw std::runtime_error("Failed while writing test Python module: " + path.string());
}

std::string indent_python(std::string_view body) {
    std::string result;
    size_t start = 0;
    while(start <= body.size()) {
        const auto end = body.find('\n', start);
        const auto line = body.substr(start, end == std::string_view::npos ? body.size() - start : end - start);
        result += "    ";
        result.append(line);
        result += '\n';
        if(end == std::string_view::npos)
            break;
        start = end + 1;
    }
    return result;
}

std::string bbx_module_source(std::string_view load_body, std::string_view predict_body) {
    std::string source = R"PY(import numpy as np
import TRex

def load_yolo(configs):
)PY";
    source += indent_python(load_body);
    source += R"PY(
def _empty_result(index):
    empty = np.empty((0,), dtype=np.float32)
    return TRex.Result(
        index,
        TRex.Boxes(empty),
        [],
        TRex.KeypointData(empty),
        TRex.ObbData(empty),
        TRex.PointData(empty),
    )

def _result_count(input):
    original_ids = input.orig_id()
    return max(original_ids) + 1 if original_ids else 0

def predict(input):
)PY";
    source += indent_python(predict_body);
    return source;
}

class PythonMlEnvironment final : public ::testing::Environment {
public:
    fs::path root;
    fs::path model;
    bool available{false};
    std::string unavailable_reason;

    void SetUp() override {
        try {
            root = fs::temp_directory_path() / ("trex-ml-failures-" + unique_suffix());
            fs::create_directories(root);
            model = root / "dummy.pt";

            write_text_file(root / "trex_init.py", "import TRex\n");
            write_text_file(root / "trex_yolo.py", "# Module presence is required by the detector reload path.\n");
            write_text_file(root / "trex_detection_model.py", "# Module presence is required by the detector reload path.\n");
            fs::create_hard_link(
                root / "trex_detection_model.py",
                root / "trex_rfdetr.py");
            write_text_file(model, "test model placeholder\n");
            write_bbx("return configs", "return [_empty_result(i) for i in range(_result_count(input))]");

            GlobalSettings::write([](Configuration& config) {
                grab::default_config::get(config);
                ::default_config::get(config);
            });
            SETTING(wd) = Path(root.string());
            register_data_locations_once();

            Python::configure_runtime(
                GlobalSettings::instance(),
                file::DataLocation::instance(),
                Python::get_instance(),
                &testTileBuffers(),
                [](auto& name, auto& mat) { tf::imshow(name, mat); },
                []() { tf::destroyAllWindows(); }
            );
            Python::ensure_python_impl_loaded();
            buffers::TileBuffers::create();

            auto ready = Python::schedule([]() {});
            if(ready.wait_for(kWaitDeadline) != std::future_status::ready) {
                unavailable_reason = "Embedded Python initialization timed out.";
                return;
            }
            ready.get();
            available = true;
        } catch(const std::exception& ex) {
            unavailable_reason = ex.what();
        }
    }

    void TearDown() override {
        if(available) {
            try {
                auto future = Python::deinit();
                if(future.valid())
                    future.get();
            } catch(const std::exception& ex) {
                ADD_FAILURE() << "Python::deinit() failed: " << ex.what();
            } catch(...) {
                ADD_FAILURE() << "Python::deinit() failed with an unknown exception.";
            }
            available = false;
        }

        std::error_code ec;
        fs::remove_all(root, ec);
    }

    void write_bbx(std::string_view load_body, std::string_view predict_body) const {
        write_text_file(root / "bbx_saved_model.py", bbx_module_source(load_body, predict_body));
    }
};

auto* python_environment = static_cast<PythonMlEnvironment*>(
    ::testing::AddGlobalTestEnvironment(new PythonMlEnvironment));

void configure_yolo_settings() {
    GlobalSettings::write([](Configuration& config) {
        grab::default_config::get(config);
        ::default_config::get(config);
    });

    SETTING(wd) = Path(python_environment->root.string());
    SETTING(detect_type) = ObjectDetectionType_t{ObjectDetectionType::yolo};
    SETTING(detect_model) = Path(python_environment->model.string());
    SETTING(detect_batch_size) = uchar(1);
    SETTING(detect_resolution) = DetectResolution{32, 32};
    SETTING(detect_tile_image) = uchar(0);
    SETTING(detect_tile_target_width) = uint16_t(0);
    SETTING(detect_tile_overlap) = float(0);
    SETTING(track_background_subtraction) = false;
    SETTING(calculate_posture) = false;
    SETTING(meta_encoding) = meta_encoding_t::gray;
    SETTING(nowindow) = true;
    SETTING(auto_quit) = false;
    SETTING(terminate) = false;
    SETTING(error_terminate) = false;
}

std::optional<std::string> deinit_detection() {
    auto future = std::async(std::launch::async, []() -> std::optional<std::string> {
        try {
            Detection::deinit();
            return std::nullopt;
        } catch(const std::exception& ex) {
            return ex.what();
        }
    });

    if(future.wait_for(kWaitDeadline) != std::future_status::ready) {
        ADD_FAILURE() << "Detection::deinit() did not finish before the deadline.";
        return "timeout";
    }
    return future.get();
}

TileImage make_tile(Frame_t frame_index, const std::shared_ptr<std::atomic_size_t>& callbacks) {
    TileImage tile;
    tile.tile_size = Size2(8, 8);
    tile.source_size = Size2(8, 8);
    tile.prepared_size = Size2(8, 8);
    tile.data.image = Image::Make(8, 8, 3);
    tile.data.image->set_index(frame_index.get());
    tile.images.emplace_back(Image::Make(8, 8, 3));
    tile.set_tile_geometries({
        TileGeometry{
            .source_region = SourceRect(0, 0, 8, 8),
            .tile_content = TileRect(0, 0, 8, 8),
            .tile_size = Size2(8, 8)
        }
    });
    tile.callback = [callbacks]() {
        callbacks->fetch_add(1);
    };
    return tile;
}

struct AppliedBatch {
    std::vector<std::future<SegmentationData>> futures;
    std::shared_ptr<std::atomic_size_t> callbacks = std::make_shared<std::atomic_size_t>(0);
};

AppliedBatch apply_tiles(size_t count) {
    AppliedBatch batch;
    std::vector<TileImage> tiles;
    tiles.reserve(count);
    batch.futures.reserve(count);

    for(size_t i = 0; i < count; ++i) {
        auto tile = make_tile(Frame_t(narrow_cast<long_t>(i)), batch.callbacks);
        tile.promise = std::make_unique<std::promise<SegmentationData>>();
        batch.futures.emplace_back(tile.promise->get_future());
        tiles.emplace_back(std::move(tile));
    }

    YOLO::apply(std::move(tiles));
    return batch;
}

void expect_soft_failures(AppliedBatch& batch, std::string_view expected_message = {}) {
    for(auto& future : batch.futures) {
        ASSERT_EQ(future.wait_for(kWaitDeadline), std::future_status::ready)
            << "A YOLO result promise was never completed.";
        try {
            (void)future.get();
            ADD_FAILURE() << "Expected the YOLO result future to fail.";
        } catch(const SoftExceptionImpl& ex) {
            if(!expected_message.empty())
                EXPECT_NE(std::string(ex.what()).find(expected_message), std::string::npos) << ex.what();
        } catch(const std::future_error& ex) {
            ADD_FAILURE() << "Promise failed as broken_promise instead of SoftException: " << ex.what();
        } catch(const std::exception& ex) {
            ADD_FAILURE() << "Promise failed with the wrong exception type: " << ex.what();
        }
    }

    const auto deadline = std::chrono::steady_clock::now() + kWaitDeadline;
    while(batch.callbacks->load() != batch.futures.size()
          && std::chrono::steady_clock::now() < deadline)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_EQ(batch.callbacks->load(), batch.futures.size());
}

void expect_success(AppliedBatch& batch) {
    for(auto& future : batch.futures) {
        ASSERT_EQ(future.wait_for(kWaitDeadline), std::future_status::ready);
        EXPECT_NO_THROW((void)future.get());
    }

    const auto deadline = std::chrono::steady_clock::now() + kWaitDeadline;
    while(batch.callbacks->load() != batch.futures.size()
          && std::chrono::steady_clock::now() < deadline)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_EQ(batch.callbacks->load(), batch.futures.size());
}

class MLFailures : public ::testing::Test {
protected:
    void SetUp() override {
        if(!python_environment->available)
            GTEST_SKIP() << python_environment->unavailable_reason;

        testTileBuffers().clear();
        buffers::TileBuffers::clear();
        configure_yolo_settings();
        python_environment->write_bbx(
            "return configs",
            "return [_empty_result(i) for i in range(_result_count(input))]");
    }

    void TearDown() override {
        if(python_environment->available
           && detect::try_pipeline_manager(ObjectDetectionType::yolo))
        {
            const auto error = deinit_detection();
            if(error)
                ADD_FAILURE() << "Detection::deinit() failed: " << *error;
        }

        testTileBuffers().clear();
        if(buffers::TileBuffers::instance_if_set())
            buffers::TileBuffers::clear();
    }
};

TEST_F(MLFailures, ModelLoadFailureThrowsFromInit) {
    python_environment->write_bbx(
        "raise RuntimeError('CUDA out of memory while loading model')",
        "return []");

    std::string message;
    try {
        Detection::init();
        FAIL() << "Detection::init() unexpectedly accepted a failing model loader.";
    } catch(const std::exception& ex) {
        message = ex.what();
    }

    EXPECT_NE(message.find("CUDA out of memory"), std::string::npos) << message;
    const auto error = deinit_detection();
    EXPECT_FALSE(error.has_value()) << (error ? *error : "");
}

TEST_F(MLFailures, ModuleImportFailureSurfacesAtPredict) {
    ASSERT_NO_THROW(Detection::init());

    // Reload failures leave the module unavailable; prediction must complete
    // every outstanding request with an error instead of dropping promises.
    write_text_file(
        python_environment->root / "bbx_saved_model.py",
        "raise RuntimeError('bbx module import exploded')\n");

    auto batch = apply_tiles(3);
    expect_soft_failures(batch);
}

TEST_F(MLFailures, ShortResultsResolveAllPromises) {
    python_environment->write_bbx("return configs", "return [_empty_result(0)]");
    ASSERT_NO_THROW(Detection::init());

    auto short_batch = apply_tiles(3);
    expect_soft_failures(short_batch, "returned 1 result");

    auto next_batch = apply_tiles(1);
    expect_success(next_batch);
}

TEST_F(MLFailures, PredictReturnsGarbageResolvesAllPromises) {
    python_environment->write_bbx("return configs", "return 42");
    ASSERT_NO_THROW(Detection::init());

    auto garbage_batch = apply_tiles(3);
    expect_soft_failures(garbage_batch);

    python_environment->write_bbx(
        "return configs",
        "return [_empty_result(i) for i in range(_result_count(input))]");
    auto next_batch = apply_tiles(1);
    expect_success(next_batch);
}

TEST_F(MLFailures, PredictRaisesAfterNFramesFullConversion) {
    python_environment->write_bbx(
        "global calls\ncalls = 0\nreturn configs",
        "global calls\ncalls += 1\nif calls > 2:\n    raise RuntimeError('CUDA out of memory during predict')\nreturn [_empty_result(i) for i in range(_result_count(input))]");

    const TempWorkspace ws = make_workspace();
    const auto source_paths = create_synthetic_sequence(ws.root / "source", 12);
    SETTING(output_dir) = Path((ws.root / "output").string());
    SETTING(filename) = Path("synthetic_ml_failure");
    SETTING(source) = PathArray(source_paths);
    SETTING(save_raw_movie) = false;
    SETTING(frame_rate) = uint32_t(25);
    SETTING(track_threshold) = int(15);
    SETTING(meta_real_width) = Float2_t(1);
    SETTING(cm_per_pixel) = Float2_t(1);
    SETTING(average_samples) = uint32_t(4);
    SETTING(video_conversion_range) = Range<long_t>(-1, -1);

    auto completion = make_completion_state();
    auto result = completion->result.get_future();

    {
        Segmenter segmenter(
            [completion]() { signal_completion(completion, "eof"); },
            [completion](std::string error) {
                signal_completion(completion, "error:" + error);
            });

        ASSERT_NO_THROW(segmenter.open_video());
        ASSERT_NO_THROW(segmenter.start());
        ASSERT_EQ(result.wait_for(kWaitDeadline), std::future_status::ready)
            << "Segmenter did not surface the Python prediction failure.";
    }

    const auto status = result.get();
    ASSERT_TRUE(status.starts_with("error:")) << status;
    EXPECT_NE(status.find("CUDA out of memory during predict"), std::string::npos) << status;
}

} // namespace
