#include <commons.pc.h>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <python/SAM3InteractiveSession.h>

using namespace cmn;
using namespace track;

namespace {

TileImage make_tile(Frame_t frame_index) {
    TileImage tile;
    tile.data.image = Image::Make(1, 1, 4);
    tile.data.image->set_index(frame_index.get());
    tile.images.emplace_back(Image::Make(1, 1, 4));
    return tile;
}

class FakeBackend final : public ISam3InteractiveBackend {
public:
    std::vector<std::string> calls;
    std::string current_runtime;

    void reset_runtime(Frame_t max_frame_index) override {
        current_runtime = "reset:" + max_frame_index.toStr();
        calls.push_back(current_runtime);
    }

    void restore_runtime(const Sam3RuntimeBlob& runtime) override {
        current_runtime = runtime.handle;
        calls.push_back("restore:" + runtime.handle);
    }

    Sam3RuntimeBlob snapshot_runtime() override {
        calls.push_back("snapshot:" + current_runtime);
        return Sam3RuntimeBlob{current_runtime};
    }

    SegmentationData predict_frame(TileImage&& tiled) override {
        const auto frame_index = tiled.data.original_index();
        current_runtime = "after:" + frame_index.toStr();
        calls.push_back("predict:" + frame_index.toStr());
        return std::move(tiled.data);
    }
};

} // namespace

TEST(Sam3InteractiveSessionTest, SameFrameRerunRestoresStoredBeforeRuntime) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    Sam3InteractiveSession session(std::move(backend));

    auto first = session.process_frame(make_tile(0_f), 0);
    session.commit_frame(Sam3ProcessedFrame{
        .frame_index = first.frame_index,
        .prompt_revision = first.prompt_revision,
        .before_runtime = first.before_runtime,
        .after_runtime = first.after_runtime
    });

    auto rerun = session.process_frame(make_tile(0_f), 1);

    ASSERT_EQ(rerun.frame_index, 0_f);
    EXPECT_EQ(rerun.before_runtime.handle, "reset:0");
    EXPECT_THAT(
        backend_ptr->calls,
        ::testing::ElementsAre(
            "reset:0",
            "snapshot:reset:0",
            "predict:0",
            "snapshot:after:0",
            "restore:reset:0",
            "predict:0",
            "snapshot:after:0"));
}

TEST(Sam3InteractiveSessionTest, NextFrameRestoresPreviousAfterRuntime) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    Sam3InteractiveSession session(std::move(backend));

    auto first = session.process_frame(make_tile(0_f), 0);
    session.commit_frame(Sam3ProcessedFrame{
        .frame_index = first.frame_index,
        .prompt_revision = first.prompt_revision,
        .before_runtime = first.before_runtime,
        .after_runtime = first.after_runtime
    });

    auto second = session.process_frame(make_tile(1_f), 0);

    ASSERT_EQ(second.frame_index, 1_f);
    EXPECT_EQ(second.before_runtime.handle, "after:0");
    EXPECT_THAT(
        backend_ptr->calls,
        ::testing::Contains("restore:after:0"));
}

TEST(Sam3InteractiveSessionTest, InvalidatingFromFrameDropsLaterPreparedStates) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    Sam3InteractiveSession session(std::move(backend));

    auto first = session.process_frame(make_tile(0_f), 0);
    session.commit_frame(Sam3ProcessedFrame{
        .frame_index = first.frame_index,
        .prompt_revision = first.prompt_revision,
        .before_runtime = first.before_runtime,
        .after_runtime = first.after_runtime
    });

    auto second = session.process_frame(make_tile(1_f), 0);
    session.commit_frame(Sam3ProcessedFrame{
        .frame_index = second.frame_index,
        .prompt_revision = second.prompt_revision,
        .before_runtime = second.before_runtime,
        .after_runtime = second.after_runtime
    });

    session.invalidate_from(1_f);
    backend_ptr->calls.clear();

    auto third = session.process_frame(make_tile(2_f), 0);

    ASSERT_EQ(third.frame_index, 2_f);
    EXPECT_EQ(third.before_runtime.handle, "reset:2");
    EXPECT_THAT(backend_ptr->calls, ::testing::ElementsAre(
        "reset:2",
        "snapshot:reset:2",
        "predict:2",
        "snapshot:after:2"));
}
