#include <commons.pc.h>
#include "gtest/gtest.h"
#include <misc/SpriteMap.h>

namespace {

class SpriteMapMoveTests : public testing::TestWithParam<bool> {};

TEST_P(SpriteMapMoveTests, TransfersAttachedGeneralAndPendingCallbacks) {
    using namespace cmn;
    std::vector<std::string> before_writes, after_writes;
    int replaced_callbacks = 0;
    sprite::Map source;
    source.insert("existing", 0);
    source.register_before_write_callback([&](std::string_view name) {
        before_writes.emplace_back(name);
    });
    auto callbacks = source.register_callbacks<sprite::RegisterInit::DONT_TRIGGER>(
        {"existing", "pending"}, [&](std::string_view name) {
            after_writes.emplace_back(name);
        });

    std::unique_ptr<sprite::Map> destination;
    if(GetParam()) {
        destination = std::make_unique<sprite::Map>();
        destination->insert("replaced", 0);
        destination->register_before_write_callback([&](std::string_view) {
            ++replaced_callbacks;
        });
        *destination = std::move(source);
    } else {
        destination = std::make_unique<sprite::Map>(std::move(source));
    }

    (*destination)["existing"] = 1;
    destination->insert("pending", 0);
    (*destination)["pending"] = 1;
    destination->insert("new", 0);
    (*destination)["new"] = 1;

    EXPECT_EQ(before_writes, (std::vector<std::string>{"existing", "pending", "new"}));
    EXPECT_EQ(after_writes, (std::vector<std::string>{"existing", "pending"}));
    EXPECT_EQ(replaced_callbacks, 0);
    ASSERT_TRUE(callbacks.is_ready());
    destination->unregister_callbacks(std::move(callbacks));
    (*destination)["existing"] = 2;
    (*destination)["pending"] = 2;
    EXPECT_EQ(after_writes.size(), 2u);

    source.insert("reused", 0);
    source["reused"] = 1;
    EXPECT_EQ(before_writes.size(), 5u);
}

INSTANTIATE_TEST_SUITE_P(MoveOperations, SpriteMapMoveTests, testing::Bool());

TEST(SpriteMapTests, SelfMovePreservesCallbacks) {
    cmn::sprite::Map map;
    int calls = 0, before_calls = 0;
    map.register_general_callback([&](std::string_view) { ++calls; });
    map.register_before_write_callback([&](std::string_view) { ++before_calls; });
    map.insert("value", 0);
    
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(before_calls, 0);
    
    auto& same = map;
    map = std::move(same);
    
    map["value"] = 1;
    EXPECT_EQ(calls, 2);
    EXPECT_EQ(before_calls, 1);
}

TEST(SpriteMapTests, CopyAssignmentPreservesBeforeWriteCallbacksOnce) {
    cmn::sprite::Map source, destination;
    source.insert("value", 1);
    destination.insert("value", 0);
    std::vector<int> before_values;
    destination.register_before_write_callback([&](std::string_view name) {
        before_values.push_back(destination.at(name).value<int>());
    });

    for(int next : {2, 3}) {
        destination = source;
        destination["value"] = next;
    }
    EXPECT_EQ(before_values, (std::vector<int>{1, 1}));
}

class CallbackAttachmentMap : public cmn::sprite::Map {
public:
    auto block_callback_attachment() {
        return std::unique_lock{pending_mutex};
    }
};

TEST(SpriteMapTests, AttachesCallbacksBeforePublishingInsertedProperties) {
    CallbackAttachmentMap map;
    int calls = 0;
    map.register_general_callback([&](std::string_view) { ++calls; });
    auto guard = map.block_callback_attachment();
    std::promise<void> started;
    auto inserted = std::async(std::launch::async, [&] {
        started.set_value();
        map.insert("value", 0);
    });
    started.get_future().wait();

    // A property must stay invisible while callback attachment is blocked.
    bool visible = false;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
    while(std::chrono::steady_clock::now() < deadline) {
        if(map.has("value")) {
            visible = true;
            break;
        }
        std::this_thread::yield();
    }

    guard.unlock();
    inserted.get();
    EXPECT_EQ(calls, 1);
    EXPECT_FALSE(visible);
    map["value"] = 1;
    EXPECT_EQ(calls, 2);
}

TEST(SpriteMapTests, InitialCallbackCanRegisterAGeneralCallback) {
    cmn::sprite::Map map;
    int calls = 0, inner_calls = 0;
    auto pending = map.register_callbacks({"value"}, [&](std::string_view) {
        map.register_general_callback([&](std::string_view) {
            ++calls;
        });
        ++inner_calls;
    });
    map.insert("value", 0);
    ASSERT_TRUE(pending.is_ready());
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(inner_calls, 1);
    map["value"] = 1;
    EXPECT_EQ(calls, 3);
    EXPECT_EQ(inner_calls, 2);
    map.unregister_callbacks(std::move(pending));
    map["value"] = 1; // value has not changed => no callbacks
    EXPECT_EQ(calls, 3);
    EXPECT_EQ(inner_calls, 2);
    
    // unregistered the outer one, so only the general callbacks receive signal
    map["value"] = 2;
    EXPECT_EQ(calls, 5);
    EXPECT_EQ(inner_calls, 2);
}

}
