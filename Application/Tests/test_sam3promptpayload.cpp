#include <commons.pc.h>

#include "gtest/gtest.h"

#include <core/GPURecognitionTypes.h>
#include <misc/zipper.h>
#include <python/SAM3PromptState.h>

using namespace cmn;
using namespace track::detect;

TEST(Sam3PromptsTest, MaterializeLegacyMultiBoxPromptsAsSeparateObjects) {
    const Sam3Prompts prompts{
        {Frame_t{}, Sam3PromptList{Sam3PromptPayload{.value = std::string("fish")}}},
        {3_f, Sam3PromptList{Sam3PromptPayload{
            .value = std::vector<Bounds>{
                Bounds(0.f, 0.f, 10.f, 10.f),
                Bounds(20.f, 20.f, 10.f, 10.f),
                Bounds(40.f, 40.f, 10.f, 10.f),
            }
        }}}
    };

    const auto materialized = track::materialize_sam3_prompt_state(3_f, std::optional<Sam3Prompts>{prompts});
    const auto flattened = track::flatten_sam3_prompt_state(materialized);

    ASSERT_EQ(materialized.shared_prompts.size(), 1u);
    ASSERT_EQ(materialized.objects.size(), 3u);
    EXPECT_EQ(materialized.shared_prompts.front().text(), "fish");
    ASSERT_EQ(flattened.size(), 4u);
    EXPECT_EQ(flattened.front().text(), "fish");
    for(size_t index = 1; index < flattened.size(); ++index) {
        ASSERT_EQ(flattened[index].type(), Sam3PromptType::boxes);
        ASSERT_EQ(flattened[index].boxes().size(), 1u);
    }
}

TEST(Sam3PromptsTest, SnapshotMaterializationCarriesEarlierSeedObjectsWithoutMergingThemBack) {
    const Sam3Prompts prompts{
        {Frame_t{}, Sam3PromptList{Sam3PromptPayload{.value = std::string("fish")}}},
        {3_f, Sam3PromptList{Sam3PromptPayload{
            .value = std::vector<Bounds>{
                Bounds(0.f, 0.f, 10.f, 10.f),
                Bounds(20.f, 20.f, 10.f, 10.f),
            }
        }}}
    };

    const auto materialized = track::materialize_sam3_prompt_snapshot_state(5_f, std::optional<Sam3Prompts>{prompts});
    const auto flattened = track::flatten_sam3_prompt_state(materialized);

    ASSERT_EQ(materialized.shared_prompts.size(), 1u);
    ASSERT_EQ(materialized.objects.size(), 2u);
    EXPECT_EQ(materialized.shared_prompts.front().text(), "fish");
    ASSERT_EQ(flattened.size(), 3u);
    EXPECT_EQ(flattened.front().text(), "fish");
    for(size_t index = 1; index < flattened.size(); ++index) {
        ASSERT_EQ(flattened[index].type(), Sam3PromptType::boxes);
        ASSERT_EQ(flattened[index].boxes().size(), 1u);
    }
}
