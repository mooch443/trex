#include <commons.pc.h>

#include "gtest/gtest.h"

#include <core/GPURecognitionTypes.h>
#include <misc/zipper.h>
#include <python/SAM3.h>

using namespace cmn;
using namespace track::detect;

namespace {

std::string json_string(const glz::json_t& json) {
    return glz::write_json(json).value();
}

}

TEST(Sam3PromptPayloadTest, FromStrParsesPlainTextPrompt) {
    const auto prompt = Sam3PromptPayload::fromStr("fish");

    ASSERT_EQ(prompt.type(), Sam3PromptType::text);
    ASSERT_TRUE(prompt.has_value());
    EXPECT_EQ(prompt.text(), "fish");
    EXPECT_EQ(prompt.toStr(), "fish");
    EXPECT_EQ(json_string(prompt.to_json()), "\"fish\"");
}

TEST(Sam3PromptPayloadTest, FromStrParsesQuotedTextPrompt) {
    const auto prompt = Sam3PromptPayload::fromStr("\"fish\"");

    ASSERT_EQ(prompt.type(), Sam3PromptType::text);
    ASSERT_TRUE(prompt.has_value());
    EXPECT_EQ(prompt.text(), "fish");
    EXPECT_EQ(prompt.toStr(), "fish");
}

TEST(Sam3PromptPayloadTest, FromStrTrimsAndParsesSingleQuotedTextPrompt) {
    const auto prompt = Sam3PromptPayload::fromStr("  'fish'  ");

    ASSERT_EQ(prompt.type(), Sam3PromptType::text);
    ASSERT_TRUE(prompt.has_value());
    EXPECT_EQ(prompt.text(), "fish");
    EXPECT_EQ(prompt.toStr(), "fish");
    EXPECT_EQ(json_string(prompt.to_json()), "\"fish\"");
}

TEST(Sam3PromptPayloadTest, FromStrParsesPointPromptArray) {
    const auto prompt = Sam3PromptPayload::fromStr("[[1,2],[3,4]]");

    ASSERT_EQ(prompt.type(), Sam3PromptType::points);
    ASSERT_EQ(prompt.points().size(), 2u);
    EXPECT_EQ(prompt.points()[0], Vec2(1, 2));
    EXPECT_EQ(prompt.points()[1], Vec2(3, 4));
    EXPECT_EQ(prompt.toStr(), "[[1,2],[3,4]]");
    EXPECT_EQ(json_string(prompt.to_json()), "[[1,2],[3,4]]");
}

TEST(Sam3PromptPayloadTest, FromStrParsesBoxPromptArray) {
    const auto prompt = Sam3PromptPayload::fromStr("[[10,20,30,40],[50,60,70,80]]");

    ASSERT_EQ(prompt.type(), Sam3PromptType::boxes);
    ASSERT_EQ(prompt.boxes().size(), 2u);
    EXPECT_EQ(prompt.boxes()[0], (Bounds(10, 20, 30, 40)));
    EXPECT_EQ(prompt.boxes()[1], (Bounds(50, 60, 70, 80)));
    EXPECT_EQ(prompt.toStr(), "[[10,20,30,40],[50,60,70,80]]");
    EXPECT_EQ(json_string(prompt.to_json()), "[[10,20,30,40],[50,60,70,80]]");
}

TEST(Sam3PromptPayloadTest, FromStrTrimsArrayPromptWhitespace) {
    const auto prompt = Sam3PromptPayload::fromStr("  [[1,2],[3,4]]  ");

    ASSERT_EQ(prompt.type(), Sam3PromptType::points);
    ASSERT_EQ(prompt.points().size(), 2u);
    EXPECT_EQ(prompt.points()[0], Vec2(1, 2));
    EXPECT_EQ(prompt.points()[1], Vec2(3, 4));
    EXPECT_EQ(prompt.toStr(), "[[1,2],[3,4]]");
}

TEST(Sam3PromptPayloadTest, FromStrTreatsEmptyTextAsNoPayload) {
    const auto prompt = Sam3PromptPayload::fromStr("  ");

    EXPECT_FALSE(prompt.has_value());
    EXPECT_EQ(prompt.type(), Sam3PromptType::none);
    EXPECT_THROW((void)prompt.toStr(), std::exception);
    EXPECT_THROW((void)prompt.to_json(), std::exception);
}

TEST(Sam3PromptPayloadTest, FromStrTreatsEmptyArrayAsNoPayload) {
    const auto prompt = Sam3PromptPayload::fromStr("[]");

    EXPECT_FALSE(prompt.has_value());
    EXPECT_EQ(prompt.type(), Sam3PromptType::none);
    EXPECT_THROW((void)prompt.toStr(), std::exception);
    EXPECT_THROW((void)prompt.to_json(), std::exception);
}

TEST(Sam3PromptPayloadTest, FromStrRejectsMixedArrayShapes) {
    EXPECT_THROW(
        (void)Sam3PromptPayload::fromStr("[[1,2],[3,4,5,6]]"),
        std::exception);
}

TEST(Sam3PromptListTest, FromStrParsesBoxPromptArray) {
    auto list = Meta::fromStr<Sam3PromptList>("['hi i bims',[[25,666],[1234,4567]],[[0,0,200,200],[200,200,210,230]]]");

    Sam3PromptList expected{
        { .value = "hi i bims" },
        { .value = std::vector<Vec2>{Vec2(25,666), Vec2(1234,4567)} },
        { .value = std::vector<Bounds>{{0.f,0.f,200.f,200.f}, {200.f,200.f,210.f,230.f}} }
    };
    
    ASSERT_EQ(expected.size(), list.size());
    for(auto&& [E, A] : Zip::Zip(expected, list)) {
        ASSERT_EQ(E, A) << E.toStr() << " != " << A.toStr();
        EXPECT_EQ(E.toStr(), A.toStr());
    }
    
}

TEST(Sam3PromptListTest, ToStrCollapsesSinglePayloadButJsonKeepsArray) {
    const Sam3PromptList list{
        Sam3PromptPayload{ .value = std::string("fish") }
    };

    EXPECT_EQ(list.toStr(), "fish");
    EXPECT_EQ(json_string(list.to_json()), "[\"fish\"]");
}

TEST(Sam3PromptListTest, ToStrKeepsArrayForMultiplePayloads) {
    const Sam3PromptList list{
        Sam3PromptPayload{ .value = std::string("fish") },
        Sam3PromptPayload{ .value = std::vector<Vec2>{Vec2(1, 2)} }
    };

    EXPECT_EQ(list.toStr(), "[fish,[[1,2]]]");
    EXPECT_EQ(json_string(list.to_json()), "[\"fish\",[[1,2]]]");
}

TEST(Sam3PromptsTest, FromStrParseSingle) {
    Sam3Prompts prompts = Meta::fromStr<Sam3Prompts>("['hi i bims',[[25,666],[1234,4567]],[[0,0,200,200],[200,200,210,230]]]");

    Sam3PromptList expected{
        { .value = "hi i bims" },
        { .value = std::vector<Vec2>{Vec2(25,666), Vec2(1234,4567)} },
        { .value = std::vector<Bounds>{{0.f,0.f,200.f,200.f}, {200.f,200.f,210.f,230.f}} }
    };
    
    Sam3Prompts expected_prompts {
        { Frame_t{}, expected }
    };
    
    /// check that move is there
    Sam3Prompts other = std::move(prompts);
    prompts = std::move(other);
    
    ASSERT_EQ(expected_prompts.size(), prompts.size());
    
    for(auto &&[expected_key, real_key] : Zip::Zip(extract_keys(expected_prompts), extract_keys(prompts))) {
        ASSERT_EQ(expected_key, real_key);
        ASSERT_EQ(expected_prompts.at(expected_key), prompts.at(real_key));
    }
}

TEST(Sam3PromptsTest, FromStrPlainTextPromptString) {
    const Sam3Prompts prompts = Sam3Prompts::fromStr("fish");

    EXPECT_TRUE(prompts.size() == 1u);
    EXPECT_EQ(prompts.toStr(), "fish");
    EXPECT_EQ(json_string(prompts.to_json()), "{\"null\":[\"fish\"]}");
}

TEST(Sam3PromptsTest, ToStrCollapsesSingleGlobalPromptListButJsonKeepsMapShape) {
    const Sam3Prompts prompts{
        { Frame_t{}, Sam3PromptList{ Sam3PromptPayload{ .value = std::string("fish") } } }
    };

    EXPECT_EQ(prompts.toStr(), "fish");
    EXPECT_EQ(json_string(prompts.to_json()), "{\"null\":[\"fish\"]}");
}

TEST(Sam3PromptsTest, ToStr) {
    const Sam3Prompts prompts{
        { 0_f, Sam3PromptList{ Sam3PromptPayload{ .value = std::string("fish") } } }
    };

    EXPECT_EQ(prompts.toStr(), "{0:fish}");
    EXPECT_EQ(json_string(prompts.to_json()), "{\"0\":[\"fish\"]}");
}

TEST(Sam3PromptsTest, FromStr) {
    const Sam3Prompts expected{
        { 0_f, Sam3PromptList{ Sam3PromptPayload{ .value = std::string("fish") } } }
    };

    EXPECT_EQ(Sam3Prompts::fromStr("{0:fish}"), expected);
    EXPECT_EQ(Sam3Prompts::fromStr("{0:'fish'}"), expected);
    EXPECT_EQ(Sam3Prompts::fromStr("{0:[fish]}"), expected);
    EXPECT_EQ(Sam3Prompts::fromStr("{0:['fish']}"), expected);
    
    const Sam3Prompts multi_expected{
        { 0_f, Sam3PromptList{
            Sam3PromptPayload{ .value = std::string("fish") },
            Sam3PromptPayload{ .value = std::string("human") }
        } }
    };
    
    EXPECT_EQ(Sam3Prompts::fromStr("{0:[fish,human]}"), multi_expected);
    
    const Sam3Prompts expected2{
        { Frame_t{}, Sam3PromptList{ Sam3PromptPayload{ .value = std::string("human") } } },
        { 0_f, Sam3PromptList{ Sam3PromptPayload{ .value = std::string("fish") } } },
        { 1_f, Sam3PromptList{ Sam3PromptPayload{ .value = std::vector<Vec2>{{10_F,12_F}} } } }
    };
    
    EXPECT_EQ(Sam3Prompts::fromStr("{null:human,0:fish,1:[[10,12]]}"), expected2) << "expected " << expected2.toStr() << " got " << Sam3Prompts::fromStr("{null:human,0:fish,1:[[10,12]]}").toStr();
}

TEST(Sam3PromptsTest, EmptyPromptRepositorySerializesAsEmptyObject) {
    const Sam3Prompts prompts;

    EXPECT_TRUE(prompts.empty());
    EXPECT_EQ(prompts.toStr(), "{}");
    EXPECT_EQ(json_string(prompts.to_json()), "{}");
}

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
