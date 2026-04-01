#include <commons.pc.h>

#include "gtest/gtest.h"

#include <core/GPURecognitionTypes.h>
#include <misc/zipper.h>

using namespace cmn;
using namespace track::detect;

TEST(Sam3PromptPayloadTest, FromStrParsesPlainTextPrompt) {
    const auto prompt = Sam3PromptPayload::fromStr("fish");

    ASSERT_EQ(prompt.type(), Sam3PromptType::text);
    ASSERT_TRUE(prompt.has_value());
    EXPECT_EQ(prompt.text(), "fish");
    EXPECT_EQ(prompt.toStr(), "fish");
}

TEST(Sam3PromptPayloadTest, FromStrParsesQuotedTextPrompt) {
    const auto prompt = Sam3PromptPayload::fromStr("\"fish\"");

    ASSERT_EQ(prompt.type(), Sam3PromptType::text);
    ASSERT_TRUE(prompt.has_value());
    EXPECT_EQ(prompt.text(), "fish");
    EXPECT_EQ(prompt.toStr(), "fish");
}

TEST(Sam3PromptPayloadTest, FromStrParsesPointPromptArray) {
    const auto prompt = Sam3PromptPayload::fromStr("[[1,2],[3,4]]");

    ASSERT_EQ(prompt.type(), Sam3PromptType::points);
    ASSERT_EQ(prompt.points().size(), 2u);
    EXPECT_EQ(prompt.points()[0], Vec2(1, 2));
    EXPECT_EQ(prompt.points()[1], Vec2(3, 4));
    EXPECT_EQ(prompt.toStr(), "[[1,2],[3,4]]");
}

TEST(Sam3PromptPayloadTest, FromStrParsesBoxPromptArray) {
    const auto prompt = Sam3PromptPayload::fromStr("[[10,20,30,40],[50,60,70,80]]");

    ASSERT_EQ(prompt.type(), Sam3PromptType::boxes);
    ASSERT_EQ(prompt.boxes().size(), 2u);
    EXPECT_EQ(prompt.boxes()[0], (Bounds(10, 20, 30, 40)));
    EXPECT_EQ(prompt.boxes()[1], (Bounds(50, 60, 70, 80)));
    EXPECT_EQ(prompt.toStr(), "[[10,20,30,40],[50,60,70,80]]");
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

TEST(Sam3PromptsTest, FromStrParseSingle) {
    auto list = Meta::fromStr<Sam3PromptList>("['hi i bims',[[25,666],[1234,4567]],[[0,0,200,200],[200,200,210,230]]]");

    Sam3PromptList expected{
        { .value = "hi i bims" },
        { .value = std::vector<Vec2>{Vec2(25,666), Vec2(1234,4567)} },
        { .value = std::vector<Bounds>{{0.f,0.f,200.f,200.f}, {200.f,200.f,210.f,230.f}} }
    };
    
    Sam3Prompts prompts {
        { Frame_t{}, expected }
    };
    
    Sam3Prompts other = std::move(prompts);
    prompts = std::move(other);
    
    ASSERT_EQ(expected.size(), list.size());
    for(auto&& [E, A] : Zip::Zip(expected, list)) {
        ASSERT_EQ(E, A) << E.toStr() << " != " << A.toStr();
        EXPECT_EQ(E.toStr(), A.toStr());
    }
    
}
