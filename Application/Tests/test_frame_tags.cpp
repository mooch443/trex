#include "gtest/gtest.h"
#include <commons.pc.h>
#include <core/FrameTags.h>

using namespace cmn;
using namespace track;

TEST(FrameTagTest, AcceptsOrdinaryTagCharacters) {
    for(const std::string_view value : {
            "alpha",
            "Alpha 2",
            "review-later",
            "group_one",
            "group one-2"
        })
    {
        const auto tag = FrameTag::fromStr(value);
        EXPECT_FALSE(tag.has_location());
        EXPECT_EQ(tag.get_name(), value);
    }
}

TEST(FrameTagTest, RejectsSymbolsAndNamesWithoutLettersOrNumbers) {
    for(const std::string_view value : {
            "",
            "   ",
            "---___",
            "[review]",
            "review!",
            "review/later",
            "review:later",
            "review\tlater"
        })
    {
        EXPECT_THROW((void)FrameTag::fromStr(value), std::exception) << value;
    }
}

TEST(FrameTagTest, ValidatesAndRoundTripsLocalizedTags) {
    const Bounds bounds{10, 20, 30, 40};
    const FrameTag original{
        .name = SpatialTag{bounds, std::string("review-later_2")}
    };

    const auto parsed = FrameTag::fromStr(original.toStr());
    EXPECT_TRUE(parsed.has_location());
    EXPECT_EQ(parsed.get_location(), bounds);
    EXPECT_EQ(parsed.get_name(), "review-later_2");

    const FrameTag invalid{
        .name = SpatialTag{bounds, std::string("[review]")}
    };
    EXPECT_THROW((void)FrameTag::fromStr(invalid.toStr()), std::exception);
}
