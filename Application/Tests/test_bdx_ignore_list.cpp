#include <commons.pc.h>
#include <gtest/gtest.h>
#include <core/TrackingSettings.h>

using namespace cmn;
using track::BdxIgnoreList;

namespace track {
std::ostream& operator<<(std::ostream& os, const BdxIgnoreList& dt)
{
    os << dt.toStr();
    return os;
}
}

using IgnoreMap = std::map<Frame_t, std::set<pv::bid>>;

static_assert(std::same_as<track::Settings::track_ignore_bdx_t, BdxIgnoreList>);

TEST(BlobIgnoreMapTest, MergeUnionsFramesAndBlobIdsWithoutChangingSource) {
    IgnoreMap target{
        {10_f, {pv::bid{2u}, pv::bid{5u}}},
        {12_f, {pv::bid{7u}}}
    };
    const IgnoreMap source{
        {10_f, {pv::bid{3u}, pv::bid{5u}}},
        {11_f, {pv::bid{9u}}},
        {12_f, {}}
    };
    const auto original_source = source;
    const IgnoreMap expected{
        {10_f, {pv::bid{2u}, pv::bid{3u}, pv::bid{5u}}},
        {11_f, {pv::bid{9u}}},
        {12_f, {pv::bid{7u}}}
    };

    BdxIgnoreList::Entry::merge_maps(target, source);

    EXPECT_EQ(target, expected);
    EXPECT_EQ(source, original_source);
}

TEST(BlobIgnoreMapTest, MergeHandlesEmptySourceAndDestination) {
    IgnoreMap target;
    const IgnoreMap source{
        {10_f, {pv::bid{2u}}},
        {12_f, {}}
    };

    BdxIgnoreList::Entry::merge_maps(target, IgnoreMap{});
    EXPECT_TRUE(target.empty());

    BdxIgnoreList::Entry::merge_maps(target, source);
    EXPECT_EQ(target, source);

    BdxIgnoreList::Entry::merge_maps(target, IgnoreMap{});
    EXPECT_EQ(target, source);
}

TEST(BlobIgnoreMapTest, RepeatedMergesAreIdempotent) {
    IgnoreMap target{{10_f, {pv::bid{2u}}}};
    const IgnoreMap source{
        {10_f, {pv::bid{3u}}},
        {11_f, {pv::bid{7u}}}
    };
    const IgnoreMap expected{
        {10_f, {pv::bid{2u}, pv::bid{3u}}},
        {11_f, {pv::bid{7u}}}
    };

    BdxIgnoreList::Entry::merge_maps(target, source);
    BdxIgnoreList::Entry::merge_maps(target, source);
    EXPECT_EQ(target, expected);
}

TEST(BlobIgnoreTest, DefaultListIgnoresNothing) {
    const BdxIgnoreList empty;
    EXPECT_EQ(empty.toStr(), "[]");
    EXPECT_FALSE(empty.is_ignored(0_f, pv::bid{0u}));
    EXPECT_FALSE(empty.merged_entries_for(0_f).has_value());
    EXPECT_FALSE(empty.first_differing_frame(empty).valid());
    EXPECT_FALSE(empty.first_differing_frame(BdxIgnoreList::fromStr("[{}]")).valid());
}

TEST(BlobIgnoreTest, MergedEntriesForReturnsUnionAndDistinguishesMissingFrames) {
    const auto list = BdxIgnoreList::fromStr("[{10:[2,5],12:[]},{10:[3,5]}]");
    const auto merged = list.merged_entries_for(10_f);
    ASSERT_TRUE(merged.has_value());
    ASSERT_NE(merged.value(), nullptr);
    EXPECT_EQ(*merged.value(), (std::set<pv::bid>{pv::bid{2u}, pv::bid{3u}, pv::bid{5u}}));

    const auto empty = list.merged_entries_for(12_f);
    ASSERT_TRUE(empty.has_value());
    ASSERT_NE(empty.value(), nullptr);
    EXPECT_TRUE(empty.value()->empty());
    EXPECT_FALSE(list.merged_entries_for(11_f).has_value());
}

TEST(BlobIgnoreTest, FirstDifferingFrameFindsEarliestChangeInEitherDirection) {
    const struct {
        std::string_view before;
        std::string_view after;
        Frame_t frame;
    } cases[]{
        {"[]", "{0:[2]}", 0_f},
        {"{10:[2]}", "{10:[3]}", 10_f},
        {"{10:[2]}", "{10:[2,3]}", 10_f},
        {"{10:[2]}", "{10:[2],20:[3]}", 20_f},
        {"{10:[2],20:[3]}", "{5:[7],10:[2]}", 5_f},
        {"{10:[2],20:[3]}", "{10:[5],15:[7],20:[3]}", 10_f},
        {"{5:[2],10:[3]}", "{10:[3],20:[7]}", 5_f}
    };

    for(const auto& test : cases) {
        SCOPED_TRACE(test.before);
        SCOPED_TRACE(test.after);
        const auto before = BdxIgnoreList::fromStr(test.before);
        const auto after = BdxIgnoreList::fromStr(test.after);
        EXPECT_EQ(after.first_differing_frame(before), test.frame);
        EXPECT_EQ(before.first_differing_frame(after), test.frame);
    }
}

TEST(BlobIgnoreTest, IgnoringTrackletUpdatesMergedCacheAndPreservesGrouping) {
    auto list = BdxIgnoreList::fromStr("[{10:[2,5],12:[7]}]");
    EXPECT_TRUE(list.ignore({{10_f, pv::bid{3u}}, {11_f, pv::bid{9u}}}));

    EXPECT_EQ(list.toStr(), "[{10:[2,5],12:[7]},{10:[3],11:[9]}]");
    EXPECT_TRUE(list.is_ignored(10_f, pv::bid{2u}));
    EXPECT_TRUE(list.is_ignored(10_f, pv::bid{3u}));
    EXPECT_TRUE(list.is_ignored(10_f, pv::bid{5u}));
    EXPECT_TRUE(list.is_ignored(11_f, pv::bid{9u}));
    EXPECT_TRUE(list.is_ignored(12_f, pv::bid{7u}));
}

TEST(BlobIgnoreTest, IgnoringExistingBlobsAndEmptyTrackletsDoesNotChangeEntries) {
    auto list = BdxIgnoreList::fromStr("[{10:[2]},{11:[3]}]");
    const auto original = list;

    EXPECT_FALSE(list.ignore(10_f, pv::bid{2u}));
    EXPECT_FALSE(list.ignore({{10_f, pv::bid{2u}}, {11_f, pv::bid{3u}}}));
    EXPECT_FALSE(list.ignore({}));
    EXPECT_EQ(list, original);
    EXPECT_TRUE(list.is_ignored(10_f, pv::bid{2u}));
    EXPECT_TRUE(list.is_ignored(11_f, pv::bid{3u}));
    EXPECT_FALSE(list.is_ignored(11_f, pv::bid{2u}));
}

TEST(BlobIgnoreTest, UnignoringBlobUpdatesEveryEntryAndMergedCache) {
    auto list = BdxIgnoreList::fromStr("[{10:[2,5],11:[2]},{10:[2,3]}]");
    EXPECT_TRUE(list.unignore(10_f, pv::bid{2u}));

    EXPECT_EQ(list.toStr(), "[{10:[5],11:[2]},{10:[3]}]");
    EXPECT_FALSE(list.is_ignored(10_f, pv::bid{2u}));
    EXPECT_TRUE(list.is_ignored(10_f, pv::bid{3u}));
    EXPECT_TRUE(list.is_ignored(10_f, pv::bid{5u}));
    EXPECT_TRUE(list.is_ignored(11_f, pv::bid{2u}));

    EXPECT_TRUE(list.unignore(10_f, pv::bid{5u}));
    const auto restored = BdxIgnoreList::fromStr(list.toStr());
    EXPECT_FALSE(restored.is_ignored(10_f, pv::bid{2u}));
    EXPECT_FALSE(restored.is_ignored(10_f, pv::bid{5u}));
    EXPECT_TRUE(restored.is_ignored(10_f, pv::bid{3u}));

    EXPECT_FALSE(list.unignore(99_f, pv::bid{2u}));
    EXPECT_FALSE(list.unignore(10_f, pv::bid{99u}));
    EXPECT_EQ(list, restored);
    EXPECT_FALSE(list.is_ignored(10_f, pv::bid{5u}));
    EXPECT_TRUE(list.is_ignored(10_f, pv::bid{3u}));

    EXPECT_TRUE(list.unignore(10_f, pv::bid{3u}));
    EXPECT_FALSE(list.is_ignored(10_f, pv::bid{3u}));
    const auto remaining = list.merged_entries_for(10_f);
    EXPECT_TRUE(not remaining || remaining.value()->empty());
    EXPECT_TRUE(list.is_ignored(11_f, pv::bid{2u}));
}

TEST(BlobIgnoreTest, IgnoringSingleBlobUpdatesCacheAndRoundTrips) {
    BdxIgnoreList list;
    EXPECT_TRUE(list.ignore(10_f, pv::bid{2u}));
    EXPECT_TRUE(list.ignore(11_f, pv::bid{3u}));

    EXPECT_TRUE(list.is_ignored(10_f, pv::bid{2u}));
    EXPECT_TRUE(list.is_ignored(11_f, pv::bid{3u}));
    EXPECT_FALSE(list.is_ignored(11_f, pv::bid{2u}));
    const auto restored = BdxIgnoreList::fromStr(list.toStr());
    EXPECT_EQ(restored, list);
    EXPECT_TRUE(restored.is_ignored(10_f, pv::bid{2u}));
    EXPECT_TRUE(restored.is_ignored(11_f, pv::bid{3u}));
}

TEST(BlobIgnoreTest, UnignoringTrackletPreservesOtherEntries) {
    auto list = BdxIgnoreList::fromStr("[{10:[2],11:[3]},{10:[5],12:[7]}]");
    EXPECT_TRUE(list.unignore_tracklet_of(10_f, pv::bid{2u}));

    EXPECT_FALSE(list.is_ignored(10_f, pv::bid{2u}));
    EXPECT_FALSE(list.is_ignored(11_f, pv::bid{3u}));
    EXPECT_TRUE(list.is_ignored(10_f, pv::bid{5u}));
    EXPECT_TRUE(list.is_ignored(12_f, pv::bid{7u}));
    const auto restored = BdxIgnoreList::fromStr(list.toStr());
    EXPECT_FALSE(restored.is_ignored(10_f, pv::bid{2u}));
    EXPECT_FALSE(restored.is_ignored(11_f, pv::bid{3u}));
    EXPECT_TRUE(restored.is_ignored(10_f, pv::bid{5u}));
    EXPECT_TRUE(restored.is_ignored(12_f, pv::bid{7u}));
    EXPECT_FALSE(list.unignore_tracklet_of(10_f, pv::bid{2u}));
    EXPECT_EQ(list, restored);
}

TEST(BlobIgnoreTest, ParsedEmptyListsIgnoreNothing) {
    for(const std::string_view input : {"[]", "{}", "[{}]", "[{10:[]},{}]"}) {
        SCOPED_TRACE(input);
        const auto list = BdxIgnoreList::fromStr(input);
        EXPECT_FALSE(list.is_ignored(0_f, pv::bid{0u}));
        EXPECT_FALSE(list.is_ignored(10_f, pv::bid{1u}));
    }

    EXPECT_EQ(BdxIgnoreList::fromStr("[]").toStr(), "[]");
    EXPECT_EQ(BdxIgnoreList::fromStr("{}").toStr(), "[]");
}

TEST(BlobIgnoreTest, LegacyMapBecomesOneEntry) {
    const auto list = BdxIgnoreList::fromStr("{12:[7],0:[0,4000000000],10:[5,2,5]}");

    EXPECT_EQ(list.toStr(), "[{0:[0]},{0:[4000000000]},{10:[2]},{10:[5]},{12:[7]}]");
    EXPECT_TRUE(list.is_ignored(0_f, pv::bid{0u}));
    EXPECT_TRUE(list.is_ignored(0_f, pv::bid{4000000000u}));
    EXPECT_TRUE(list.is_ignored(10_f, pv::bid{2u}));
    EXPECT_TRUE(list.is_ignored(10_f, pv::bid{5u}));
    EXPECT_TRUE(list.is_ignored(12_f, pv::bid{7u}));
    EXPECT_FALSE(list.is_ignored(10_f, pv::bid{7u}));
    EXPECT_FALSE(list.is_ignored(11_f, pv::bid{5u}));
}

TEST(BlobIgnoreTest, OverlappingEntriesContributeToMergedLookup) {
    const auto list = BdxIgnoreList::fromStr(
        "[{10:[5,2,5],12:[7]},{10:[3,5],11:[9]},{10:[2,5],12:[7]}]");

    EXPECT_EQ(list.toStr(),
        "[{10:[2,5],12:[7]},{10:[3,5],11:[9]},{10:[2,5],12:[7]}]");
    EXPECT_TRUE(list.is_ignored(10_f, pv::bid{2u}));
    EXPECT_TRUE(list.is_ignored(10_f, pv::bid{3u}));
    EXPECT_TRUE(list.is_ignored(10_f, pv::bid{5u}));
    EXPECT_TRUE(list.is_ignored(11_f, pv::bid{9u}));
    EXPECT_TRUE(list.is_ignored(12_f, pv::bid{7u}));
    EXPECT_FALSE(list.is_ignored(10_f, pv::bid{9u}));
    EXPECT_FALSE(list.is_ignored(11_f, pv::bid{2u}));
    EXPECT_FALSE(list.is_ignored(13_f, pv::bid{7u}));
}

TEST(BlobIgnoreTest, EmptyEntriesDoNotEraseOtherEntries) {
    const auto list = BdxIgnoreList::fromStr("[{},{10:[2]},{10:[]},{}]");

    EXPECT_EQ(list.toStr(), "[{},{10:[2]},{10:[]},{}]");
    EXPECT_TRUE(list.is_ignored(10_f, pv::bid{2u}));
    EXPECT_FALSE(list.is_ignored(10_f, pv::bid{3u}));
}

TEST(BlobIgnoreTest, AcceptsInnerWhitespaceAndQuotedFrameKeys) {
    const auto list = BdxIgnoreList::fromStr(
        R"([ { "10" : [ 5, 2, 5 ], "12" : [ ] }, { "11" : [ 7 ] } ])");

    EXPECT_EQ(list.toStr(), "[{10:[2,5],12:[]},{11:[7]}]");
    EXPECT_TRUE(list.is_ignored(10_f, pv::bid{2u}));
    EXPECT_TRUE(list.is_ignored(10_f, pv::bid{5u}));
    EXPECT_TRUE(list.is_ignored(11_f, pv::bid{7u}));
    EXPECT_FALSE(list.is_ignored(12_f, pv::bid{7u}));
}

TEST(BlobIgnoreTest, MetaRoundTripPreservesEntriesAndRebuildsLookup) {
    const std::string input = "[{10:[2,5]},{10:[3],11:[7]},{}]";
    const auto original = Meta::fromStr<BdxIgnoreList>(input);
    const auto restored = Meta::fromStr<BdxIgnoreList>(Meta::toStr(original));

    EXPECT_EQ(Meta::toStr(restored), input);
    EXPECT_TRUE(restored.is_ignored(10_f, pv::bid{2u}));
    EXPECT_TRUE(restored.is_ignored(10_f, pv::bid{3u}));
    EXPECT_TRUE(restored.is_ignored(10_f, pv::bid{5u}));
    EXPECT_TRUE(restored.is_ignored(11_f, pv::bid{7u}));
    EXPECT_FALSE(restored.is_ignored(11_f, pv::bid{3u}));
    EXPECT_FALSE(restored.is_ignored(12_f, pv::bid{7u}));
}

TEST(BlobIgnoreTest, ParsedValuesOwnTheirInput) {
    std::string input = "prefix[{10:[2,5]},{11:[7]}]suffix";
    const auto list = BdxIgnoreList::fromStr(std::string_view(input).substr(6, input.size() - 12));
    input.assign(input.size(), 'x');

    EXPECT_EQ(list.toStr(), "[{10:[2,5]},{11:[7]}]");
    EXPECT_TRUE(list.is_ignored(10_f, pv::bid{2u}));
    EXPECT_TRUE(list.is_ignored(11_f, pv::bid{7u}));
}

TEST(BlobIgnoreTest, JsonRoundTripPreservesEntriesAndLookup) {
    const auto original = BdxIgnoreList::fromStr("[{10:[2,5]},{10:[3],11:[7]},{}]");
    const auto value = original.to_json();
    ASSERT_TRUE(value.is_array());
    ASSERT_EQ(value.size(), 3u);
    ASSERT_TRUE(value[2].is_object());
    const auto json = glz::write_json(value);
    ASSERT_TRUE(json.has_value());
    const auto restored = BdxIgnoreList::fromStr(json.value());

    EXPECT_EQ(restored, original);
    EXPECT_EQ(restored.toStr(), "[{10:[2,5]},{10:[3],11:[7]},{}]");
    EXPECT_TRUE(restored.is_ignored(10_f, pv::bid{2u}));
    EXPECT_TRUE(restored.is_ignored(10_f, pv::bid{3u}));
    EXPECT_TRUE(restored.is_ignored(11_f, pv::bid{7u}));
    EXPECT_FALSE(restored.is_ignored(11_f, pv::bid{2u}));
}

TEST(BlobIgnoreTest, EqualityPreservesEntryGrouping) {
    const auto legacy = BdxIgnoreList::fromStr("{10:[2,5]}");
    const auto single_entry = BdxIgnoreList::fromStr("[{10:[5,2,2]}]");
    const auto separate_entries = BdxIgnoreList::fromStr("[{10:[2]},{10:[5]}]");

    EXPECT_EQ(legacy, separate_entries);
    EXPECT_NE(single_entry, separate_entries);
    EXPECT_FALSE(legacy.first_differing_frame(single_entry).valid());
    EXPECT_FALSE(single_entry.first_differing_frame(separate_entries).valid());
    EXPECT_FALSE(separate_entries.first_differing_frame(single_entry).valid());
    EXPECT_TRUE(single_entry.is_ignored(10_f, pv::bid{2u}));
    EXPECT_TRUE(single_entry.is_ignored(10_f, pv::bid{5u}));
    EXPECT_TRUE(separate_entries.is_ignored(10_f, pv::bid{2u}));
    EXPECT_TRUE(separate_entries.is_ignored(10_f, pv::bid{5u}));
}

TEST(BlobIgnoreTest, RejectsInvalidOuterSyntax) {
    for(const std::string_view input : {
            "", " ", "null", "42", "{10:[1]", "[{10:[1]}", "{10:[1]}]"
        })
    {
        SCOPED_TRACE(input);
        EXPECT_THROW((void)BdxIgnoreList::fromStr(input), std::exception);
    }
}

TEST(BlobIgnoreTest, RejectsMalformedEntries) {
    for(const std::string_view input : {
            "[1]", "{bad:[1]}", "[{10:[bad]}]", "{10:1}",
            "[{10:[1]},,{11:[2]}]", "[{10:[1,,2]}]"
        })
    {
        SCOPED_TRACE(input);
        EXPECT_THROW((void)BdxIgnoreList::fromStr(input), std::exception);
    }
}
