#include "gtest/gtest.h"

#include <misc/OptionsList.h>
#include <tracking/OutputLibraryTypes.h>

TEST(OptionsListTest, DistinctModifierSetsCanBeUsedAsMapKeys) {
    Output::Options_t left;
    left.push(Output::Modifiers::SMOOTH);
    left.push(Output::Modifiers::WEIGHTED_CENTROID);

    Output::Options_t right;
    right.push(Output::Modifiers::CENTROID);
    right.push(Output::Modifiers::POSTURE_CENTROID);

    std::map<Output::Options_t, int> values;
    values[left] = 1;
    values[right] = 2;

    ASSERT_EQ(values.size(), 2u);
    EXPECT_EQ(values.at(left), 1);
    EXPECT_EQ(values.at(right), 2);
}
