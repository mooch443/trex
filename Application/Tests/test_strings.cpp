#include "gtest/gtest.h"
#include <commons.pc.h>
#include <misc/parse_parameter_lists.h>
#include <misc/format.h>
#include <misc/Timer.h>


#ifdef NDEBUG
#undef NDEBUG
#endif

#include <misc/checked_casts.h>

using namespace cmn;
using namespace utils;
// Tests for the split function.
TEST(SplitTest, TestBasicSplit) {
    std::string s = "foo,bar,baz";
    std::vector<std::string> expected = {"foo", "bar", "baz"};
    EXPECT_EQ(split(s, ','), expected);

    std::wstring ws = L"hello,world";
    std::vector<std::wstring> expected_ws = {L"hello", L"world"};
    EXPECT_EQ(split(ws, ','), expected_ws);
}

TEST(SplitTest, CombinedTests) {
    // Test data
    std::vector<std::tuple<std::string, char, bool, bool, std::vector<std::string>>> test_data{
        {"a,b,c,d,e", ',', false, false, {"a", "b", "c", "d", "e"}},
        {"a,,b,c,d,,e,", ',', true, false, {"a", "b", "c", "d", "e"}},
        {" a , b , c , d , e ", ',', false, true, {"a", "b", "c", "d", "e"}},
        {" a ,, b , c , d ,, e ,", ',', true, true, {"a", "b", "c", "d", "e"}},
        {"", ',', false, false, {""}},
        {"abcdef", ',', false, false, {"abcdef"}}
    };

    // Test for normal string
    for (const auto& [input, separator, skip_empty, trim, expected] : test_data) {
        EXPECT_EQ(split(input, separator, skip_empty, trim), expected);
    }

    // Test data for wide string
    std::vector<std::tuple<std::wstring, char, bool, bool, std::vector<std::wstring>>> test_data_wide{
        {L"a,b,c,d,e", ',', false, false, {L"a", L"b", L"c", L"d", L"e"}},
        {L"a,,b,c,d,,e,", ',', true, false, {L"a", L"b", L"c", L"d", L"e"}},
        {L" a , b , c , d , e ", ',', false, true, {L"a", L"b", L"c", L"d", L"e"}},
        {L" a ,, b , c , d ,, e ,", ',', true, true, {L"a", L"b", L"c", L"d", L"e"}},
        {L"", ',', false, false, {L""}},
        {L"abcdef", ',', false, false, {L"abcdef"}}
    };

    // Test for wide string
    for (const auto& [input, separator, skip_empty, trim, expected] : test_data_wide) {
        EXPECT_EQ(split(input, separator, skip_empty, trim), expected);
    }
}

TEST(SplitTest, ComplexTests) {
    // Test data
    std::vector<std::tuple<std::string, char, bool, bool, std::vector<std::string>>> test_data{
        // Test case with multiple characters between separators and mixed casing
        {"HeLLo:WorlD:::AnoTher:Word", ':', true, false, {"HeLLo", "WorlD", "AnoTher", "Word"}},

        // Test case with spaces and tabs around separators, and trimming enabled
        {" apple , banana  , cherry, kiwi , peach ", ',', false, true, {"apple", "banana", "cherry", "kiwi", "peach"}},

        // Test case with different consecutive separators, and skip_empty enabled
        {"this||is|||a||test|||", '|', true, false, {"this", "is", "a", "test"}},

        // Test case with varying spaces between words and trimming enabled
        {"first    second   third       fourth", ' ', true, true, {"first", "second", "third", "fourth"}},
    };

    // Test for normal string
    for (const auto& [input, separator, skip_empty, trim, expected] : test_data) {
        EXPECT_EQ(split(input, separator, skip_empty, trim), expected);
    }

    // Test data for wide string
    std::vector<std::tuple<std::wstring, char, bool, bool, std::vector<std::wstring>>> test_data_wide{
        // Test case with multiple characters between separators and mixed casing (wide string)
        {L"HeLLo:WorlD:::AnoTher:Word", ':', true, false, {L"HeLLo", L"WorlD", L"AnoTher", L"Word"}},

        // Test case with spaces and tabs around separators, and trimming enabled (wide string)
        {L" apple , banana     , cherry, kiwi , peach ", ',', false, true, {L"apple", L"banana", L"cherry", L"kiwi", L"peach"}},

        // Test case with different consecutive separators, and skip_empty enabled (wide string)
        {L"this||is|||a||test|||", '|', true, false, {L"this", L"is", L"a", L"test"}},

        // Test case with varying spaces between words and trimming enabled (wide string)
        {L"first   second      third    fourth", ' ', true, true, {L"first", L"second", L"third", L"fourth"}},
        
        // Test case with varying spaces between words and trimming disabled (wide string)
        {L"first   second      third    fourth", ' ', true, true, {L"first", L"second", L"third", L"fourth"}},
    };

    // Test for wide string
    for (const auto& [input, separator, skip_empty, trim, expected] : test_data_wide) {
        EXPECT_EQ(split(input, separator, skip_empty, trim), expected);
    }
}



TEST(SplitTest, TestEmptyString) {
    std::string s = "";
    std::vector<std::string> expected = {""};
    EXPECT_EQ(split(s, ','), expected);

    std::wstring ws = L"";
    std::vector<std::wstring> expected_ws = {L""};
    EXPECT_EQ(split(ws, ','), expected_ws);
}

TEST(SplitTest, TestSingleDelimiter) {
    std::string s = ",";
    std::vector<std::string> expected = {"", ""};
    EXPECT_EQ(split(s, ',', false, false), expected);

    std::wstring ws = L",";
    std::vector<std::wstring> expected_ws = {L"", L""};
    EXPECT_EQ(split(ws, ',', false, false), expected_ws);
}

TEST(SplitTest, TestNoDelimiter) {
    std::string s = "foobar";
    std::vector<std::string> expected = {"foobar"};
    EXPECT_EQ(split(s, ','), expected);

    std::wstring ws = L"hello";
    std::vector<std::wstring> expected_ws = {L"hello"};
    EXPECT_EQ(split(ws, ','), expected_ws);
}

TEST(SplitTest, TestMultipleDelimiters) {
    std::string s = "foo,,bar,,baz";
    std::vector<std::string> expected = {"foo", "", "bar", "", "baz"};
    EXPECT_EQ(split(s, ','), expected);

    std::wstring ws = L"hello, ,world";
    std::vector<std::wstring> expected_ws = {L"hello", L" ", L"world"};
    EXPECT_EQ(split(ws, ','), expected_ws);
}

TEST(SplitTest, TestTrimming) {
    std::string s = "  foo , bar ,  baz  ";
    std::vector<std::string> expected = {"foo", "bar", "baz"};
    EXPECT_EQ(split(s, ',', false, true), expected);

    std::wstring ws = L"  hello  ,  world  ";
    std::vector<std::wstring> expected_ws = {L"hello", L"world"};
    EXPECT_EQ(split(ws, ',', false, true), expected_ws);
}

TEST(SplitTest, TestSkipEmpty) {
    std::string s = "foo,,bar,,baz";
    std::vector<std::string> expected = {"foo", "bar", "baz"};
    EXPECT_EQ(split(s, ',', true, false), expected);

    std::wstring ws = L"hello, ,world";
    std::vector<std::wstring> expected_ws = {L"hello", L" ", L"world"};
    EXPECT_EQ(split(ws, ',', false, false), expected_ws);
}

// repeat

TEST(RepeatTest, TestBasicRepeat) {
  std::string s = "hello";
  std::string expected = "hellohellohellohellohello";
  EXPECT_EQ(repeat(s, 5), expected);
}

TEST(RepeatTest, TestEmptyString) {
  std::string s = "";
  std::string expected = "";
  EXPECT_EQ(repeat(s, 10), expected);
}

TEST(RepeatTest, TestZeroRepetitions) {
  std::string s = "world";
  std::string expected = "";
  EXPECT_EQ(repeat(s, 0), expected);
}

TEST(RepeatTest, TestLargeRepetitions) {
  std::string s = "abc";
  std::string expected = "";
  for (int i = 0; i < 1000000; i++) {
    expected += s;
  }
  EXPECT_EQ(repeat(s, 1000000), expected);
}

// find_replace

TEST(FindReplaceTest, BasicTest) {
    std::string str = "The quick brown fox jumps over the lazy dog.";
    std::vector<std::tuple<std::string, std::string>> search_strings = {
        {"quick", "fast"},
        {"brown", "red"},
        {"jumps", "leaps"},
        {"lazy", "sleepy"}
    };
    std::string expected = "The fast red fox leaps over the sleepy dog.";
    EXPECT_EQ(find_replace(str, search_strings), expected);
}

TEST(FindReplaceTest, EmptyInput) {
    std::string str = "";
    std::vector<std::tuple<std::string, std::string>> search_strings = {
        {"quick", "fast"},
        {"brown", "red"},
        {"jumps", "leaps"},
        {"lazy", "sleepy"}
    };
    std::string expected = "";
    EXPECT_EQ(find_replace(str, search_strings), expected);
}

TEST(FindReplaceTest, NoMatches) {
    std::string str = "The quick brown fox jumps over the lazy dog.";
    std::vector<std::tuple<std::string, std::string>> search_strings = {
        {"foo", "bar"},
        {"baz", "qux"}
    };
    std::string expected = "The quick brown fox jumps over the lazy dog.";
    EXPECT_EQ(find_replace(str, search_strings), expected);
}

TEST(FindReplaceTest, OverlappingMatches) {
    std::string str = "The quick brown fox jumps over the lazy dog.";
    std::vector<std::tuple<std::string, std::string>> search_strings = {
        {"the", "THE"},
        {"THE", "the"},
    };
    std::string expected = "The quick brown fox jumps over THE lazy dog.";
    EXPECT_EQ(find_replace(str, search_strings), expected);
}



TEST(FindReplaceTest, EmptyInputString) {
    std::string input = "";
    std::vector<std::tuple<std::string, std::string>> search_strings = {{"abc", "xyz"}, {"def", "uvw"}};
    std::string result = find_replace(input, search_strings);
    ASSERT_EQ(result, "");
}


TEST(FindReplaceTest, EmptySearchStrings) {
    std::string str = "The quick brown fox jumps over the lazy dog.";
    std::vector<std::tuple<std::string, std::string>> search_strings = {};
    std::string expected = "The quick brown fox jumps over the lazy dog.";
    EXPECT_EQ(find_replace(str, search_strings), expected);
    
    {
        std::string input = "abcdefgh";
        std::vector<std::tuple<std::string, std::string>> search_strings;
        std::string result = find_replace(input, search_strings);
        ASSERT_EQ(result, "abcdefgh");
    }
}

TEST(FindReplaceTest, EmptyInputAndSearchStrings) {
    std::string input = "";
    std::vector<std::tuple<std::string, std::string>> search_strings;
    std::string result = find_replace(input, search_strings);
    ASSERT_EQ(result, "");
}

TEST(FindReplaceTest, NoMatchingSearchStrings) {
    std::string input = "abcdefgh";
    std::vector<std::tuple<std::string, std::string>> search_strings = {{"ijk", "xyz"}, {"lmn", "uvw"}};
    std::string result = find_replace(input, search_strings);
    ASSERT_EQ(result, "abcdefgh");
}

TEST(FindReplaceTest, SomeMatchingSearchStrings) {
    std::string input = "abcdefgh";
    std::vector<std::tuple<std::string, std::string>> search_strings = {{"abc", "xyz"}, {"lmn", "uvw"}};
    std::string result = find_replace(input, search_strings);
    ASSERT_EQ(result, "xyzdefgh");
}

TEST(FindReplaceTest, AllMatchingSearchStrings) {
    std::string input = "abcdefgh";
    std::vector<std::tuple<std::string, std::string>> search_strings = {{"abc", "xyz"}, {"def", "uvw"}};
    std::string result = find_replace(input, search_strings);
    ASSERT_EQ(result, "xyzuvwgh");
}

TEST(FindReplaceTest, MultipleInstancesOfSearchStrings) {
    std::string input = "abcdeabc";
    std::vector<std::tuple<std::string, std::string>> search_strings = {{"abc", "xyz"}};
    std::string result = find_replace(input, search_strings);
    ASSERT_EQ(result, "xyzdexyz");
}

TEST(FindReplaceTest, SpecialCharactersAndDigits) {
    std::string input = "a$b%c123";
    std::vector<std::tuple<std::string, std::string>> search_strings = {{"$", "X"}, {"%", "Y"}, {"1", "2"}};
    std::string result = find_replace(input, search_strings);
    ASSERT_EQ(result, "aXbYc223");
}

TEST(FindReplaceTest, IdenticalReplacements) {
    std::string input = "abcabc";
    std::vector<std::tuple<std::string, std::string>> search_strings = {{"abc", "abc"}};
    std::string result = find_replace(input, search_strings);
    ASSERT_EQ(result, "abcabc");
}

TEST(FindReplaceTest, ComplexSearchStrings) {
    std::string input = "A quick brown fox jumps over the lazy dog, and the dog returns the favor.";
    std::vector<std::tuple<std::string, std::string>> search_strings = {
        {"quick", "swift"},
        {"the lazy dog", "a sleeping canine"},
        {"the favor", "a compliment"}
    };
    std::string expected = "A swift brown fox jumps over a sleeping canine, and the dog returns a compliment.";
    EXPECT_EQ(find_replace(input, search_strings), expected);
}

TEST(FindReplaceTest, UnicodeCharacters) {
    std::string input = "こんにちは世界";
    std::vector<std::tuple<std::string, std::string>> search_strings = {{"こんにちは", "さようなら"}, {"世界", "宇宙"}};
    std::string expected = "さようなら宇宙";
    EXPECT_EQ(find_replace(input, search_strings), expected);
}

TEST(FindReplaceTest, CaseSensitivity) {
    std::string input = "The Quick Brown Fox Jumps Over The Lazy Dog.";
    std::vector<std::tuple<std::string, std::string>> search_strings = {{"The", "A"}, {"Quick", "Swift"}, {"Brown", "Red"}};
    std::string expected = "A Swift Red Fox Jumps Over A Lazy Dog.";
    EXPECT_EQ(find_replace(input, search_strings), expected);
}

TEST(FindReplaceTest, ComplexOverlappingSearchStrings) {
    std::string input = "appleappletreeapple";
    std::vector<std::tuple<std::string, std::string>> search_strings = {{"apple", "orange"}, {"appletree", "peachtree"}};
    std::string expected = "orangeorangetreeorange";
    EXPECT_EQ(find_replace(input, search_strings), expected);
}

TEST(FindReplaceTest, RespectOrderOfSearchStrings) {
    std::string input = "this is an orangetree";
    std::vector<std::tuple<std::string, std::string>> search_strings = {
        {"orange", "apple"},
        {"orangetree", "appletree"},
    };
    std::string expected = "this is an appletree";
    EXPECT_EQ(find_replace(input, search_strings), expected);
}

TEST(FindReplaceTest, MultipleReplacementsInARow) {
    std::string input = "helloorangeapplegrape";
    std::vector<std::tuple<std::string, std::string>> search_strings = {
        {"orange", "banana"},
        {"apple", "cherry"},
        {"grape", "kiwi"},
    };
    std::string expected = "hellobananacherrykiwi";
    EXPECT_EQ(find_replace(input, search_strings), expected);
}

TEST(FindReplaceTest, OverlappingSearchStrings) {
    std::string input = "abcde";
    std::vector<std::tuple<std::string, std::string>> search_strings = {
        {"abc", "xyz"},
        {"bcd", "uv"},
    };
    std::string expected = "xyzde";
    EXPECT_EQ(find_replace(input, search_strings), expected);
}

TEST(FindReplaceTest, ReplaceSubstringsWithDifferentLengths) {
    std::string input = "this is an orangetree";
    std::vector<std::tuple<std::string, std::string>> search_strings = {
        {"orange", "app"},
        {"orangetree", "appletree"},
    };
    std::string expected = "this is an apptree";
    EXPECT_EQ(find_replace(input, search_strings), expected);
}


// more complex parsing

TEST(ParseSetMetaTest, HandlesSimpleKeyValuePairs) {
    std::string input = "option1=value1;option2=value2";
    auto result = parse_set_meta(input);

    ASSERT_EQ(2, result.size());
    ASSERT_EQ("value1", result["option1"]);
    ASSERT_EQ("value2", result["option2"]);
}

TEST(ParseSetMetaTest, HandlesQuotedValues) {
    std::string input = R"(option1="value1 with spaces";option2='value2 with spaces')";
    auto result = parse_set_meta(input);

    ASSERT_EQ(2, result.size());
    ASSERT_EQ("value1 with spaces", result["option1"]);
    ASSERT_EQ("value2 with spaces", result["option2"]);
}

TEST(ParseSetMetaTest, HandlesValuesWithEqualSigns) {
    std::string input = R"(option1="value=1";option2='value=2 with spaces';option3=value3=extra)";
    auto result = parse_set_meta(input);

    ASSERT_EQ(3, result.size());
    ASSERT_EQ("value=1", result["option1"]);
    ASSERT_EQ("value=2 with spaces", result["option2"]);
    ASSERT_EQ("value3=extra", result["option3"]);
}

TEST(ParseSetMetaTest, HandlesEscapedQuotes) {
    std::string input = R"(option1="value1 \"with escaped\" quotes";option2='value2 \'with escaped\' quotes')";
    auto result = parse_set_meta(input);

    ASSERT_EQ(2, result.size());
    ASSERT_EQ(R"(value1 "with escaped" quotes)", result["option1"]);
    ASSERT_EQ(R"(value2 'with escaped' quotes)", result["option2"]);
}

TEST(ParseSetMetaTest, HandlesEmptyInput) {
    std::string input = "";
    auto result = parse_set_meta(input);

    ASSERT_EQ(0, result.size());
}

std::string which_from(const auto& indexes, const auto& corpus) {
    std::string r = "[";
    bool first = true;
    for(auto index : indexes) {
        if(not first) {
            r += ',';
        }
        r += size_t(index) < corpus.size() ? "'"+corpus.at(index)+"'" : "<invalid:"+std::to_string(index)+">";
        first = false;
    }
    
    return r+"] from the corpus";
}

bool compare_to_first_n_elements(const std::vector<int>& vec1, const std::vector<int>& vec2) {
    if (vec1.size() > vec2.size()) {
        return false;
    }

    return std::equal(vec1.begin(), vec1.end(), vec2.begin());
}

TEST(TextSearchTest, HandlesMultipleWords) {
    std::vector<std::string> corpus = {"apples", "oranges", "apples_oranges"};

    // Test case: search for "apples oranges", should return only "apples_oranges" (index 2)
    auto result = text_search("apples oranges", corpus);
    decltype(result) expected{2, 1, 0};
    ASSERT_TRUE(compare_to_first_n_elements(expected, result)) << "selecting " << which_from(result, corpus) << " from " << Meta::toStr(corpus) << " instead of " << which_from(expected, corpus);
}

TEST(TextSearchTest, HandlesSingleWord) {
    std::vector<std::string> corpus = {"apples", "oranges", "apples_oranges"};

    // Test case: search for "apples", should return "apples" (index 0) and "apples_oranges" (index 2)
    auto result = text_search("apples", corpus);
    decltype(result) expected{0, 2};
    ASSERT_TRUE(compare_to_first_n_elements(expected, result)) << "selecting " << which_from(result, corpus) << " from " << Meta::toStr(corpus) << " instead of " << which_from(expected, corpus);
}

TEST(TextSearchTest, HandlesMisspelledWord) {
    std::vector<std::string> corpus = {"apple", "oranges", "apples_oranges"};

    // Test case: search for "organes" (misspelled), should return "oranges" (index 1) and "apples_oranges" (index 2)
    auto result = text_search("organes", corpus);
    decltype(result) expected{1,2};
    ASSERT_TRUE(compare_to_first_n_elements(expected, result)) << "selecting " << which_from(result, corpus) << " from " << Meta::toStr(corpus) << " instead of " << which_from(expected, corpus) << " for search term 'organes'";
}

TEST(SearchText, HandlesPhrasesWithCommonWords) {
    std::vector<std::string> corpus = {
        "apple pie recipe",
        "banana bread recipe",
        "apple and banana smoothie",
        "banana split",
        "apple and orange juice",
        "orange chicken recipe"
    };

    auto result = text_search("apple banana", corpus);
    decltype(result) expected = {2};
    ASSERT_TRUE(compare_to_first_n_elements(expected, result)) << "selecting " << which_from(result, corpus) << " from " << Meta::toStr(corpus) << " instead of " << which_from(expected, corpus) << " for search term 'apple banana'";
}

TEST(SearchText, HandlesMultipleMisspellings) {
    std::vector<std::string> corpus = {
        "apple pie recipe",
        "banana bread recipe",
        "apple and banana smoothie",
        "banana split",
        "apple and orange juice",
        "orange chicken recipe"
    };

    auto result = text_search("aplpe banan", corpus);
    decltype(result) expected = {3,2,1};
    ASSERT_TRUE(compare_to_first_n_elements(expected, result)) << "selecting " << which_from(result, corpus) << " from " << Meta::toStr(corpus) << " instead of " << which_from(expected, corpus) << " for search term 'aplpe banan'";
}

TEST(SearchText, HandlesPunctuationAndSpaces) {
    std::vector<std::string> corpus = {
        "apple-pie recipe",
        "banana_bread recipe",
        "apple and banana smoothie",
        "banana split",
        "apple, and orange juice",
        "orange chicken recipe"
    };

    auto result = text_search("apple, banana", corpus);
    decltype(result) expected = {2};
    ASSERT_TRUE(compare_to_first_n_elements(expected, result)) << "selecting " << which_from(result, corpus) << " from " << Meta::toStr(corpus) << " instead of " << which_from(expected, corpus) << " for search term 'apple, banana'";
}

TEST(SearchText, HandlesPartialParameterNames) {
    std::vector<std::string> corpus = {
        "track_threshold",
        "track_posture_threshold",
        "track_threshold2",
        "app_version"
    };
    
    {
        auto result = text_search("track_th", corpus);
        decltype(result) expected = {0,2,1};
        ASSERT_TRUE(compare_to_first_n_elements(expected, result)) << "selecting " << which_from(result, corpus) << " from " << Meta::toStr(corpus) << " instead of " << which_from(expected, corpus) << " for search term 'track_th'";
    }
    
    {
        auto result = text_search("track_threshold", corpus);
        decltype(result) expected = {0,2,1};
        ASSERT_TRUE(compare_to_first_n_elements(expected, result)) << "selecting " << which_from(result, corpus) << " from " << Meta::toStr(corpus) << " instead of " << which_from(expected, corpus) << " for search term 'track_threshold'";
    }
}

TEST(SearchText, HandlesPartialWordMatches) {
    std::vector<std::string> corpus = {
        "apple pie recipe",
        "banana bread recipe",
        "apple and banana smoothie",
        "banana split",
        "apple and orange juice",
        "orange chicken recipe",
        "track_threshold",
        "track_posture_threshold",
        "track_threshold2"
    };
    
    {
        auto result = text_search("track_th", corpus);
        decltype(result) expected = {6, 8, 7};
        ASSERT_TRUE(compare_to_first_n_elements(expected, result)) << "selecting " << which_from(result, corpus) << " from " << Meta::toStr(corpus) << " instead of " << which_from(expected, corpus) << " for search term 'track_th'";
    }
    {
        auto result = text_search("rec", corpus);
        decltype(result) expected = {0, 1, 5};
        ASSERT_TRUE(compare_to_first_n_elements(expected, result)) << "selecting " << which_from(result, corpus) << " from " << Meta::toStr(corpus) << " instead of " << which_from(expected, corpus) << " for search term 'rec'";
    }
    
    
    
}

// Test narrow_cast with warn_on_error
TEST(NarrowCastTest, WarnOnError) {
    EXPECT_NO_THROW({
        int negative_value = -5;
        unsigned int result = narrow_cast<unsigned int>(negative_value, tag::warn_on_error{});
        EXPECT_EQ(result, static_cast<unsigned int>(negative_value));
    });

    EXPECT_NO_THROW({
        int large_value = 100000;
        short result = narrow_cast<short>(large_value, tag::warn_on_error{});
        EXPECT_EQ(result, static_cast<short>(large_value));
    });
}

// Test narrow_cast with fail_on_error
TEST(NarrowCastTest, FailOnError) {
    int negative_value = -5;
    ASSERT_ANY_THROW(narrow_cast<unsigned int>(negative_value, tag::fail_on_error{}));

    int large_value = 100000;
    ASSERT_ANY_THROW(narrow_cast<short>(large_value, tag::fail_on_error{}));
}

// Test narrow_cast without tags
TEST(NarrowCastTest, NoTags) {
    int value = 42;
    short result = narrow_cast<short>(value);
    EXPECT_EQ(result, static_cast<short>(value));

    value = -42;
    result = narrow_cast<short>(value);
    EXPECT_EQ(result, static_cast<short>(value));
}
