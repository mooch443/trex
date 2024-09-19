#include "gtest/gtest.h"
#include <commons.pc.h>
#include <misc/parse_parameter_lists.h>
#include <misc/Timer.h>
#include <file/Path.h>
#include <misc/default_settings.h>
#include <gui/types/StaticText.h>
#include <tracking/IndividualCache.h>
#include <misc/detail.h>
#include <misc/RecentItems.h>

static constexpr auto lower = cmn::utils::lowercase("hiIbImS");

using namespace cmn;
using namespace utils;

/**
 
 { "type":"vlayout", "pos":[20,10], "pad":[5,5,5,5],
   "children": [
   { "type": "hlayout", "pad":[0,0,10,5], "children": [
       { "type": "button", "text": "Quit", "action": "QUIT"},
       { "type": "collection", "children": [
         {"type": "text", "text": "Hello World!"}
       ]}
   ]},
   { "type": "textfield",
     "size":[500,40],
     "color":[255,0,255,255],
     "action":"QUIT"
   },
   {"type":"stext",
   "text":"{mouse.x} {window_size.w} {video_length} -- {*:{video_length}:{/:{mouse.x}:{window_size.w}}}"
 }
 
 
 */


namespace dyn {
enum class Align {
    left, center, right, verticalcenter
};

struct Font {
    std::optional<Float2_t> size;
    std::optional<dyn::Align> align;
};

using Color = std::vector<uint8_t>;
using Loc = std::array<Float2_t, 2>;

struct Template {
    std::string text;
    std::optional<std::string> detail;
    std::optional<std::string> action;
};

struct Object {
    std::string type;
    std::optional<dyn::Loc> pos, max_size, size;
    std::optional<std::array<Float2_t, 4>> pad;
    std::optional<dyn::Color> color, line, fill, horizontal_clr, vertical_clr;
    std::optional<std::vector<Object>> children;
    std::optional<std::string> text, action, var;
    std::optional<bool> clickable, foldable, folded;
    std::optional<Template> schablone;
    std::optional<dyn::Font> font;
    std::optional<std::vector<std::string>> modules;
    std::shared_ptr<Object> then;
    std::shared_ptr<Object> _else;
    std::shared_ptr<Object> preview;
};

struct MainFile {
    std::vector<Object> objects;
    std::optional<glz::json_t> defaults;
};


}

template <>
struct glz::meta<dyn::Object> {
   using T = dyn::Object;
   static constexpr auto value = object(&T::type,  //
                                        &T::pos, &T::max_size, &T::size,
                                        &T::pad,
                                        &T::color, &T::line, &T::fill, &T::horizontal_clr, &T::vertical_clr,
                                        &T::children,
                                        &T::text, &T::action, &T::var,
                                        &T::clickable, &T::foldable, &T::folded,
                                        &T::schablone,
                                        &T::font,
                                        &T::modules,
                                        &T::then,
                                        "else", &T::_else,
                                        &T::preview);
};

template <>
struct glz::meta<dyn::Align> {
   using enum dyn::Align;
   static constexpr auto value = enumerate(left,
                                           center,
                                           right,
                                           verticalcenter
   );
};

template <>
struct glz::meta<Vec2> {
    static constexpr auto value = object(&Vec2::x, &Vec2::y);
};

struct StructTest {
    uint64_t number;
    std::string text;
    std::optional<std::map<std::string, double>> numbers;
};

static const auto recent_items_test = R"({"entries":[{"created":"1722898354365058","filename":"/Users/user/Downloads/MatrixIssues/20240407_130252","modified":"1722898656173724","name":"/Users/user/Downloads/MatrixIssues/20240407_130252","output_dir":"","output_prefix":"","settings":{"cam_matrix":[0.9162667552083333,0,0.5045229020833334,0,1.6345315185185185,0.5236356972222223,0,0,1],"cam_undistort":true,"cam_undistort_vector":[-0.314080328,-0.0967427775,-0.000260259724,0.000191272304,0.476336027],"cm_per_pixel":0.019999999552965164,"detect_skeleton":["human",[[0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]]],"enable_absolute_difference":false,"function_test":null,"individual_image_scale":0.6000000238418579,"midline_stiff_percentage":0.10000000149011612,"output_dir":"","source":"/Users/user/Downloads/MatrixIssues/20240407_130252.MOV","threshold":40,"track_absolute_difference":false,"track_include":[[[1501,817],[463,841],[461,351],[1489,337]],[[1498,803],[1497,847],[456,861],[459,828]],[[1517,300],[1492,347],[456,364],[444,305]],[[498,853],[432,857],[430,316],[490,337]],[[1540,841],[1492,834],[1479,317],[1510,323]]],"track_max_individuals":2,"track_max_speed":40,"track_posture_threshold":9,"track_size_filter":[[0.25,1.5]],"track_threshold":40}},{"created":"1720779303324089","filename":"","modified":"1720779303324089","name":"/Users/user/Videos/tmp/juvenile_birchmanni_4_Trial12_UN_UN","output_dir":"","output_prefix":"","settings":{"blob_split_algorithm":"fill","calculate_posture":false,"cwd":"/Users/user/trex/Application/beta/Debug","detect_model":"yolov8x-pose","individual_image_normalization":"moments","meta_encoding":"r3g3b2","meta_source_path":"/Users/user/Downloads/juvenile_birchmanni_4_Trial12_UN_UN.mov","source":"/Users/user/Downloads/juvenile_birchmanni_4_Trial12_UN_UN.mov","track_do_history_split":false,"track_max_reassign_time":1}},{"created":"1720779303320210","filename":"","modified":"1720779303320210","name":"/Users/user/Videos/tmp/002_full_pilot","output_dir":"","output_prefix":"","settings":{"cm_per_pixel":0.019999999552965164,"detect_skeleton":["human",[[0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]]],"gpu_torch_no_fixes":true,"output_dir":"/Users/user/Videos/tmp","recognition_segment_add_factor":1,"source":"/Users/user/Downloads/002_full_pilot.MP4","track_max_individuals":4,"track_threshold":15}},{"created":"1720779303312230","filename":"","modified":"1720779303312230","name":"/Users/user/Downloads/002_full_pilot","output_dir":"","output_prefix":"","settings":{"cm_per_pixel":0.10000000149011612,"detect_skeleton":["human",[[0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]]],"gpu_torch_no_fixes":true,"gui_focus_group":[2],"gui_show_individual_preview":true,"heatmap_ids":[2],"individual_image_size":[80,80],"manual_matches":{"0":{"0":331361215,"1":362298148,"2":365449625,"3":50918454}},"manual_splits":{"10145":[1798340233],"10147":[1794669897],"10392":[1616404981],"10755":[432130367],"11032":[429400456],"11477":[1531453472],"11488":[1524637286],"11966":[152144770],"11968":[148474138],"12968":[183080834],"12993":[230779254],"12996":[228680390],"13308":[227114888],"14421":[174613835],"14426":[171468834],"14944":[1822521131],"15312":[1574971339],"15318":[1578641100],"16071":[66138639],"1653":[67681770],"16850":[913424684],"17002":[44104078],"1732":[67152321],"17605":[1929459522],"17896":[216569868],"17897":[207132831],"17944":[222846982],"17946":[228613767],"18586":[1881244567],"20015":[175739857],"20510":[465689085],"20807":[340865054],"20811":[343485149],"2238":[469772213],"23193":[53004033],"23197":[52479054],"24079":[97563986],"25172":[1928388496],"25174":[1925242007],"27190":[75032149],"27266":[72926034],"278":[1093151078],"280":[1094199649],"2891":[1940964054],"29018":[1131421509],"30853":[1362219416],"31401":[743976203],"31406":[745024848],"31460":[743452091],"3195":[1486995894],"34363":[229209693],"3446":[307347713],"3626":[77129362],"37473":[1757510170],"38899":[1612717393],"3899":[578302611],"44644":[1779992133],"44680":[1758495877],"45275":[109088291],"4540":[1935218378],"4601":[1924228299],"4603":[1903257168],"4821":[1159321968],"4825":[1155127728],"50231":[267946708],"50759":[1803052086],"50932":[1842905049],"54123":[37802975],"5500":[436217029],"5506":[433071617],"55981":[688400379],"57154":[1944147094],"57158":[1937328989],"6038":[922830723],"6039":[923879310],"61508":[217617076],"62288":[1944102445],"62304":[1904777172],"62306":[1900582039],"62552":[1461727497],"62557":[1458582283],"6325":[1518979461],"63657":[1690912047],"67618":[84490233],"6805":[1075946738],"68194":[313629355],"68199":[285841900],"68400":[75552263],"7599":[1828216105],"7649":[1832936187],"7681":[1843423414],"7726":[1867022998],"8488":[370266431],"8495":[351391185],"8496":[402771064],"8499":[382847024],"8555":[144804749],"9027":[1344294991],"9280":[1530968058],"9464":[40954656],"9470":[40955829]},"output_dir":"","segment_size_filter":[[0.10000000149011612,100]],"source":"/Users/user/Downloads/002_full_pilot.MP4","threshold":15,"track_ignore":[[[679,2097],[6,2147],[3,1663]],[[499,10],[6,351],[2,3]],[[3834,2152],[3284,2130],[3832,1723]]],"track_max_individuals":4,"track_max_speed":100,"track_size_filter":[[1.5,10]],"track_threshold":15}}],"modified":"1723213416841714"})";

TEST(ConversionTest, FileObjects) {
    //dyn::MainFile object;
    //std::string buffer = file::Path("/Users/tristan/trex/Application/src/commons/examples/test_gui.json").read_file();
    
    //glz::error_ctx error;
    //ASSERT_EQ(error = glz::read_json(object, buffer), glz::error_code::none) << glz::format_error(error, buffer);
    
    RecentItemFile object;
    std::string buffer = recent_items_test;//file::Path("/Users/tristan/trex/Application/beta/Debug/TRex.app/Contents/MacOS/../Resources/../Resources/.trex_recent_files").read_file();
    
    glz::error_ctx error;
    ASSERT_EQ(error = glz::read_json(object, buffer), glz::error_code::none) << glz::format_error(error, buffer);
}

TEST(ConversionTest, RealObjects) {
    /*glz::json_t json{};
    std::string buffer = R"([18446744073709551615,"Hello World",{"pi":3.14}])";
    glz::read_json(json, buffer);
    ASSERT_EQ(json[1].get<std::string>(), "Hello World");*/
    
    dyn::Object object;
    std::string buffer = R"({
        "type":"vlayout",
        "pos":[10,123],
        "children":[
            { "type":"button", 
              "color":[255,255,255],
              "font": {
                "size":1,
                "align":"left"
              }
            }
        ]
    })";
    glz::error_ctx error;
    ASSERT_EQ(error = glz::read_json(object, buffer), glz::error_code::none) << glz::format_error(error, buffer);
    //ASSERT_EQ(test.at("array").number, 18446744073709551615llu);
    
    //ASSERT_EQ(json[0].get<int64_t>(), 5);
    //ASSERT_EQ(json[2]["pi"].get<double>(), 3.14);
    //ASSERT_EQ(test.number.toStr(), "5");
}

TEST(ConversionTest, Website) {
    /*glz::json_t json{};
    std::string buffer = R"([18446744073709551615,"Hello World",{"pi":3.14}])";
    glz::read_json(json, buffer);
    ASSERT_EQ(json[1].get<std::string>(), "Hello World");*/
    
    std::map<std::string, StructTest> test;
    std::string buffer = R"({"array":{"number":18446744073709551615,"text":"Hello World"}})";
    glz::error_ctx error;
    ASSERT_EQ(error = glz::read_json(test, buffer), glz::error_code::none) << glz::format_error(error, buffer);
    ASSERT_EQ(test.at("array").number, 18446744073709551615llu);
    
    //ASSERT_EQ(json[0].get<int64_t>(), 5);
    //ASSERT_EQ(json[2]["pi"].get<double>(), 3.14);
    //ASSERT_EQ(test.number.toStr(), "5");
}

/*TEST(ConversionTest, Basic) {
    glz::json_t tuple_test = { 5.5, 5.5 };
    
    std::string s{};
    double dbl = 3.14;
    glz::json_t dblj = dbl;
    glz::write_json(dblj, s);
    
    uint64_t nr = 55;

    static_assert(glz::detail::num_t<uint64_t>);
    glz::json_t j = nr;
    ASSERT_TRUE(j.is_number());
    ASSERT_TRUE(j.holds<uint64_t>());
    glz::write_json(j, s);

    EXPECT_EQ(s, R"(55)");
    
    glz::write_json(j, s);
    
    glz::json_t json = 64u;
    glz::error_ctx error;
    EXPECT_NO_THROW(error = glz::read_json(json, s));
    EXPECT_EQ(error, glz::error_code::none) << glz::format_error(error, s);
    EXPECT_TRUE(json.is_number());
    EXPECT_EQ(json.get_uint(), 55);
}

template <typename T>
class NumberConversionTest : public ::testing::Test {
protected:
    void test_conversion(T value) {
        std::string s;
        glz::json_t json = value;
        glz::write_json(json, s);

        std::string expected;
        if constexpr(std::is_floating_point_v<T>) {
            std::stringstream ss;
            ss << value;
            expected = ss.str();
        } else {
            expected = Meta::toStr(value);
        }

        EXPECT_EQ(s, expected);
        
        glz::json_t json_read;
        auto error = glz::read_json(json_read, s);
        EXPECT_EQ(error, glz::error_code::none) << glz::format_error(error, s);
        EXPECT_EQ(json_read.get<T>(), value);
    }
};

typedef ::testing::Types< uint64_t, int64_t, double> NumberTypes;
TYPED_TEST_SUITE(NumberConversionTest, NumberTypes);

TYPED_TEST(NumberConversionTest, BasicConversion) {
    this->test_conversion(static_cast<TypeParam>(42));
    this->test_conversion(static_cast<TypeParam>(-42));
    this->test_conversion(static_cast<TypeParam>(0));
    this->test_conversion(static_cast<TypeParam>(std::numeric_limits<TypeParam>::max()));
    this->test_conversion(static_cast<TypeParam>(std::numeric_limits<TypeParam>::min()));
}

TEST(FloatDoubleConversionTest, PrecisionCheck) {
    std::string s;
    double dbl = 3.141592653589793;
    glz::json_t dblj = dbl;
    glz::write_json(dblj, s);
    
    EXPECT_EQ(s, "3.141592653589793");

    glz::json_t json;
    auto error = glz::read_json(json, s);
    EXPECT_EQ(error, glz::error_code::none) << glz::format_error(error, s);
    EXPECT_EQ(json.get_float(), 3.141592653589793);
}

TEST(FloatDoubleConversionTest, ScientificNotation) {
    std::string s;
    double dbl = 1.23e-10;
    glz::json_t dblj = dbl;
    glz::write_json(dblj, s);
    
    EXPECT_EQ(s, "1.23E-10");

    glz::json_t json;
    auto error = glz::read_json(json, s);
    EXPECT_EQ(error, glz::error_code::none) << glz::format_error(error, s);
    EXPECT_EQ(json.get_float(), 1.23e-10);
}

TEST(IntegerConversionTest, LargeIntegers) {
    std::string s;
    uint64_t large_uint = 18446744073709551615ULL; // Max value for uint64_t
    glz::json_t json_uint = large_uint;
    glz::write_json(json_uint, s);
    
    EXPECT_EQ(s, "18446744073709551615");

    glz::json_t json;
    auto error = glz::read_json(json, s);
    EXPECT_EQ(error, glz::error_code::none) << glz::format_error(error, s);
    EXPECT_EQ(json.get<uint64_t>(), 18446744073709551615ULL);

    int64_t large_int = 9223372036854775807LL; // Max value for int64_t
    glz::json_t json_int = large_int;
    glz::write_json(json_int, s);
    
    EXPECT_EQ(s, "9223372036854775807");

    glz::json_t json_i;
    error = glz::read_json(json_i, s);
    EXPECT_EQ(error, glz::error_code::none) << glz::format_error(error, s);
    EXPECT_EQ(json_i.get<int64_t>(), 9223372036854775807LL);
}

TEST(ErrorHandlingTest, InvalidJson) {
    std::string invalid_json = R"({"invalid": )"; // malformed JSON
    glz::json_t input;
    auto error = glz::read_json(input, invalid_json);
    EXPECT_NE(error, glz::error_code::none);
}

TEST(ErrorHandlingTest, NonNumberToNumber) {
    std::string invalid_json = R"("not a number")";
    glz::json_t input;
    auto error = glz::read_json(input, invalid_json);
    EXPECT_NE(error, glz::error_code::none);
}

TEST(ErrorHandlingTest, ValidStringToNumber) {
    std::string valid_number = "123";
    glz::json_t input;
    auto error = glz::read_json(input, valid_number);
    EXPECT_EQ(error, glz::error_code::none) << glz::format_error(error, valid_number);
    EXPECT_EQ(input.get<int64_t>(), 123);
}*/


TEST(CacheSizeTest, Basic) {
    UNUSED(lower);
    
    using namespace track;
    //static_assert(std::is_trivial_v<TrivialOptional<uint32_t>>);
    static_assert(std::is_trivially_copyable_v<TrivialOptional<uint32_t>>);
    //Print(sizeof(IndividualCache), " ", sizeof(float), " ", sizeof(Vec2), " ", sizeof(Frame_t), " ", sizeof(Match::prob_t), " ", sizeof(TrivialOptional<uint32_t>));
    //static_assert(std::is_trivial_v<Frame_t>);
}

TEST(JSONTest, TestBasicJSON) {
    std::vector<std::pair<std::string, std::vector<std::string>>> object {
        {{"X", {"RAW", "WCENTROID"}}}
    };
    SETTING(graphs) = object;
    
    auto json = SETTING(graphs).get().to_json();
    ASSERT_EQ(Meta::fromStr<std::string>(glz::write_json(json).value()), SETTING(graphs).get().valueString());
}

TEST(JSONTest, TestSkeletonJSON) {
    blob::Pose::Skeleton object{
        "skeleton",
        {
            {0, 1, "first"},
            {1, 2, "second"}
        }
    };
    SETTING(skeleton) = object;
    
    auto json = SETTING(skeleton).get().to_json();
    ASSERT_EQ(Meta::fromStr<std::string>(glz::write_json(json).value()), SETTING(skeleton).get().valueString());
}

TEST(JSONTest, TestVec2JSON) {
    std::vector<Vec2> object {
        Vec2(10,25)
    };
    SETTING(vectors) = object;
    SETTING(number) = 5;
    SETTING(big_number) = uint64_t(std::numeric_limits<uint64_t>::max());
    
    /// the strings will not be exactly the same.
    auto json = SETTING(vectors).get().to_json();
    ASSERT_EQ(Meta::fromStr<std::vector<Vec2>>(Meta::fromStr<std::string>(glz::write_json(json))), object);
    
    /// check whether it removes trailing zeros
    auto s = glz::write_json(json);
    ASSERT_STREQ(s.c_str(), "[[10,25]]");
    
    json = SETTING(number).get().to_json();
    s = glz::write_json(json).value();
    ASSERT_STREQ(s.c_str(), "5");
    
    json = SETTING(big_number).get().to_json();
    s = glz::write_json(json).value();
    /// currently not achievable - only in custom structs
    //ASSERT_EQ(s, SETTING(big_number).get().valueString());
}

// Tests for the split function.
TEST(SplitTest, TestBasicSplit) {
    std::string s = "foo,bar,baz";
    std::vector<std::string_view> expected = {"foo", "bar", "baz"};
    auto result = split(s, ',');
    static_assert(are_the_same<decltype(result), decltype(expected)>, "Has to be a vector of string_view");
    EXPECT_EQ(result, expected);
    
    for(auto &view : result) {
        EXPECT_GE(view.data(), s.data());
        EXPECT_LE(view.data() + view.length(), s.data() + s.length());
    }

    std::wstring ws = L"hello,world";
    std::vector<std::wstring_view> expected_ws = {L"hello", L"world"};
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
    std::vector<std::string_view> expected = {""};
    EXPECT_EQ(split(s, ','), expected);

    std::wstring ws = L"";
    std::vector<std::wstring_view> expected_ws = {L""};
    EXPECT_EQ(split(ws, ','), expected_ws);
}

TEST(SplitTest, TestSingleDelimiter) {
    std::string s = ",";
    std::vector<std::string_view> expected = {"", ""};
    EXPECT_EQ(split(s, ',', false, false), expected);

    std::wstring ws = L",";
    std::vector<std::wstring_view> expected_ws = {L"", L""};
    EXPECT_EQ(split(ws, ',', false, false), expected_ws);
}

TEST(SplitTest, TestNoDelimiter) {
    std::string s = "foobar";
    std::vector<std::string_view> expected = {"foobar"};
    EXPECT_EQ(split(s, ','), expected);

    std::wstring ws = L"hello";
    std::vector<std::wstring_view> expected_ws = {L"hello"};
    EXPECT_EQ(split(ws, ','), expected_ws);
}

TEST(SplitTest, TestMultipleDelimiters) {
    std::string s = "foo,,bar,,baz";
    std::vector<std::string_view> expected = {"foo", "", "bar", "", "baz"};
    auto result = split(s, ',');
    EXPECT_EQ(result, expected);
    
    for(auto &view : result) {
        EXPECT_GE(view.data(), s.data());
        EXPECT_LE(view.data() + view.length(), s.data() + s.length());
    }

    std::wstring ws = L"hello, ,world";
    std::vector<std::wstring_view> expected_ws = {L"hello", L" ", L"world"};
    EXPECT_EQ(split(ws, ','), expected_ws);
}

TEST(SplitTest, TestTrimming) {
    std::string s = "  foo , bar ,  baz  ";
    std::vector<std::string_view> expected = {"foo", "bar", "baz"};
    EXPECT_EQ(split(s, ',', false, true), expected);

    std::wstring ws = L"  hello  ,  world  ";
    std::vector<std::wstring_view> expected_ws = {L"hello", L"world"};
    EXPECT_EQ(split(ws, ',', false, true), expected_ws);
}

TEST(StringTrimTests, LTrimBasicTest) {
    std::string str1 = "    leading";
    auto result1 = ltrim(str1);
    ASSERT_EQ(result1, "leading");

    std::string_view str2 = "    leading";
    auto result2 = ltrim(str2);
    ASSERT_EQ(result2, "leading");
}

TEST(StringTrimTests, LTrimNoTrimTest) {
    std::string str1 = "noleading";
    auto result1 = ltrim(str1);
    ASSERT_EQ(result1, "noleading");

    std::string_view str2 = "noleading";
    auto result2 = ltrim(str2);
    ASSERT_EQ(result2, "noleading");
}

TEST(StringTrimTests, RTrimBasicTest) {
    std::string str1 = "trailing    ";
    auto result1 = rtrim(str1);
    ASSERT_EQ(result1, "trailing");

    std::string_view str2 = "trailing    ";
    auto result2 = rtrim(str2);
    ASSERT_EQ(result2, "trailing");
}

TEST(StringTrimTests, RTrimNoTrimTest) {
    std::string str1 = "notrailing";
    auto result1 = rtrim(str1);
    ASSERT_EQ(result1, "notrailing");

    std::string_view str2 = "notrailing";
    auto result2 = rtrim(str2);
    ASSERT_EQ(result2, "notrailing");
}

TEST(StringTrimTests, LTrimConstRefTest) {
    const std::string str = "    leading";
    auto result = ltrim(str);
    ASSERT_EQ(result, "leading");
}

TEST(StringTrimTests, RTrimConstRefTest) {
    const std::string str = "trailing    ";
    auto result = rtrim(str);
    ASSERT_EQ(result, "trailing");
}

TEST(StringTrimTests, LTrimRValueTest) {
    auto result = ltrim(std::string("    leading"));
    ASSERT_EQ(result, "leading");
}

TEST(StringTrimTests, RTrimRValueTest) {
    auto result = rtrim(std::string("trailing    "));
    ASSERT_EQ(result, "trailing");
}

TEST(StringTrimTests, LTrimEmptyTest) {
    std::string str = "";
    auto result = ltrim(str);
    ASSERT_EQ(result, "");

    std::string_view str2 = "";
    auto result2 = ltrim(str2);
    ASSERT_EQ(result2, "");
}

TEST(StringTrimTests, RTrimEmptyTest) {
    std::string str = "";
    auto result = rtrim(str);
    ASSERT_EQ(result, "");

    std::string_view str2 = "";
    auto result2 = rtrim(str2);
    ASSERT_EQ(result2, "");
}

TEST(StringTrimTests, TrimOnlySpacesTest) {
    std::string str1 = "    ";
    auto result1 = trim(str1);
    ASSERT_EQ(result1, "");

    std::string_view str2 = "    ";
    auto result2 = trim(str2);
    ASSERT_EQ(result2, "");
}

TEST(StringTrimTests, TrimMixedSpacesTest) {
    std::string str1 = " \t \r \n ";
    auto result1 = trim(str1);
    ASSERT_EQ(result1, "");

    std::string_view str2 = " \t \r \n ";
    auto result2 = trim(str2);
    ASSERT_EQ(result2, "");
}

TEST(StringTrimTests, TrimConstRefTest) {
    const std::string str = "  both  ";
    auto result = trim(str);
    ASSERT_EQ(result, "both");
}

TEST(StringTrimTests, TrimRValueTest) {
    auto result = trim(std::string("  both  "));
    ASSERT_EQ(result, "both");
}

TEST(StringTrimTests, TrimEmptyTest) {
    std::string str1 = "";
    auto result1 = trim(str1);
    ASSERT_EQ(result1, "");

    std::string_view str2 = "";
    auto result2 = trim(str2);
    ASSERT_EQ(result2, "");
}

TEST(SplitTest, TestSkipEmpty) {
    std::string s = "foo,,bar,,baz";
    std::vector<std::string_view> expected = {"foo", "bar", "baz"};
    EXPECT_EQ(split(s, ',', true, false), expected);

    std::wstring ws = L"hello, ,world";
    std::vector<std::wstring_view> expected_ws = {L"hello", L" ", L"world"};
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

// upper lower
// Test with identical strings
TEST(StringLowercaseEqual, IdenticalStrings) {
    std::string_view str1 = "hello";
    const char str2[] = "hello";
    EXPECT_TRUE(utils::lowercase_equal_to(str1, str2));
}

// Test with different strings
TEST(StringLowercaseEqual, DifferentStrings) {
    std::string_view str1 = "hello";
    const char str2[] = "world";
    EXPECT_FALSE(utils::lowercase_equal_to(str1, str2));
}

// Test with strings that differ only in case
TEST(StringLowercaseEqual, CaseInsensitive) {
    std::string_view str1 = "Hello";
    const char str2[] = "hello";
    EXPECT_TRUE(utils::lowercase_equal_to(str1, str2));
}

// Test with one string as a substring of the other
TEST(StringLowercaseEqual, Substring) {
    std::string_view str1 = "hello";
    const char str2[] = "hell";
    EXPECT_FALSE(utils::lowercase_equal_to(str1, str2));
}

// Test with empty strings
TEST(StringLowercaseEqual, EmptyStrings) {
    std::string_view str1 = "";
    const char str2[] = "";
    EXPECT_TRUE(utils::lowercase_equal_to(str1, str2));
}

// Test with empty strings
TEST(StringLowercaseEqual, RawTest) {
    std::string_view str1 = "RAW";
    EXPECT_TRUE(utils::lowercase_equal_to(str1, "raw"));
}

// Test with non-alphabetic characters
TEST(StringLowercaseEqual, NonAlphabetic) {
    std::string_view str1 = "123@!";
    const char str2[] = "123@!";
    EXPECT_TRUE(utils::lowercase_equal_to(str1, str2));
}

// Test with strings containing uppercase non-alphabetic characters
TEST(StringLowercaseEqual, MixedCaseNonAlphabetic) {
    std::string_view str1 = "HeLlO123@!";
    const char str2[] = "hello123@!";
    EXPECT_TRUE(utils::lowercase_equal_to(str1, str2));
}

TEST(StringUtilTest, Lowercase) {
    // Test with std::string
    EXPECT_EQ(lowercase(std::string("Hello")), "hello");
    
    // Test with std::wstring
    EXPECT_EQ(lowercase(std::wstring(L"Hello")), L"hello");
    
    // Test with std::string_view
    EXPECT_EQ(lowercase(std::string_view("Hello")), "hello");
    
    // Test with const char*
    EXPECT_EQ(lowercase("Hello"), "hello");
    
    // Test with const wchar_t*
    EXPECT_EQ(lowercase(L"Hello"), L"hello");

    // Test with an empty std::string
    EXPECT_EQ(lowercase(std::string("")), "");

    // Test with an empty std::string_view
    EXPECT_EQ(lowercase(std::string_view("")), "");
}

TEST(StringUtilTest, Uppercase) {
    // Test with std::string
    EXPECT_EQ(uppercase(std::string("Hello")), "HELLO");
    
    // Test with std::wstring
    EXPECT_EQ(uppercase(std::wstring(L"Hello")), L"HELLO");
    
    // Test with std::string_view
    EXPECT_EQ(uppercase(std::string_view("Hello")), "HELLO");
    
    // Test with const char*
    EXPECT_EQ(uppercase("Hello"), "HELLO");
    
    // Test with const wchar_t*
    EXPECT_EQ(uppercase(L"Hello"), L"HELLO");

    // Test with an empty std::string
    EXPECT_EQ(uppercase(std::string("")), "");

    // Test with an empty std::string_view
    EXPECT_EQ(uppercase(std::string_view("")), "");
}

// find_replace

TEST(FindReplaceTest, BasicTest) {
    std::string str = "The quick brown fox jumps over the lazy dog.";
    std::vector<std::pair<std::string_view, std::string_view>> search_strings = {
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
    std::vector<std::pair<std::string_view, std::string_view>> search_strings = {
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
    std::vector<std::pair<std::string_view, std::string_view>> search_strings = {
        {"foo", "bar"},
        {"baz", "qux"}
    };
    std::string expected = "The quick brown fox jumps over the lazy dog.";
    EXPECT_EQ(find_replace(str, search_strings), expected);
}

TEST(FindReplaceTest, OverlappingMatches) {
    std::string str = "The quick brown fox jumps over the lazy dog.";
    std::vector<std::pair<std::string_view, std::string_view>> search_strings = {
        {"the", "THE"},
        {"THE", "the"},
    };
    std::string expected = "The quick brown fox jumps over THE lazy dog.";
    EXPECT_EQ(find_replace(str, search_strings), expected);
}



TEST(FindReplaceTest, EmptyInputString) {
    std::string input = "";
    std::vector<std::pair<std::string_view, std::string_view>> search_strings = {{"abc", "xyz"}, {"def", "uvw"}};
    std::string result = find_replace(input, search_strings);
    ASSERT_EQ(result, "");
}


TEST(FindReplaceTest, EmptySearchStrings) {
    std::string str = "The quick brown fox jumps over the lazy dog.";
    std::vector<std::pair<std::string_view, std::string_view>> search_strings = {};
    std::string expected = "The quick brown fox jumps over the lazy dog.";
    EXPECT_EQ(find_replace(str, search_strings), expected);
    
    {
        std::string input = "abcdefgh";
        std::vector<std::pair<std::string_view, std::string_view>> search_strings;
        std::string result = find_replace(input, search_strings);
        ASSERT_EQ(result, "abcdefgh");
    }
}

TEST(FindReplaceTest, EmptyInputAndSearchStrings) {
    std::string input = "";
    std::vector<std::pair<std::string_view, std::string_view>> search_strings;
    std::string result = find_replace(input, search_strings);
    ASSERT_EQ(result, "");
}

TEST(FindReplaceTest, NoMatchingSearchStrings) {
    std::string input = "abcdefgh";
    std::vector<std::pair<std::string_view, std::string_view>> search_strings = {{"ijk", "xyz"}, {"lmn", "uvw"}};
    std::string result = find_replace(input, search_strings);
    ASSERT_EQ(result, "abcdefgh");
}

TEST(FindReplaceTest, SomeMatchingSearchStrings) {
    std::string input = "abcdefgh";
    std::vector<std::pair<std::string_view, std::string_view>> search_strings = {{"abc", "xyz"}, {"lmn", "uvw"}};
    std::string result = find_replace(input, search_strings);
    ASSERT_EQ(result, "xyzdefgh");
}

TEST(FindReplaceTest, AllMatchingSearchStrings) {
    std::string input = "abcdefgh";
    std::vector<std::pair<std::string_view, std::string_view>> search_strings = {{"abc", "xyz"}, {"def", "uvw"}};
    std::string result = find_replace(input, search_strings);
    ASSERT_EQ(result, "xyzuvwgh");
}

TEST(FindReplaceTest, MultipleInstancesOfSearchStrings) {
    std::string input = "abcdeabc";
    std::vector<std::pair<std::string_view, std::string_view>> search_strings = {{"abc", "xyz"}};
    std::string result = find_replace(input, search_strings);
    ASSERT_EQ(result, "xyzdexyz");
}

TEST(FindReplaceTest, SpecialCharactersAndDigits) {
    std::string input = "a$b%c123";
    std::vector<std::pair<std::string_view, std::string_view>> search_strings = {{"$", "X"}, {"%", "Y"}, {"1", "2"}};
    std::string result = find_replace(input, search_strings);
    ASSERT_EQ(result, "aXbYc223");
}

TEST(FindReplaceTest, IdenticalReplacements) {
    std::string input = "abcabc";
    std::vector<std::pair<std::string_view, std::string_view>> search_strings = {{"abc", "abc"}};
    std::string result = find_replace(input, search_strings);
    ASSERT_EQ(result, "abcabc");
}

TEST(FindReplaceTest, ComplexSearchStrings) {
    std::string input = "A quick brown fox jumps over the lazy dog, and the dog returns the favor.";
    std::vector<std::pair<std::string_view, std::string_view>> search_strings = {
        {"quick", "swift"},
        {"the lazy dog", "a sleeping canine"},
        {"the favor", "a compliment"}
    };
    std::string expected = "A swift brown fox jumps over a sleeping canine, and the dog returns a compliment.";
    EXPECT_EQ(find_replace(input, search_strings), expected);
}

TEST(FindReplaceTest, UnicodeCharacters) {
    std::string input = "こんにちは世界";
    std::vector<std::pair<std::string_view, std::string_view>> search_strings = {{"こんにちは", "さようなら"}, {"世界", "宇宙"}};
    std::string expected = "さようなら宇宙";
    EXPECT_EQ(find_replace(input, search_strings), expected);
}

TEST(FindReplaceTest, CaseSensitivity) {
    std::string input = "The Quick Brown Fox Jumps Over The Lazy Dog.";
    std::vector<std::pair<std::string_view, std::string_view>> search_strings = {{"The", "A"}, {"Quick", "Swift"}, {"Brown", "Red"}};
    std::string expected = "A Swift Red Fox Jumps Over A Lazy Dog.";
    EXPECT_EQ(find_replace(input, search_strings), expected);
}

TEST(FindReplaceTest, ComplexOverlappingSearchStrings) {
    std::string input = "appleappletreeapple";
    std::vector<std::pair<std::string_view, std::string_view>> search_strings = {{"apple", "orange"}, {"appletree", "peachtree"}};
    std::string expected = "orangeorangetreeorange";
    EXPECT_EQ(find_replace(input, search_strings), expected);
}

TEST(FindReplaceTest, RespectOrderOfSearchStrings) {
    std::string input = "this is an orangetree";
    std::vector<std::pair<std::string_view, std::string_view>> search_strings = {
        {"orange", "apple"},
        {"orangetree", "appletree"},
    };
    std::string expected = "this is an appletree";
    EXPECT_EQ(find_replace(input, search_strings), expected);
}

TEST(FindReplaceTest, MultipleReplacementsInARow) {
    std::string input = "helloorangeapplegrape";
    std::vector<std::pair<std::string_view, std::string_view>> search_strings = {
        {"orange", "banana"},
        {"apple", "cherry"},
        {"grape", "kiwi"},
    };
    std::string expected = "hellobananacherrykiwi";
    EXPECT_EQ(find_replace(input, search_strings), expected);
}

TEST(FindReplaceTest, OverlappingSearchStrings) {
    std::string input = "abcde";
    std::vector<std::pair<std::string_view, std::string_view>> search_strings = {
        {"abc", "xyz"},
        {"bcd", "uv"},
    };
    std::string expected = "xyzde";
    EXPECT_EQ(find_replace(input, search_strings), expected);
}

TEST(FindReplaceTest, ReplaceSubstringsWithDifferentLengths) {
    std::string input = "this is an orangetree";
    std::vector<std::pair<std::string_view, std::string_view>> search_strings = {
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

std::string which_from(const auto& indexes, const auto& corpus, size_t len = 0) {
    std::string r = "[";
    bool first = true;
    size_t I{0u};
    for(auto index : indexes) {
        if(not first) {
            r += ',';
        }
        r += size_t(index) < corpus.size() ? "'"+corpus.at(index)+"'" : "<invalid:"+std::to_string(index)+">";
        first = false;
        
        if(len > 0u && ++I >= len)
            break;
    }
    
    return r+"] from the corpus";
}

bool compare_to_first_n_elements(const std::vector<int>& vec1, const std::vector<int>& vec2) {
    if (vec1.size() > vec2.size()) {
        return false;
    }

    return std::equal(vec1.begin(), vec1.end(), vec2.begin());
}

bool first_elements_should_contain(const std::vector<int>& vec1, const std::vector<int>& vec2) {
    if (vec1.size() > vec2.size()) {
        return false;
    }
    auto sub = std::vector<int>(vec2.begin(), vec2.begin() + vec1.size());
    for(auto& v : vec1)
        if(not contains(sub, v))
            return false;
    return true;
}

TEST(TextSearchTest, HandlesMultipleWords) {
    std::vector<std::string> corpus = {"apples", "oranges", "apples_oranges"};

    // Test case: search for "apples oranges", should return only "apples_oranges" (index 2)
    auto result = text_search("apples oranges", corpus);
    decltype(result) expected{2, 1, 0};
    ASSERT_TRUE(compare_to_first_n_elements(expected, result)) << "selecting " << which_from(result, corpus, expected.size()) << " from " << Meta::toStr(corpus) << " instead of " << which_from(expected, corpus, expected.size());
}

TEST(TextSearchTest, ReturnsLongerStringFirst) {
    std::vector<std::string> corpus = {
        "potoo6511_small_abstract_t-rex_silhouette_black_and_white_minim_d1dd143b-dsg-4ca8-fsd46-fgd43234234.png",
        "640-yolov8n-seg-2023-09-14-10_MouseSourveillance-2-mAP5095_0.93182-mAP50_0.97449.pt",
        "2023-02-21-raspios-bullseye-armhf-lite.img.xz",
        "yolov7-tiny-640p-20230418T075703Z-001.zip",
        "reversals3m_1024_dotbot_20181025_105202.stitched.pv",
        "65MP01_10Kmarching_01_2023-03-29_10-10-24-924.jpg",
        "20230622_ContRamp_60deg_0.5degmin_R1_Myrmica_rubra_C3_C2_1920x1080@20FPS_11-42-01.mp4",
        "640-yolov8n-2023-07-18-13_mais-2-mAP5095_0.94445-mAP50_0.995.pt"
    };

    // Test case: search for "apples oranges", should return only "apples_oranges" (index 2)
    auto result = text_search("yolov8n 640 seg mouse", corpus);
    decltype(result) expected{1,7,3};
    ASSERT_TRUE(compare_to_first_n_elements(expected, result)) << "selecting " << which_from(result, corpus, expected.size()) << " from " << Meta::toStr(corpus) << " instead of " << which_from(expected, corpus, expected.size());
}

TEST(TextSearchTest, HandlesSingleWord) {
    std::vector<std::string> corpus = {"apples", "oranges", "apples_oranges"};

    // Test case: search for "apples", should return "apples" (index 0) and "apples_oranges" (index 2)
    auto search = "apples";
    auto result = text_search(search, corpus);
    decltype(result) expected{0, 2};
    ASSERT_TRUE(compare_to_first_n_elements(expected, result)) << "selecting " << which_from(result, corpus, expected.size()) << " from " << Meta::toStr(corpus) << " instead of " << which_from(expected, corpus, expected.size()) << " for search '" << search << "'";
}

TEST(TextSearchTest, HandlesSingleCharacter) {
    std::vector<std::string> corpus = {"$"};

    // Test case: search for "apples", should return "apples" (index 0) and "apples_oranges" (index 2)
    auto result = text_search("$", corpus);
    decltype(result) expected{0};
    ASSERT_TRUE(compare_to_first_n_elements(expected, result)) << "selecting " << which_from(result, corpus, expected.size()) << " from " << Meta::toStr(corpus) << " instead of " << which_from(expected, corpus, expected.size());
}

TEST(TextSearchTest, HandlesMisspelledWord) {
    std::vector<std::string> corpus = {"apple", "oranges", "apples_oranges"};

    // Test case: search for "organes" (misspelled), should return "oranges" (index 1) and "apples_oranges" (index 2)
    auto result = text_search("organes", corpus);
    decltype(result) expected{1,2};
    ASSERT_TRUE(compare_to_first_n_elements(expected, result)) << "selecting " << which_from(result, corpus, expected.size()) << " from " << Meta::toStr(corpus) << " instead of " << which_from(expected, corpus, expected.size()) << " for search term 'organes'";
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
    ASSERT_TRUE(compare_to_first_n_elements(expected, result)) << "selecting " << which_from(result, corpus, expected.size()) << " from " << Meta::toStr(corpus) << " instead of " << which_from(expected, corpus, expected.size()) << " for search term 'apple banana'";
}

TEST(SearchText, HandlesPaths) {
    std::vector<std::string> corpus = {
        "apple/pie/recipe",
        "/rooted/path/to/folder",
        "/folder/ending/"
    };

    auto input = "rooted banana";
    auto result = text_search(input, corpus);
    decltype(result) expected = {1};
    ASSERT_TRUE(compare_to_first_n_elements(expected, result)) << "selecting " << which_from(result, corpus, expected.size()) << " from " << Meta::toStr(corpus) << " instead of " << which_from(expected, corpus, expected.size()) << " for search term '"<<input<<"'";
    
    input = "rooted/folder";
    result = text_search(input, corpus);
    expected = {1,2};
    ASSERT_TRUE(compare_to_first_n_elements(expected, result)) << "selecting " << which_from(result, corpus, expected.size()) << " from " << Meta::toStr(corpus) << " instead of " << which_from(expected, corpus, expected.size()) << " for search term '"<<input<<"'";
}

TEST(SearchText, MatchesPathsCorrectly) {
    std::vector<std::string> corpus = {
        "apple/pie/recipe",
        "/rooted/path/to/folder/2023-02-21-raspios-bullseye-armhf-lite.img.xz",
        "/rooted/path/to/folder/yolov7-tiny-640p-20230418T075703Z-001.zip",
        "/folder/ending/640-yolov8n-seg-2023-09-14-10_MouseSourveillance-2-mAP5095_0.93182-mAP50_0.97449.pt"
    };

    auto input = "640 yolov8n";
    auto result = text_search(input, corpus);
    auto expected = {3,2};
    ASSERT_TRUE(compare_to_first_n_elements(expected, result)) << "selecting " << which_from(result, corpus, expected.size()) << " from " << Meta::toStr(corpus) << " instead of " << which_from(expected, corpus, expected.size()) << " for search term '"<<input<<"'";
}

TEST(SearchText, EmptyInput) {
    std::vector<std::string> corpus = {
        "apple pie recipe",
        "banana bread recipe",
        "apple and banana smoothie",
        "banana split",
        "apple and orange juice",
        "orange chicken recipe"
    };

    auto result = text_search("", corpus);
    decltype(result) expected = {0,1,2,3,4,5};
    ASSERT_TRUE(compare_to_first_n_elements(expected, result)) << "selecting " << which_from(result, corpus, expected.size()) << " from " << Meta::toStr(corpus) << " instead of " << which_from(expected, corpus, expected.size()) << " for search term 'aplpe banan'";
}

TEST(SearchText, AlwaysReturnAllItems) {
    std::vector<std::string> corpus = {
        "apple pie recipe",
        "banana bread recipe",
        "apple and banana smoothie",
        "banana split",
        "apple and orange juice",
        "orange chicken recipe"
    };

    auto result = text_search("banana", corpus);
    ASSERT_EQ(result.size(), corpus.size()) << " only returned " << which_from(result, corpus) << " instead of the entire corpus of " << Meta::toStr(corpus);
}

TEST(SearchText, ExactInput) {
    std::vector<std::string> corpus = {
        "apple pie recipe",
        "banana bread recipe",
        "apple and banana smoothie",
        "banana split",
        "apple and orange juice",
        "orange chicken recipe",
        "$"
    };

    auto search_term = "$";
    auto result = text_search(search_term, corpus);
    decltype(result) expected = {6};
    ASSERT_TRUE(compare_to_first_n_elements(expected, result)) << "selecting " << which_from(result, corpus, expected.size()) << " from " << Meta::toStr(corpus) << " instead of " << which_from(expected, corpus, expected.size()) << " for search term '"<<search_term<<"'";
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
    ASSERT_TRUE(first_elements_should_contain(expected, result)) << "selecting " << which_from(result, corpus, expected.size()) << " from " << Meta::toStr(corpus) << " instead of " << which_from(expected, corpus, expected.size()) << " for search term 'aplpe banan'";
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
    ASSERT_TRUE(compare_to_first_n_elements(expected, result)) << "selecting " << which_from(result, corpus, expected.size()) << " from " << Meta::toStr(corpus) << " instead of " << which_from(expected, corpus, expected.size()) << " for search term 'apple, banana'";
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
        ASSERT_TRUE(first_elements_should_contain(expected, result)) << "selecting " << which_from(result, corpus, expected.size()) << " from " << Meta::toStr(corpus) << " instead of " << which_from(expected, corpus, expected.size()) << " for search term 'track_th'";
    }
    
    {
        auto result = text_search("track_threshold", corpus);
        decltype(result) expected = {0,2,1};
        ASSERT_TRUE(first_elements_should_contain(expected, result)) << "selecting " << which_from(result, corpus, expected.size()) << " from " << Meta::toStr(corpus) << " instead of " << which_from(expected, corpus, expected.size()) << " for search term 'track_threshold'";
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
        ASSERT_TRUE(first_elements_should_contain(expected, result)) << "selecting " << which_from(result, corpus, expected.size()) << " from " << Meta::toStr(corpus) << " instead of " << which_from(expected, corpus, expected.size()) << " for search term 'track_th'";
    }
    {
        auto result = text_search("rec", corpus);
        decltype(result) expected = {0, 1, 5};
        ASSERT_TRUE(first_elements_should_contain(expected, result)) << "selecting " << which_from(result, corpus, expected.size()) << " from " << Meta::toStr(corpus) << " instead of " << which_from(expected, corpus, expected.size()) << " for search term 'rec'";
    }
    
    
    
}

#ifndef NDEBUG
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
#endif

// Test suite for round-trip Meta::toStr -> Meta::fromStr -> Meta::toStr
TEST(ToStrFromStrRoundTripTest, BasicTest) {
    std::string original1 = "hello";
    std::string original2 = R"(world)";
    std::string original3 = R"(\n\t\\)";
    
    std::string intermediate1 = Meta::toStr<std::string>(original1);
    std::string intermediate2 = Meta::toStr<std::string>(original2);
    std::string intermediate3 = Meta::toStr<std::string>(original3);
    
    std::string result1 = Meta::toStr<std::string>(Meta::fromStr<std::string>(intermediate1));
    std::string result2 = Meta::toStr<std::string>(Meta::fromStr<std::string>(intermediate2));
    std::string result3 = Meta::toStr<std::string>(Meta::fromStr<std::string>(intermediate3));
    
    EXPECT_EQ(result1, intermediate1);
    EXPECT_EQ(result2, intermediate2);
    EXPECT_EQ(result3, intermediate3);
}

// Test suite for round-trip Meta::toStr -> Meta::fromStr -> Meta::toStr
TEST(ToStrFromStrRoundTripTest, EscapingTest) {
    std::vector<std::string> testStrings = {
        R"(\\\test\\)",
        R"(\)",
        R"(testwithend\)",
        R"("trailing spaces beyond end"    )",
        R"(hello)",
        R"(world)",
        R"('single quotes')",
        R"("double quotes")",
        R"(\"escaped double quotes\")",
        R"('\'escaped single quotes\'')",
        R"(\\)",
        R"(\\\")",
        R"(\\\\)",
        R"(\n\t\b\r\f\a)",
        R"(\\n\\t\\b\\r\\f\\a)",
        R"(   leading spaces)",
        R"(trailing spaces   )",
        R"("trailing spaces beyond end"    )",
        R"(    "leading spaces beyond start")"
    };
    
    for (const auto& original : testStrings) {
        std::string intermediate = Meta::toStr<std::string>(original);
        Print("original ", original.c_str());
        Print("intermed ", intermediate.c_str());
        
        // Assuming your fromStr function trims spaces beyond the enclosing quotes
        std::string trimmed_intermediate = intermediate;
        
        std::string result = Meta::toStr<std::string>(Meta::fromStr<std::string>(trimmed_intermediate));
        
        Print("result   ", result.c_str());
        
        EXPECT_EQ(result, intermediate) << "Failed round-trip test for string: " << original;
        
        std::string fully_reversed = Meta::fromStr<std::string>(result);
        Print("fully r  ", fully_reversed.c_str());
        EXPECT_EQ(fully_reversed, original) << "Failed round-trip test for string: " << original;
    }
}

bool isWindowsOS() {
#if defined(_WIN32) || defined(_WIN64)
    return true;
#else
    return false;
#endif
}

TEST(PathSerializationTest, CrossOSPath) {
    file::Path path("/unix/style/path");
    
    std::string serialized = Meta::toStr(path);
    
    // Debug output
    Print("Debug Info: Path = ", path.str());
    Print("Serialized: ", serialized);
    
    if (isWindowsOS()) {
        EXPECT_EQ(serialized, "\"\\\\unix\\\\style\\\\path\"");
    } else {
        EXPECT_EQ(serialized, "\"/unix/style/path\"");
    }
}

TEST(PathSerializationTest, WindowsPath) {
    file::Path path("C:\\windows\\style\\path");
    
    std::string serialized = Meta::toStr(path);
    
    // Debug output
    Print("Debug Info: Path = ", path.str());
    Print("Serialized: ", serialized);
    
    if (isWindowsOS()) {
        EXPECT_EQ(serialized, "\"C:\\\\windows\\\\style\\\\path\"");
    } else {
        EXPECT_EQ(serialized, "\"C:/windows/style/path\"");
    }
}

TEST(PathSerializationTest, RoundTripCrossOS) {
    file::Path original_path("/unix/or/windows/path");
    
    std::string serialized = Meta::toStr(original_path);
    file::Path deserialized_path = Meta::fromStr<file::Path>(serialized);
    
    // Debug output
    Print("Debug Info: Original Path = ", original_path.str());
    Print("Serialized: ", serialized);
    Print("Debug Info: Deserialized Path = ", deserialized_path.str());
    
    if (isWindowsOS()) {
        EXPECT_EQ(deserialized_path.str(), "\\unix\\or\\windows\\path");
    } else {
        EXPECT_EQ(deserialized_path.str(), "/unix/or/windows/path");
    }
}

using namespace util;
TEST(ParseArrayParts, BasicTest) {
    auto result = parse_array_parts("a,b,c");
    ASSERT_EQ(result.size(), 3);
    EXPECT_EQ(result[0], "a");
    EXPECT_EQ(result[1], "b");
    EXPECT_EQ(result[2], "c");
}

TEST(ParseArrayParts, BasicTest2) {
    auto result = parse_array_parts("[a,b,c]");
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0], "[a,b,c]");
}

TEST(ParseArrayParts, WithWhiteSpace) {
    auto result = parse_array_parts(" a , b , c ");
    ASSERT_EQ(result.size(), 3);
    EXPECT_EQ(result[0], "a");
    EXPECT_EQ(result[1], "b");
    EXPECT_EQ(result[2], "c");
}

TEST(ParseArrayParts, WithNestedBrackets) {
    auto result = parse_array_parts("a, [b,c], {d,e}");
    ASSERT_EQ(result.size(), 3);
    EXPECT_EQ(result[0], "a");
    EXPECT_EQ(result[1], "[b,c]");
    EXPECT_EQ(result[2], "{d,e}");
}

TEST(ParseArrayParts, WithEscapedDelimiter) {
    auto result = parse_array_parts(R"(a, "b,c", 'd,e')");
    ASSERT_EQ(result.size(), 3);
    EXPECT_EQ(result[0], "a");
    EXPECT_EQ(result[1], R"("b,c")");
    EXPECT_EQ(result[2], R"('d,e')");
}

TEST(ParseArrayParts, SingleValue) {
    auto result = parse_array_parts("a");
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0], "a");
}

TEST(ParseArrayParts, EmptyString) {
    auto result = parse_array_parts("");
    ASSERT_EQ(result.size(), 0);
}

TEST(ParseArrayParts, OnlyDelimiters) {
    auto result = parse_array_parts(",,");
    ASSERT_EQ(result.size(), 3);
    EXPECT_EQ(result[0], "");
    EXPECT_EQ(result[1], "");
    EXPECT_EQ(result[2], "");
}

TEST(ParseArrayParts, WithDifferentTypes) {
    auto result1 = parse_array_parts(std::string("a,b,c"));
    ASSERT_EQ(result1.size(), 3);

    auto result2 = parse_array_parts("a,b,c");
    ASSERT_EQ(result2.size(), 3);

    std::string_view sv("a,b,c");
    auto result3 = parse_array_parts(sv);
    ASSERT_EQ(result3.size(), 3);
}


TEST(TruncateTests, HandlesCurlyBracesInStdString) {
    std::string input = "{test}";
    auto result = truncate(input);
    ASSERT_EQ(result, "test");
}

TEST(TruncateTests, HandlesSquareBracketsInStdString) {
    std::string input = "[test]";
    auto result = truncate(input);
    ASSERT_EQ(result, "test");
}

TEST(TruncateTests, HandlesCurlyBracesInStdStringView) {
    std::string_view input = "{test}";
    auto result = truncate(input);
    ASSERT_EQ(result, "test");
}

TEST(TruncateTests, HandlesSquareBracketsInStdStringView) {
    std::string_view input = "[test]";
    auto result = truncate(input);
    ASSERT_EQ(result, "test");
}

TEST(TruncateTests, HandlesCurlyBracesInConstCharPointer) {
    const char* input = "{test}";
    auto result = truncate(input);
    ASSERT_EQ(result, "test");
}

TEST(TruncateTests, HandlesSquareBracketsInConstCharPointer) {
    const char* input = "[test]";
    auto result = truncate(input);
    ASSERT_EQ(result, "test");
}

TEST(TruncateTests, HandlesCurlyBracesInConstStdString) {
    const std::string input = "{test}";
    auto result = truncate(input);
    ASSERT_EQ(result, "test");
}

TEST(TruncateTests, HandlesSquareBracketsInConstStdString) {
    const std::string input = "[test]";
    auto result = truncate(input);
    ASSERT_EQ(result, "test");
}

TEST(TruncateTests, HandlesNestedCurlyBraces) {
    std::string input = "{{nested}}";
    auto result = truncate(input);
    ASSERT_EQ(result, "{nested}");
}

TEST(TruncateTests, HandlesNestedSquareBrackets) {
    std::string input = "[[nested]]";
    auto result = truncate(input);
    ASSERT_EQ(result, "[nested]");
}

TEST(TruncateTests, HandlesEmptyCurlyBraces) {
    std::string input = "{}";
    auto result = truncate(input);
    ASSERT_EQ(result, "");
}

TEST(TruncateTests, HandlesEmptySquareBrackets) {
    std::string input = "[]";
    auto result = truncate(input);
    ASSERT_EQ(result, "");
}

TEST(TruncateTests, HandlesWhitespaceInCurlyBraces) {
    std::string input = "{  }";
    auto result = truncate(input);
    ASSERT_EQ(result, "");
}

TEST(TruncateTests, HandlesWhitespaceInSquareBrackets) {
    std::string input = "[  ]";
    auto result = truncate(input);
    ASSERT_EQ(result, "");
}

TEST(ToStringTest, BasicTests) {
    EXPECT_EQ(to_string(1.0), "1");
    EXPECT_EQ(to_string(1.01), "1.01");
    EXPECT_EQ(to_string(1.0100000), "1.01");
    EXPECT_EQ(to_string(0.0), "0");
    EXPECT_EQ(to_string(0.1), "0.1");
    EXPECT_EQ(to_string(0.00001), "0.00001");
    EXPECT_EQ(to_string(100.0), "100");
    EXPECT_EQ(to_string(100.01), "100.01");
    EXPECT_EQ(to_string(100.01000), "100.01");
    EXPECT_EQ(to_string(1080.0), "1080");
}

TEST(ToStringTest, NegativeNumbers) {
    EXPECT_EQ(to_string(-1.0), "-1");
    EXPECT_EQ(to_string(-1.01), "-1.01");
    EXPECT_EQ(to_string(-1.0100000), "-1.01");
    EXPECT_EQ(to_string(-0.1), "-0.1");
    EXPECT_EQ(to_string(-0.00001), "-0.00001");
    EXPECT_EQ(to_string(-100.0), "-100");
    EXPECT_EQ(to_string(-100.01), "-100.01");
    EXPECT_EQ(to_string(-100.01000), "-100.01");
}

TEST(ToStringTest, EdgeCases) {
    EXPECT_EQ(to_string(1.00001), "1.00001");
    EXPECT_EQ(to_string(1.99999), "1.99999");
}

TEST(ToStringTest, ExceptionCase) {
    // Assuming the buffer is of size 128 as in your example.
    // Here we test if a really big float that cannot fit in our buffer would throw the exception.
    float bigNumber = 1e40;
    EXPECT_EQ(to_string(bigNumber), "inf");
}

TEST(ToStringTest, HandleCompileTimeUnsignedInt) {
    // only check using static_assert if not using GCC
#if defined(__GNUC__)
    assert(to_string(1234) == "1234");
    assert(to_string(0) == "0");
#else
    static_assert(to_string(1234) == "1234");
    static_assert(to_string(0) == "0");
#endif
}

TEST(ToStringTest, HandleRunTimeUnsignedInt) {
    std::vector<uint> v{
        9876, 10
    };
    ASSERT_EQ(to_string(v.front()), "9876");
    ASSERT_EQ(to_string(v.back()), "10");
}

TEST(ToStringTest, HandleCompileTimeSignedInt) {
#if defined(__GNUC__)
    assert(to_string(-1234) == "-1234");
    assert(to_string(0) == "0");
#else
    static_assert(to_string(-1234) == "-1234");
    static_assert(to_string(0) == "0");
#endif
}

TEST(ToStringTest, HandleRunTimeSignedInt) {
    ASSERT_EQ(to_string(-9876), "-9876");
    ASSERT_EQ(to_string(-10), "-10");
}

// printto function for gunit automatic printing of conststring_t
namespace cmn::util {
    void PrintTo(const cmn::util::ConstString_t& str, std::ostream* os) {
        *os << str.view();
    }
}

// Test suite for ConstexprString
TEST(ConstexprStringTest, ViewMethod) {
    constexpr ConstString_t myStr("Hello");
    static_assert(myStr.view() == "Hello", "View method failed");
    EXPECT_EQ(myStr.view(), "Hello");
    EXPECT_EQ(myStr.size(), 5);
    EXPECT_EQ(myStr[0], 'H');
    EXPECT_EQ(myStr[1], 'e');
    EXPECT_EQ(myStr[2], 'l');
    EXPECT_EQ(myStr[3], 'l');
    EXPECT_EQ(myStr[4], 'o');
    EXPECT_THROW(myStr[5], std::out_of_range);  // Ensure null terminator at the end
}

TEST(ConstexprStringTest, SizeMethod) {
    constexpr ConstString_t myStr("Hello");
    static_assert(myStr.size() == 5, "Size method failed");
    EXPECT_EQ(myStr.size(), 5);
    EXPECT_THROW(myStr[5], std::out_of_range);  // Ensure null terminator at the end
    static_assert(myStr[5] == 0, "Constant evaluation does not trigger an exception.");
}

TEST(ConstexprStringTest, SquareBracketsOperator) {
    constexpr ConstString_t myStr("Hello");
    static_assert(myStr[1] == 'e', "Square brackets operator failed");
    EXPECT_EQ(myStr[1], 'e');
    EXPECT_THROW(myStr[5], std::out_of_range);  // Ensure null terminator at the end
    static_assert(myStr[5] == 0, "Constant evaluation does not trigger an exception.");
}

TEST(ConstexprStringTest, EqualityOperator) {
    constexpr ConstString_t myStr("Hello");
    constexpr ConstString_t myStr2("Hello");
    static_assert(myStr == myStr2.view(), "Equality operator failed");
    EXPECT_TRUE(myStr == myStr2.view());
    EXPECT_THROW(myStr[5], std::out_of_range);  // Ensure null terminator at the end
    static_assert(myStr[5] == 0, "Constant evaluation does not trigger an exception.");
}

TEST(ConstexprStringTest, ToStringConversion) {
    constexpr ConstString_t myStr("Hello");
    std::string stdStr = static_cast<std::string>(myStr);
    static_assert(myStr == "Hello", "ToString conversion failed");
    EXPECT_EQ(stdStr, "Hello");
    EXPECT_THROW(myStr[5], std::out_of_range);  // Ensure null terminator at the end
    static_assert(myStr[5] == 0, "Constant evaluation does not trigger an exception.");
}

// Additional tests to ensure everything outside the valid range is zero or inaccessible
TEST(ConstexprStringTest, OutOfBoundsAccess) {
    constexpr ConstString_t myStr("Hello");
    EXPECT_EQ(myStr.size(), 5);

    // Check that the elements after the size are zero
    for (size_t i = myStr.size(); i < myStr.capacity(); ++i) {
        EXPECT_EQ(myStr.data()[i], 0);
    }

    // Ensure out-of-bounds access is not possible (depends on implementation, this checks manually)
    EXPECT_THROW(myStr[5], std::out_of_range); // Null terminator at the end
    EXPECT_EQ(myStr.data()[5], 0);
    EXPECT_EQ(myStr.data()[6], 0);
    EXPECT_EQ(myStr.data()[127], 0);  // Last element in the array
}

TEST(ConstexprStringTest, ConstructorFromArray) {
    constexpr const char testStr[] = "Hello";
    constexpr ConstexprString<128> myStr(testStr);
    static_assert(myStr.size() == 5, "Length is incorrect");
    EXPECT_EQ(myStr.size(), 5);
    EXPECT_EQ(myStr.view(), "Hello");
}

TEST(ConstexprStringTest, ConstructorFromStdArray) {
    constexpr std::array<char, 6> testArray = {'H', 'e', 'l', 'l', 'o', '\0'};
    constexpr ConstexprString<128> myStr(testArray);
    static_assert(myStr.size() == 5, "Length is incorrect");
    EXPECT_EQ(myStr.size(), 5);
    EXPECT_EQ(myStr.view(), "Hello");
}

TEST(ConstexprStringTest, ConstructorFromAnotherConstexprString) {
    constexpr const char testStr[] = "World";
    constexpr ConstexprString<128> originalStr(testStr);
    constexpr ConstexprString<128> myStr(originalStr, [](char c) { return c; });
    static_assert(myStr.size() == 5, "Length is incorrect");
    EXPECT_EQ(myStr.size(), 5);
    EXPECT_EQ(myStr.view(), "World");
}

TEST(ConstexprStringTest, ConstructorFromArrayWithFunction) {
    constexpr const char testStr[] = "Hello";
    constexpr ConstexprString<128> myStr(testStr, [](char c) { return c == 'e' ? 'E' : c; });
    static_assert(myStr.size() == 5, "Length is incorrect");
    EXPECT_EQ(myStr.size(), 5);
    EXPECT_EQ(myStr.view(), "HEllo");
}

TEST(ConstexprStringTest, DefaultConstructor) {
    constexpr ConstexprString<128> myStr;
    static_assert(myStr.size() == 0, "Length is incorrect");
    EXPECT_EQ(myStr.size(), 0);
    EXPECT_EQ(myStr.view(), "");
}

TEST(ConstexprStringTest, AppendMethod) {
    constexpr const char testStr1[] = "Hello";
    constexpr const char testStr2[] = "World";
    constexpr ConstexprString<128> str1(testStr1);
    constexpr ConstexprString<128> str2(testStr2);
    constexpr auto result = str1.append(str2);
    static_assert(result.size() == 10, "Length is incorrect");
    EXPECT_EQ(result.size(), 10);
    EXPECT_EQ(result.view(), "HelloWorld");
}

TEST(ConstexprStringTest, AppendCharArrayMethod) {
    constexpr const char testStr1[] = "Hello";
    constexpr const char testStr2[] = "World";
    constexpr ConstexprString<128> str1(testStr1);
    constexpr auto result = str1.append(testStr2);
    static_assert(result.size() == 10, "Length is incorrect");
    EXPECT_EQ(result.size(), 10);
    EXPECT_EQ(result.view(), "HelloWorld");
}

TEST(ConstexprStringTest, FillMethod) {
    constexpr const char testStr[] = "Hello";
    constexpr ConstexprString<128> str(testStr);
    constexpr auto result = str.fill<'*'>();
    static_assert(result.size() == 128, "Length is incorrect");
    EXPECT_EQ(result.size(), 128);

    // Check that the elements after the size are zero
    for (size_t i = 0; i < result.capacity(); ++i) {
        EXPECT_EQ(result.data()[i], '*');
    }
}

TEST(ConstexprStringTest, FillMethodWithZeroTermination) {
    constexpr const char testStr[] = "Hello";
    constexpr ConstexprString<128> str(testStr);
    constexpr auto result = str.fill<'\0'>();
    static_assert(result.size() == 0, "Length should be zero when filled with null characters");
    EXPECT_EQ(result.size(), 0);

    // Ensure zero termination and no further modifications
    EXPECT_EQ(result.data()[0], '\0');

    // Check that the elements after the size are zero
    for (size_t i = result.size() + 1; i < result.capacity(); ++i) {
        EXPECT_EQ(result.data()[i], 0);
    }
}

TEST(ConstexprStringTest, ApplyMethod) {
    constexpr const char testStr[] = "Hello";
    constexpr ConstexprString<128> str(testStr);
    constexpr auto result = str.apply([](char c) { return c == 'e' ? 'E' : c; });
    static_assert(result.size() == 5, "Length is incorrect");
    EXPECT_EQ(result.size(), 5);
    EXPECT_EQ(result.view(), "HEllo");

    // Check that the elements after the size are zero
    for (size_t i = result.size(); i < result.capacity(); ++i) {
        EXPECT_EQ(result.data()[i], 0);
    }
}

TEST(ConstexprStringTest, ApplyMethodWithZeroTermination) {
    constexpr const char testStr[] = "Hello";
    constexpr ConstexprString<128> str(testStr);
    
    size_t call_count = 0;
    auto result = str.apply([&call_count](char c) {
        ++call_count;
        return c == 'e' ? 0 : c;
    });

    EXPECT_EQ(result.size(), 1) << "Length should be truncated to the position of the zero termination";
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(call_count, 2); // Ensure the lambda is called only for 'H' and 'e'

    // Ensure zero termination and no further modifications
    EXPECT_EQ(result.data()[0], 'H');
    EXPECT_EQ(result.data()[1], '\0'); // Ensure the string remains null-terminated
}

TEST(ConstexprStringTest, ApplyToCapacityMethod) {
    constexpr const char testStr[] = "Hello";
    constexpr ConstexprString<128> str(testStr);
    constexpr auto result = str.apply_to_capacity([](char c) { return c == 0 ? '#' : c; });

    // Check the first few characters and the null terminator space
    //static_assert(result.size() == 128, "Length is incorrect");
    EXPECT_EQ(result.size(), 128);
    EXPECT_EQ(result.view().substr(0,5), "Hello");

    // Check that the elements after the valid characters and applied-to capacity are correct
    for (size_t i = 5; i < result.capacity(); ++i) {
        EXPECT_EQ(result.data()[i], '#') << "Result should have set everything to #(i="<<i<<"): " << result.data()[i];
    }
}


TEST(ToStringTest, ConstExprFloat) {
    static constexpr auto sv0i = to_string(0u);
    static_assert(sv0i == "0", "Zero conversion failed");
    
    static constexpr auto sv1000i = to_string(1000u);
    static_assert(sv1000i == "1000", "1000 conversion failed");
    
    static constexpr auto sv1045i = to_string(1045);
    static_assert(sv1045i == "1045", "1045 conversion failed" );
    
    // Simple whole number
    constexpr auto number = 1.0f;
    static constexpr auto sv = to_string(number);
    static_assert(sv == "1", "Simple whole number conversion failed");
    EXPECT_EQ(sv, "1");
    
    // Zero
    static constexpr auto sv0 = to_string(0.f);
    static_assert(sv0 == "0", "Zero conversion failed");

    // Fractional number, no trailing zeros
    static constexpr auto sv1 = to_string(123.456);
    static_assert(sv1 == "123.456", "Fractional number conversion failed");
    EXPECT_EQ(sv1, "123.456");

    // Fractional number with trailing zeros
    static constexpr auto sv2 = to_string(45.600);
    static_assert(sv2 == "45.6", "Fractional number with trailing zeros conversion failed");
    EXPECT_EQ(sv2, "45.6");

    // Negative number
    static constexpr auto sv3 = to_string(-3.21f);
    static_assert(sv3 == "-3.21", "Negative number conversion failed");
    EXPECT_EQ(sv3, "-3.21");

    // NaN
    static constexpr auto nan_float = std::numeric_limits<float>::quiet_NaN();
    static constexpr auto sv4 = to_string(nan_float);
    static_assert(sv4 == "nan", "NaN conversion failed");

    // Infinity
    static constexpr auto inf_float = std::numeric_limits<float>::infinity();
    static constexpr auto sv5 = to_string(inf_float);
    static_assert(sv5 == "inf", "Infinity conversion failed");

    // Negative infinity
    static constexpr auto neg_inf_float = -std::numeric_limits<float>::infinity();
    static constexpr auto sv6 = to_string(neg_inf_float);
    static_assert(sv6 == "-inf", "Negative infinity conversion failed");
    EXPECT_EQ(sv6, "-inf");
}

using namespace settings;
TEST(HtmlifyTests, EmptyDocument) {
    std::string doc = "";
    std::string result = htmlify(doc, false);
    ASSERT_EQ(result, "");
}

TEST(HtmlifyTests, Complex) {
    std::string doc = R"(["/Volumes/Public/work/yolov7-seg.pt","/Volumes/Public/work/yolov7-tiny.pt","/Volumes/Public/work/yolov8l-seg.pt","/Volumes/Public/work/yolov8m-seg.pt","/Volumes/Public/work/yolov8n-seg.pt","/Volumes/Public/work/yolov8n.pt","/Volumes/Public/work/yolov8s-seg.pt","/Volumes/Public/work/yolov8x-seg.pt"])";
    std::string expected = R"([<str>"/Volumes/Public/work/yolov7-seg.pt"</str>,<str>"/Volumes/Public/work/yolov7-tiny.pt"</str>,<str>"/Volumes/Public/work/yolov8l-seg.pt"</str>,<str>"/Volumes/Public/work/yolov8m-seg.pt"</str>,<str>"/Volumes/Public/work/yolov8n-seg.pt"</str>,<str>"/Volumes/Public/work/yolov8n.pt"</str>,<str>"/Volumes/Public/work/yolov8s-seg.pt"</str>,<str>"/Volumes/Public/work/yolov8x-seg.pt"</str>])";
    
    std::string result = htmlify(doc, false);
    ASSERT_EQ(result, expected);
}

TEST(HtmlifyTests, QuotedText) {
    std::string doc = "This \"is 'plain' text\".";
    std::string result = htmlify(doc, false);
    ASSERT_EQ(result, "This <str>\"is 'plain' text\"</str>.");
}

TEST(HtmlifyTests, PlainText) {
    std::string doc = "This is plain text.";
    std::string result = htmlify(doc, false);
    ASSERT_EQ(result, "This is plain text.");
}

TEST(HtmlifyTests, Keywords) {
    std::string doc = "true int";
    std::string result = htmlify(doc, false);
    ASSERT_EQ(result, "<key>true</key> <key>int</key>");
}

TEST(HtmlifyTests, NumberFormatting) {
    std::string doc = "123 45.67";
    std::string result = htmlify(doc, false);
    ASSERT_EQ(result, "<nr>123</nr> <nr>45.67</nr>");
}

TEST(HtmlifyTests, Links) {
    std::string doc = "`http://example.com`";
    std::string result = htmlify(doc, true);
    ASSERT_EQ(result, "<a href=\"http://example.com\" target=\"_blank\">http://example.com</a>");
}

TEST(HtmlifyTests, Newlines) {
    std::string doc = "Line 1\nLine 2";
    std::string result = htmlify(doc, false);
    ASSERT_EQ(result, "Line <nr>1</nr><br/>Line <nr>2</nr>");
}

TEST(HtmlifyTests, Quotations) {
    std::string doc = "'quoted' \"also quoted\"";
    std::string result = htmlify(doc, false);
    ASSERT_EQ(result, "<str>'quoted'</str> <str>\"also quoted\"</str>");
}

TEST(HtmlifyTests, Headings) {
    std::string doc = "$Heading$";
    std::string result = htmlify(doc, false);
    ASSERT_EQ(result, "<h4>Heading</h4>\n");
}

TEST(HtmlifyTests, UnknownHeadings) {
    std::string doc = "<test>Heading</test>";
    std::string result = htmlify(doc, false);
    ASSERT_EQ(result, "&lt;test&gt;Heading&lt;/test&gt;");
}

TEST(ContainsTest, StringAndChar) {
    std::string s = "hello world";
    ASSERT_TRUE(contains(s, 'e'));
    ASSERT_FALSE(contains(s, 'z'));
}

TEST(ContainsTest, StringAndString) {
    std::string s = "hello world";
    ASSERT_TRUE(contains(s, "world"));
    ASSERT_FALSE(contains(s, "universe"));
}

TEST(ContainsTest, CStringAndChar) {
    const char* s = "hello world";
    ASSERT_TRUE(contains(s, 'e'));
    ASSERT_FALSE(contains(s, 'z'));
}

TEST(ContainsTest, CStringAndCString) {
    const char* s = "hello world";
    ASSERT_TRUE(contains(s, "world"));
    ASSERT_FALSE(contains(s, "universe"));
}

TEST(ContainsTest, StringViewAndChar) {
    std::string_view sv = "hello world";
    ASSERT_TRUE(contains(sv, 'e'));
    ASSERT_FALSE(contains(sv, 'z'));
}

TEST(ContainsTest, StringViewAndStringView) {
    std::string_view sv = "hello world";
    std::string_view needle = "world";
    ASSERT_TRUE(contains(sv, needle));
    ASSERT_FALSE(contains(sv, "universe"));
}

TEST(ContainsTest, EmptyStrings) {
    std::string s = "hello world";
    ASSERT_FALSE(contains(s, ""));
    ASSERT_FALSE(contains("", s));
    ASSERT_FALSE(contains("", ""));
}

template<typename T>
class ContainsTest : public ::testing::TestWithParam<T> {
};

using ContainsTestTypes = ::testing::Types<
    std::tuple<std::string, std::string>,
    std::tuple<std::string, const char*>,
    std::tuple<std::string, std::string_view>,
    std::tuple<std::string, char>,
    std::tuple<const char*, std::string>,
    std::tuple<const char*, const char*>,
    std::tuple<const char*, std::string_view>,
    std::tuple<const char*, char>
>;

TYPED_TEST_SUITE(ContainsTest, ContainsTestTypes);

TYPED_TEST(ContainsTest, EmptyCases) {
    using StrType = typename std::tuple_element<0, TypeParam>::type;
    using NeedleType = typename std::tuple_element<1, TypeParam>::type;

    StrType emptyStr{};
    if constexpr(std::is_same_v<StrType, const char*>)
        emptyStr = "";
    NeedleType emptyNeedle{};
    if constexpr(std::is_same_v<NeedleType, const char*>)
        emptyNeedle = "";
    
    EXPECT_FALSE(contains(emptyStr, emptyNeedle));
    EXPECT_FALSE(contains(emptyStr, "needle"));
    EXPECT_FALSE(contains("haystack", emptyNeedle));
}

TYPED_TEST(ContainsTest, SingleCharacterCases) {
    using StrType = typename std::tuple_element<0, TypeParam>::type;
    using NeedleType = typename std::tuple_element<1, TypeParam>::type;

    StrType str{"a"};
    NeedleType needle;
    if constexpr(std::is_same_v<std::remove_cvref_t<NeedleType>, char>)
        needle = 'a';
    else needle = "a";
    
    EXPECT_TRUE(contains(str, needle));
    EXPECT_FALSE(contains(str, "b"));
}

TYPED_TEST(ContainsTest, LongerCases) {
    using StrType = typename std::tuple_element<0, TypeParam>::type;
    using NeedleType = typename std::tuple_element<1, TypeParam>::type;

    if constexpr(std::is_same_v<std::remove_cvref_t<NeedleType>, char>)
        return;
    else {
        StrType str{"abcdef"};
        NeedleType needle{"cd"};
        
        EXPECT_TRUE(contains(str, needle));
        EXPECT_FALSE(contains(str, "gh"));
    }
}

TYPED_TEST(ContainsTest, StartEndCases) {
    using StrType = typename std::tuple_element<0, TypeParam>::type;
    using NeedleType = typename std::tuple_element<1, TypeParam>::type;

    StrType str{"abcdef"};

    if constexpr(std::is_same_v<std::remove_cvref_t<NeedleType>, char>) {
        NeedleType needleStart{'a'};
        NeedleType needleEnd{'f'};
        EXPECT_TRUE(contains(str, needleStart));
        EXPECT_TRUE(contains(str, needleEnd));
    } else {
        NeedleType needleStart{"ab"};
        NeedleType needleEnd{"ef"};
        EXPECT_TRUE(contains(str, needleStart));
        EXPECT_TRUE(contains(str, needleEnd));
    }
}

TYPED_TEST(ContainsTest, MultipleOccurrences) {
    using StrType = typename std::tuple_element<0, TypeParam>::type;
    using NeedleType = typename std::tuple_element<1, TypeParam>::type;

    StrType str{"abcabcabc"};

    if constexpr(std::is_same_v<std::remove_cvref_t<NeedleType>, char>) {
        NeedleType needle{'a'};
        EXPECT_TRUE(contains(str, needle));
    } else {
        NeedleType needle{"abc"};
        EXPECT_TRUE(contains(str, needle));
    }
}

TYPED_TEST(ContainsTest, AllSameCharacters) {
    using StrType = typename std::tuple_element<0, TypeParam>::type;
    using NeedleType = typename std::tuple_element<1, TypeParam>::type;

    StrType str{"aaaaaa"};

    if constexpr(std::is_same_v<std::remove_cvref_t<NeedleType>, char>) {
        NeedleType needle{'a'};
        EXPECT_TRUE(contains(str, needle));
    } else {
        NeedleType needle{"aaa"};
        EXPECT_TRUE(contains(str, needle));
    }
}

TYPED_TEST(ContainsTest, NeedleLongerThanStr) {
    using StrType = typename std::tuple_element<0, TypeParam>::type;
    using NeedleType = typename std::tuple_element<1, TypeParam>::type;

    StrType str{"short"};

    if constexpr(not std::is_same_v<std::remove_cvref_t<NeedleType>, char>) {
        NeedleType needle{"very long needle"};
        EXPECT_FALSE(contains(str, needle));
    } else {
        // Do nothing, not applicable for char type
    }
}

// Test suite for to_string function template
TEST(ToStringTest, HandleNormalFloat) {
    auto str = to_string(123.456);
    EXPECT_EQ(str, "123.456") << str.view() << " != " << "123.456";
}

TEST(ToStringTest, HandleNaN) {
    float nanValue = std::numeric_limits<float>::quiet_NaN();
    auto str = to_string(nanValue);
    EXPECT_EQ(str, "nan");
}

TEST(ToStringTest, HandleInfinity) {
    float infValue = std::numeric_limits<float>::infinity();
    auto str = to_string(infValue);
    EXPECT_EQ(str, "inf");
}

TEST(ToStringTest, PrecisionTest) {
    auto str = to_string(0.0001);
    EXPECT_EQ(str, "0.0001");
    
    str = to_string(0.00001);
    EXPECT_EQ(str, "0.00001");
    
    str = to_string(0.0001f);
    EXPECT_EQ(str, "0.0001");
    
    str = to_string(9.999999f);
    EXPECT_EQ(str, "9.99999");
    
    str = to_string(9.9999);
    EXPECT_EQ(str, "9.9999");
    
    str = to_string(9.9999f);
    EXPECT_EQ(str, "9.99989"); /// float cant handle this
    
    str = to_string(99.99999);
    EXPECT_EQ(str, "99.99999");
    
    str = to_string(99.99999f);
    EXPECT_EQ(str, "99.99999");
    
    str = to_string(9999.99999999);
    EXPECT_EQ(str, "10000"); /// too many decimals
    
    str = to_string(9999.99999999f);
    EXPECT_EQ(str, "10000"); /// same here
    
    str = to_string(0.99999);
    EXPECT_EQ(str, "0.99999");
    
    str = to_string(1.23456789);
    EXPECT_EQ(str, "1.23456"); /// precision limited to 6 decimal places
    
    str = to_string(-1.23456789);
    EXPECT_EQ(str, "-1.23456"); /// precision limited to 6 decimal places

    str = to_string(1.23456789f);
    EXPECT_EQ(str, "1.23456"); /// same as above but for float

    str = to_string(123456789.0);
    EXPECT_EQ(str, "123456789"); /// scientific notation for large numbers

    str = to_string(1234567.0f);
    EXPECT_EQ(str, "1234567"); /// same as above but for float

    str = to_string(0.000000123456789);
    EXPECT_EQ(str, "0"); /// small number is too small

    str = to_string(0.000000123456789f);
    EXPECT_EQ(str, "0"); /// same as above but for float

    str = to_string(1.0 / 3.0);
    EXPECT_EQ(str, "0.33333"); /// repeating decimal limited to precision

    str = to_string(static_cast<float>(1.0 / 3.0));
    EXPECT_EQ(str, "0.33333"); /// same as above but for float

    str = to_string(2.0 / 3.0);
    EXPECT_EQ(str, "0.66666"); /// another repeating decimal

    str = to_string(static_cast<float>(2.0 / 3.0));
    EXPECT_EQ(str, "0.66666"); /// same as above but for float

    str = to_string(1.7976931348623157e+308); // Max double
    EXPECT_EQ(str, "inf");

    str = to_string(3.4028235e+38f); // Max float
    EXPECT_EQ(str, "inf");
    str = to_string(-3.4028235e+38f); // Max float
    EXPECT_EQ(str, "-inf");

    str = to_string(5e-324); // Min positive double
    EXPECT_EQ(str, "0");

    str = to_string(1.4e-45f); // Min positive float
    EXPECT_EQ(str, "0");

    str = to_string(0.1 + 0.2);
    EXPECT_EQ(str, "0.3"); // Precision test for floating point addition

    str = to_string(0.1f + 0.2f);
    EXPECT_EQ(str, "0.3"); // Same as above for float

    str = to_string(-0.0000001);
    EXPECT_EQ(str, "-0"); // Negative small number

    str = to_string(-0.0000001f);
    EXPECT_EQ(str, "-0"); // Same as above for float
}

TEST(StaticParseTest, invalid_label) {
    std::string fail = R"(<c><b><nr>1</nr></b>:<unknown:0></c> (p=<i>0.92 s=16.76</i>) (Drawable*)Label<[460.19254,142.63751,843.00757,832.82635], '<c><b><nr>1</nr></b>:<unknown:0></c> (p=<i>0.92 s=16.93</i>) (Drawable*)Label<[460.19254,142.63751,843.00757,832.82635], '<c><b><nr>1</nr></b>:<unknown:0></c> (p=<i>0.92 s=16.93</i>) (Drawable*)Label<[460.19254,142.63751,843.00757,832.82635], '<c><b><nr>1</nr></b>:<unknown:0></c> (p=<i>0.92 s=16.93</i>) (Drawable*)Label<[464.265,144.67377,836.8988,830.7901], '<c><b><nr>1</nr></b>:<unknown:0></c> (p=<i>0.92 s=16.76</i>) (Drawable*)Label<[464.265,146.71002,836.8988,828.75385], '<c><b><nr>1</nr></b>:<unknown:0></c> (p=<i>0.92 s=16.72</i>) (Drawable*)Label<[464.265,146.71002,836.8988,828.75385], '<c><b><nr>1</nr></b>:<unknown:0></c> (p=<i>0.92 s=16.72</i>) (Drawable*)Label<[460.19254,142.63751,843.00757,832.82635], '<c><b><nr>1</nr></b>:<unknown:0></c> (p=<i>0.92 s=16.93</i>) (Drawable*)Label<[460.19254,142.63751,843.00757,832.82635], '<c><b><nr>1</nr></b>:<unknown:0></c> (p=<i>0.92 s=16.93</i>) (Drawable*)Label<[460.19254,144.67377,843.00757,830.7901], '<c><b><nr>1</nr></b>:<unknown:0></c> (p=<i>0.92 s=16.89</i>) (Drawable*)Label<[460.19254,144.67377,843.00757,830.7901], '<c><b><nr>1</nr></b>:<unknown:0></c> (p=<i>0.92 s=16.89</i>) (Drawable*)Label<[460.19254,144.67377,843.00757,830.7901], '<c><b><nr>1</nr></b>:<unknown:0></c> (p=<i>0.92 s=16.89</i>) (Drawable*)Label<[464.265,142.63751,836.8988,832.82635], '<c><b><nr>1</nr></b>:<unknown:0></c> (p=<i>0.92 s=16.80</i>) (Drawable*)Label<[460.19254,140.60126,840.9713,834.8626], '<c><b><nr>1</nr></b>:<unknown:0></c> (p=<i>0.92 s=16.93</i>) (Drawable*)Label<[460.19254,140.60126,840.9713,834.8626], '<c><b><nr>1</nr></b>:<unknown:0></c> (p=<i>0.92 s=16.93</i>) (Drawable*)Label<[464.265,142.63751,836.8988,832.82635], '<c><b><nr>1</nr></b>:<unknown:0></c> (p=<i>0.92 s=16.80</i>) (Drawable*)Label<[464.265,142.63751,836.8988,832.82635], '<c><b><nr>1</nr></b>:<unknown:0></c> (p=<i>0.92 s=16.80</i>) (Drawable*)Label<[460.19254,138.56502,843.00757,836.89886], '<c><b><nr>1</nr></b>:<unknown:0></c> (p=<i>0.92 s=17.01</i>) (Drawable*)Label<[460.19254,138.56502,843.00757,836.89886], '<c><b><nr>1</nr></b>:<unknown:0></c> (p=<i>0.92 s=17.01</i>) (Drawable*)Label<[464.265,140.60126,836.8988,834.8626], '<c><b><nr>1</nr></b>:<unknown:0></c> (p=<i>0.92 s=16.85</i>) (Drawable*)Label<[464.265,142.63751,836.8988,832.82635], '<c><b><nr>1</nr></b>:<unknown:0></c> (p=<i>0.92 s=16.80</i>) (Drawable*)Label<[464.265,142.63751,836.8988,832.82635], '<c><b><nr>1</nr></b>:<unknown:0></c> (p=<i>0.92 s=16.80</i>) (Drawable*)Label<[464.265,142.63751,836.8988,832.82635], '<c><b><nr>1</nr></b>:<unknown:0></c> (p=<i>0.92 s=16.80</i>) (Drawable*)Label<[464.265,142.63751,836.8988,832.82635], '<c><b><nr>1</nr></b>:<unknown:0></c> (p=<i>0.92 s=16.80</i>) (Drawable*)Label<[464.265,142.63751,836.8988,832.82635], '<c><b><nr>1</nr></b>:<unknown:0></c> (p=<i>0.92 s=16.80</i>) (Drawable*)Label<[464.265,142.63751,836.8988,832.82635], '<c><b><nr>1</nr></b>:<unknown:0></c> (p=<i>0.92 s=16.80</i>) (Drawable*)Label<[464.265,142.63751,836.8988,832.82635], '<c><b><nr>1</nr></b>:<unknown:0></c> (p=<i>0.92 s=16.80</i>) (Drawable*)Label<[464.265,140.60126,836.8988,834.8626], '<c><b><nr>1</nr></b>:<unknown:0></c> (p=<i>0.92 s=16.85</i>) (Drawable*)Label<[462.2288,142.63751,838.93506,832.82635], '<c><b><nr>1</nr></b>:<unknown:0></c> (p=<i>0.92 s=16.85</i>) (Drawable*)Label<[462.2288,142.63751,838.93506,832.82635], '<c><b><nr>1</nr></b>:<unknown:0></c> (p=<i>0.92 s=16.85</i>) (Drawable*)Label<[460.19254,140.60126,843.00757,834.8626], '<c><b><nr>1</nr></b>:<unknown:0></c> (p=<i>0.92 s=16.97</i>) (Drawable*)Label<[460.19254,140.60126,843.00757,834.8626], '<c><b><nr>1</nr></b>:<unknown:0></c> (p=<i>0.92 s=16.97</i>) (Drawable*)Label<[462.2288,142.63751,840.9713,832.82635], '<c><b><nr>1</nr></b>:<unknown:0></c> (p=<i>0.92 s=16.89</i>) (Drawable*)Label<[462.2288,142.63751,840.9713,832.82635], '<c><b><nr>1</nr></b>:<unknown:0></c> (p=<i>0.92 s=16.89</i>) (Drawable*)Label<[0,0,0,0], ''>'>'>'>'>'>'>'>'>'>'>'>'>'>'>'>'>'>'>'>'>'>'>'>'>'>'>'>'>'>'>'>'>'>)";

    using namespace gui;
    auto ranges = StaticText::to_tranges(fail);

}
