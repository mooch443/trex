#include "gtest/gtest.h"
#include <commons.pc.h>
#include <misc/parse_parameter_lists.h>
#include <misc/format.h>
#include <misc/Timer.h>
#include <file/PathArray.h>
#include <gmock/gmock.h>
#include <misc/checked_casts.h>
#include <file/Path.h>

#ifdef NDEBUG
#undef NDEBUG
#endif

using namespace file;
bool matched;
using namespace cmn;

// Integer to Integer Conversions
TEST(NarrowCast, IntToIntSameSignedness) {
    EXPECT_TRUE(check_narrow_cast<int>(12345L));  // long to int, within range
    EXPECT_TRUE(check_narrow_cast<short>(-12345));  // int to short, within range
    unsigned int largeUnsigned = static_cast<unsigned int>(std::numeric_limits<int>::max()) + 1;
    EXPECT_FALSE(check_narrow_cast<int>(largeUnsigned));  // unsigned int to int, out of range
}

TEST(NarrowCast, SignedToUnsigned) {
    EXPECT_TRUE(check_narrow_cast<unsigned int>(12345));  // int to unsigned int, positive value
    EXPECT_FALSE(check_narrow_cast<unsigned int>(-1));  // int to unsigned int, negative value
}

TEST(NarrowCast, UnsignedToSigned) {
    EXPECT_TRUE(check_narrow_cast<int>(12345u));  // unsigned int to int, within range
    EXPECT_FALSE(check_narrow_cast<int>(std::numeric_limits<unsigned int>::max()));  // unsigned int to int, out of range
}

// Floating-Point to Integer Conversions
TEST(NarrowCast, FloatToSignedInt) {
    EXPECT_TRUE(check_narrow_cast<int>(12345.67f));  // float to int, within range
    EXPECT_TRUE(check_narrow_cast<int>(-12345.67f));  // float to int, within range
    EXPECT_FALSE(check_narrow_cast<int>(static_cast<float>(std::numeric_limits<int>::max()) + 10000.0f));  // float to int, out of range
}

TEST(NarrowCast, FloatToUnsignedInt) {
    EXPECT_TRUE(check_narrow_cast<unsigned int>(12345.67f));  // float to unsigned int, within range
    EXPECT_FALSE(check_narrow_cast<unsigned int>(-1.0f));  // float to unsigned int, negative
    EXPECT_FALSE(check_narrow_cast<unsigned int>(static_cast<float>(std::numeric_limits<unsigned int>::max()) + 10000.0f));  // float to unsigned int, out of range
}

// Integer to Floating-Point Conversions
TEST(NarrowCast, SignedIntToFloat) {
    EXPECT_TRUE(check_narrow_cast<float>(12345));  // int to float, within range
    EXPECT_TRUE(check_narrow_cast<float>(-12345));  // int to float, within range
}

TEST(NarrowCast, UnsignedIntToFloat) {
    EXPECT_TRUE(check_narrow_cast<float>(12345u));  // unsigned int to float, within range
}

// Floating-Point to Floating-Point Conversions
TEST(NarrowCast, FloatToDouble) {
    EXPECT_TRUE(check_narrow_cast<double>(123.456f));  // float to double, valid conversion
    EXPECT_TRUE(check_narrow_cast<double>(-123.456f));  // float to double, valid conversion
}

// Edge Cases
TEST(NarrowCast, EdgeCases) {
    EXPECT_FALSE(check_narrow_cast<int>(static_cast<int64_t>(std::numeric_limits<int64_t>::min())));  // long to int, out of range
    EXPECT_FALSE(check_narrow_cast<int>(static_cast<int64_t>(std::numeric_limits<int64_t>::max())));  // long to int, out of range
    EXPECT_TRUE(check_narrow_cast<long>(std::numeric_limits<int>::min()));  // int to long, within range
    EXPECT_TRUE(check_narrow_cast<long>(std::numeric_limits<int>::max()));  // int to long, within range
}

TEST(PathArrayTest, AddPath) {
    file::_PathArray<> pa;
#if defined(_WIN32)
    pa.add_path("C:\\path\\to\\file1");
    pa.add_path("C:\\path\\to\\file2");
#else
    pa.add_path("/path/to/file1");
    pa.add_path("/path/to/file2");
#endif
    EXPECT_FALSE(pa.empty());
}

TEST(PathArrayTest, IteratorBeginAndEnd) {
    file::_PathArray<> pa;
#if defined(_WIN32)
    pa.add_path("C:\\path\\to\\file1");
    pa.add_path("C:\\path\\to\\file2");
#else
    pa.add_path("/path/to/file1");
    pa.add_path("/path/to/file2");
#endif
    auto it = pa.begin();
#if defined(_WIN32)
    EXPECT_EQ(it->str(), "C:\\path\\to\\file1"); // Assuming file::Path has a str() function to return its string representation
#else
    EXPECT_EQ(it->str(), "/path/to/file1"); // Assuming file::Path has a str() function to return its string representation
#endif
    ++it;
#if defined(_WIN32)
    EXPECT_EQ(it->str(), "C:\\path\\to\\file2");
#else
    EXPECT_EQ(it->str(), "/path/to/file2");
#endif
}

TEST(PathArrayTest, GetPaths) {
    file::_PathArray<> pa;
#if defined(_WIN32)
    pa.add_path("C:\\path\\to\\file1");
#else
    pa.add_path("/path/to/file1");
#endif
    const auto& paths = pa.get_paths();
    EXPECT_EQ(paths.size(), 1);
#if defined(_WIN32)
    EXPECT_EQ(paths[0].str(), "C:\\path\\to\\file1"); // Again, assuming file::Path has a str() function
#else
    EXPECT_EQ(paths[0].str(), "/path/to/file1"); // Again, assuming file::Path has a str() function
#endif
}

TEST(PathArrayTest, ConstructorAndEmpty) {
    // Custom mock filesystem for this specific test case
    struct LocalMockFilesystem : public file::FilesystemInterface {
        std::set<file::Path> find_files(const file::Path& path) const override {
            // Mock implementation specific to this test
            return {};
        }

        bool is_folder(const file::Path& path) const override {
            // Mock implementation specific to this test
            return true;
        }
        
        bool exists(const file::Path& path) const override {
            // Mock implementation specific to this test
            return false;
        }
    };

    // Using local mock filesystem
    file::_PathArray<LocalMockFilesystem> pa;
    EXPECT_TRUE(pa.empty());
}

template<typename FS>
std::vector<file::Path> resolve_paths_artificially(std::string pattern) {
    using Defer = file::_PathArray<FS>::DeferredFileCheck;

    std::vector<Defer> checks;
    std::optional<std::vector<std::string>> to_be_resolved;
    bool has_to_be_filtered{ false };
    FS fs{};
    auto parsed_paths = file::_PathArray<FS>::parse_path(pattern, matched, to_be_resolved, checks, has_to_be_filtered);
    file::_PathArray<FS>::ensure_loaded_static(parsed_paths, checks, to_be_resolved, has_to_be_filtered, fs);
    return parsed_paths;
}

TEST(PathArrayTest, ParsePath) {
    // Custom mock filesystem for this specific test case
    struct LocalMockFilesystem : public file::FilesystemInterface {
        std::set<file::Path> find_files(const file::Path& path) const override {
            // Mock implementation specific to this test
#if defined(_WIN32)
            return { file::Path("C:\\path\\to\\file00"), file::Path("C:\\path\\to\\file01"), file::Path("C:\\path\\to\\file02") };
#else
            return { file::Path("/path/to/file00"), file::Path("/path/to/file01"), file::Path("/path/to/file02") };
#endif
        }

        bool is_folder(const file::Path& path) const override {
            // Mock implementation specific to this test
            return true;
        }
        
        bool exists(const file::Path& path) const override {
            // Mock implementation specific to this test
            return contains(find_files(""), path);
        }
    };

    // Custom mock filesystem for this specific test case
    LocalMockFilesystem mockFs;

#if defined(_WIN32)
    print("Parsing path: C:\\path\\to\\file%0.2d => ", mockFs.find_files(""));
    auto parsed_paths = resolve_paths_artificially<LocalMockFilesystem>("C:\\path\\to\\file%0.2d");
    EXPECT_EQ(parsed_paths[0].str(), "C:\\path\\to\\file00");
    EXPECT_EQ(parsed_paths[1].str(), "C:\\path\\to\\file01");
    EXPECT_EQ(parsed_paths[2].str(), "C:\\path\\to\\file02");
#else
    print("Parsing path: /path/to/file%0.2d => ", mockFs.find_files(""));
    auto parsed_paths = resolve_paths_artificially<LocalMockFilesystem>("/path/to/file%0.2d");
    EXPECT_EQ(parsed_paths[0].str(), "/path/to/file00");
    EXPECT_EQ(parsed_paths[1].str(), "/path/to/file01");
    EXPECT_EQ(parsed_paths[2].str(), "/path/to/file02");
#endif

    EXPECT_EQ(parsed_paths.size(), 3);
}

// Test pattern matching with %10.100.6d
#if defined(_WIN32)
#include <windows.h>
#endif

#include <gtest/gtest.h>
// ... other includes ...

TEST(PathArrayTest, ParsePath_ConsecutiveFiles_10_100) {
    struct LocalMockFilesystem : public file::FilesystemInterface {
        std::set<file::Path> find_files(const file::Path& path) const override {
            // Generate mock files from 10 to 100
            std::set<file::Path> mock_files;
            for (int i = 10; i <= 100; ++i) {
                std::stringstream ss;
                ss << std::setw(6) << std::setfill('0') << i;  // Zero padding to 6 digits
#if defined(_WIN32)
                mock_files.insert(file::Path("C:\\path\\to\\file" + ss.str() + ".mp4"));
#else
                mock_files.insert(file::Path("/path/to/file" + ss.str() + ".mp4"));
#endif
            }
            return mock_files;
        }

        bool is_folder(const file::Path& path) const override {
            if(path == "C:\\path\\to" || path == "/path/to")
                return true;
            if(path == "C:\\path" || path == "/path")
				return true;
            if(path == "C:\\" || path == "/")
                return true;
            return false;
        }

        bool exists(const file::Path& path) const override {
            return find_files(path).count(path) > 0;
        }
    };
     
#if defined(_WIN32)
    auto parsed_paths = resolve_paths_artificially<LocalMockFilesystem>("C:\\path\\to\\file%10.100.6d.mp4");
#else
    auto parsed_paths = resolve_paths_artificially<LocalMockFilesystem>("/path/to/file%10.100.6d.mp4");
#endif

    EXPECT_EQ(parsed_paths.size(), 91); // 100 - 10 + 1 = 91 files

    for (size_t i = 0; i < parsed_paths.size(); ++i) {
        // Generate the expected file name
        int file_number = 10 + i; // Starting from 10
        std::stringstream ss;
        ss << std::setw(6) << std::setfill('0') << file_number;  // Zero padding to 6 digits

#if defined(_WIN32)
        std::string expected_file_name = "C:\\path\\to\\file" + ss.str() + ".mp4";
#else
        std::string expected_file_name = "/path/to/file" + ss.str() + ".mp4";
#endif

        // Compare
        EXPECT_EQ(parsed_paths[i].str(), expected_file_name);
    }
}

// Test pattern matching with %10.3d
TEST(PathArrayTest, ParsePath_From10ToEnd) {
    struct LocalMockFilesystem : public file::FilesystemInterface {
        std::set<file::Path> find_files(const file::Path& path) const override {
#if defined(_WIN32)
            return { file::Path("C:\\path\\to\\file010"), file::Path("C:\\path\\to\\file011") };
#else
            return { file::Path("/path/to/file010"), file::Path("/path/to/file011") };
#endif
        }
        bool is_folder(const file::Path& path) const override {
            return true;
        }
        bool exists(const file::Path& path) const override {
            return find_files(path).count(path) > 0;
        }
    };

#if defined(_WIN32)
    auto parsed_paths = resolve_paths_artificially<LocalMockFilesystem>("C:\\path\\to\\file%10.3d");
    EXPECT_EQ(parsed_paths[0].str(), "C:\\path\\to\\file010");
    EXPECT_EQ(parsed_paths[1].str(), "C:\\path\\to\\file011");
#else
    auto parsed_paths = resolve_paths_artificially<LocalMockFilesystem>("/path/to/file%10.3d");
    EXPECT_EQ(parsed_paths[0].str(), "/path/to/file010");
    EXPECT_EQ(parsed_paths[1].str(), "/path/to/file011");
#endif

    EXPECT_EQ(parsed_paths.size(), 2);
}

// Custom mock filesystem for this specific test case
struct RootMockFilesystem : public file::FilesystemInterface {
    std::set<file::Path> find_files(const file::Path& path) const override {
#if defined(_WIN32)
        if (path == file::Path("C:\\")) {
            return { file::Path("C:\\file1.txt"), file::Path("C:\\folder1") };
        }
#else
        if (path == file::Path("/")) {
            return { file::Path("/file1.txt"), file::Path("/folder1") };
        }
#endif
        return {};
    }

    bool is_folder(const file::Path& path) const override {
        return path == file::Path("C:\\") || path == file::Path("/");
    }

    bool exists(const file::Path& path) const override {
        return find_files(path.remove_filename()).contains(path) > 0;
    }
};

TEST(PathArrayTest, RootFolderTest) {
    file::_PathArray<RootMockFilesystem> pa;
#if defined(_WIN32)
    pa.add_path("C:\\*");
#else
    pa.add_path("/*");
#endif
    const auto& paths = pa.get_paths();
    EXPECT_EQ(paths.size(), 2);
#if defined(_WIN32)
    EXPECT_EQ(paths[0].str(), "C:\\file1.txt");
    EXPECT_EQ(paths[1].str(), "C:\\folder1");
#else
    EXPECT_EQ(paths[0].str(), "/file1.txt");
    EXPECT_EQ(paths[1].str(), "/folder1");
#endif
}

// Test pattern matching with %3d
TEST(PathArrayTest, ParsePath_3DigitsPadded) {
    struct LocalMockFilesystem : public file::FilesystemInterface {
        std::set<file::Path> find_files(const file::Path& path) const override {
#if defined(_WIN32)
            return { file::Path("C:\\path\\to\\file000"), file::Path("C:\\path\\to\\file001") };
#else
            return { file::Path("/path/to/file000"), file::Path("/path/to/file001") };
#endif
        }
        bool is_folder(const file::Path& path) const override {
            return true;
        }
        bool exists(const file::Path& path) const override {
            return find_files(path).count(path) > 0;
        }
    };

#if defined(_WIN32)
    auto parsed_paths = resolve_paths_artificially<LocalMockFilesystem>("C:\\path\\to\\file%3d");
    EXPECT_EQ(parsed_paths[0].str(), "C:\\path\\to\\file000");
    EXPECT_EQ(parsed_paths[1].str(), "C:\\path\\to\\file001");
#else
    auto parsed_paths = resolve_paths_artificially<LocalMockFilesystem>("/path/to/file%3d");
    EXPECT_EQ(parsed_paths[0].str(), "/path/to/file000");
    EXPECT_EQ(parsed_paths[1].str(), "/path/to/file001");
#endif

    EXPECT_EQ(parsed_paths.size(), 2);
}

// Test pattern matching with %03d (should be the same as %3d)
TEST(PathArrayTest, ParsePath_03DigitsPadded) {
    // Custom mock filesystem for this specific test case
    struct LocalMockFilesystem : public file::FilesystemInterface {
        std::set<file::Path> find_files(const file::Path& path) const override {
#if defined(_WIN32)
            return { file::Path("C:\\path\\to\\file000"), file::Path("C:\\path\\to\\file001") };
#else
            return { file::Path("/path/to/file000"), file::Path("/path/to/file001") };
#endif
        }
        bool is_folder(const file::Path& path) const override {
            return true;
        }
        bool exists(const file::Path& path) const override {
            return find_files(path).count(path) > 0;
        }
    };

#if defined(_WIN32)
    auto parsed_paths = resolve_paths_artificially<LocalMockFilesystem>("C:\\path\\to\\file%03d");
    EXPECT_EQ(parsed_paths[0].str(), "C:\\path\\to\\file000");
    EXPECT_EQ(parsed_paths[1].str(), "C:\\path\\to\\file001");
#else
    auto parsed_paths = resolve_paths_artificially<LocalMockFilesystem>("/path/to/file%03d");
    EXPECT_EQ(parsed_paths[0].str(), "/path/to/file000");
    EXPECT_EQ(parsed_paths[1].str(), "/path/to/file001");
#endif

    EXPECT_EQ(parsed_paths.size(), 2);
}

// Test pattern matching with filenames having spaces
TEST(PathArrayTest, ParsePath_FilenamesWithSpaces) {
    // Custom mock filesystem for this specific test case
    struct LocalMockFilesystem : public file::FilesystemInterface {
        std::set<file::Path> find_files(const file::Path& path) const override {
#if defined(_WIN32)
            return { file::Path("C:\\path to\\file 000"), file::Path("C:\\path to\\file 001") };
#else
            return { file::Path("/path to/file 000"), file::Path("/path to/file 001") };
#endif
        }
        bool is_folder(const file::Path& path) const override {
            return true;
        }
        bool exists(const file::Path& path) const override {
            return find_files(path).count(path) > 0;
        }
    };

#if defined(_WIN32)
    auto parsed_paths = resolve_paths_artificially<LocalMockFilesystem>("C:\\path to\\file %3d");
    EXPECT_EQ(parsed_paths[0].str(), "C:\\path to\\file 000");
    EXPECT_EQ(parsed_paths[1].str(), "C:\\path to\\file 001");
#else
    auto parsed_paths = resolve_paths_artificially<LocalMockFilesystem>("/path to/file %3d");
    EXPECT_EQ(parsed_paths[0].str(), "/path to/file 000");
    EXPECT_EQ(parsed_paths[1].str(), "/path to/file 001");
#endif

    EXPECT_EQ(parsed_paths.size(), 2);
}

// Test pattern matching with %[10,20,30]
TEST(PathArrayTest, ParsePath_Array) {
    struct LocalMockFilesystem : public file::FilesystemInterface {
        std::set<file::Path> find_files(const file::Path& path) const override {
#if defined(_WIN32)
            return { file::Path("C:\\path\\to\\file010"), file::Path("C:\\path\\to\\file020"), file::Path("C:\\path\\to\\file030") };
#else
            return { file::Path("/path/to/file010"), file::Path("/path/to/file020"), file::Path("/path/to/file030") };
#endif
        }
        bool is_folder(const file::Path& path) const override {
            return true;
        }
        bool exists(const file::Path& path) const override {
            return find_files(path).count(path) > 0;
        }
    };

    // Custom mock filesystem for this specific test case
    LocalMockFilesystem mockFs;

/*#if defined(_WIN32)
    auto parsed_paths = resolve_paths_artificially<LocalMockFilesystem>("C:\\path\\to\\file%10.30.3d");
    EXPECT_EQ(parsed_paths[0].str(), "C:\\path\\to\\file010");
    EXPECT_EQ(parsed_paths[1].str(), "C:\\path\\to\\file020");
    EXPECT_EQ(parsed_paths[2].str(), "C:\\path\\to\\file030");
#else
    auto parsed_paths = resolve_paths_artificially<LocalMockFilesystem>("/path/to/file%[10,20,30]");
    EXPECT_EQ(parsed_paths[0].str(), "/path/to/file010");
    EXPECT_EQ(parsed_paths[1].str(), "/path/to/file020");
    EXPECT_EQ(parsed_paths[2].str(), "/path/to/file030");
#endif

    EXPECT_EQ(parsed_paths.size(), 3);*/
}

// Test pattern matching with array
TEST(PathArrayTest, ParsePath_ArrayFormat) {
    struct LocalMockFilesystem : public file::FilesystemInterface {
        std::set<file::Path> find_files(const file::Path& path) const override {
#if defined(_WIN32)
            return { file::Path("path\\to\\file1"), file::Path("C:\\other\\path") };
#else
            return { file::Path("path/to/file1"), file::Path("/other/path") };
#endif
        }

        bool is_folder(const file::Path& path) const override {
            return true;
        }

        bool exists(const file::Path& path) const override {
            return find_files("").count(path) > 0;
        }
    };
#if defined(_WIN32)
    auto parsed_paths = file::_PathArray<LocalMockFilesystem>("[\"path\\to\\file1\",\"C:\\other\\path\"]");
#else
    auto parsed_paths = file::_PathArray<LocalMockFilesystem>("[\"path/to/file1\",\"/other/path\"]");
#endif

    EXPECT_EQ(parsed_paths.size(), 2);

#if defined(_WIN32)
    EXPECT_EQ(parsed_paths[0].str(), "path\\to\\file1");
    EXPECT_EQ(parsed_paths[1].str(), "C:\\other\\path");
#else
    EXPECT_EQ(parsed_paths[0].str(), "path/to/file1");
    EXPECT_EQ(parsed_paths[1].str(), "/other/path");
#endif
}

// Test pattern matching with *
TEST(PathArrayTest, ParsePath_Star) {
    struct LocalMockFilesystem : public file::FilesystemInterface {
        std::set<file::Path> find_files(const file::Path& path) const override {
#if defined(_WIN32)
            return { file::Path("C:\\path\\to\\file1"), file::Path("C:\\path\\to\\file2"), file::Path("C:\\path\\to\\file3") };
#else
            return { file::Path("/path/to/file1"), file::Path("/path/to/file2"), file::Path("/path/to/file3") };
#endif
        }
        bool is_folder(const file::Path& path) const override {
            return true;
        }
        bool exists(const file::Path& path) const override {
            return find_files(path).count(path) > 0;
        }
    };

#if defined(_WIN32)
    auto parsed_paths = resolve_paths_artificially<LocalMockFilesystem>("C:\\path\\to\\file*");
    EXPECT_EQ(parsed_paths[0].str(), "C:\\path\\to\\file1");
    EXPECT_EQ(parsed_paths[1].str(), "C:\\path\\to\\file2");
    EXPECT_EQ(parsed_paths[2].str(), "C:\\path\\to\\file3");
#else
    auto parsed_paths = resolve_paths_artificially<LocalMockFilesystem>("/path/to/file*");
    EXPECT_EQ(parsed_paths[0].str(), "/path/to/file1");
    EXPECT_EQ(parsed_paths[1].str(), "/path/to/file2");
    EXPECT_EQ(parsed_paths[2].str(), "/path/to/file3");
#endif

    EXPECT_EQ(parsed_paths.size(), 3);
}

TEST(FindBasenameTest, EmptyPathArray) {
  file::PathArray emptyArray;
  EXPECT_EQ("", find_basename(emptyArray));
}

TEST(FindBasenameTest, SingleElement) {
  file::PathArray singleElement = { "/path/to/file.txt" };
  EXPECT_EQ("file", find_basename(singleElement));
}

TEST(FindBasenameTest, MultipleElements) {
    file::PathArray multipleElements{
        { "/path/to/file.txt", "/path/to/another_file.txt", "/path/to/yet_another.txt" }
    };
  // Replace with your expected result
  EXPECT_EQ("to", find_basename(multipleElements));
}

TEST(FindBasenameTest, MultipleElementsAndSamePrefix) {
    file::PathArray multipleElements{
        { "/path/to/file0001.txt", "/path/to/file0002.txt", "/path/to/file0005.txt" }
    };
  // Replace with your expected result
  EXPECT_EQ("to", find_basename(multipleElements));
}

TEST(FindBasenameTest, DifferentDirectories) {
    file::PathArray differentDirs{
        { "/first/path/to/file.txt", "/second/path/to/file.txt" }
    };
  // Replace with your expected result
  EXPECT_EQ("file", find_basename(differentDirs));
}

TEST(SanitizeFilenameTest, EmptyString) {
  EXPECT_EQ("", sanitize_filename(""));
}

TEST(SanitizeFilenameTest, NoSpecialChars) {
  EXPECT_EQ("valid_filename.txt", sanitize_filename("valid_filename.txt"));
}

TEST(SanitizeFilenameTest, SpecialChars) {
  EXPECT_EQ("sanitized_filename.txt", sanitize_filename("s/a*n:i?t|i<z>e>d_filename.txt"));
}

TEST(SanitizeFilenameTest, TrailingSpaces) {
  EXPECT_EQ("filename.txt", sanitize_filename("filename.txt  "));
}

// Test concatenation of two relative paths
TEST(PathConcatenation, RelativeToRelative) {
    Path lhs("path/to");
    Path rhs("relative");
    EXPECT_EQ((lhs / rhs).str(), file::make_path("path","to","relative"));  // Expect paths to be concatenated with separator
}

// Test concatenation where the lhs path has a trailing separator
TEST(PathConcatenation, TrailingSeparator) {
    Path lhs("path/to/");
    Path rhs("relative");
    EXPECT_EQ((lhs / rhs).str(), file::make_path("path","to","relative"));  // Trailing separator should be removed from lhs
}

// Test concatenation where the rhs is an absolute path
TEST(PathConcatenation, AbsoluteRhs) {
    Path lhs("path/to");
    Path rhs("/absolute/path");
    EXPECT_THROW(lhs / rhs, std::invalid_argument);  // Should throw due to absolute rhs
}

// Test concatenation where the lhs is empty
TEST(PathConcatenation, EmptyLhs) {
    Path lhs("");
    Path rhs("relative");
    EXPECT_EQ((lhs / rhs).str(), "relative");  // Empty lhs should result in only rhs
}

// Test concatenation where the rhs is empty
TEST(PathConcatenation, EmptyRhs) {
    Path lhs("path/to");
    Path rhs("");
    EXPECT_EQ((lhs / rhs).str(), lhs.str());  // Empty rhs should result in only lhs
}

// Test concatenation where both paths are empty
TEST(PathConcatenation, BothEmpty) {
    Path lhs("");
    Path rhs("");
    EXPECT_EQ((lhs / rhs).str(), "");  // Both paths empty should result in empty
}

// Test concatenation with various special characters in paths
TEST(PathConcatenation, SpecialCharacters) {
    Path lhs("path/with space");
    Path rhs("file@123");
    EXPECT_EQ((lhs / rhs).str(), file::make_path("path","with space", "file@123"));  // Special characters should be handled correctly
}

// Test concatenation with dot and dot-dot segments
/*TEST(PathConcatenation, DotAndDotDot) {
    Path lhs("path/to/.");
    Path rhs("../relative");
    EXPECT_EQ((lhs / rhs).str(), "path/relative");  // Dot and dot-dot should be preserved
}

// Test handling of multiple separators
TEST(PathConcatenation, MultipleSeparators) {
    Path lhs("path//to");
    Path rhs("//relative");
    EXPECT_EQ((lhs / rhs).str(), "path/to/relative");  // Multiple separators should be handled correctly
}*/

// Test handling of multiple separators
TEST(PathConcatenation, AbsoluteToRelative) {
    Path lhs("/path/to");
    Path rhs("relative");
    EXPECT_EQ((lhs / rhs).str(), file::make_path("","path","to","relative"));  // Multiple separators should be handled correctly
}

TEST(ExtensionTests, DotsInPath) {
    Path lhs("/path/to/a.file.with.dots/thatsalso.with.dots");
    EXPECT_EQ(lhs.extension(), "dots");
}

TEST(ExtensionTests, NoDotsInPath) {
    Path lhs("/path/to/afilewithoutdots/thatsalso.with.dots");
    EXPECT_EQ(lhs.extension(), "dots");
}

TEST(ExtensionTests, CommonExtensions) {
    Path lhs1("/path/to/file.txt");
    EXPECT_EQ(lhs1.extension(), "txt");

    Path lhs2("/another/path/image.jpeg");
    EXPECT_EQ(lhs2.extension(), "jpeg");
}

TEST(ExtensionTests, NoExtension) {
    Path lhs("/path/to/afilewithoutextension/");
    EXPECT_EQ(lhs.extension(), "");
}

TEST(ExtensionTests, MultipleDotsInFilename) {
    Path lhs("/path/to.a/file.with.many.dots.ext");
    EXPECT_EQ(lhs.extension(), "ext");
}

/*TEST(ExtensionTests, HiddenFiles) {
    Path lhs("/path/.hiddenfile");
    EXPECT_EQ(lhs.extension(), "");
    
    Path lhs2("/path/to/.another.hidden.ext");
    EXPECT_EQ(lhs2.extension(), "ext");
}*/

TEST(ExtensionTests, DotAtEnd) {
    Path lhs("/path/to/filewithdotatend.");
    EXPECT_EQ(lhs.extension(), "");
}

TEST(ExtensionTests, MultipleExtensions) {
    Path lhs("/path/to/file.tar.gz");
    EXPECT_EQ(lhs.extension(), "gz");
}

TEST(ExtensionTests, NoBasename) {
    Path lhs("/path/to/.ext");
    EXPECT_EQ(lhs.extension(), "ext");
}

TEST(ExtensionTests, UnusualCharactersInExtension) {
    Path lhs("/path/to/file.@#$.weird");
    EXPECT_EQ(lhs.extension(), "weird");
}

TEST(ExtensionTests, EmptyString) {
    Path lhs("");
    EXPECT_EQ(lhs.extension(), "");
}

TEST(ExtensionTests, OnlyDots) {
    Path lhs("....");
    EXPECT_EQ(lhs.extension(), "");
    
    Path lhs2("/path/to/....");
    EXPECT_EQ(lhs2.extension(), "");
}

TEST(ExtensionTests, SpacesInPath) {
    Path lhs("/path/ to /file with spaces.ext");
    EXPECT_EQ(lhs.extension(), "ext");
}

TEST(ExtensionTests, SpecialCharactersInPath) {
    Path lhs("/path/to/file!@#$%^&*().ext");
    EXPECT_EQ(lhs.extension(), "ext");
}

TEST(ExtensionTests, DotsOnlyInPath) {
    Path lhs("/path/to/a.file.without/dots");
    EXPECT_EQ(lhs.extension(), "");
}

// Test the constructor and normalization of path separators
TEST(PathNormalization, SeparatorNormalization) {
    auto s = file::make_path({ "path","to","directory" });
    Path p1(s);
    EXPECT_EQ(p1.str(), s);  // Backslashes should be converted to OS_SEP

    Path p2(s + std::string(1, file::Path::os_sep()));  // Trailing separator
    EXPECT_EQ(p1.str(), s);  // Trailing separator should be removed
}

// Test the constructor with only a root slash
TEST(PathNormalization, RootSlashPreservation) {
    Path root("/");
    EXPECT_EQ(root.str(), "/");  // Single root slash should be preserved
}

// Test concatenation with an absolute right-hand side path
TEST(PathConcatenation, AbsoluteRhsError) {
    Path lhs("path/to");
    Path rhs("/absolute/path");
    EXPECT_THROW(lhs / rhs, std::invalid_argument);  // Should throw due to absolute rhs
}

// Test concatenation where both paths are empty
TEST(PathConcatenation, BothEmpty2) {
    Path lhs("");
    Path rhs("");
    EXPECT_EQ((lhs / rhs).str(), "");  // Both paths empty should result in empty
}

// Test concatenation with empty rhs and lhs having a trailing separator
TEST(PathConcatenation, EmptyRhsWithTrailingSeparator) {
    Path lhs("path/to/");
    Path rhs("");
    EXPECT_EQ((lhs / rhs).str(), file::make_path("path","to"));  // Trailing separator should be removed, and only lhs returned
}

// Test concatenation with both paths non-empty
TEST(PathConcatenation, NonEmptyPaths) {
    Path lhs("path/to");
    Path rhs("next/dir");
    EXPECT_EQ((lhs / rhs).str(), file::make_path("path","to", "next", "dir"));  // Paths should be concatenated with separator
}

// Test existence checks with empty paths
TEST(PathExistence, EmptyPathCheck) {
    Path emptyPath("");
    EXPECT_FALSE(emptyPath.exists());  // Empty path should not exist
    EXPECT_FALSE(emptyPath.is_folder());  // Empty path should not be considered a folder
    EXPECT_FALSE(emptyPath.is_regular());  // Empty path should not be considered a regular file
}

// Define a custom mock filesystem for the test cases
struct MockFilesystem : public file::FilesystemInterface {
    std::set<file::Path> find_files(const file::Path& path) const override {
        // Define the mock behavior for finding files
#if defined(_WIN32)
        return { file::Path("C:\\mock\\path1"), file::Path("C:\\mock\\path2") };
#else
        return { file::Path("/mock/path1"), file::Path("/mock/path2") };
#endif
    }
    bool is_folder(const file::Path& path) const override {
        // Define the mock behavior for checking if a path is a folder
        return true;
    }
    bool exists(const file::Path& path) const override {
        // Define the mock behavior for checking if a path exists
        return find_files(path).count(path) > 0;
    }
};

// Test constructor with empty string
TEST(PathArrayTest, EmptyConstructor) {
    file::PathArray paths;
    EXPECT_TRUE(paths.get_paths().empty());  // Expect no paths
}

// Test constructor with a single path
TEST(PathArrayTest, SinglePathConstructor) {
    file::_PathArray<MockFilesystem> paths("path/to/file");
    EXPECT_EQ(paths.get_paths().size(), 1);  // Expect one path
    EXPECT_EQ(paths.get_paths().front().str(), file::make_path("path", "to", "file"));  // Check the path is correct
}

// Test constructor with array of paths as string
TEST(PathArrayTest, ArrayOfStringConstructor) {
    file::_PathArray<MockFilesystem> paths("[\"path/to/file1\", \"path/to/file2\"]");
    EXPECT_EQ(paths.get_paths().size(), 2);  // Expect two paths
    EXPECT_EQ(paths.get_paths()[0].str(), file::make_path("path", "to", "file1"));  // Check the first path
    EXPECT_EQ(paths.get_paths()[1].str(), file::make_path("path", "to", "file2"));  // Check the second path
}

// Test constructor with vector of strings
TEST(PathArrayTest, VectorOfStringConstructor) {
    std::vector<std::string> vec_paths = {"path/to/file1", "path/to/file2"};
    file::_PathArray<MockFilesystem> paths(vec_paths);
    EXPECT_EQ(paths.get_paths().size(), 2);  // Expect two paths
    EXPECT_EQ(paths.get_paths()[0].str(), file::make_path("path","to","file1"));  // Check the first path
    EXPECT_EQ(paths.get_paths()[1].str(), file::make_path("path","to","file2"));  // Check the second path
}

// Test copy and move constructors
TEST(PathArrayTest, CopyAndMoveConstructors) {
    file::_PathArray<MockFilesystem> original("path/to/file");
    auto copied = original;  // Copy constructor
    EXPECT_EQ(copied.get_paths().size(), 1);  // Expect one path in the copied object

    file::_PathArray<MockFilesystem> moved = std::move(original);  // Move constructor
    EXPECT_EQ(moved.get_paths().size(), 1);  // Expect one path in the moved object
    EXPECT_TRUE(original.get_paths().empty());  // Original should be empty after move
}


// Define a struct to hold the parameters
struct PathTestParams {
    std::string path;
    std::string expectedStr;
    std::string expectedFilename;
};

// Define PrintTo for PathTestParams
void PrintTo(const PathTestParams& params, std::ostream* os) {
    *os << "PathTestParams{ path: \"" << params.path
        << "\", expectedStr: \"" << params.expectedStr
        << "\", expectedFilename: \"" << params.expectedFilename << "\" }";
}

// Optionally, define PrintTo for the Path class if you want more detailed output
void PrintTo(const Path& p, std::ostream* os) {
    *os << "Path{ str: \"" << p.str() << "\", filename: \"" << p.filename() << "\" }";
}


// Define the test fixture class
class PathTest : public ::testing::TestWithParam<PathTestParams> {
protected:
    // You can set up variables here that will be used in each test
};

// Now, use the TEST_P macro to create your parameterized test cases
TEST_P(PathTest, CorrectlyHandlesSeparators) {
    auto params = GetParam();
    Path p(params.path);
    EXPECT_EQ(p.str(), params.expectedStr);
    EXPECT_EQ(p.filename(), params.expectedFilename);
}

// Instantiate the test cases with the parameters for OS_SEP and NOT_OS_SEP
INSTANTIATE_TEST_SUITE_P(Default, PathTest, ::testing::Values(
  PathTestParams{file::make_path({"path", "to", "file"}),
                 file::make_path({"path", "to", "file"}), "file"},

  PathTestParams{file::make_path({"path", "to", "file"}, UseNotOsSep{}),
                 file::make_path({"path", "to", "file"}), "file"},
  PathTestParams{file::make_path({"path", "to.with.ext", "file"}, UseNotOsSep{}),
                 file::make_path({"path", "to.with.ext", "file"}), "file"},
  PathTestParams{file::make_path({"path", "to.with.ext", "file"}),
                 file::make_path({"path", "to.with.ext", "file"}), "file"},

  PathTestParams{file::make_path({"path", "to", "directory", ""}),
                 file::make_path({"path", "to", "directory"}), "directory"},

  PathTestParams{file::make_path({"path", "to", "directory", ""}, UseNotOsSep{}),
                 file::make_path({"path", "to", "directory"}), "directory"},

  PathTestParams{file::make_path({""}), file::make_path({""}), ""},
  PathTestParams{file::make_path({""}, UseNotOsSep{}), file::make_path({""}), ""},
  PathTestParams{"file", "file", "file"},
  PathTestParams{"", "", ""},
  
  PathTestParams{file::make_path({"path", "to", ""}),
                 file::make_path({"path", "to"}), "to"},

  PathTestParams{file::make_path({"path", "to", ""}, UseNotOsSep{}),
                 file::make_path({"path", "to"}), "to"},

  PathTestParams{file::make_path({"path", "to", ".", "..", "file"}),
                 file::make_path({"path", "to", ".", "..", "file"}), "file"},

  PathTestParams{file::make_path({"path", "to", ".", "..", "file"}, UseNotOsSep{}),
                 file::make_path({"path", "to", ".", "..", "file"}), "file"}
  // Add more test cases as necessary
));

struct MakePathTestParams {
    char separator;
    std::vector<const char*> parts;
    std::string expected;
};

class PathUtilTest : public ::testing::TestWithParam<MakePathTestParams> {
protected:
    // Setup and teardown, if needed
};

TEST_P(PathUtilTest, CorrectlyCreatesPath) {
    auto &params = GetParam();
    auto path = (params.separator == file::Path::os_sep()) ?
        file::make_path(params.parts) :
        file::make_path(params.parts, file::UseNotOsSep());
    
    EXPECT_EQ(path, params.expected);
}

INSTANTIATE_TEST_SUITE_P(
    Default,
    PathUtilTest,
    ::testing::Values(
        // Assuming file::Path::os_sep() returns '/' and file::Path::not_os_sep() returns '\\'
      MakePathTestParams{file::Path::os_sep(), 
        {"home", "user", "documents", "file.txt"}, 
        "home" + std::string(1, file::Path::os_sep()) + "user" + std::string(1, file::Path::os_sep()) + "documents" + std::string(1, file::Path::os_sep()) + "file.txt"},

      MakePathTestParams{file::Path::not_os_sep(), 
        {"home", "user", "documents", "file.txt"}, 
        "home" + std::string(1, file::Path::not_os_sep()) + "user" + std::string(1, file::Path::not_os_sep()) + "documents" + std::string(1, file::Path::not_os_sep()) + "file.txt"},

      MakePathTestParams{file::Path::os_sep(), 
        {"", "", "home", "file.txt"}, 
        std::string(1, file::Path::os_sep()) + "home" + std::string(1, file::Path::os_sep()) + "file.txt"},

      MakePathTestParams{file::Path::not_os_sep(), 
        {"home", "", "", "file.txt"}, 
        "home" + std::string(1, file::Path::not_os_sep()) + "file.txt"},

      MakePathTestParams{file::Path::os_sep(), 
        {"home", "file.txt", "", ""}, 
        "home" + std::string(1, file::Path::os_sep()) + "file.txt"},

      MakePathTestParams{file::Path::os_sep(), 
        {"", "home", "file.txt", "", ""}, 
        std::string(1, file::Path::os_sep()) + "home" + std::string(1, file::Path::os_sep()) + "file.txt"}
    )
);

// Define PrintTo for PathTestParams
void PrintTo(const MakePathTestParams& params, std::ostream* os) {
    *os << "PathTestParams{ path: " << Meta::toStr(params.parts)
        << ", expectedStr: \"" << params.expected<< "\" }";
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
