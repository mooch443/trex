#include "gtest/gtest.h"
#include <commons.pc.h>
#include <misc/parse_parameter_lists.h>
#include <misc/format.h>
#include <misc/Timer.h>
#include <file/PathArray.h>
#include <gmock/gmock.h>

#ifdef NDEBUG
#undef NDEBUG
#endif

using namespace file;

TEST(PathArrayTest, AddPath) {
    file::_PathArray<> pa;
    pa.add_path("/path/to/file1");
    pa.add_path("/path/to/file2");
    EXPECT_FALSE(pa.empty());
}

TEST(PathArrayTest, IteratorBeginAndEnd) {
    file::_PathArray<> pa;
    pa.add_path("/path/to/file1");
    pa.add_path("/path/to/file2");
    auto it = pa.begin();
    EXPECT_EQ(it->str(), "/path/to/file1"); // Assuming file::Path has a str() function to return its string representation
    ++it;
    EXPECT_EQ(it->str(), "/path/to/file2");
}

TEST(PathArrayTest, GetPaths) {
    file::_PathArray<> pa;
    pa.add_path("/path/to/file1");
    const auto& paths = pa.get_paths();
    EXPECT_EQ(paths.size(), 1);
    EXPECT_EQ(paths[0].str(), "/path/to/file1"); // Again, assuming file::Path has a str() function
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

TEST(PathArrayTest, ParsePath) {
    // Custom mock filesystem for this specific test case
    struct LocalMockFilesystem : public file::FilesystemInterface {
        std::set<file::Path> find_files(const file::Path& path) const override {
            // Mock implementation specific to this test
            return {file::Path("/path/to/file00"), file::Path("/path/to/file01"), file::Path("/path/to/file02")};
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

    auto parsed_paths = file::_PathArray<LocalMockFilesystem>::parse_path("/path/to/file%0.2d");
    EXPECT_EQ(parsed_paths.size(), 3);
    EXPECT_EQ(parsed_paths[0].str(), "/path/to/file00");
    EXPECT_EQ(parsed_paths[1].str(), "/path/to/file01");
    EXPECT_EQ(parsed_paths[2].str(), "/path/to/file02");
}

// Test pattern matching with %10.100.6d
TEST(PathArrayTest, ParsePath_ConsecutiveFiles_10_100) {
    struct LocalMockFilesystem : public file::FilesystemInterface {
        std::set<file::Path> find_files(const file::Path& path) const override {
            // Generate mock files from 10 to 100
            std::set<file::Path> mock_files;
            for (int i = 10; i <= 100; ++i) {
                std::stringstream ss;
                ss << std::setw(6) << std::setfill('0') << i;  // Zero padding to 6 digits
                mock_files.insert(file::Path("/path/to/file" + ss.str()));
            }
            return mock_files;
        }
        bool is_folder(const file::Path& path) const override {
            return true;
        }
        bool exists(const file::Path& path) const override {
            return find_files(path).count(path) > 0;
        }
    };

    auto parsed_paths = file::_PathArray<LocalMockFilesystem>::parse_path("/path/to/file%10.100.6d");
    EXPECT_EQ(parsed_paths.size(), 91); // 100 - 10 + 1 = 91 files

    for (size_t i = 0; i < parsed_paths.size(); ++i) {
        // Generate the expected file name
        int file_number = 10 + i; // Starting from 10
        std::stringstream ss;
        ss << std::setw(6) << std::setfill('0') << file_number;  // Zero padding to 6 digits
        std::string expected_file_name = "/path/to/file" + ss.str();

        // Compare
        EXPECT_EQ(parsed_paths[i].str(), expected_file_name);
    }
}

// Test pattern matching with %10.3d
TEST(PathArrayTest, ParsePath_From10ToEnd) {
    struct LocalMockFilesystem : public file::FilesystemInterface {
        std::set<file::Path> find_files(const file::Path& path) const override {
            return {file::Path("/path/to/file010"), file::Path("/path/to/file011")};
        }
        bool is_folder(const file::Path& path) const override {
            return true;
        }
        bool exists(const file::Path& path) const override {
            return find_files(path).count(path) > 0;
        }
    };

    auto parsed_paths = file::_PathArray<LocalMockFilesystem>::parse_path("/path/to/file%10.3d");
    EXPECT_EQ(parsed_paths.size(), 2);
    EXPECT_EQ(parsed_paths[0].str(), "/path/to/file010");
    EXPECT_EQ(parsed_paths[1].str(), "/path/to/file011");
}

// Test pattern matching with %3d
TEST(PathArrayTest, ParsePath_3DigitsPadded) {
    struct LocalMockFilesystem : public file::FilesystemInterface {
        std::set<file::Path> find_files(const file::Path& path) const override {
            return {file::Path("/path/to/file000"), file::Path("/path/to/file001")};
        }
        bool is_folder(const file::Path& path) const override {
            return true;
        }
        bool exists(const file::Path& path) const override {
            return find_files(path).count(path) > 0;
        }
    };

    auto parsed_paths = file::_PathArray<LocalMockFilesystem>::parse_path("/path/to/file%3d");
    EXPECT_EQ(parsed_paths.size(), 2);
    EXPECT_EQ(parsed_paths[0].str(), "/path/to/file000");
    EXPECT_EQ(parsed_paths[1].str(), "/path/to/file001");
}

// Test pattern matching with %03d (should be the same as %3d)
TEST(PathArrayTest, ParsePath_03DigitsPadded) {
    struct LocalMockFilesystem : public file::FilesystemInterface {
        std::set<file::Path> find_files(const file::Path& path) const override {
            return {file::Path("/path/to/file000"), file::Path("/path/to/file001")};
        }
        bool is_folder(const file::Path& path) const override {
            return true;
        }
        bool exists(const file::Path& path) const override {
            return find_files(path).count(path) > 0;
        }
    };

    auto parsed_paths = file::_PathArray<LocalMockFilesystem>::parse_path("/path/to/file%03d");
    EXPECT_EQ(parsed_paths.size(), 2);
    EXPECT_EQ(parsed_paths[0].str(), "/path/to/file000");
    EXPECT_EQ(parsed_paths[1].str(), "/path/to/file001");
}

// Test pattern matching with filenames having spaces
TEST(PathArrayTest, ParsePath_FilenamesWithSpaces) {
    struct LocalMockFilesystem : public file::FilesystemInterface {
        std::set<file::Path> find_files(const file::Path& path) const override {
            return {file::Path("/path to/file 000"), file::Path("/path to/file 001")};
        }
        bool is_folder(const file::Path& path) const override {
            return true;
        }
        bool exists(const file::Path& path) const override {
            return find_files(path).count(path) > 0;
        }
    };

    auto parsed_paths = file::_PathArray<LocalMockFilesystem>::parse_path("/path to/file %3d");
    EXPECT_EQ(parsed_paths.size(), 2);
    EXPECT_EQ(parsed_paths[0].str(), "/path to/file 000");
    EXPECT_EQ(parsed_paths[1].str(), "/path to/file 001");
}

// Test pattern matching with %[10,20,30]
TEST(PathArrayTest, ParsePath_Array) {
    struct LocalMockFilesystem : public file::FilesystemInterface {
        std::set<file::Path> find_files(const file::Path& path) const override {
            return {file::Path("/path/to/file010"), file::Path("/path/to/file020"), file::Path("/path/to/file030")};
        }
        bool is_folder(const file::Path& path) const override {
            return true;
        }
        bool exists(const file::Path& path) const override {
            return find_files(path).count(path) > 0;
        }
    };

    auto parsed_paths = file::_PathArray<LocalMockFilesystem>::parse_path("/path/to/file%[10,20,30]");
    EXPECT_EQ(parsed_paths.size(), 3);
    EXPECT_EQ(parsed_paths[0].str(), "/path/to/file010");
    EXPECT_EQ(parsed_paths[1].str(), "/path/to/file020");
    EXPECT_EQ(parsed_paths[2].str(), "/path/to/file030");
}

// Test pattern matching with array
TEST(PathArrayTest, ParsePath_ArrayFormat) {
    struct LocalMockFilesystem : public file::FilesystemInterface {
        std::set<file::Path> find_files(const file::Path& path) const override {
            return {file::Path("path/to/file1"), file::Path("/other/path")};
        }
        bool is_folder(const file::Path& path) const override {
            return true;
        }
        bool exists(const file::Path& path) const override {
            return find_files("").count(path) > 0;
        }
    };

    auto parsed_paths = file::_PathArray<LocalMockFilesystem>("[\"path/to/file1\",\"/other/path\"]");
    EXPECT_EQ(parsed_paths.size(), 2);
    EXPECT_EQ(parsed_paths[0].str(), "path/to/file1");
    EXPECT_EQ(parsed_paths[1].str(), "/other/path");
}


// Test pattern matching with *
TEST(PathArrayTest, ParsePath_Star) {
    struct LocalMockFilesystem : public file::FilesystemInterface {
        std::set<file::Path> find_files(const file::Path& path) const override {
            return {file::Path("/path/to/file1"), file::Path("/path/to/file2"), file::Path("/path/to/file3")};
        }
        bool is_folder(const file::Path& path) const override {
            return true;
        }
        bool exists(const file::Path& path) const override {
            return find_files(path).count(path) > 0;
        }
    };

    auto parsed_paths = file::_PathArray<LocalMockFilesystem>::parse_path("/path/to/file*");
    EXPECT_EQ(parsed_paths.size(), 3);
    EXPECT_EQ(parsed_paths[0].str(), "/path/to/file1");
    EXPECT_EQ(parsed_paths[1].str(), "/path/to/file2");
    EXPECT_EQ(parsed_paths[2].str(), "/path/to/file3");
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
        std::vector<file::Path>{ "/first/path/to/file.txt", "/second/path/to/file.txt" }
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


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
