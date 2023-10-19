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
bool matched;

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
        if (path.str() == file::Path("/")) {
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
