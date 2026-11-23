#pragma once

#include <commons.pc.h>
#include <misc/Path.h>

namespace trex_test {

class TestFilesystem : public cmn::file::FilesystemInterface {
    using Path = cmn::file::Path;
    std::map<Path, std::set<std::string>> directories{{".", {}}};

    static Path parent_of(const Path& path) {
        auto parent = path.remove_filename();
        return parent.empty() ? Path(".") : parent;
    }

public:
    TestFilesystem(std::initializer_list<Path> files) {
        for(auto path : files) {
            while(not path.filename().empty() && path != ".") {
                const auto parent = parent_of(path);
                directories[parent].insert(path.filename());
                path = parent;
            }
        }
    }

    std::set<Path> find_files(const Path& folder) const override {
        const auto& names = directories.at(folder);
        std::set<Path> matches;
        for(const auto& name : names)
            matches.insert(folder / name);
        return matches;
    }

    bool is_folder(const Path& folder) const override {
        return directories.contains(folder);
    }

    bool exists(const Path& path) const override {
        if(is_folder(path))
            return true;
        const auto folder = directories.find(parent_of(path));
        return folder != directories.end() && folder->second.contains(path.filename());
    }
};

}
