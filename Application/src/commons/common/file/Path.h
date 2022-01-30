#ifndef _PATH_H
#define _PATH_H

#include <misc/defines.h>
#include <string_view>

namespace file {
    using namespace cmn;
    
    class Path {
        //! The full path without any trailing slashes,
        //  but with file extension and filename
        GETTER(std::string, str)
        
    public:
        Path(const std::string& s = "");
        Path(const char* c);
        Path(std::string_view sv);
        
        const char* c_str() const { return _str.c_str(); }
        
        //! Concatenation
        //  Path p = "Users" / username / "Desktop"
        Path& operator/=(const Path&);
        Path& append(const Path&);
        
        static char os_sep();
        bool is_absolute() const;
        bool empty() const { return _str.empty(); }
        
        //! Returns the last filename, removes the path
        std::string_view filename() const;
        
        //! Returns path, removes last filename
        Path remove_filename() const;
        
        std::pair<Path, Path> split(size_t position) const;
        
        //! Checks if the given file exists
        bool exists() const;

        //! Returns the file size and throws exceptions if it does not exist or is a dir
        uint64_t file_size() const;
        
        //! Recursively creates the folders for this path
        bool create_folder() const;
        
        //! Check whether the given filename exists and is a folder
        bool is_folder() const;
        
        //! Finds all files under the given path. Exception if the given
        //  path is not a folder.
        std::set<file::Path> find_files(const std::string& filter_extension = "") const;
        
        //! Checks whether the given path is a regular file.
        bool is_regular() const;
        
        //! Deletes the file if it exists and is a file - otherwise returns false.
        bool delete_file() const;
        
        //! Deletes the folder, if this file is a folder.
        bool delete_folder(bool recursively) const;
        
        //! Moves the file to given destination
        bool move_to(const file::Path& to);
        
        Path replace_extension(std::string_view ext) const;
        Path add_extension(std::string_view ext) const;
        bool has_extension() const;
        Path remove_extension() const;
        
        FILE* fopen(const std::string& access_rights) const;
        
        std::string_view extension() const;
        
        bool operator<(const Path& other) const;
        bool operator<=(const Path& other) const;
        bool operator>(const Path& other) const;
        bool operator>=(const Path& other) const;
        
        explicit operator std::string() const { return str(); }
        static std::string class_name() { return "path"; }
        static file::Path fromStr(const std::string& str);
    };

    Path operator/( const Path& lhs, const Path& rhs );
    Path operator+( const Path& lhs, const Path& rhs);
    bool operator==(const Path& lhs, const Path& rhs);
    bool operator!=(const Path& lhs, const Path& rhs);
    
    std::string exec(const char* cmd);

    bool valid_extension(const file::Path&, const std::string& filter_extension);
}

std::ostream& operator<<(std::ostream& os, const file::Path& p);

#endif
