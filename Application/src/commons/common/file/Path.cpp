#include "Path.h"
#include <cstdlib>
#if WIN32
#include "../dirent.h"
#define OS_SEP '\\'
#define NOT_OS_SEP '/'
#else
#include <dirent.h>
#define OS_SEP '/'
#define NOT_OS_SEP '\\'
#endif
#include <errno.h>

#ifdef __APPLE__
#include <Availability.h>

#ifdef __MAC_OS_X_VERSION_MAX_ALLOWED
#if __MAC_OS_X_VERSION_MAX_ALLOWED < 101500
#undef __cpp_lib_filesystem
#endif
#endif
#endif

#ifdef __cpp_lib_filesystem
#include <filesystem>
#endif

#include <misc/metastring.h>

namespace file {
    char Path::os_sep() { return OS_SEP; }
    
    Path::Path(const std::string& s)
        : _str(s)
    {
        // remove trailing slashes
        while(utils::endsWith(_str, OS_SEP))
            _str = _str.substr(0, _str.length()-1);
		for (size_t i = 0; i < _str.length(); ++i)
			if (_str[i] == NOT_OS_SEP) _str[i] = OS_SEP;
    }

    Path::Path(const char* c)
        : Path(std::string(c))
    {}

    Path::Path(std::string_view sv)
        : Path(std::string(sv))
    {}

    Path& Path::operator/=(const Path &other) {
        _str += other.str();
        return *this;
    }
    
    bool Path::is_absolute() const {
#if WIN32
        return (_str.length() >= 3 && _str[1] == ':');
#else
        return !_str.empty() && _str[0] == OS_SEP;
#endif
    }

std::string_view Path::filename() const {
        if(empty())
            return _str;
        
        const char *ptr = _str.data() + _str.length() - 1;
        for(; ptr >= _str.data(); ptr--) {
            if(*ptr == OS_SEP)
                return std::string_view(ptr+1u, size_t(_str.data() + _str.length() - (ptr+1u)));
        }
        
        return _str;
    }
    
    std::pair<Path, Path> Path::split(size_t position) const
    {
        size_t before_len = 0, count = 0;
        
        for (size_t i=0; i<=_str.length(); i++) {
            if(i == _str.length() || _str[i] == OS_SEP) {
                if(++count > position) {
                    before_len = i;
                    break;
                }
            }
        }
        
        if(count != position+1)
            U_EXCEPTION("Path only contains %d segments (requested split at %d).", count, position);
        
        return {Path(std::string_view(_str.data(), before_len)), Path(std::string_view(_str.data() + before_len, _str.length() - before_len))};
    }

    Path Path::remove_filename() const {
        return Path(std::string_view(_str.data(), _str.length() - filename().length()));
    }

    Path operator/(const Path& lhs, const Path& rhs) {
        return Path(lhs.str() + OS_SEP + rhs.str());
    }

    Path operator+(const Path& lhs, const Path& rhs) {
        return Path(lhs.str() + rhs.str());
    }

    bool Path::exists() const {
        return file_exists(_str);
    }

    uint64_t Path::file_size() const {
#if defined(WIN32)
        WIN32_FILE_ATTRIBUTE_DATA fInfo;

        DWORD ftyp = GetFileAttributesEx(c_str(), GetFileExInfoStandard, &fInfo);
        if (INVALID_FILE_ATTRIBUTES == ftyp || fInfo.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
        {
            U_EXCEPTION("Cannot stat file '%S'.", &str());
        }
        return (static_cast<ULONGLONG>(fInfo.nFileSizeHigh) << sizeof(fInfo.nFileSizeLow) * 8) | fInfo.nFileSizeLow;
#else
        struct stat sbuf;
        if (stat(c_str(), &sbuf) == -1)
            U_EXCEPTION("Cannot stat file '%S'.", &str());
        return narrow_cast<uint64_t>(sbuf.st_size);
#endif
    }

    std::string_view Path::extension() const {
        if(empty())
            return std::string_view(_str);
        
        const char *ptr = &_str.back();
        for(; ptr != _str.data(); --ptr) {
            if (*ptr == '.') {
                return std::string_view(ptr+1, size_t(&_str.back() - ptr));
            }
        }
        
        return std::string_view(_str.data() + _str.length() - 1, 0);
    }
    
    bool Path::has_extension() const {
        return !extension().empty();
    }
    
    bool Path::create_folder() const {
        if(exists())
            return true;
        
        std::deque<std::string> folders;
        Path tmp = *this;
        while (!tmp.empty()) {
            if (tmp.exists()) {
                break;
            }
            
            folders.push_front(tmp.str());
            tmp = tmp.remove_filename();
        }
        
        for(auto &folder : folders) {
#if WIN32
			if (!folder.empty() && folder.back() == ':')
				continue;
			if(CreateDirectory(folder.c_str(), NULL)) {
#else
            if(mkdir(folder.c_str(), ACCESSPERMS) != 0) {
#endif
                Except("Cannot create folder '%S'.", &folder);
                return false;
            }
        }
        
        return true;
    }
    
    bool Path::is_folder() const {
#if WIN32
        DWORD ftyp = GetFileAttributesA(str().c_str());
        if (ftyp == INVALID_FILE_ATTRIBUTES)
            return false;  //something is wrong with your path!

        if (ftyp & FILE_ATTRIBUTE_DIRECTORY)
            return true;   // this is a directory!

        return false;
#else
        struct stat path_stat;
        if(stat(empty() ? "/" : str().c_str(), &path_stat) != 0)
            return false;
        return S_ISDIR(path_stat.st_mode);
#endif
    }
    
    bool Path::is_regular() const {
        struct stat path_stat;
        if(stat(str().c_str(), &path_stat) != 0)
            return false;
        return S_ISREG(path_stat.st_mode);
    }
    
    bool Path::delete_file() const {
        if(!exists())
            return false;
        
        if(!is_regular())
            U_EXCEPTION("Cannot delete non-regular file '%S'.", &str());
        
        return std::remove(str().c_str()) == 0;
    }
    
    bool Path::delete_folder(bool recursively) const {
        if(!exists())
            return false;
        
        if(!is_folder())
            U_EXCEPTION("Cannot folder-delete a non-folder '%S'.", &str());
        
        if(recursively) {
            auto files = find_files();
            for(auto &file : files) {
                if(file.is_folder()) {
                    if(!file.delete_folder(true))
                        U_EXCEPTION("Cannot delete folder '%S'.", &file.str());
                    
                } else if(file.is_regular()) {
                    if(!file.delete_file())
                        U_EXCEPTION("Cannot delete file '%S'.", &file.str());
                    
                } else
                    U_EXCEPTION("Unknown file type '%S'.", &file.str());
            }
        }
#if defined(WIN32)
        return RemoveDirectory(str().c_str());
#else
        return rmdir(str().c_str()) == 0;
#endif
    }
        
        bool valid_extension(const file::Path& path, const std::string& filter_extension) {
            if(filter_extension.empty())
                return true;
            
            auto extensions = utils::split(utils::lowercase(filter_extension), ';');
            if(path.has_extension()) {
                return contains(extensions, utils::lowercase((std::string)path.extension()));
            }
            
            return false;
        }
    
    std::set<Path> Path::find_files(const std::string& filter_extension) const {
        if(!is_folder())
            U_EXCEPTION("The path '%S' is not a folder and can not be iterated on.", &str());
        if(!empty() && !exists())
            U_EXCEPTION("The path '%S' does not exist.", &str());
        
        std::set<Path> result;
        DIR *dir;
        struct dirent *ent;
        if ((dir = opendir (empty() ? "/" : str().c_str())) != NULL) {
            while ((ent = readdir (dir)) != NULL) {
                std::string file(ent->d_name);
                if(file == "." || file == "..")
                    continue;
                
                if(ent->d_type & DT_DIR || valid_extension(file::Path(ent->d_name), filter_extension))
                    result.insert(*this / ent->d_name);
            }
            closedir (dir);
            
        } else
            U_EXCEPTION("Folder '%S' exists but cannot be read.", &str());
        
        return result;
    }

    Path Path::replace_extension(std::string_view ext) const {
        auto e = extension();
        return std::string_view(_str.data(), size_t(max(0, e.data() - _str.data() - 1))) + "." + ext;
    }

    Path Path::add_extension(std::string_view ext) const {
        auto current = extension();
        if(ext != current)
            return Path(_str + "." + ext);
        return *this;
    }
    
    Path Path::remove_extension() const {
        if(has_extension())
            return Path(_str.substr(0, _str.length() - extension().length() - 1));
        return *this;
    }
    
    int copy_file( const char* srce_file, const char* dest_file )
    {
#ifndef __cpp_lib_filesystem
        try {
            std::ifstream srce( srce_file, std::ios::binary ) ;
            std::ofstream dest( dest_file, std::ios::binary ) ;
            dest << srce.rdbuf() ;
        } catch (std::exception& e) {
            Except("Caught an exception copying '%s' to '%s': %s", srce_file, dest_file, e.what());
            return 1;
        }
        
        return 0;
#else
        try {
            std::filesystem::copy(srce_file, dest_file);
            
        } catch(const std::filesystem::filesystem_error& e) {
            Except("Caught an exception copying '%s' to '%s': %s", srce_file, dest_file, e.what());
            return 1;
        }
        return 0;
#endif
    }
    
    bool Path::move_to(const file::Path &to) {
#ifndef __cpp_lib_filesystem
        if(std::rename(c_str(), to.c_str()) == 0) {
#else
        try {
            std::filesystem::rename(c_str(), to.c_str());
#endif
            assert(!exists() && to.exists());
            return true;
#ifdef __cpp_lib_filesystem
        } catch (std::filesystem::filesystem_error& e) {
            // do nothing
    #ifndef NDEBUG
            Warning("Filesystem error: '%s'", e.what());
    #endif
#endif
        }
        
        if(copy_file(c_str(), to.c_str()) != 0) {
            Except("Failed renaming file '%S' to '%S', so I tried copying. That also failed. Make sure the second location is writable (this means that the file is still available in the first location, it just failed to be moved).", &str(), &to.str());
            
        } else {
            if(!delete_file())
                Warning("Cannot remove file '%S' from its original location after moving.", &str());
            return true;
        }
        
        return false;
    }
    
    bool Path::operator<(const Path& other) const {
        return str() < other.str();
    }
    
    bool Path::operator<=(const Path& other) const {
        return str() <= other.str();
    }
    
    bool Path::operator>(const Path& other) const {
        return str() > other.str();
    }
    
    bool Path::operator>=(const Path& other) const {
        return str() >= other.str();
    }
    
    bool operator==(const Path& lhs, const Path& rhs) {
        return lhs.str() == rhs.str();
    }
    
    bool operator!=(const Path& lhs, const Path& rhs) {
        return lhs.str() != rhs.str();
    }
    
    FILE* Path::fopen(const std::string &access_rights) const {
        auto f = ::fopen(c_str(), access_rights.c_str());
        if(!f)
            Error("fopen failed, errno = %d\n", errno);
        return f;
    }
        
#ifdef WIN32
#define pclose _pclose
#define popen _popen
#endif
        
    std::string exec(const char* cmd) {
        std::array<char, 256> buffer;
        std::string result;
        std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
        if (!pipe) {
            throw std::runtime_error("popen() failed!");
        }
        while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
            result += buffer.data();
        }
        if(!result.empty() && result[result.length()-1] == '\n')
            result = result.substr(0,result.length()-1);
        if(!result.empty() && result[0] == '\"')
            result = result.substr(1);
        
        return result;
    }
}

std::ostream& operator<<(std::ostream& os, const file::Path& p) {
    return os << p.str();
}

