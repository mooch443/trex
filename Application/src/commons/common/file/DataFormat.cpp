#include "DataFormat.h"
#ifndef WIN32
#include <sys/mman.h>
#endif
#include <fcntl.h>
#ifdef __cpp_lib_filesystem
#include <filesystem>
namespace fs = std::filesystem;
#endif

#if __APPLE__
#include <Availability.h>
#include <AvailabilityMacros.h>
#endif

using namespace cmn;

#if defined(WIN32)

char* memGetAddr( mappedRegion *hReg)
{
    return (char *)((*hReg)->addr);
}

int memmap(bool create, const file::Path& path, mappedRegion* hReg, uint64_t length = 0) {
    unsigned int i;
    int exists = 1;
    char buffer[1024];
    
    PSECURITY_DESCRIPTOR pSD;
    HANDLE hFile, hMem;
    SECURITY_ATTRIBUTES  sa;
    DWORD sz;

    if (!path.exists())
        return -1;

    auto reg = (mappedRegion)malloc( sizeof( struct mappedRegionS));
    if (NULL == reg) return -3;
    *hReg = reg;

    if (exists) reg->length = path.file_size();
    else reg->length = length;
    
    /* create security descriptor (needed for Windows NT) */
    pSD = (PSECURITY_DESCRIPTOR) malloc( SECURITY_DESCRIPTOR_MIN_LENGTH );
    if( pSD == NULL ) return -2;

    InitializeSecurityDescriptor(pSD, SECURITY_DESCRIPTOR_REVISION);
    SetSecurityDescriptorDacl(pSD, true, (PACL) NULL, false);

    sa.nLength = sizeof(sa);
    sa.lpSecurityDescriptor = pSD;
    sa.bInheritHandle = true;

    /* create or open file */
    if (!create) {
        assert(path.exists());
        
        hFile = CreateFile ( path.c_str(), GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL, NULL);
    }
    else {
        hFile = CreateFile ( path.c_str(), GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_ALWAYS,
            FILE_ATTRIBUTE_NORMAL, NULL);
    }
    if (hFile == INVALID_HANDLE_VALUE) {
        free( pSD);
        return -3;
    }
    if (! exists) {
        /* ensure that file is long enough and filled with zero */
        memset( buffer, 0, sizeof(buffer));
        for (i = 0; i < reg->length/sizeof(buffer); ++i) {
            if (! WriteFile( hFile, buffer, sizeof(buffer), &sz, NULL)) {
                return -3;
            }
        }
        if (! WriteFile( hFile, buffer, reg->length, &sz, NULL)) {
            return -3;
        }
    }

    ULARGE_INTEGER maxSize;
    maxSize.QuadPart = reg->length;

    hMem = CreateFileMapping( hFile, &sa, PAGE_READWRITE, maxSize.HighPart, maxSize.LowPart, NULL);
    free( pSD);
    if (NULL == hMem) return -3;

    reg->addr = MapViewOfFile( hMem, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    if (NULL == reg->addr) return -3;

    CloseHandle( hFile);
    CloseHandle( hMem);
    
    return 0;
}

void memunmap( mappedRegion *hReg)
{
    mappedRegion reg = *hReg;
    if (reg) {
        if (reg->addr) {
            UnmapViewOfFile( reg->addr);
        }
        free( reg);
    }
    *hReg = 0;
}
#endif

DataFormat::~DataFormat() {
    if(f || _mmapped)
        close();
}

void DataFormat::close() {
    if(f) {
        fclose(f);
        f = NULL;
    }
    
    if(_mmapped) {
#if !defined(WIN32)
        munmap((void*)_data, _reading_file_size);
        ::close(fd);
#else
        memunmap(&reg);
#endif
        _mmapped = false;
    }
    
    _supports_fast = false;
    _file_offset = 0;
}

void DataFormat::start_modifying() {
    if(open())
        close();
    
    f = fopen(_filename.c_str(), "r+");
    if(!f)
        U_EXCEPTION("Cannot open file '%S'.", &_filename);
    
    _open_for_modifying = true;
    _open_for_writing = false;
    _file_offset = 0;
    _read_header();
}

uint64_t DataFormat::current_offset() const {
	return _file_offset; 
}
uint64_t DataFormat::tell() const {
	return current_offset(); /*ftell(f);*/ 
}

void DataFormat::start_reading() {
    if (open())
        close();
    
/*#if defined(WIN32)
	_supports_fast = false;
	_mmapped = false;

	f = _filename.fopen("rb");
	if (!f)
		U_EXCEPTION("Cannot open file '%S'.", &_filename);
#else*/

    if (!_filename.exists())
        U_EXCEPTION("Cannot find file '%S'.", &_filename);

    _reading_file_size = _filename.file_size();
    
#if defined(WIN32)
    int result = memmap(false, _filename, &reg);
    if (result != 0) {
        if (reg)
            memunmap(&reg);
        U_EXCEPTION("Cannot mmap file '%S' (%d).", &_filename, result);
    }
    _data = memGetAddr(&reg);
#else
    struct stat sbuf;
    if (stat(_filename.c_str(), &sbuf) == -1)
        U_EXCEPTION("Cannot stat file '%S'.", &_filename);

    if ((fd = ::open(_filename.c_str(), O_RDONLY)) == -1)
        U_EXCEPTION("Cannot open file '%S'.", &_filename);

    if ((_data = (char*)mmap((caddr_t)0, (size_t)sbuf.st_size, PROT_READ, MAP_SHARED, fd, 0)) == (caddr_t)(-1))
        U_EXCEPTION("Cannot mmap file '%S'.", &_filename);
#endif
    _supports_fast = true;
    _mmapped = true;
//#endif
    _open_for_modifying = false;
    _open_for_writing = false;
    _file_offset = 0;
    _read_header();
}

void DataFormat::start_writing(bool overwrite) {
    if (open())
        close();
    
    if (_filename.exists() && !overwrite)
        U_EXCEPTION("File already exists '%S'.", &_filename);
    
    f = _filename.fopen("wb");
    if(!f)
        U_EXCEPTION("Cannot open file '%S'.", &_filename);
    
    _file_offset = 0;
    _open_for_writing = true;
    _open_for_modifying = false;
    _header_written = true;
    
    _write_header();
}

void DataFormat::stop_writing() {
    assert(_open_for_writing);
    close();
    
    _open_for_writing = false;
    _header_written = false;
}

uint64_t DataFormat::read_data(uint64_t num_bytes, char *buffer) {
    if(!_mmapped) {
        std::lock_guard<std::mutex> guard(_internal_modification);
        if(!f)
            U_EXCEPTION("File is not opened yet.");
        if(feof(f))
            U_EXCEPTION("File is over.");
        
        uint64_t ret = fread(buffer, sizeof(char), num_bytes, f);
        if(ret != num_bytes)
            U_EXCEPTION("Read an unexpected number of bytes (%lu != %lu) at position %lu / %lu.", ret, num_bytes, _file_offset, _reading_file_size);
        
        _file_offset += ret;
        /*printf("Reading %ld bytes: ", num_bytes);
        for(int i=0; i<num_bytes; i++)
            printf("0x%X ", (uint)buffer[i] & 0xffU);
        printf("\n");*/
        
        return ret;
        
    } else {
        uint64_t offset;
        {
            std::lock_guard<std::mutex> guard(_internal_modification);
            if(_file_offset + num_bytes > _reading_file_size)
                U_EXCEPTION("Out of range %lu", _file_offset+num_bytes);
            
            offset = _file_offset;
            _file_offset += num_bytes;
        }
        
        memcpy(buffer, _data + offset, num_bytes);
        return num_bytes;
    }
}

const char* DataFormat::read_data_fast(uint64_t num_bytes) {
    uint64_t offset;
    
    {
        std::lock_guard<std::mutex> guard(_internal_modification);
        if(!_mmapped)
            U_EXCEPTION("Only available for mmapped files.");
        if(uint64_t(_file_offset + num_bytes) > _reading_file_size) {
            Error("Exceeding reported file size of %lu at %lu reading %lu bytes.", _reading_file_size, _file_offset, num_bytes);
        }
        
        offset = _file_offset;
        _file_offset += num_bytes;
    }
    
    return _data + offset;
}

const char* ReadonlyMemoryWrapper::read_data_fast(uint64_t num_bytes) {
    const uchar *ptr = _data + pos;
    seek(pos + num_bytes);
    return (const char*)ptr;
}

uint64_t DataFormat::write_data(uint64_t num_bytes, const char *buffer) {
    if(!f)
        U_EXCEPTION("File is not opened yet.");
    
    uint64_t written;
    if((written = fwrite(buffer, sizeof(char), num_bytes, f)) != num_bytes) {
#ifdef __cpp_lib_filesystem
// code only compiled when targeting Mac OS X and not iPhone
// note use of 1050 instead of __MAC_10_5
#if !__APPLE__ || (defined(__MAC_OS_X_VERSION_MIN_REQUIRED) && __MAC_OS_X_VERSION_MIN_REQUIRED >= 101500)
        try {
            fs::space_info space = fs::space(_filename.remove_filename().str());
            if(space.available <= num_bytes)
                U_EXCEPTION("Cannot write %lu bytes to disk ('%S') since there is not enough empty space available on disk (%lu).", num_bytes, &_filename.str(), space.available);
        } catch(const fs::filesystem_error& ex)
        {
            Warning("An exception occurred while handling another exception: '%s'.", ex.what());
        }
#endif
#endif
		U_EXCEPTION("Did not expect to not write %lu bytes (instead of %lu) o.O", written, num_bytes);
    }
    
    uint64_t before = _file_offset;
    _file_offset += num_bytes;
    return before;
}

/**
 * STRING IN/OUT
 */
template<>
uint64_t Data::write(const std::string& val) {
    return write_data(val.length() + 1, val.data());
}

template<>
void Data::read(std::string& str) {
    std::stringstream ss;
    uchar c = UCHAR_MAX;
    while (c != 0) {
        read<uchar>(c);
        //read_data(1, (char*)&c);
        if(c != 0)
            ss << c;
    }
    str = ss.str();
}

/**
 * cv::Size IN/OUT
 */
template<>
uint64_t Data::write(const cv::Size& val) {
    assert(val.width <= USHRT_MAX && val.height <= USHRT_MAX);
    assert(val.width >= 0 && val.height >= 0);
    
    uint64_t pos =
    write(uint16_t(val.width));
    write(uint16_t(val.height));
    
    return pos;
}

template<>
void Data::read(cv::Size& size) {
    uint16_t width, height;
    read<uint16_t>(width); read<uint16_t>(height);
    size.width = width;
    size.height = height;
}

/**
 * cv::Point IN/OUT
 */
template<>
uint64_t Data::write(const cv::Point& val) {
    assert(val.x <= USHRT_MAX && val.y <= USHRT_MAX);
    assert(val.x >= 0 && val.y >= 0);
    
    uint64_t pos =
    write(uint16_t(val.x));
    write(uint16_t(val.y));
    
    return pos;
}

template<>
void Data::read(cv::Point& pt) {
    uint16_t x, y;
    read<uint16_t>(x);
    read<uint16_t>(y);
    pt.x = x;
    pt.y = y;
}

template<>
uint64_t Data::write(const cv::Point2f& val) {
    uint64_t pos =
    write(float(val.x));
    write(float(val.y));
    
    return pos;
}

template<>
void Data::read(cv::Point2f& pt) {
    read<float>(pt.x);
    read<float>(pt.y);
}

template<>
uint64_t Data::write(const Size2& val) {
    uint64_t pos =
    write(float(val.width));
    write(float(val.height));
    
    return pos;
}

template<>
void Data::read(Size2& pt) {
    if constexpr(std::is_same<Float2_t, float>::value) {
        read<float>(pt.width);
        read<float>(pt.height);
        
    } else {
        float x,y;
        read<float>(x);
        read<float>(y);
        pt.width = x;
        pt.height = y;
    }
}

template<>
uint64_t Data::write(const Vec2& val) {
    uint64_t pos =
    write(float(val.x));
    write(float(val.y));
    
    return pos;
}

template<>
void Data::read(Vec2& pt) {
    if constexpr(std::is_same<Float2_t, float>::value) {
        read<float>(pt.x);
        read<float>(pt.y);
        
    } else {
        float x,y;
        read<float>(x);
        read<float>(y);
        pt.x = x;
        pt.y = y;
    }
}

template<>
void Data::read(uint8_t& c) {
    constexpr auto size = sizeof(uint8_t);
                
    if(_supports_fast) {
        memcpy(&c, read_data_fast(size), 1);
        return;
    }
    
    const uint64_t read_size = read_data(size, (char*)&c);
    if(read_size != size)
        U_EXCEPTION("Read unexpected number of bytes (%d/%d).", read_size, size);
}

/**
 * cv::Rect2i IN/OUT
 */
template<>
uint64_t Data::write(const cv::Rect2i& val) {
    assert(val.x <= USHRT_MAX && val.y <= USHRT_MAX && val.width <= USHRT_MAX && val.height <= USHRT_MAX);
    assert(val.x >= 0 && val.y >= 0 && val.width >= 0 && val.height >= 0);
    
    uint64_t pos =
    write(uint16_t(val.x));
    write(uint16_t(val.y));
    write(uint16_t(val.width));
    write(uint16_t(val.height));
    
    return pos;
}

template<>
void Data::read(cv::Rect2i& r) {
    uint16_t x,y,w,h;
    read<uint16_t>(x);
    read<uint16_t>(y);
    read<uint16_t>(w);
    read<uint16_t>(h);
    r.x = x;
    r.y = y;
    r.width = w;
    r.height = h;
}

template<>
uint64_t Data::write(const CropOffsets& o) {
    auto val = o.toPixels(Size2(8000, 8000));

    assert(val.x <= USHRT_MAX && val.y <= USHRT_MAX && val.width <= USHRT_MAX && val.height <= USHRT_MAX);
    assert(val.x >= 0 && val.y >= 0 && val.width >= 0 && val.height >= 0);

    uint64_t pos =
        write(uint16_t(val.x));
    write(uint16_t(val.y));
    write(uint16_t(val.width));
    write(uint16_t(val.height));

    return pos;
}

template<>
void Data::read(CropOffsets& r) {
    uint16_t x, y, w, h;
    read<uint16_t>(x);
    read<uint16_t>(y);
    read<uint16_t>(w);
    read<uint16_t>(h);

    r = CropOffsets(
        float(x) / 8000.f, 
        float(y) / 8000.f, 
        1 - float(w) / 8000.f,
        1 - float(h) / 8000.f);
}

/**
 * DataPackage OUT
 */
template<>
uint64_t Data::write(const DataPackage& val) {
    return write_data(val.size(), val.data());
}

uint64_t DataPackage::write_data(uint64_t num_bytes, const char *buffer) {
    if(pos + num_bytes >= _capacity) {
        // heuristic for adding potentially even more bytes than needed
        resize(max(pos + num_bytes * 2, _capacity * 2));
    }
    
    uint64_t before = pos;
    memcpy(_data + pos, buffer, num_bytes);
    seek(pos + num_bytes);
    
    return before;
}

