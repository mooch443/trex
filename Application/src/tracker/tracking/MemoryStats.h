#pragma once

#include <commons.pc.h>
#include <tracking/Individual.h>
#include <tracking/OutputLibrary.h>
#include <misc/idx_t.h>

namespace mem {
using namespace track;

struct MemoryStats {
    MemoryStats();
    
    Idx_t id;
    uint64_t bytes;
    std::map<std::string, uint64_t> sizes;
    std::map<std::string, std::map<std::string, uint64_t>> details;
    
    template <typename T>
    uint64_t get_memory_size(const T&, const std::string&) {
        return sizeof(typename cmn::remove_cvref<T>::type);
    }
    
    void operator +=(const MemoryStats& other) {
        if(!id.valid()) {
            *this = other;
            return;
        }
        
        bytes += other.bytes;
        for(auto & [key, size] : other.sizes) {
            sizes[key] += size;
        }
        
        for(auto & [name, map] : other.details) {
            for(auto & [key, size] : map) {
                details[name][key] += size;
            }
        }
        
        id = Idx_t(std::numeric_limits<uint32_t>::max()-1);
    }
    
    void clear() {
        bytes = 0;
        id = Idx_t();
        sizes.clear();
    }
    virtual void print() const;
    virtual ~MemoryStats() {}
};

struct IndividualMemoryStats : public MemoryStats {
    IndividualMemoryStats(Individual* = nullptr);
    void print() const override;
};

struct TrackerMemoryStats : public MemoryStats {
    TrackerMemoryStats();
    void print() const override;
};

struct OutputLibraryMemoryStats : public MemoryStats {
    OutputLibraryMemoryStats(Output::LibraryCache::Ptr = nullptr);
    void print() const override;
};

}
