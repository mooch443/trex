#pragma once

#include <types.h>
#include <tracking/Individual.h>
#include <misc/OutputLibrary.h>

namespace mem {
using namespace track;

struct MemoryStats {
    MemoryStats();
    
    idx_t id;
    uint64_t bytes;
    std::map<std::string, uint64_t> sizes;
    std::map<std::string, std::map<std::string, uint64_t>> details;
    
    template <typename T>
    uint64_t get_memory_size(T, const std::string&) {
        return sizeof(typename remove_cvref<T>::type);
    }
    
    void operator +=(const MemoryStats& other) {
        if(id == -1) {
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
        
        id = -2;
    }
    
    void clear() {
        bytes = 0;
        id = -1;
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
