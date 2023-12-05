#pragma once

#include <commons.pc.h>

using useMat_t = cv::Mat;

struct GPUMatPtr {
    std::unique_ptr<useMat_t> ptr;
#ifndef NDEBUG
    cmn::source_location loc;
#endif
    
    GPUMatPtr(nullptr_t) {}
    GPUMatPtr() { }
    GPUMatPtr(GPUMatPtr&& other) noexcept 
#ifdef NDEBUG
        : ptr(std::move(other.ptr)) 
#endif
    {
#ifndef NDEBUG
        *this = std::move(other);
#endif
        assert(ptr);
    }
    GPUMatPtr& operator=(GPUMatPtr&& ptr) noexcept {
#ifndef NDEBUG
        if(this->ptr) {
            cmn::print("Destroyed buffer ",loc.file_name(),"::",loc.function_name(),":",loc.line());
        }
        
        //print("Moving buffer ",ptr.loc.file_name(),"::",ptr.loc.function_name(),":",ptr.loc.line());
        this->loc = ptr.loc;
#endif
        this->ptr = std::move(ptr.ptr);
        return *this;
    }
    
    static GPUMatPtr Make(cmn::source_location loc) {
        //print("Created buffer ", loc.file_name(),"::",loc.function_name(),":",loc.line());
        GPUMatPtr ptr;
        ptr.ptr = std::make_unique<useMat_t>();
#ifndef NDEBUG
        if(not ptr.ptr)
			throw cmn::U_EXCEPTION("Failed to allocate buffer");

        ptr.loc = loc;
        
        static std::atomic<size_t> counter{0u};
        ++counter;
        if(counter % 100 == 0)
            cmn::thread_print("Counted ", counter.load());
#else
        UNUSED(loc);
#endif
        return ptr;
    }
    
    
#ifndef NDEBUG
    ~GPUMatPtr() {
        if(ptr)
            cmn::print("Destroyed buffer ",loc.file_name(),"::",loc.function_name(),":",loc.line());
    }
#endif
    
    /*operator useMat_t&() {
        return *ptr;
    }*/
    
    useMat_t* get() const noexcept {
        return ptr.get();
    }
    
    useMat_t& operator*() {
        return *ptr;
    }
    
    useMat_t* operator->() const noexcept {
        return ptr.get();
    }
    
    operator bool() const noexcept {
        return ptr != nullptr;
    }
};

#define MAKE_GPU_MAT std::make_unique<useMat_t>() // GPUMatPtr::Make(cmn::source_location::current())
#define MAKE_GPU_MAT_LOC(loc) std::make_unique<useMat_t>() // GPUMatPtr::Make(loc)

using useMatPtr_t = std::unique_ptr<useMat_t>;//GPUMatPtr;
