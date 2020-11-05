#pragma once

#pragma warning(push, 0)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wall"
#pragma clang diagnostic ignored "-Wextra"
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wimplicit-int-conversion"
#pragma clang diagnostic ignored "-Wimplicit-float-conversion"
#pragma clang diagnostic ignored "-Wfloat-conversion"

#ifdef WIN32
#include <windows.h>
#endif

#include <ctime>
#include <iomanip>
#include <locale>
#include <type_traits>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <map>
#include <sstream>
#include <cstdlib>
#include <algorithm>
#include <thread>
#include <functional>
#include <memory>
#include <chrono>
#include <set>
#include <unordered_set>
#include <queue>
#include <stdexcept>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <future>
#include <ostream>
#include <array>
#include <cstdint>
#include <exception>
#include <stdarg.h>

#ifdef WIN32
#define _USE_MATH_DEFINES
#include <cmath>

#if (_MSC_VER <= 1916)
    // visual studio 2017 does not have __builtin_clzl
    #include <intrin.h>

    static inline int __builtin_clz(unsigned x) {
        return (int)__lzcnt(x);
    }

    static inline int __builtin_clzll(unsigned long long x) {
        return (int)__lzcnt64(x);
    }

    static inline int __builtin_clzl(unsigned long x) {
        return sizeof(x) == 8 ? __builtin_clzll(x) : __builtin_clz((uint32_t)x);
    }
#endif

#endif

typedef int32_t long_t;

#ifndef _MSC_VER
#define CV_STATIC_ANALYSIS 0
#define CV_ErrorNoReturn(code, msg) cv::errorNoReturn( code, msg, CV_Func, __FILE__, __LINE__ )

#include <opencv2/opencv.hpp>
#endif

#include <commons/common/cpputils/cpputils.h>
#include <commons/common/cpputils/debug/Debug.h>
#include <commons/common/cpputils/utilsexception.h>
#include <commons/common/cpputils/debug/Printable.h>
#include <cnpy.h>

// Code that produces warnings...
#pragma clang diagnostic pop
#pragma warning(pop)
